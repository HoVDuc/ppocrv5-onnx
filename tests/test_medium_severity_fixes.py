from __future__ import annotations

import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import ppocrv5_onnx.config as config_module
from ppocrv5_onnx.config import ModelPreset
from ppocrv5_onnx.pipeline import OCRPipeline
from ppocrv5_onnx.utils import (
    DEFAULT_CUDA_PROVIDER_OPTIONS,
    build_providers,
    create_session,
)


# --- ONNX Runtime providers / session options ---------------------------------


def test_build_providers_defaults_to_cpu() -> None:
    assert build_providers(None) == ["CPUExecutionProvider"]


def test_build_providers_applies_safe_cuda_defaults() -> None:
    providers = build_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert providers == [
        ("CUDAExecutionProvider", DEFAULT_CUDA_PROVIDER_OPTIONS),
        "CPUExecutionProvider",
    ]
    assert providers[0][1] is not DEFAULT_CUDA_PROVIDER_OPTIONS


def test_build_providers_keeps_explicit_cuda_options() -> None:
    explicit = ("CUDAExecutionProvider", {"gpu_mem_limit": 1024})

    assert build_providers([explicit]) == [explicit]


def test_create_session_passes_sess_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object, object]] = []

    def fake_session(path: str, sess_options: object = None, providers: object = None):
        calls.append((path, sess_options, providers))
        return "session"

    monkeypatch.setattr("ppocrv5_onnx.utils.ort.InferenceSession", fake_session)
    options = object()

    assert create_session(Path("m.onnx"), ["CPUExecutionProvider"], options) == "session"
    assert calls == [("m.onnx", options, ["CPUExecutionProvider"])]


# --- Input colour order --------------------------------------------------------


class _RecordingDetector:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def detect(self, image: np.ndarray) -> tuple[list[np.ndarray], list[list[float]]]:
        self.images.append(image)
        return [np.zeros((0, 4, 2), dtype=np.int16)], [[]]


def _stub_pipeline() -> OCRPipeline:
    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.detector = _RecordingDetector()
    pipeline.recognizer = None
    pipeline.visualizer = None
    pipeline.config = None
    return pipeline


BGR_PIXEL = (10, 120, 250)  # blue, green, red


def test_path_and_cv2_array_inputs_reach_detector_identically(tmp_path: Path) -> None:
    """A file path and the same image loaded with cv2.imread give identical model input."""
    image_path = tmp_path / "img.png"
    cv2.imwrite(str(image_path), np.full((8, 8, 3), BGR_PIXEL, np.uint8))
    pipeline = _stub_pipeline()

    pipeline(str(image_path))
    pipeline(image_path)
    pipeline(cv2.imread(str(image_path)))

    first, second, third = pipeline.detector.images
    assert tuple(first[0, 0]) == BGR_PIXEL
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)


def test_pil_rgb_input_is_converted_to_bgr() -> None:
    pipeline = _stub_pipeline()
    rgb_pixel = BGR_PIXEL[::-1]

    pipeline(Image.new("RGB", (8, 8), rgb_pixel))
    pipeline(Image.new("RGBA", (8, 8), (*rgb_pixel, 128)))

    for image in pipeline.detector.images:
        assert tuple(image[0, 0]) == BGR_PIXEL


def test_bgra_array_keeps_channel_order() -> None:
    pipeline = _stub_pipeline()

    pipeline(np.full((8, 8, 4), (*BGR_PIXEL, 255), np.uint8))

    assert tuple(pipeline.detector.images[0][0, 0]) == BGR_PIXEL


def test_unreadable_path_error_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"

    with pytest.raises(ValueError, match="missing.png"):
        _stub_pipeline()(str(missing))


# --- Model download ------------------------------------------------------------


PRESET = ModelPreset(model_dir="Fake_det", release_tag="v0", zip_name="Fake_det.zip")


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models = tmp_path / "models"
    monkeypatch.setattr(config_module, "MODELS_DIR", models)
    return models


def _serve_zip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, members: dict[str, bytes]
) -> None:
    source = tmp_path / "source.zip"
    _make_zip(source, members)

    def fake_urlretrieve(url: str, filename: Path) -> None:
        Path(filename).write_bytes(source.read_bytes())

    monkeypatch.setattr(config_module.urllib.request, "urlretrieve", fake_urlretrieve)


def _cache_entries(models_dir: Path) -> list[str]:
    """Cache contents, ignoring the per-model lock files that are kept on purpose."""
    return sorted(p.name for p in models_dir.iterdir() if p.suffix != ".lock")


VALID_MEMBERS = {
    "Fake_det/inference.onnx": b"onnx",
    "Fake_det/inference.yml": b"Global: {}\n",
}


def test_download_places_model_and_leaves_no_staging(
    models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_zip(monkeypatch, tmp_path, VALID_MEMBERS)

    config_module._download_if_missing(PRESET)

    assert PRESET.model_path.read_bytes() == b"onnx"
    assert _cache_entries(models_dir) == ["Fake_det"]


def test_download_replaces_incomplete_leftover(
    models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    PRESET.model_dir_path.mkdir(parents=True)
    PRESET.model_path.write_bytes(b"trunc")  # no inference.yml -> incomplete
    _serve_zip(monkeypatch, tmp_path, VALID_MEMBERS)

    config_module._download_if_missing(PRESET)

    assert PRESET.model_path.read_bytes() == b"onnx"
    assert (PRESET.model_dir_path / "inference.yml").exists()


def test_interrupted_download_leaves_cache_untouched(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_urlretrieve(url: str, filename: Path) -> None:
        Path(filename).write_bytes(b"partial")
        raise ConnectionError("network dropped")

    monkeypatch.setattr(config_module.urllib.request, "urlretrieve", failing_urlretrieve)

    with pytest.raises(ConnectionError):
        config_module._download_if_missing(PRESET)

    assert _cache_entries(models_dir) == []


def test_zip_without_model_files_raises_and_leaves_cache_untouched(
    models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_zip(monkeypatch, tmp_path, {"Fake_det/inference.yml": b"Global: {}\n"})

    with pytest.raises(FileNotFoundError, match="does not contain"):
        config_module._download_if_missing(PRESET)

    assert _cache_entries(models_dir) == []


def test_concurrent_downloads_install_once_and_never_remove_a_complete_model(
    models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlapping downloads (with an incomplete leftover in place) are serialised.

    Only one worker downloads; the others reuse its result, and once the model is
    complete it never disappears again, so a pipeline loading it cannot fail.
    """
    PRESET.model_dir_path.mkdir(parents=True)
    PRESET.model_path.write_bytes(b"trunc")  # incomplete leftover
    source = tmp_path / "source.zip"
    _make_zip(source, VALID_MEMBERS)
    downloads: list[int] = []

    def slow_urlretrieve(url: str, filename: Path) -> None:
        downloads.append(threading.get_ident())
        time.sleep(0.2)
        Path(filename).write_bytes(source.read_bytes())

    monkeypatch.setattr(config_module.urllib.request, "urlretrieve", slow_urlretrieve)

    done = threading.Event()
    regressions: list[str] = []

    def watch() -> None:
        seen_complete = False
        while not done.is_set():
            complete = config_module._is_complete(PRESET)
            if seen_complete and not complete:
                regressions.append("complete model disappeared")
            seen_complete = seen_complete or complete

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            config_module._download_if_missing(PRESET)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    watcher = threading.Thread(target=watch)
    watcher.start()
    workers = [threading.Thread(target=worker) for _ in range(4)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    done.set()
    watcher.join()

    assert errors == []
    assert len(downloads) == 1
    assert regressions == []
    assert PRESET.model_path.read_bytes() == b"onnx"
    assert _cache_entries(models_dir) == ["Fake_det"]
