from __future__ import annotations

import shutil
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
    assert sorted(p.name for p in models_dir.iterdir()) == ["Fake_det"]


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

    assert list(models_dir.iterdir()) == []


def test_zip_without_model_files_raises_and_leaves_cache_untouched(
    models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_zip(monkeypatch, tmp_path, {"Fake_det/inference.yml": b"Global: {}\n"})

    with pytest.raises(FileNotFoundError, match="does not contain"):
        config_module._download_if_missing(PRESET)

    assert list(models_dir.iterdir()) == []


def test_download_keeps_copy_finished_by_another_process(
    models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def concurrent_urlretrieve(url: str, filename: Path) -> None:
        # Simulate another process completing the same model meanwhile.
        PRESET.model_dir_path.mkdir(parents=True)
        PRESET.model_path.write_bytes(b"other")
        (PRESET.model_dir_path / "inference.yml").write_text("Global: {}\n")
        _make_zip(Path(filename), VALID_MEMBERS)

    monkeypatch.setattr(
        config_module.urllib.request, "urlretrieve", concurrent_urlretrieve
    )

    config_module._download_if_missing(PRESET)

    assert PRESET.model_path.read_bytes() == b"other"
    assert sorted(p.name for p in models_dir.iterdir()) == ["Fake_det"]


def test_download_never_deletes_a_model_another_process_just_completed(
    models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another process may place a complete model right after any completeness check."""
    _serve_zip(monkeypatch, tmp_path, VALID_MEMBERS)
    real_is_complete = config_module._is_complete
    calls = {"n": 0}

    def is_complete_then_other_process_finishes(preset: ModelPreset) -> bool:
        result = real_is_complete(preset)
        calls["n"] += 1
        if calls["n"] == 2 and not PRESET.model_dir_path.exists():
            PRESET.model_dir_path.mkdir(parents=True)
            PRESET.model_path.write_bytes(b"other")
            (PRESET.model_dir_path / "inference.yml").write_text("Global: {}\n")
        return result

    deleted_complete: list[Path] = []
    real_rmtree = shutil.rmtree

    def recording_rmtree(path, *args, **kwargs):
        if Path(path) == PRESET.model_dir_path and real_is_complete(PRESET):
            deleted_complete.append(Path(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(config_module, "_is_complete", is_complete_then_other_process_finishes)
    monkeypatch.setattr(shutil, "rmtree", recording_rmtree)

    config_module._download_if_missing(PRESET)

    assert deleted_complete == []
    assert real_is_complete(PRESET)
    assert sorted(p.name for p in models_dir.iterdir()) == ["Fake_det"]
