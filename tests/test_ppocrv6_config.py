from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ppocrv5_onnx import OCRPipeline
from ppocrv5_onnx.config import OCRConfig
from ppocrv5_onnx.inference_config import CHARACTER_DICT_FILENAME

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODELS = PROJECT_ROOT / "ppocrv5_onnx/models"
V6_MODELS = PROJECT_ROOT / "model"
V6_DET_MODEL = V6_MODELS / "PP-OCRv6_medium_det_onnx/inference.onnx"
V6_REC_MODEL = V6_MODELS / "PP-OCRv6_medium_rec_onnx/inference.onnx"
DEMO_IMAGE = PROJECT_ROOT / "img/demo.png"


@pytest.fixture(autouse=True)
def disable_model_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep config tests deterministic and offline."""
    monkeypatch.setattr("ppocrv5_onnx.config._download_if_missing", lambda _preset: None)


def test_ppocrv6_dict_packaged(tmp_path: Path) -> None:
    """Install to a target dir and verify the PP-OCRv6 dictionary is present."""
    install_dir = tmp_path / "install"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            ".",
            "--no-deps",
            "--target",
            str(install_dir),
            "--force-reinstall",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    dict_path = install_dir / "ppocrv5_onnx/data/dict/ppocrv6_dict.txt"
    assert dict_path.exists()
    assert len(dict_path.read_text(encoding="utf-8").splitlines()) == 18708


@pytest.mark.skipif(
    not (
        (PACKAGE_MODELS / "PP-OCRv5_mobile_det/inference.onnx").exists()
        and V6_REC_MODEL.exists()
    ),
    reason="Packaged v5 detector and local PP-OCRv6 recognizer are required.",
)
def test_from_mix_yaml(unified_models_dir: Path) -> None:
    """YAML det_model/rec_model entries resolve through the preset registry."""
    _ = unified_models_dir
    config = OCRConfig.from_yaml(PROJECT_ROOT / "config_mix_v5det_v6rec.yaml")

    assert config.det.path.endswith("PP-OCRv5_mobile_det/inference.onnx")
    assert config.det.thresh == 0.3
    assert config.rec.path.endswith("PP-OCRv6_medium_rec_onnx/inference.onnx")
    assert config.rec.input_shape == [3, 48, 320]
    assert Path(config.rec.dict_path).name == CHARACTER_DICT_FILENAME


def test_full_yaml_override_does_not_require_preset_downloads() -> None:
    """Detailed YAML model blocks override presets without resolver side effects."""
    config = OCRConfig.from_yaml(PROJECT_ROOT / "config_ppocrv6.yaml")

    assert config.det.path == "./model/PP-OCRv6_medium_det_onnx/inference.onnx"
    assert config.det.thresh == 0.2
    assert config.rec.path == "./model/PP-OCRv6_medium_rec_onnx/inference.onnx"
    assert config.rec.input_shape == [3, 48, 320]
    assert config.rec.dict_path == "./ppocrv5_onnx/data/dict/ppocrv6_dict.txt"


def test_from_model_paths_detects_v6_rec_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct path factory defaults to the v6 dict and shape for v6 rec paths."""
    package_dir = PROJECT_ROOT / "ppocrv5_onnx"

    class FakeComponent:
        """Minimal component stub that avoids ONNX Runtime session creation."""

        def __init__(self, config: object, providers: list[str] | None = None) -> None:
            self.config = config
            self.providers = providers

    monkeypatch.setattr("ppocrv5_onnx.pipeline.Detector", FakeComponent)
    monkeypatch.setattr("ppocrv5_onnx.pipeline.Recognizer", FakeComponent)

    config_pipeline = OCRPipeline.from_model_paths(
        det_model_path=str(V6_DET_MODEL),
        rec_model_path=str(V6_REC_MODEL),
    )

    assert config_pipeline.config.rec.input_shape == [3, 48, 320]
    assert config_pipeline.config.rec.dict_path == str(
        package_dir / "data/dict/ppocrv6_dict.txt"
    )


def test_pipeline_uses_safe_default_recognition_batch_size() -> None:
    """Default recognition batching avoids large all-crops inference batches."""

    class FakeCropPoly:
        def get_minarea_rect_crop(
            self,
            image: np.ndarray,
            pts: np.ndarray,
        ) -> np.ndarray:
            return image

    class FakeDetector:
        crop_poly = FakeCropPoly()

        def detect(self, image: np.ndarray) -> list[list[np.ndarray]]:
            points = np.zeros((10, 4, 2), dtype=np.int16)
            return [[points]]

    class FakeRecognizer:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def recognize(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
            self.batch_sizes.append(len(images))
            return [("text", 1.0)] * len(images)

    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.detector = FakeDetector()
    pipeline.recognizer = FakeRecognizer()
    pipeline.visualizer = None
    pipeline.config = None

    results = pipeline(np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(results) == 10
    assert pipeline.recognizer.batch_sizes == [8, 2]


def test_pipeline_rejects_invalid_recognition_batch_size() -> None:
    """Invalid recognition batch sizes fail before model inference."""
    pipeline = OCRPipeline.__new__(OCRPipeline)

    with pytest.raises(ValueError, match="rec_batch_size must be >= 1"):
        pipeline(np.zeros((8, 8, 3), dtype=np.uint8), rec_batch_size=0)


@pytest.mark.skipif(
    os.environ.get("RUN_OCR_SMOKE") != "1",
    reason="Set RUN_OCR_SMOKE=1 to run local ONNX inference smoke tests.",
)
@pytest.mark.skipif(
    not (V6_DET_MODEL.exists() and V6_REC_MODEL.exists() and DEMO_IMAGE.exists()),
    reason="Local PP-OCRv6 models or demo image are not available.",
)
def test_ppocrv6_inference_smoke() -> None:
    """Run a local PP-OCRv6 end-to-end smoke test when explicitly enabled."""
    pipeline = OCRPipeline.from_config(PROJECT_ROOT / "config_ppocrv6.yaml")
    results = pipeline(str(DEMO_IMAGE))

    assert len(results) > 0
