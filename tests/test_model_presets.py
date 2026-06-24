from __future__ import annotations

from pathlib import Path

import pytest

from ppocrv5_onnx.config import (
    DET_PRESETS,
    OCRConfig,
    REC_PRESETS,
    list_det_presets,
    list_rec_presets,
)
from ppocrv5_onnx.inference_config import (
    CHARACTER_DICT_FILENAME,
    detector_config_from_inference_yml,
    recognizer_config_from_inference_yml,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODELS = PROJECT_ROOT / "ppocrv5_onnx/models"
LOCAL_V6_MODELS = PROJECT_ROOT / "model"


@pytest.fixture(autouse=True)
def disable_model_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep preset tests offline."""
    monkeypatch.setattr("ppocrv5_onnx.config._download_if_missing", lambda _preset: None)


def test_all_det_presets_have_release_urls() -> None:
    """Every detector preset maps to a GitHub release asset."""
    assert list_det_presets() == sorted(DET_PRESETS)
    for name, preset in DET_PRESETS.items():
        assert preset.download_url.endswith(preset.zip_name), name
        assert preset.release_tag in {"v1.0.0", "v1.1.0"}, name


def test_all_rec_presets_have_release_urls() -> None:
    """Every recognizer preset maps to a GitHub release asset."""
    assert list_rec_presets() == sorted(REC_PRESETS)
    for name, preset in REC_PRESETS.items():
        assert preset.download_url.endswith(preset.zip_name), name
        assert preset.release_tag in {"v1.0.0", "v1.1.0"}, name


def test_v5_mobile_detector_config_from_inference_yml() -> None:
    """Detector defaults are parsed from the bundled v5 mobile inference.yml."""
    model_dir = PACKAGE_MODELS / "PP-OCRv5_mobile_det"
    onnx_path = model_dir / "inference.onnx"
    config = detector_config_from_inference_yml(model_dir, onnx_path=onnx_path)

    assert config.path.endswith("PP-OCRv5_mobile_det/inference.onnx")
    assert config.resize_long == 960
    assert config.thresh == 0.3
    assert config.box_thresh == 0.6
    assert config.unclip_ratio == 1.5
    assert config.max_candidates == 1000


def test_v5_mobile_recognizer_config_from_inference_yml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recognizer defaults are parsed from the bundled v5 mobile inference.yml."""
    model_dir = tmp_path / "PP-OCRv5_mobile_rec"
    source_dir = PACKAGE_MODELS / "PP-OCRv5_mobile_rec"
    model_dir.mkdir()
    (model_dir / "inference.yml").write_text(
        (source_dir / "inference.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    onnx_path = model_dir / "inference.onnx"
    onnx_path.write_bytes(b"onnx")

    config = recognizer_config_from_inference_yml(model_dir, onnx_path=onnx_path)

    assert config.input_shape == [3, 32, 128]
    assert config.dict_path.endswith(CHARACTER_DICT_FILENAME)
    assert Path(config.dict_path).exists()
    assert len(Path(config.dict_path).read_text(encoding="utf-8").splitlines()) == 18383


@pytest.mark.skipif(
    not (LOCAL_V6_MODELS / "PP-OCRv6_medium_det_onnx/inference.yml").exists(),
    reason="Local PP-OCRv6 medium detector yml is not available.",
)
def test_v6_medium_detector_config_from_inference_yml() -> None:
    """Detector defaults for PP-OCRv6 medium come from inference.yml."""
    model_dir = LOCAL_V6_MODELS / "PP-OCRv6_medium_det_onnx"
    config = detector_config_from_inference_yml(
        model_dir,
        onnx_path=model_dir / "inference.onnx",
    )

    assert config.thresh == 0.2
    assert config.box_thresh == 0.45
    assert config.unclip_ratio == 1.4
    assert config.max_candidates == 3000


@pytest.mark.skipif(
    not (LOCAL_V6_MODELS / "PP-OCRv6_medium_rec_onnx/inference.yml").exists(),
    reason="Local PP-OCRv6 medium recognizer yml is not available.",
)
def test_v6_medium_recognizer_config_from_inference_yml(
    tmp_path: Path,
) -> None:
    """Recognizer defaults for PP-OCRv6 medium come from inference.yml."""
    source_dir = LOCAL_V6_MODELS / "PP-OCRv6_medium_rec_onnx"
    model_dir = tmp_path / source_dir.name
    model_dir.mkdir()
    (model_dir / "inference.yml").write_text(
        (source_dir / "inference.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    onnx_path = model_dir / "inference.onnx"
    onnx_path.write_bytes(b"onnx")

    config = recognizer_config_from_inference_yml(model_dir, onnx_path=onnx_path)

    assert config.input_shape == [3, 48, 320]
    assert Path(config.dict_path).name == CHARACTER_DICT_FILENAME
    assert len(Path(config.dict_path).read_text(encoding="utf-8").splitlines()) == 18708


def test_from_mix_v5_mobile_uses_inference_yml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preset resolution for v5 mobile reads config from packaged inference.yml."""
    monkeypatch.setattr("ppocrv5_onnx.config.MODELS_DIR", PACKAGE_MODELS)

    config = OCRConfig.from_mix(det_model="mobile", rec_model="mobile")

    assert config.det.path.endswith("PP-OCRv5_mobile_det/inference.onnx")
    assert config.det.resize_long == 960
    assert config.rec.path.endswith("PP-OCRv5_mobile_rec/inference.onnx")
    assert config.rec.input_shape == [3, 32, 128]
    assert Path(config.rec.dict_path).name == CHARACTER_DICT_FILENAME


@pytest.mark.skipif(
    not (LOCAL_V6_MODELS / "PP-OCRv6_medium_det_onnx/inference.onnx").exists(),
    reason="Local PP-OCRv6 medium models are not available.",
)
def test_from_pretrained_ppocrv6_medium_uses_inference_yml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PP-OCRv6 medium preset resolves paths and yml-driven defaults."""
    monkeypatch.setattr("ppocrv5_onnx.config.MODELS_DIR", LOCAL_V6_MODELS)

    config = OCRConfig.from_pretrained("ppocrv6_medium")

    assert config.det.path.endswith("PP-OCRv6_medium_det_onnx/inference.onnx")
    assert config.det.thresh == 0.2
    assert config.rec.path.endswith("PP-OCRv6_medium_rec_onnx/inference.onnx")
    assert config.rec.input_shape == [3, 48, 320]
    assert Path(config.rec.dict_path).name == CHARACTER_DICT_FILENAME


def test_from_mix_unknown_preset_raises() -> None:
    """Unknown presets fail with a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown detector preset"):
        OCRConfig.from_mix(det_model="missing", rec_model="ppocrv6_medium")

    with pytest.raises(ValueError, match="Unknown recognizer preset"):
        OCRConfig.from_mix(det_model="ppocrv5_mobile", rec_model="missing")


@pytest.mark.skipif(
    not (
        (PACKAGE_MODELS / "PP-OCRv5_mobile_det/inference.onnx").exists()
        and (LOCAL_V6_MODELS / "PP-OCRv6_medium_rec_onnx/inference.onnx").exists()
    ),
    reason="Packaged v5 detector and local PP-OCRv6 recognizer are required.",
)
def test_from_mix_v5det_v6rec(unified_models_dir: Path) -> None:
    """Mixing presets resolves independent model directories."""
    _ = unified_models_dir
    config = OCRConfig.from_mix(det_model="mobile", rec_model="ppocrv6_medium")

    assert config.det.path.endswith("PP-OCRv5_mobile_det/inference.onnx")
    assert config.rec.path.endswith("PP-OCRv6_medium_rec_onnx/inference.onnx")