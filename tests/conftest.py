from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODELS = PROJECT_ROOT / "ppocrv5_onnx/models"
LOCAL_V6_MODELS = PROJECT_ROOT / "model"

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def unified_models_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Build a single models cache containing packaged v5 and local v6 artifacts."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    for model_name in ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"):
        source = PACKAGE_MODELS / model_name
        if source.exists():
            shutil.copytree(source, models_dir / model_name)

    for model_name in (
        "PP-OCRv6_medium_det_onnx",
        "PP-OCRv6_medium_rec_onnx",
    ):
        source = LOCAL_V6_MODELS / model_name
        if source.exists():
            shutil.copytree(source, models_dir / model_name)

    monkeypatch.setattr("ppocrv5_onnx.config.MODELS_DIR", models_dir)
    monkeypatch.setattr(
        "ppocrv5_onnx.config._download_if_missing",
        lambda _preset: None,
    )
    return models_dir