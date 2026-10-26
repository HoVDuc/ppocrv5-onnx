from __future__ import annotations


import numpy as np
import pytest
from PIL import Image

from ppocrv5_onnx.config import DetectorConfig, OCRConfig, RecognizerConfig
from ppocrv5_onnx.pipeline import OCRPipeline
from ppocrv5_onnx.schema import OCRResult


def test_ocr_result_xyxy_and_to_dict_with_ndarray_point() -> None:
    """OCRResult helpers work with the ndarray points produced by the pipeline."""
    point = np.array([[10, 20], [50, 22], [49, 40], [9, 38]], dtype=np.int16)
    result = OCRResult(text="abc", score=0.9, point=point)

    assert result.xyxy() == (9, 20, 50, 40)
    assert result.to_dict() == {"text": "abc", "score": 0.9, "box": point.tolist()}


def test_ocr_result_xyxy_with_list_point() -> None:
    """OCRResult.xyxy still accepts list-based points."""
    result = OCRResult(text="abc", score=0.9, point=[[1, 2], [5, 2], [5, 7], [1, 7]])

    assert result.xyxy() == (1, 2, 5, 7)


class _RecordingDetector:
    """Detector stub that records the image shape and returns no boxes."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def detect(self, image: np.ndarray) -> tuple[list[np.ndarray], list[list[float]]]:
        self.shapes.append(image.shape)
        return [np.zeros((0, 4, 2), dtype=np.int16)], [[]]


def _stub_pipeline() -> OCRPipeline:
    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.detector = _RecordingDetector()
    pipeline.recognizer = None
    pipeline.visualizer = None
    pipeline.config = None
    return pipeline


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((16, 24), dtype=np.uint8),
        np.zeros((16, 24, 1), dtype=np.uint8),
        np.zeros((16, 24, 4), dtype=np.uint8),
        Image.new("L", (24, 16)),
        Image.new("RGBA", (24, 16)),
    ],
    ids=["gray", "gray-1ch", "rgba", "pil-L", "pil-RGBA"],
)
def test_pipeline_converts_inputs_to_three_channels(image: object) -> None:
    """Grayscale and alpha inputs reach the detector as HxWx3 arrays."""
    pipeline = _stub_pipeline()

    assert pipeline(image) == []
    assert pipeline.detector.shapes == [(16, 24, 3)]


def test_pipeline_rejects_unsupported_channel_count() -> None:
    """Unexpected channel counts fail with a clear error."""
    with pytest.raises(ValueError, match="Unsupported image shape"):
        _stub_pipeline()(np.zeros((16, 24, 2), dtype=np.uint8))


@pytest.fixture
def fake_presets(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Replace preset resolution with in-memory configs and record calls."""
    calls: list[tuple[str, str]] = []

    def fake_det(name: str) -> DetectorConfig:
        calls.append(("det", name))
        return DetectorConfig(path=f"/models/{name}_det.onnx", thresh=0.3)

    def fake_rec(name: str) -> RecognizerConfig:
        calls.append(("rec", name))
        return RecognizerConfig(
            path=f"/models/{name}_rec.onnx",
            input_shape=[3, 48, 320],
            dict_path=f"/models/{name}/character_dict.txt",
        )

    monkeypatch.setattr("ppocrv5_onnx.config._resolve_det_preset", fake_det)
    monkeypatch.setattr("ppocrv5_onnx.config._resolve_rec_preset", fake_rec)
    return calls


def test_partial_yaml_override_merges_onto_preset(
    fake_presets: list[tuple[str, str]],
) -> None:
    """A model block without a path keeps preset values and overrides the rest."""
    config = OCRConfig.from_dict(
        {
            "engine": {
                "det_model": "ppocrv6_medium",
                "model": {"det": {"thresh": 0.2}, "rec": {"input_shape": [3, 32, 320]}},
            }
        }
    )

    assert config.det.path == "/models/ppocrv6_medium_det.onnx"
    assert config.det.thresh == 0.2
    assert config.rec.path == "/models/ppocrv6_medium_rec.onnx"
    assert config.rec.input_shape == [3, 32, 320]
    assert config.rec.dict_path == "/models/ppocrv6_medium/character_dict.txt"
    assert fake_presets == [("det", "ppocrv6_medium"), ("rec", "ppocrv6_medium")]


def test_yaml_override_with_path_skips_preset(
    fake_presets: list[tuple[str, str]],
) -> None:
    """A model block with a path is used as-is without resolving the preset."""
    config = OCRConfig.from_dict(
        {"engine": {"model": {"det": {"path": "./det.onnx", "thresh": 0.25}}}}
    )

    assert config.det == DetectorConfig(path="./det.onnx", thresh=0.25)
    assert fake_presets == [("rec", "ppocrv5_mobile")]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"model_name": "ppocrv6_medium", "det_model": "ppocrv5_mobile"},
            ("ppocrv5_mobile", "ppocrv6_medium"),
        ),
        (
            {"model_name": "ppocrv6_medium", "rec_model": "ppocrv5_server"},
            ("ppocrv6_medium", "ppocrv5_server"),
        ),
        (
            {"det_model": "ppocrv6_small", "rec_model": "ppocrv6_tiny"},
            ("ppocrv6_small", "ppocrv6_tiny"),
        ),
    ],
)
def test_from_pretrained_mixed_falls_back_to_model_name(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, str],
    expected: tuple[str, str],
) -> None:
    """The component not given explicitly comes from model_name, not the other component."""
    seen: list[tuple[str, str]] = []

    def fake_from_mix(*, det_model: str, rec_model: str) -> str:
        seen.append((det_model, rec_model))
        return "config"

    monkeypatch.setattr(OCRConfig, "from_mix", staticmethod(fake_from_mix))
    monkeypatch.setattr(OCRPipeline, "__init__", lambda self, *args: None)

    OCRPipeline.from_pretrained(**kwargs)

    assert seen == [expected]
