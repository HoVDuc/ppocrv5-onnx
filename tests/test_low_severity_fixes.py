from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from ppocrv5_onnx.pipeline import OCRPipeline
from ppocrv5_onnx.text_detector.postprocess import DBPostProcess


def _prob_map_with_box(height: int, width: int) -> np.ndarray:
    """Return a (1, 1, H, W) probability map containing one text-like region."""
    pred = np.zeros((1, 1, height, width), dtype=np.float32)
    pred[0, 0, 10:30, 10:120] = 0.9
    return pred


def test_boxes_use_int32_for_large_images() -> None:
    """Box coordinates beyond the int16 range are preserved."""
    post = DBPostProcess(thresh=0.3, box_thresh=0.6, unclip_ratio=1.5)
    # Original image is 1000x larger than the probability map.
    img_shape = np.array([40_000, 160_000, 1.0, 1.0])

    boxes, _ = post([_prob_map_with_box(40, 160)], [img_shape])

    assert boxes[0].dtype == np.int32
    assert boxes[0][..., 0].max() > np.iinfo(np.int16).max


def test_explicit_zero_threshold_is_respected() -> None:
    """thresh=0 / box_thresh=0 overrides are not replaced by the defaults."""
    post = DBPostProcess(thresh=0.95, box_thresh=0.95)
    pred = _prob_map_with_box(40, 160)
    img_shape = np.array([40, 160, 1.0, 1.0])

    default_boxes, _ = post([pred], [img_shape])
    zero_boxes, _ = post([pred], [img_shape], thresh=0.0, box_thresh=0.0)

    assert len(default_boxes[0]) == 0
    assert len(zero_boxes[0]) > 0


def test_unsupported_box_type_fails_early() -> None:
    with pytest.raises(ValueError, match="box_type='quad'"):
        DBPostProcess(box_type="poly")


def test_postprocess_does_not_modify_model_output() -> None:
    post = DBPostProcess()
    pred = _prob_map_with_box(40, 160)
    original = pred.copy()

    post([pred], [np.array([40, 160, 1.0, 1.0])])

    np.testing.assert_array_equal(pred, original)


class _OneBoxDetector:
    class crop_poly:
        @staticmethod
        def get_minarea_rect_crop(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
            return image

    def detect(self, image: np.ndarray):
        return [np.array([[[0, 0], [4, 0], [4, 4], [0, 4]]], dtype=np.int32)], [[1.0]]


class _FakeRecognizer:
    def recognize(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        return [("text", 1.0)] * len(images)


class _FakeVisualizer:
    def draw(self, image: np.ndarray, results: list) -> dict:
        return {"ocr_res_img": Image.new("RGB", (4, 4))}


def _visualizing_pipeline(save_dir: Path) -> OCRPipeline:
    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.detector = _OneBoxDetector()
    pipeline.recognizer = _FakeRecognizer()
    pipeline.visualizer = _FakeVisualizer()
    pipeline.config = SimpleNamespace(visualize=SimpleNamespace(save_dir=str(save_dir)))
    return pipeline


def test_visualization_does_not_overwrite_previous_results(tmp_path: Path) -> None:
    save_dir = tmp_path / "out"
    first, second = tmp_path / "first.png", tmp_path / "second.jpg"
    for path in (first, second):
        cv2.imwrite(str(path), np.full((8, 8, 3), 255, np.uint8))
    pipeline = _visualizing_pipeline(save_dir)

    pipeline(str(first))
    pipeline(second)
    pipeline(np.zeros((8, 8, 3), np.uint8))
    pipeline(np.zeros((8, 8, 3), np.uint8))
    pipeline(str(first))  # re-running the same file replaces its own result

    assert sorted(p.name for p in save_dir.iterdir()) == [
        "first_ocr_result.png",
        "ocr_result.png",
        "ocr_result_1.png",
        "second_ocr_result.png",
    ]
