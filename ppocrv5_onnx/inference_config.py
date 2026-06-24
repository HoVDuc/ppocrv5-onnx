from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CHARACTER_DICT_FILENAME = "character_dict.txt"
DEFAULT_RESIZE_LONG = 960
DEFAULT_DET_POSTPROCESS: dict[str, float | int] = {
    "thresh": 0.3,
    "box_thresh": 0.6,
    "unclip_ratio": 1.5,
    "max_candidates": 1000,
}
DEFAULT_REC_INPUT_SHAPE = [3, 32, 320]


def load_inference_yaml(model_dir: Path) -> dict[str, Any]:
    """Load the PaddleOCR inference YAML shipped with an ONNX model."""
    yaml_path = model_dir / "inference.yml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing inference.yml in {model_dir}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid inference.yml payload in {yaml_path}")
    return payload


def detector_config_from_inference_yml(
    model_dir: Path,
    *,
    onnx_path: Path,
) -> "DetectorConfig":
    """Build detector config defaults from the bundled inference.yml."""
    from .config import DetectorConfig

    payload = load_inference_yaml(model_dir)
    postprocess = payload.get("PostProcess", {})
    if not isinstance(postprocess, dict):
        postprocess = {}

    return DetectorConfig(
        path=str(onnx_path),
        resize_long=_extract_resize_long(payload),
        thresh=float(postprocess.get("thresh", DEFAULT_DET_POSTPROCESS["thresh"])),
        box_thresh=float(
            postprocess.get("box_thresh", DEFAULT_DET_POSTPROCESS["box_thresh"])
        ),
        unclip_ratio=float(
            postprocess.get("unclip_ratio", DEFAULT_DET_POSTPROCESS["unclip_ratio"])
        ),
        max_candidates=int(
            postprocess.get(
                "max_candidates", DEFAULT_DET_POSTPROCESS["max_candidates"]
            )
        ),
    )


def recognizer_config_from_inference_yml(
    model_dir: Path,
    *,
    onnx_path: Path,
    packaged_dict_paths: dict[str, Path] | None = None,
) -> "RecognizerConfig":
    """Build recognizer config defaults from the bundled inference.yml."""
    from .config import RecognizerConfig

    payload = load_inference_yaml(model_dir)
    return RecognizerConfig(
        path=str(onnx_path),
        input_shape=_extract_rec_image_shape(payload),
        dict_path=str(
            _resolve_character_dict_path(model_dir, payload, packaged_dict_paths)
        ),
    )


def _extract_resize_long(payload: dict[str, Any]) -> int:
    """Read detector resize_long from DetResizeForTest when present."""
    preprocess = payload.get("PreProcess", {})
    if not isinstance(preprocess, dict):
        return DEFAULT_RESIZE_LONG

    transform_ops = preprocess.get("transform_ops", [])
    if not isinstance(transform_ops, list):
        return DEFAULT_RESIZE_LONG

    for transform in transform_ops:
        if not isinstance(transform, dict):
            continue
        resize_for_test = transform.get("DetResizeForTest")
        if resize_for_test is None:
            continue
        if isinstance(resize_for_test, dict) and "resize_long" in resize_for_test:
            return int(resize_for_test["resize_long"])
        if resize_for_test is not None:
            break

    return DEFAULT_RESIZE_LONG


def _extract_rec_image_shape(payload: dict[str, Any]) -> list[int]:
    """Read recognizer image_shape from RecResizeImg in the inference YAML."""
    preprocess = payload.get("PreProcess", {})
    if not isinstance(preprocess, dict):
        return list(DEFAULT_REC_INPUT_SHAPE)

    transform_ops = preprocess.get("transform_ops", [])
    if not isinstance(transform_ops, list):
        return list(DEFAULT_REC_INPUT_SHAPE)

    for transform in transform_ops:
        if not isinstance(transform, dict):
            continue
        resize_img = transform.get("RecResizeImg")
        if not isinstance(resize_img, dict):
            continue
        image_shape = resize_img.get("image_shape")
        if isinstance(image_shape, list) and len(image_shape) == 3:
            return [int(value) for value in image_shape]

    return list(DEFAULT_REC_INPUT_SHAPE)


def _resolve_character_dict_path(
    model_dir: Path,
    payload: dict[str, Any],
    packaged_dict_paths: dict[str, Path] | None,
) -> Path:
    """Resolve or materialize the recognizer dictionary for a model directory."""
    cached_dict_path = model_dir / CHARACTER_DICT_FILENAME
    if cached_dict_path.exists():
        return cached_dict_path

    postprocess = payload.get("PostProcess", {})
    if isinstance(postprocess, dict):
        character_dict = postprocess.get("character_dict")
        if isinstance(character_dict, list) and character_dict:
            cached_dict_path.write_text(
                "\n".join(str(character) for character in character_dict) + "\n",
                encoding="utf-8",
            )
            return cached_dict_path

    packaged = packaged_dict_paths or {}
    model_name = ""
    global_section = payload.get("Global", {})
    if isinstance(global_section, dict):
        model_name = str(global_section.get("model_name", "")).lower()

    model_dir_name = model_dir.name.lower()
    if "v6" in model_name or "v6" in model_dir_name or "pp-ocrv6" in model_dir_name:
        fallback = packaged.get("v6")
        if fallback is not None:
            return fallback

    fallback = packaged.get("v5")
    if fallback is not None:
        return fallback

    raise FileNotFoundError(
        f"Could not resolve character dictionary for {model_dir}. "
        "Expected PostProcess.character_dict in inference.yml or a packaged fallback."
    )