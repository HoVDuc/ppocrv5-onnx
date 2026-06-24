from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import os
import urllib.request
import zipfile

from loguru import logger
import yaml

PACKAGE_DIR = Path(__file__).parent


def _default_models_dir() -> Path:
    """Return the default ONNX model cache outside the importable package."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "ppocrv5_onnx" / "models"


MODELS_DIR = _default_models_dir()
V5_DICT_PATH = PACKAGE_DIR / "data/dict/ppocrv5_dict.txt"
V6_DICT_PATH = PACKAGE_DIR / "data/dict/ppocrv6_dict.txt"
FONT_PATH = PACKAGE_DIR / "data/fonts/simfang.ttf"
RELEASE_BASE_URL = "https://github.com/HoVDuc/ppocrv5-onnx/releases/download"


@dataclass(frozen=True)
class ModelPreset:
    """Release artifact metadata for a downloadable ONNX model."""

    model_dir: str
    release_tag: str
    zip_name: str

    @property
    def model_path(self) -> Path:
        """Return the expected ONNX model path inside the package cache."""
        return MODELS_DIR / self.model_dir / "inference.onnx"

    @property
    def model_dir_path(self) -> Path:
        """Return the directory that should contain inference.yml and ONNX files."""
        return MODELS_DIR / self.model_dir

    @property
    def download_url(self) -> str:
        """Return the GitHub release download URL for this artifact."""
        return f"{RELEASE_BASE_URL}/{self.release_tag}/{self.zip_name}"


DET_PRESETS: dict[str, ModelPreset] = {
    "ppocrv5_mobile": ModelPreset(
        model_dir="PP-OCRv5_mobile_det",
        release_tag="v1.0.0",
        zip_name="PP-OCRv5_mobile_det.zip",
    ),
    "ppocrv5_server": ModelPreset(
        model_dir="PP-OCRv5_server_det",
        release_tag="v1.0.0",
        zip_name="PP-OCRv5_server_det.zip",
    ),
    "ppocrv6_medium": ModelPreset(
        model_dir="PP-OCRv6_medium_det_onnx",
        release_tag="v1.1.0",
        zip_name="PP-OCRv6_medium_det_onnx.zip",
    ),
    "ppocrv6_small": ModelPreset(
        model_dir="PP-OCRv6_small_det_onnx",
        release_tag="v1.1.0",
        zip_name="PP-OCRv6_small_det_onnx.zip",
    ),
    "ppocrv6_tiny": ModelPreset(
        model_dir="PP-OCRv6_tiny_det_onnx",
        release_tag="v1.1.0",
        zip_name="PP-OCRv6_tiny_det_onnx.zip",
    ),
}


REC_PRESETS: dict[str, ModelPreset] = {
    "ppocrv5_mobile": ModelPreset(
        model_dir="PP-OCRv5_mobile_rec",
        release_tag="v1.0.0",
        zip_name="PP-OCRv5_mobile_rec.zip",
    ),
    "ppocrv5_server": ModelPreset(
        model_dir="PP-OCRv5_server_rec",
        release_tag="v1.0.0",
        zip_name="PP-OCRv5_server_rec.zip",
    ),
    "ppocrv6_medium": ModelPreset(
        model_dir="PP-OCRv6_medium_rec_onnx",
        release_tag="v1.1.0",
        zip_name="PP-OCRv6_medium_rec_onnx.zip",
    ),
    "ppocrv6_small": ModelPreset(
        model_dir="PP-OCRv6_small_rec_onnx",
        release_tag="v1.1.0",
        zip_name="PP-OCRv6_small_rec_onnx.zip",
    ),
    "ppocrv6_tiny": ModelPreset(
        model_dir="PP-OCRv6_tiny_rec_onnx",
        release_tag="v1.1.0",
        zip_name="PP-OCRv6_tiny_rec_onnx.zip",
    ),
}


PRESET_ALIASES: dict[str, str] = {
    "mobile": "ppocrv5_mobile",
    "server": "ppocrv5_server",
}


@dataclass
class DetectorConfig:
    """Configuration for a text detection model."""

    path: str
    resize_long: int = 960
    thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5
    max_candidates: int = 1000


@dataclass
class RecognizerConfig:
    """Configuration for a text recognition model."""

    path: str
    input_shape: list[int] = field(default_factory=lambda: [3, 32, 320])
    dict_path: str = "data/dict/ppocrv5_dict.txt"


@dataclass
class VisualizeConfig:
    """Configuration for OCR result visualization."""

    font_path: str | None = "data/fonts/simfang.ttf"
    save_dir: str = "output"
    box_thickness: int = 2


@dataclass
class OCRConfig:
    """Main OCR configuration."""

    det: DetectorConfig
    rec: RecognizerConfig
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "OCRConfig":
        """Create an OCR configuration from a dictionary."""
        engine = config_dict.get("engine", {})
        model = engine.get("model", {})

        det_overrides = model.get("det")
        rec_overrides = model.get("rec")
        det_model = str(
            engine.get("det_model", engine.get("model_name", "ppocrv5_mobile"))
        )
        rec_model = str(engine.get("rec_model", det_model))

        det_config = (
            DetectorConfig(**det_overrides)
            if det_overrides
            else _resolve_det_preset(det_model)
        )
        if rec_overrides:
            _warn_on_dict_override(rec_model, rec_overrides)
            rec_config = RecognizerConfig(**rec_overrides)
        else:
            rec_config = _resolve_rec_preset(rec_model)

        vis_config = VisualizeConfig(**config_dict.get("visualize", {}))
        return cls(det=det_config, rec=rec_config, visualize=vis_config)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "OCRConfig":
        """Load an OCR configuration from a YAML file."""
        with Path(yaml_path).open("r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}
        return cls.from_dict(config_dict)

    @classmethod
    def from_pretrained(cls, model_name: str = "mobile") -> "OCRConfig":
        """Load a synchronized detector and recognizer preset."""
        preset_name = _normalize_preset_name(model_name)
        return cls.from_mix(det_model=preset_name, rec_model=preset_name)

    @classmethod
    def from_mix(
        cls,
        *,
        det_model: str = "ppocrv5_mobile",
        rec_model: str = "ppocrv5_mobile",
    ) -> "OCRConfig":
        """Create a config by independently choosing detector and recognizer."""
        det = _resolve_det_preset(det_model)
        rec = _resolve_rec_preset(rec_model)
        return cls(
            det=det,
            rec=rec,
            visualize=VisualizeConfig(font_path=str(FONT_PATH)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert this configuration to a plain dictionary."""
        return asdict(self)

    def save_yaml(self, yaml_path: str | Path) -> None:
        """Save this configuration to a YAML file."""
        with Path(yaml_path).open("w", encoding="utf-8") as handle:
            yaml.dump(self.to_dict(), handle, default_flow_style=False)


def list_det_presets() -> list[str]:
    """Return sorted detector preset names."""
    return sorted(DET_PRESETS)


def list_rec_presets() -> list[str]:
    """Return sorted recognizer preset names."""
    return sorted(REC_PRESETS)


def _normalize_preset_name(name: str) -> str:
    """Return the canonical preset name for a user-facing alias."""
    return PRESET_ALIASES.get(name, name)


def _packaged_dict_paths() -> dict[str, Path]:
    """Return packaged dictionary fallbacks keyed by OCR generation."""
    return {"v5": V5_DICT_PATH, "v6": V6_DICT_PATH}


def _resolve_det_preset(name: str) -> DetectorConfig:
    """Resolve a detector preset using the bundled inference.yml defaults."""
    from .inference_config import detector_config_from_inference_yml

    preset_name = _normalize_preset_name(name)
    preset = _get_preset(DET_PRESETS, preset_name, "detector")
    _ensure_model_artifacts(preset)
    return detector_config_from_inference_yml(
        preset.model_dir_path,
        onnx_path=preset.model_path,
    )


def _resolve_rec_preset(name: str) -> RecognizerConfig:
    """Resolve a recognizer preset using the bundled inference.yml defaults."""
    from .inference_config import recognizer_config_from_inference_yml

    preset_name = _normalize_preset_name(name)
    preset = _get_preset(REC_PRESETS, preset_name, "recognizer")
    _ensure_model_artifacts(preset)
    return recognizer_config_from_inference_yml(
        preset.model_dir_path,
        onnx_path=preset.model_path,
        packaged_dict_paths=_packaged_dict_paths(),
    )


def _get_preset(
    presets: dict[str, ModelPreset],
    name: str,
    kind: str,
) -> ModelPreset:
    """Return a preset or raise a helpful error."""
    try:
        return presets[name]
    except KeyError as exc:
        available = ", ".join(sorted(presets))
        raise ValueError(
            f"Unknown {kind} preset: {name!r}. Available presets: {available}"
        ) from exc


def _ensure_model_artifacts(preset: ModelPreset) -> None:
    """Ensure ONNX and inference.yml exist for a preset."""
    _download_if_missing(preset)
    if not preset.model_path.exists():
        raise FileNotFoundError(
            f"Expected ONNX model at {preset.model_path} after extracting "
            f"{preset.zip_name}"
        )
    if not (preset.model_dir_path / "inference.yml").exists():
        raise FileNotFoundError(
            f"Missing inference.yml in {preset.model_dir_path} after extracting "
            f"{preset.zip_name}"
        )


def _download_if_missing(preset: ModelPreset) -> None:
    """Download and extract a model artifact when it is absent."""
    if preset.model_path.exists() and (preset.model_dir_path / "inference.yml").exists():
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading {} from {}", preset.model_dir_path, preset.download_url)

    temp_zip = MODELS_DIR / preset.zip_name
    urllib.request.urlretrieve(preset.download_url, temp_zip)
    try:
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(MODELS_DIR)
    finally:
        temp_zip.unlink(missing_ok=True)


def _warn_on_dict_override(rec_model: str, rec_config: dict[str, Any]) -> None:
    """Warn when a recognizer preset appears to be paired with another dict."""
    dict_path = rec_config.get("dict_path")
    if dict_path is None:
        return

    preset_name = _normalize_preset_name(rec_model)
    expected_name = (
        "ppocrv6_dict.txt"
        if preset_name.startswith("ppocrv6")
        else "ppocrv5_dict.txt"
    )
    dict_name = Path(str(dict_path)).name
    if dict_name not in {expected_name, "character_dict.txt"}:
        logger.warning(
            "Recognizer preset '{}' usually uses '{}' or a cached character_dict.txt, "
            "but dict_path was set to '{}'",
            preset_name,
            expected_name,
            dict_path,
        )