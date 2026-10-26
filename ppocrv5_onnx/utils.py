from typing import Any, List, Optional, Sequence, Tuple, Union

import onnxruntime as ort
from omegaconf import OmegaConf

# Load config and using via class

def load_config(config_path):
    return OmegaConf.load(config_path)

def build_config(config_path):
    return load_config(config_path)


# Detection and recognition inputs change shape on every image/batch. With the
# CUDA EP defaults, the BFC arena grows in power-of-two steps and cuDNN runs an
# exhaustive algorithm search with maximum workspace for every new shape, so GPU
# memory keeps growing until OOM (issue #5). These defaults keep it bounded.
DEFAULT_CUDA_PROVIDER_OPTIONS = {
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "HEURISTIC",
    "cudnn_conv_use_max_workspace": "0",
}

Provider = Union[str, Tuple[str, dict]]


def build_providers(providers: Optional[Sequence[Provider]] = None) -> List[Provider]:
    """Return ONNX Runtime providers, applying safe defaults to a bare CUDA provider.

    Providers given as ``(name, options)`` tuples are passed through unchanged.
    """
    resolved: List[Provider] = []
    for provider in providers or ["CPUExecutionProvider"]:
        if provider == "CUDAExecutionProvider":
            provider = (provider, dict(DEFAULT_CUDA_PROVIDER_OPTIONS))
        resolved.append(provider)
    return resolved


def create_session(
    model_path: Any,
    providers: Optional[Sequence[Provider]] = None,
    sess_options: Optional[ort.SessionOptions] = None,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session for a model."""
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=build_providers(providers),
    )
