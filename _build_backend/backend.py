"""In-tree PEP 517 backend: setuptools, with compat-mode editable installs by default.

A plain ``pip install -e .`` with setuptools >= 64 installs an import-hook
``.pth`` file that editor tooling (Pylance/Pyright, Jedi) cannot follow, so
there is no autocompletion for ``ppocrv5_onnx``. ``editable_mode=compat`` writes
the repository path into the ``.pth`` file instead. Passing another mode via
``--config-settings editable_mode=...`` still takes precedence.

Regular wheel and sdist builds are delegated to setuptools unchanged.
"""

from setuptools import build_meta as _setuptools_backend
from setuptools.build_meta import *  # noqa: F401,F403


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    config_settings = dict(config_settings or {})
    config_settings.setdefault("editable_mode", "compat")
    return _setuptools_backend.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
