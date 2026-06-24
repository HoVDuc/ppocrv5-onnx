# Repository Guidelines

## Build, Test, and Development Commands

- Run all project commands inside the `survey` conda environment. In non-interactive shells, initialize conda first: `source /home/anlab/anaconda3/etc/profile.d/conda.sh && conda activate survey`.
- `source /home/anlab/anaconda3/etc/profile.d/conda.sh && conda activate survey`

## Packaging Guardrails

- When adding Python modules/packages or editing `pyproject.toml`, verify the installed artifact, not only source-tree imports.
- Required packaging smoke test:
  `source /home/anlab/anaconda3/etc/profile.d/conda.sh && conda activate survey && python -m pip install . --no-deps --target /tmp/ppocrv5-install-check --force-reinstall`
- After that install check, confirm `/tmp/ppocrv5-install-check/ppocrv5_onnx` contains the `.py` modules and subpackages, especially `__init__.py`, `pipeline.py`, `config.py`, `text_detector/`, `text_recognizer/`, `tools/`, `data/dict/ppocrv5_dict.txt`, and `data/dict/ppocrv6_dict.txt`.
- Run `pytest tests/test_packaging.py -q` after packaging changes. This prevents the broken install state where only `data/` and `models/` appear under `site-packages/ppocrv5_onnx`.
