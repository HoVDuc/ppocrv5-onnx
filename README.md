# PP-OCRv5 ONNX (uv-based workflow)

This repo runs PaddleOCR v5 (detection + recognition) exported to ONNX with onnxruntime, using uv for dependency management and execution.

## Requirements
- Linux
- Python 3.12+
- uv (https://docs.astral.sh/uv/) – fast Python package/dependency manager

## Install uv
Choose one:

- Via installer:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Or via pipx:
```sh
pipx install uv
```

Ensure `~/.local/bin` (or the path printed by the installer) is on your `PATH`.

## Setup
Sync the environment (installs deps from `pyproject.toml` / `uv.lock`):
```sh
uv sync
```

If you are on a headless server and encounter display issues, you can swap OpenCV with:
```sh
uv remove opencv-python && uv add opencv-python-headless
```

For GPU builds (CUDA), replace onnxruntime with the GPU variant:
```sh
uv remove onnxruntime && uv add onnxruntime-gpu
```

## Configuration
Models and the character dictionary are configured in `config.yaml`:
```yaml
engine:
  model:
    det:
      path: ./models/PP-OCRv5_mobile_det/inference.onnx
      input_shape: [3, 640, 640]
    rec:
      path: ./models/PP-OCRv5_mobile_rec/inference.onnx
      input_shape: [3, 32, 320]
      dict_path: ./dict/ppocrv5_dict.txt
```
Adjust paths if you relocate models or the dictionary.

## Run
You can run either via the console script or directly with Python.

- Console script (after `uv sync`):
```sh
uv sync  # re-sync after changes to pyproject
uv run ppocrv5-onnx path/to/image.jpg --config config.yaml
```

- Detection only:
```sh
uv run ppocrv5-onnx path/to/image.jpg --det-only
```

- Recognition only:
```sh
uv run ppocrv5-onnx path/to/image.jpg --rec-only
```

- Select ONNX Runtime providers (e.g., CUDA + CPU):
```sh
uv run ppocrv5-onnx path/to/image.jpg --providers CUDAExecutionProvider CPUExecutionProvider
```

- Run via Python module directly:
```sh
uv run python main.py path/to/image.jpg --config config.yaml
```

- Programmatic usage:
```python
from utils import load_config
from engine import Detector, Recognizer, run_ocr

cfg = load_config('config.yaml')
detector = Detector(cfg)
recognizer = Recognizer(cfg)

results = run_ocr('path/to/image.jpg', det=True, rec=True, detector=detector, recognizer=recognizer)
print(results)
```

## Common uv commands
- Install/sync env from lockfile: `uv sync`
- Add/remove a package: `uv add <pkg>`, `uv remove <pkg>`
- Run a script in the project env: `uv run <cmd>`

## Project structure (excerpt)
```
config.yaml
engine.py
main.py
pyproject.toml
utils.py
models/
  PP-OCRv5_mobile_det/inference.onnx
  PP-OCRv5_mobile_rec/inference.onnx
ppocr/
  det/ ...
  rec/ ...
dict/ppocrv5_dict.txt
```

## Troubleshooting
- Missing dependency? `uv add <name>` and re-run `uv sync`.
- Path errors for models/dict? Verify the paths in `config.yaml` are correct relative to the repo root.
- OpenCV display errors on servers? Use `opencv-python-headless`.
- GPU not used? Ensure CUDA drivers/runtime are installed and pass `--providers CUDAExecutionProvider`.
