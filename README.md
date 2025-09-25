# PP-OCRv5 ONNX (uv-based workflow)

This repo runs PaddleOCR v5 (detection + recognition) exported to ONNX with onnxruntime, using uv for dependency management and execution.

## Requirements
- Python 3.10+
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
visualize:
  font_path: fonts/simfang.ttf
  save_dir: output
  box_thickness: 2
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

- With visualization:
```sh
uv run ppocrv5-onnx path/to/image.jpg --vis
```

- Run via Python module directly:
```sh
uv run python main.py path/to/image.jpg --config config.yaml
```

- Programmatic usage:
```python
from ppocrv5_onnx.utils import load_config
from ppocrv5_onnx.engine import Detector, Recognizer, run_ocr

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
main.py
pyproject.toml
src/
  ppocrv5_onnx/
    __init__.py
    cli.py
    engine.py
    utils.py
models/
  PP-OCRv5_mobile_det/inference.onnx
  PP-OCRv5_mobile_rec/inference.onnx
ppocr/
  det/ ...
  rec/ ...
dict/ppocrv5_dict.txt
fonts/simfang.ttf
```

## Model export (Paddle2ONNX)
Clone the PaddleOCR repo if you haven't already:
```sh
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

Step 1
Export model to inference format using PaddlePaddle tools. Example for detection model:
```sh
cd PaddleOCR
python3 tools/export_model.py -c=configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o \
        Global.pretrained_model=/path/PP-OCRv5_server_rec_pretrained.pdparams \
        Global.save_inference_dir=./PP-OCRv5_server_rec/
```

Step 2
Convert the exported model to ONNX format using `paddle2onnx`. Example command:
```sh
paddle2onnx --model_dir /path/PP-OCRv5_server_rec_infer \
--model_filename inference.json \
--params_filename inference.pdiparams \
--save_file model.onnx \
--opset_version 17 \
--enable_onnx_checker True
```


## Troubleshooting
- Missing dependency? `uv add <name>` and re-run `uv sync`.
- Path errors for models/dict? Verify the paths in `config.yaml` are correct relative to the repo root.
- OpenCV display errors on servers? Use `opencv-python-headless`.
- GPU not used? Ensure CUDA drivers/runtime are installed and pass `--providers CUDAExecutionProvider`.
