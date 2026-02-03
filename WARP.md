# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview
This is a PP-OCRv5 ONNX runtime implementation that uses `uv` for dependency management. It provides text detection and recognition using ONNX models exported from PaddleOCR v5, running via `onnxruntime` without requiring the full PaddlePaddle framework.

## Dependency Management and Environment

### Setup
```sh
uv sync
```

### Common Commands
- Add dependency: `uv add <package>`
- Remove dependency: `uv remove <package>`
- Run in project env: `uv run <command>`

### Platform-Specific Variants
- Headless servers: `uv remove opencv-python && uv add opencv-python-headless`
- GPU acceleration: `uv remove onnxruntime && uv add onnxruntime-gpu`

## Running OCR

### Basic Usage
```sh
# Full OCR (detection + recognition)
uv run ppocrv5-onnx path/to/image.jpg

# Detection only
uv run ppocrv5-onnx path/to/image.jpg --det-only

# Recognition only
uv run ppocrv5-onnx path/to/image.jpg --rec-only

# With visualization (saves to output/ directory)
uv run ppocrv5-onnx path/to/image.jpg --vis

# GPU execution
uv run ppocrv5-onnx path/to/image.jpg --providers CUDAExecutionProvider CPUExecutionProvider
```

### Alternative Entry Point
```sh
uv run python main.py path/to/image.jpg --config config.yaml
```

## Configuration
Model paths and settings are in `config.yaml`:
- Detection model: `engine.model.det.path` and `input_shape`
- Recognition model: `engine.model.rec.path` and `input_shape`
- Character dictionary: `engine.model.rec.dict_path`
- Visualization: `visualize.font_path`, `save_dir`, `box_thickness`

All paths in config are relative to repository root.

## Architecture

### Core Components

**Engine Layer** (`src/ppocrv5_onnx/engine.py`):
- `Detector`: Wraps detection ONNX model with pre/post-processing
- `Recognizer`: Wraps recognition ONNX model with CTC decoding
- `run_ocr()`: Main pipeline function orchestrating detection and recognition
- `Result`: Dataclass containing text, bounding box, and confidence score

**Pre/Post Processing** (`ppocr/`):
This directory contains PaddleOCR preprocessing and postprocessing utilities adapted for ONNX inference:
- `ppocr/det/preprocess.py`: Detection image resizing and normalization
- `ppocr/det/postprocess.py`: `DBPostProcess` - converts detection heatmaps to polygons
- `ppocr/det/crop_poly.py`: `CropPoly` - extracts rotated text regions from detection boxes
- `ppocr/rec/postprocess.py`: `CTCLabelDecode` - decodes CTC output to text using character dictionary
- `ppocr/tools/visualizer.py`: Draws bounding boxes and text on images

**Entry Points**:
- `src/ppocrv5_onnx/cli.py`: Console script entry (registered as `ppocrv5-onnx` command)
- `main.py`: Alternative Python module entry point

### Data Flow
1. Load image with OpenCV
2. **Detection** (if enabled):
   - Preprocess: resize to target size, normalize (HWC→CHW format)
   - Run ONNX inference
   - Postprocess: apply DBNet postprocessing to extract text polygon boxes
3. **Recognition** (if enabled):
   - For each detected box: crop rotated region using minimum area rectangle
   - Preprocess: resize with aspect ratio preservation, pad, normalize
   - Run ONNX inference
   - Postprocess: CTC decode using character dictionary to extract text and confidence
4. **Visualization** (optional): Draw boxes and text on image, save to `output/`

### Model Export Workflow
Models are exported from PaddleOCR using a two-step process:
1. Export PaddlePaddle model to inference format using `tools/export_model.py`
2. Convert to ONNX using `paddle2onnx` CLI with opset 17

See README section "Model export (Paddle2ONNX)" for detailed commands.

## Programmatic Usage
```python
from ppocrv5_onnx.utils import load_config
from ppocrv5_onnx.engine import Detector, Recognizer, run_ocr

cfg = load_config('config.yaml')
detector = Detector(cfg)
recognizer = Recognizer(cfg)

results = run_ocr('path/to/image.jpg', det=True, rec=True, 
                  detector=detector, recognizer=recognizer)
# results is List[Result] with .text, .box, .score attributes
```

## Project Structure
- `src/ppocrv5_onnx/`: Main package (engine, CLI, utilities)
- `ppocr/`: PaddleOCR preprocessing/postprocessing utilities
- `models/`: ONNX model files (not in git)
- `dict/`: Character dictionaries for recognition
- `fonts/`: Font files for visualization
- `output/`: Visualization output directory (created on demand)
- `config.yaml`: Model paths and runtime configuration
