from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contains_python_modules(tmp_path: Path) -> None:
    """Build a wheel and ensure it contains importable package modules."""
    project_root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-build-isolation",
            "--no-deps",
            "-w",
            str(wheel_dir),
        ],
        cwd=project_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    wheels = sorted(wheel_dir.glob("ppocrv5_onnx-*.whl"))
    assert wheels, "Expected pip wheel to create a ppocrv5_onnx wheel"

    required_members = {
        "ppocrv5_onnx/__init__.py",
        "ppocrv5_onnx/config.py",
        "ppocrv5_onnx/inference_config.py",
        "ppocrv5_onnx/pipeline.py",
        "ppocrv5_onnx/schema.py",
        "ppocrv5_onnx/text_detector/__init__.py",
        "ppocrv5_onnx/text_detector/detector.py",
        "ppocrv5_onnx/text_recognizer/__init__.py",
        "ppocrv5_onnx/text_recognizer/recognizer.py",
        "ppocrv5_onnx/tools/__init__.py",
        "ppocrv5_onnx/tools/visualizer.py",
        "ppocrv5_onnx/data/__init__.py",
        "ppocrv5_onnx/data/dict/ppocrv5_dict.txt",
        "ppocrv5_onnx/data/dict/ppocrv6_dict.txt",
        "ppocrv5_onnx/data/fonts/simfang.ttf",
    }
    forbidden_members = {
        "ppocrv5_onnx/ppocrv6_pipeline.py",
        "ppocrv5_onnx/yomitoku_pipeline.py",
    }

    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_members = set(wheel.namelist())

    assert required_members <= wheel_members
    assert forbidden_members.isdisjoint(wheel_members)


def test_target_install_contains_python_modules(tmp_path: Path) -> None:
    """Regular install into a target dir must ship importable package modules."""
    project_root = Path(__file__).resolve().parents[1]
    target = tmp_path / "site-packages"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            ".",
            "--no-deps",
            "--force-reinstall",
            "--target",
            str(target),
        ],
        cwd=project_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    required_files = [
        "ppocrv5_onnx/__init__.py",
        "ppocrv5_onnx/config.py",
        "ppocrv5_onnx/inference_config.py",
        "ppocrv5_onnx/pipeline.py",
        "ppocrv5_onnx/text_detector/detector.py",
        "ppocrv5_onnx/text_recognizer/recognizer.py",
        "ppocrv5_onnx/tools/visualizer.py",
        "ppocrv5_onnx/data/__init__.py",
        "ppocrv5_onnx/data/dict/ppocrv5_dict.txt",
        "ppocrv5_onnx/data/dict/ppocrv6_dict.txt",
    ]
    forbidden_dirs = [
        target / "ppocrv5_onnx" / "models",
    ]

    for relative_path in required_files:
        assert (target / relative_path).is_file(), relative_path

    for forbidden_dir in forbidden_dirs:
        assert not forbidden_dir.exists(), forbidden_dir
