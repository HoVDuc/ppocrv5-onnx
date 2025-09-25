import argparse
from loguru import logger

from utils import load_config
from src.ppocrv5_onnx.engine import Detector, Recognizer, run_ocr


def main():
    parser = argparse.ArgumentParser(description="PP-OCRv5 ONNX runner")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--det-only", action="store_true", help="Run detection only")
    parser.add_argument("--rec-only", action="store_true", help="Run recognition only")
    parser.add_argument("--vis", action="store_true", help="Visualize the results")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="ONNX Runtime providers, e.g. CUDAExecutionProvider CPUExecutionProvider",
    )
    args = parser.parse_args()

    if args.det_only and args.rec_only:
        parser.error("--det-only and --rec-only are mutually exclusive")

    cfg = load_config(args.config)

    det = not args.rec_only
    rec = not args.det_only

    detector = Detector(cfg, providers=args.providers)
    recognizer = Recognizer(cfg, providers=args.providers)

    result = run_ocr(
        img_path=args.image,
        det=det,
        rec=rec,
        detector=detector if det else None,
        recognizer=recognizer if rec else None,
        visualize=args.vis,
    )

    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
