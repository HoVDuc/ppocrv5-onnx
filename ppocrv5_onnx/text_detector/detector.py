import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from typing import List, Optional
from .preprocess import DetResize, NormalizeImage
from .crop_poly import CropPoly
from .postprocess import DBPostProcess


class Detector:
    def __init__(self, config, providers: Optional[List[str]] = None):
        model_path = config.path
        if not str(model_path).endswith(".onnx"):
            raise ValueError(f"Model path must end with .onnx: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get detection config
        
        # Use resize_long for dynamic shape support (same as official inference.yml)
        # Default to 960 as per official PP-OCRv5 config
        self.resize_long = getattr(config, 'resize_long', 960)
        
        # For backward compatibility: if input_shape is specified, calculate target_size from it
        if hasattr(config, 'input_shape'):
            det_shape = list(config.input_shape)
            if len(det_shape) == 3:
                _, det_h, det_w = det_shape
                self.resize_long = max(int(det_h), int(det_w))
                logger.warning(
                    f"Using deprecated 'input_shape' config. Please use 'resize_long: {self.resize_long}' instead."
                )
        
        # Pre/Post processors with configurable parameters (aligned with official config)
        self.preprocessor = DetResize(resize_long=self.resize_long)
        self.normalize = NormalizeImage(order="hwc")
        
        # PostProcess parameters from config (with official defaults)
        thresh = getattr(config, 'thresh', 0.3)
        box_thresh = getattr(config, 'box_thresh', 0.6)
        unclip_ratio = getattr(config, 'unclip_ratio', 1.5)
        max_candidates = getattr(config, 'max_candidates', 1000)
        
        self.postprocessor = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            max_candidates=max_candidates
        )
        self.crop_poly = CropPoly()
        
        logger.info(
            f"Detector initialized with resize_long={self.resize_long}, "
            f"thresh={thresh}, box_thresh={box_thresh}, unclip_ratio={unclip_ratio}"
        )

    def detect(self, image: np.ndarray):
        if image is None:
            raise ValueError("Input image is None")
        logger.info(f"Image shape: {image.shape}")

        # Use resize_long strategy (keep aspect ratio, resize longest side)
        data = self.preprocessor([image], limit_side_len=self.resize_long, limit_type="resize_long")
        input_tensor = data[0]
        input_tensor = self.normalize(input_tensor)
        input_tensor = input_tensor[0].transpose((2, 0, 1))  # HWC -> CHW
        shape_input = data[1]
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        boxes = self.postprocessor(outputs, shape_input)
        return boxes