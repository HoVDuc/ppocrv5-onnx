from dataclasses import dataclass
from typing import List, Optional
import time
import math

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

from ppocr.det.crop_poly import CropPoly
from ppocr.det.postprocess import DBPostProcess
from ppocr.det.preprocess import DetResizeForTest, NormalizeImage
from ppocr.rec.postprocess import CTCLabelDecode


@dataclass
class Result:
    text: str
    box: np.ndarray


class Detector:
    def __init__(self, config, providers: Optional[List[str]] = None):
        model_path = config.engine.model.det.path
        if not str(model_path).endswith(".onnx"):
            raise ValueError(f"Model path must end with .onnx: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Pre/Post
        self.preprocessor = DetResizeForTest()
        self.normalize = NormalizeImage(order="hwc")
        self.postprocessor = DBPostProcess(unclip_ratio=1.0)
        self.crop_poly = CropPoly()

        # Use configured input shape to pick a reasonable target size
        # Expecting [C, H, W]
        det_shape = list(config.engine.model.det.input_shape)
        if len(det_shape) != 3:
            raise ValueError(f"det.input_shape must be [C, H, W], got: {det_shape}")
        _, det_h, det_w = det_shape
        self.target_size = max(int(det_h), int(det_w))

    def detect(self, image: np.ndarray):
        if image is None:
            raise ValueError("Input image is None")
        logger.info(f"Image shape: {image.shape}")

        data = self.preprocessor([image], self.target_size)
        input_tensor = data[0]
        input_tensor = self.normalize(input_tensor)
        input_tensor = input_tensor[0].transpose((2, 0, 1))  # HWC -> CHW
        shape_input = data[1]
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        boxes = self.postprocessor(outputs, shape_input)
        return boxes


class Recognizer:
    def __init__(self, config, providers: Optional[List[str]] = None):
        charset_path = config.engine.model.rec.dict_path
        model_path = config.engine.model.rec.path
        if not str(model_path).endswith(".onnx"):
            raise ValueError(f"Model path must end with .onnx: {model_path}")

        # Rec input shape from config [C, H, W]
        rec_shape = list(config.engine.model.rec.input_shape)
        if len(rec_shape) != 3:
            raise ValueError(f"rec.input_shape must be [C, H, W], got: {rec_shape}")
        self.imgC, self.imgH, self.imgW = map(int, rec_shape)

        self.processor = CTCLabelDecode(
            character_dict_path=str(charset_path),
            use_space_char=True,
        )

        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def resize_norm_img(self, img: np.ndarray) -> np.ndarray:
        """Resize to configured (C,H,W) with aspect ratio padding and normalize."""
        imgC, imgH, imgW = self.imgC, self.imgH, self.imgW
        h, w = img.shape[:2]
        wh_ratio = w / float(h)

        # Max width allowed for the configured height
        max_wh_ratio = imgW / float(imgH)
        max_wh_ratio = max(max_wh_ratio, wh_ratio)

        assert imgC == img.shape[2], f"Expected {imgC} channels, got {img.shape[2]}"
        dyn_imgW = int(imgH * max_wh_ratio)
        dyn_imgW = min(dyn_imgW, 3200)  # safety cap

        if dyn_imgW >= 3200:
            resized_image = cv2.resize(img, (3200, imgH))
            resized_w = 3200
            pad_w = 3200
        else:
            ratio = w / float(h)
            resized_w = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))
            resized_w = min(resized_w, imgW)
            resized_image = cv2.resize(img, (resized_w, imgH))
            pad_w = imgW

        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((imgC, imgH, pad_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def recognize(self, image: np.ndarray):
        if image is None:
            raise ValueError("Input image is None")
        h, w = image.shape[:2]
        input_tensor = self.resize_norm_img(image)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        return_word_box = False
        wh_ratio = w / float(h)
        max_wh_ratio = max(self.imgW / float(self.imgH), wh_ratio)
        text = self.processor(
            outputs,
            return_word_box=return_word_box,
            wh_ratio_list=[wh_ratio],
            max_wh_ratio=max_wh_ratio,
        )
        return text


def run_ocr(
    img_path: str,
    det: bool = True,
    rec: bool = True,
    detector: Optional[Detector] = None,
    recognizer: Optional[Recognizer] = None,
):
    """Run OCR pipeline on a single image path.

    Returns:
        - If det and rec: List[Result]
        - If det only: raw boxes as returned by Detector.detect
        - If rec only: recognized text on the whole image
    """
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")

    if det and rec:
        if detector is None or recognizer is None:
            raise ValueError("Both detector and recognizer must be provided when det and rec are True")
        t0 = time.time()
        # NOTE: The postprocessor structure returns nested lists; keep indexing to maintain behavior
        boxes = detector.detect(image)[0][0]
        results: List[Result] = []
        for box in boxes:
            crop = detector.crop_poly.get_minarea_rect_crop(image, box)
            text = recognizer.recognize(crop)
            results.append(Result(text=text, box=box))
        t1 = time.time()
        logger.info(f"Processing time: {t1 - t0:.2f} seconds")
        return results

    if det:
        if detector is None:
            raise ValueError("Detector must be provided when det=True and rec=False")
        return detector.detect(image)

    if rec:
        if recognizer is None:
            raise ValueError("Recognizer must be provided when rec=True and det=False")
        return recognizer.recognize(image)

    raise ValueError("At least one of det or rec must be True")