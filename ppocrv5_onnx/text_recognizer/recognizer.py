import cv2
import math
import numpy as np
import onnxruntime as ort
from typing import List, Optional
from .postprocess import CTCLabelDecode
from typing import List, Union, Tuple, Optional

class Recognizer:
    def __init__(self, config, providers: Optional[List[str]] = None):
        charset_path = config.dict_path
        model_path = config.path
        if not str(model_path).endswith(".onnx"):
            raise ValueError(f"Model path must end with .onnx: {model_path}")

        # Rec input shape from config [C, H, W]
        rec_shape = list(config.input_shape)
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

    def resize_norm_img_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[float], float]:
        """
        Batch preprocessing:
        1. Calculate target width for each image maintaining aspect ratio.
        2. Find the max width in the batch (capped at 3200).
        3. Resize, Normalize, and Pad all images to (B, C, H, Max_W).
        """
        imgC, imgH, imgW = self.imgC, self.imgH, self.imgW
        batch_size = len(images)
        
        # Calculate aspect ratios and target widths
        ratios = []
        resized_widths = []
        
        for img in images:
            h, w = img.shape[:2]
            ratio = w / float(h)
            ratios.append(ratio)
            
            # Logic to ensure minimum width matches config, or expand if image is wider
            max_wh_ratio = max(imgW / float(imgH), ratio)
            
            # Calculate width required for this image
            current_dyn_width = int(imgH * max_wh_ratio)
            resized_widths.append(current_dyn_width)

        # Determine the maximum width needed for this batch
        # We limit the max width to 3200 to prevent OOM or extremely slow inference on outliers
        max_batch_w = max(resized_widths)
        max_batch_w = min(max_batch_w, 3200)

        # Pre-allocate batch tensor: [B, C, H, W]
        batch_imgs = np.zeros((batch_size, imgC, imgH, max_batch_w), dtype=np.float32)

        for i, img in enumerate(images):
            h, w = img.shape[:2]
            
            # Calculate actual resize width for this specific image
            # We need to resize strictly by aspect ratio first
            ratio = w / float(h)
            resized_w = int(math.ceil(imgH * ratio))
            
            # Cap width if it exceeds our batch maximum
            if resized_w > max_batch_w:
                resized_w = max_batch_w

            # Resize
            resized_image = cv2.resize(img, (resized_w, imgH))
            resized_image = resized_image.astype("float32")
            
            # Normalize: (H, W, C) -> (C, H, W) and scale
            resized_image = resized_image.transpose((2, 0, 1)) / 255.0
            resized_image -= 0.5
            resized_image /= 0.5

            # Pad: Copy the resized image into the pre-allocated zero tensor
            # The rest of the width (max_batch_w - resized_w) remains 0 (padding)
            batch_imgs[i, :, :, 0:resized_w] = resized_image

        # Calculate max_wh_ratio for the whole batch (used for decoding boxes if needed)
        batch_max_wh_ratio = max_batch_w / float(imgH)
        
        return batch_imgs, ratios, batch_max_wh_ratio

    def recognize(self, images: Union[np.ndarray, List[np.ndarray]]):
        """
        Run OCR recognition on a batch of images or a single image.
        Args:
            images: Single np.ndarray image or List of np.ndarray images.
        Returns:
            List of results [(text, score), ...]
        """
        # Handle single image input for backward compatibility
        if isinstance(images, np.ndarray):
            images = [images]
            
        if not images:
            raise ValueError("Input images list is empty")

        # Preprocess batch
        input_tensor, ratios, max_wh_ratio = self.resize_norm_img_batch(images)

        # Run Inference
        # ONNX Runtime handles the batch dimension naturally [B, C, H, W]
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # Post-process (Decode)
        return_word_box = False
        
        # The processor (CTCLabelDecode) iterates through the batch dimension internally
        text = self.processor(
            outputs,
            return_word_box=return_word_box,
            wh_ratio_list=ratios,
            max_wh_ratio=max_wh_ratio,
        )
        
        return text