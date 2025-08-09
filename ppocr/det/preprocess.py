from typing import Union
import numpy as np 
import cv2
from loguru import logger as loguru_logger
from loguru import logger as logging
import math

class DetResizeForTest:
    """DetResizeForTest"""

    def __init__(self, input_shape=None, max_side_limit=4000, **kwargs):
        self.resize_type = 0
        self.keep_ratio = False
        if input_shape is not None:
            self.input_shape = input_shape
            self.resize_type = 3
        elif "image_shape" in kwargs:
            self.image_shape = kwargs["image_shape"]
            self.resize_type = 1
            if "keep_ratio" in kwargs:
                self.keep_ratio = kwargs["keep_ratio"]
        elif "limit_side_len" in kwargs:
            self.limit_side_len = kwargs["limit_side_len"]
            self.limit_type = kwargs.get("limit_type", "min")
        elif "resize_long" in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get("resize_long", 960)
        else:
            self.limit_side_len = 736
            self.limit_type = "min"

        self.max_side_limit = max_side_limit

    def __call__(
        self,
        imgs,
        limit_side_len: Union[int, None] = None,
        limit_type: Union[str, None] = None,
        max_side_limit: Union[int, None] = None,
    ):
        """apply"""
        max_side_limit = (
            max_side_limit if max_side_limit is not None else self.max_side_limit
        )
        resize_imgs, img_shapes = [], []
        for ori_img in imgs:
            img, shape = self.resize(
                ori_img, limit_side_len, limit_type, max_side_limit
            )
            resize_imgs.append(img)
            img_shapes.append(shape)
        return resize_imgs, img_shapes

    def resize(
        self,
        img,
        limit_side_len: Union[int, None],
        limit_type: Union[str, None],
        max_side_limit: Union[int, None] = None,
    ):
        loguru_logger.info(
            "Resize image with limit_side_len: {}, limit_type: {}, max_side_limit: {}".format(
                limit_side_len, limit_type, max_side_limit
            )
        )
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        loguru_logger.info("resize_type: {}".format(self.resize_type))
        img, [ratio_h, ratio_w] = self.resize_image_type0(
            img, limit_side_len, limit_type, max_side_limit
        )
        
        return img, np.array([src_h, src_w, ratio_h, ratio_w])


    def image_padding(self, im, value=0):
        """padding image"""
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image_type0(
        self,
        img,
        limit_side_len: Union[int, None],
        limit_type: Union[str, None],
        max_side_limit: Union[int, None] = None,
    ):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = limit_side_len or self.limit_side_len
        limit_type = limit_type or self.limit_type
        h, w, c = img.shape

        # limit the max side
        if limit_type == "max":
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif limit_type == "min":
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception("not support limit type, image ")
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        if max(resize_h, resize_w) > max_side_limit:
            logging.warning(
                f"Resized image size ({resize_h}x{resize_w}) exceeds max_side_limit of {max_side_limit}. "
                f"Resizing to fit within limit."
            )
            ratio = float(max_side_limit) / max(resize_h, resize_w)
            resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        if resize_h == h and resize_w == w:
            return img, [1.0, 1.0]

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            logging.info(img.shape, resize_w, resize_h)
            raise

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        loguru_logger.info(
            f"Resize image from ({h}, {w}) to ({resize_h}, {resize_w}) with ratios ({ratio_h}, {ratio_w})."
        )
        return img, [ratio_h, ratio_w]

class NormalizeImage:
    """normalize image such as subtract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw"):
        super().__init__()
        if isinstance(scale, str):
            scale = eval(scale)
        self.order = order

        scale = scale if scale is not None else 1.0 / 255.0
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        self.alpha = [scale / std[i] for i in range(len(std))]
        self.beta = [-mean[i] / std[i] for i in range(len(std))]

    def __call__(self, imgs):
        """apply"""

        def _norm(img):
            if self.order == "chw":
                img = np.transpose(img, (2, 0, 1))

            split_im = list(cv2.split(img))
            for c in range(img.shape[2]):
                split_im[c] = split_im[c].astype(np.float32)
                split_im[c] *= self.alpha[c]
                split_im[c] += self.beta[c]

            res = cv2.merge(split_im)

            if self.order == "chw":
                res = np.transpose(res, (1, 2, 0))
            return res

        return [_norm(img) for img in imgs]
