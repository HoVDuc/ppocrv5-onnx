import traceback
import cv2
import random
import numpy as np
import math
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from ppocrv5_onnx.utils import load_config

config = load_config("config.yaml")
visualize = config.get("visualize", {})
def get_minarea_rect(points: np.ndarray) -> np.ndarray:
    """
    Get the minimum area rectangle for the given points using OpenCV.

    Args:
        points (np.ndarray): An array of 2D points.

    Returns:
        np.ndarray: An array of 2D points representing the corners of the minimum area rectangle
                    in a specific order (clockwise or counterclockwise starting from the top-left corner).
    """
    bounding_box = cv2.minAreaRect(points)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = np.array(
        [points[index_a], points[index_b], points[index_c], points[index_d]]
    ).astype(np.int32)

    return box

def create_font_vertical(
    txt: str, sz: tuple, font_path: str, scale=1.2
) -> ImageFont.FreeTypeFont:
    n = len(txt) if len(txt) > 0 else 1
    base_font_size = int(sz[1] / n * 0.8 * scale)
    base_font_size = max(base_font_size, 10)
    font = ImageFont.truetype(font_path, base_font_size, encoding="utf-8")

    max_char_width = max([font.getlength(c) for c in txt])

    if max_char_width > sz[0]:
        new_size = int(base_font_size * sz[0] / max_char_width)
        new_size = max(new_size, 10)
        font = ImageFont.truetype(font_path, new_size, encoding="utf-8")

    return font

def create_font(txt: str, sz: tuple, font_path: str) -> ImageFont:
    """
    Create a font object with specified size and path, adjusted to fit within the given image region.

    Parameters:
    txt (str): The text to be rendered with the font.
    sz (tuple): A tuple containing the height and width of an image region, used for font size.
    font_path (str): The path to the font file.

    Returns:
    ImageFont: An ImageFont object adjusted to fit within the given image region.
    """

    font_size = int(sz[1] * 0.8)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getlength(txt)

    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font

def draw_vertical_text(draw, position, text, font, fill=(0, 0, 0), line_spacing=2):
    x, y = position
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        bbox = font.getbbox(char)
        char_height = bbox[3] - bbox[1]
        y += char_height + line_spacing

def draw_box_txt_fine(
    img_size: tuple, box: np.ndarray, txt: str, font_path: str
) -> np.ndarray:
    """
    Draws text in a box on an image with fine control over size and orientation.

    Args:
        img_size (tuple): The size of the output image (width, height).
        box (np.ndarray): A 4x2 numpy array defining the corners of the box in (x, y) order.
        txt (str): The text to draw inside the box.
        font_path (str): The path to the font file to use for drawing the text.

    Returns:
        np.ndarray: An image with the text drawn in the specified box.
    """
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font_vertical(txt, (box_width, box_height), font_path)
            draw_vertical_text(
                draw_text, (0, 0), txt, font, fill=(0, 0, 0), line_spacing=2
            )
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text


def draw_image(image: np.ndarray, 
               boxes: np.ndarray,
               txts: np.ndarray,
               det_only=False) -> np.ndarray:
    """
    Converts the internal data to a PIL Image with detection and recognition results.

    Returns:
        Dict[Image.Image]: A dictionary containing two images: 'doc_preprocessor_res' and 'ocr_res_img'.
    """


    h, w = image.shape[0:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_left = Image.fromarray(image_rgb)
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        try:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            box = np.array(box)
            if len(box) > 4:
                pts = [(x, y) for x, y in box.tolist()]
                draw_left.polygon(pts, outline=color, width=8)
                box = get_minarea_rect(box)
                height = int(0.5 * (max(box[:, 1]) - min(box[:, 1])))
                box[:2, 1] = np.mean(box[:, 1])
                box[2:, 1] = np.mean(box[:, 1]) + min(20, height)
            box_pts = [(int(x), int(y)) for x, y in box.tolist()]
            draw_left.polygon(box_pts, fill=color)
            img_right_text = draw_box_txt_fine(
                (w, h), box, txt, visualize.get("font_path", "")
            )
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_right_text, [pts], True, color, visualize.get("box_thickness", 2))
            img_right = cv2.bitwise_and(img_right, img_right_text)
        except Exception:
            logger.error(traceback.print_exc())
            continue

    img_left = Image.blend(Image.fromarray(image_rgb), img_left, 0.5)
    isW = True
    
    if not det_only:
        if isW:
            img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
            img_show.paste(img_left, (0, 0, w, h))
            img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
        else:
            img_show = Image.new("RGB", (w, h*2), (255, 255, 255))
            img_show.paste(img_left, (0, 0, w, h))
            img_show.paste(Image.fromarray(img_right), (0, h, w, h*2))
    else:
        img_show = Image.new("RGB", (w, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))

    res_img_dict = {"ocr_res_img": img_show}
    return res_img_dict