"""Utility functions for image processing."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as ImageType

COLORS = [
    (158, 218, 229),
    (219, 219, 141),
    (23, 190, 207),
    (188, 189, 34),
    (199, 199, 199),
    (247, 182, 210),
    (127, 127, 127),
    (227, 119, 194),
    (196, 156, 148),
    (197, 176, 213),
    (140, 86, 75),
    (148, 103, 189),
    (255, 152, 150),
    (152, 223, 138),
    (214, 39, 40),
    (44, 160, 44),
    (255, 187, 120),
    (174, 199, 232),
    (255, 127, 14),
    (31, 119, 180),
]


def b64_to_pil(b64_str: str) -> ImageType:
    r"""Convert a base64 string to a PIL Image.

    Parameters:
        b64_str: the base64 encoded image

    Returns:
        The decoded PIL Image
    """
    # , can't be encoded in b64 data so must be part of prefix
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(b64_str)))


def get_image_size(data: Union[str, Path, np.ndarray, ImageType]) -> Tuple[int, ...]:
    r"""Get the size of an image.

    Parameters:
        data: the input image

    Returns:
        The size of the image in the form (height, width)
    """
    if isinstance(data, (str, Path)):
        data = Image.open(data)

    return data.size[::-1] if isinstance(data, Image.Image) else data.shape[:2]


def convert_to_b64(data: Union[str, Path, np.ndarray, ImageType]) -> str:
    r"""Convert an image to a base64 string.

    Parameters:
        data: the input image

    Returns:
        The base64 encoded image
    """
    if data is None:
        raise ValueError(f"Invalid input image: {data}. Input image can't be None.")
    if isinstance(data, (str, Path)):
        data = Image.open(data)
    if isinstance(data, Image.Image):
        buffer = BytesIO()
        data.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        arr_bytes = data.tobytes()
        return base64.b64encode(arr_bytes).decode("utf-8")


def overlay_bboxes(
    image: Union[str, Path, np.ndarray, ImageType], bboxes: Dict
) -> ImageType:
    r"""Plots bounding boxes on to an image.

    Parameters:
        image: the input image
        bboxes: the bounding boxes to overlay

    Returns:
        The image with the bounding boxes overlayed
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(bboxes["labels"])}

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    if "bboxes" not in bboxes:
        return image.convert("RGB")

    for label, box in zip(bboxes["labels"], bboxes["bboxes"]):
        box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
        draw.rectangle(box, outline=color[label], width=3)
        label = f"{label}"
        text_box = draw.textbbox((box[0], box[1]), text=label, font=font)
        draw.rectangle(text_box, fill=color[label])
        draw.text((text_box[0], text_box[1]), label, fill="black", font=font)
    return image.convert("RGB")


def overlay_masks(
    image: Union[str, Path, np.ndarray, ImageType], masks: Dict, alpha: float = 0.5
) -> ImageType:
    r"""Plots masks on to an image.

    Parameters:
        image: the input image
        masks: the masks to overlay
        alpha: the transparency of the overlay

    Returns:
        The image with the masks overlayed
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(masks["labels"])}
    if "masks" not in masks:
        return image.convert("RGB")

    for label, mask in zip(masks["labels"], masks["masks"]):
        if isinstance(mask, str):
            mask = np.array(Image.open(mask))
        np_mask = np.zeros((image.size[1], image.size[0], 4))
        np_mask[mask > 0, :] = color[label] + (255 * alpha,)
        mask_img = Image.fromarray(np_mask.astype(np.uint8))
        image = Image.alpha_composite(image.convert("RGBA"), mask_img)
    return image.convert("RGB")
