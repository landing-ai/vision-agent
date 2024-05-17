"""Utility functions for image processing."""

import base64
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Union

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


def normalize_bbox(
    bbox: List[Union[int, float]], image_size: Tuple[int, ...]
) -> List[float]:
    r"""Normalize the bounding box coordinates to be between 0 and 1."""
    x1, y1, x2, y2 = bbox
    x1 = round(x1 / image_size[1], 2)
    y1 = round(y1 / image_size[0], 2)
    x2 = round(x2 / image_size[1], 2)
    y2 = round(y2 / image_size[0], 2)
    return [x1, y1, x2, y2]


def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    r"""Decode a run-length encoded mask. Returns numpy array, 1 - mask, 0 - background.

    Parameters:
        mask_rle: Run-length as string formated (start length)
        shape: The (height, width) of array to return
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


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
    elif isinstance(data, np.ndarray):
        data = Image.fromarray(data)

    if isinstance(data, Image.Image):
        buffer = BytesIO()
        data.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(
            f"Invalid input image: {data}. Input image must be a PIL Image or a numpy array."
        )


def denormalize_bbox(
    bbox: List[Union[int, float]], image_size: Tuple[int, ...]
) -> List[float]:
    r"""DeNormalize the bounding box coordinates so that they are in absolute values."""

    if len(bbox) != 4:
        raise ValueError("Bounding box must be of length 4.")

    arr = np.array(bbox)
    if np.all((arr >= 0) & (arr <= 1)):
        x1, y1, x2, y2 = bbox
        x1 = round(x1 * image_size[1])
        y1 = round(y1 * image_size[0])
        x2 = round(x2 * image_size[1])
        y2 = round(y2 * image_size[0])
        return [x1, y1, x2, y2]
    else:
        return bbox


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

    if "bboxes" not in bboxes:
        return image.convert("RGB")

    color = {
        label: COLORS[i % len(COLORS)] for i, label in enumerate(set(bboxes["labels"]))
    }

    width, height = image.size
    fontsize = max(12, int(min(width, height) / 40))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        str(resources.files("vision_agent.fonts").joinpath("default_font_ch_en.ttf")),
        fontsize,
    )

    for label, box, scores in zip(bboxes["labels"], bboxes["bboxes"], bboxes["scores"]):
        box = [
            int(box[0] * width),
            int(box[1] * height),
            int(box[2] * width),
            int(box[3] * height),
        ]
        draw.rectangle(box, outline=color[label], width=4)
        text = f"{label}: {scores:.2f}"
        text_box = draw.textbbox((box[0], box[1]), text=text, font=font)
        draw.rectangle((box[0], box[1], text_box[2], text_box[3]), fill=color[label])
        draw.text((box[0], box[1]), text, fill="black", font=font)
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

    if "masks" not in masks:
        return image.convert("RGB")

    if "labels" not in masks:
        masks["labels"] = [""] * len(masks["masks"])

    color = {
        label: COLORS[i % len(COLORS)] for i, label in enumerate(set(masks["labels"]))
    }

    for label, mask in zip(masks["labels"], masks["masks"]):
        if isinstance(mask, str) or isinstance(mask, Path):
            mask = np.array(Image.open(mask))
        np_mask = np.zeros((image.size[1], image.size[0], 4))
        np_mask[mask > 0, :] = color[label] + (255 * alpha,)
        mask_img = Image.fromarray(np_mask.astype(np.uint8))
        image = Image.alpha_composite(image.convert("RGBA"), mask_img)
    return image.convert("RGB")


def overlay_heat_map(
    image: Union[str, Path, np.ndarray, ImageType], heat_map: Dict, alpha: float = 0.8
) -> ImageType:
    r"""Plots heat map on to an image.

    Parameters:
        image: the input image
        masks: the heatmap to overlay
        alpha: the transparency of the overlay

    Returns:
        The image with the heatmap overlayed
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if "heat_map" not in heat_map or len(heat_map["heat_map"]) == 0:
        return image.convert("RGB")

    image = image.convert("L")
    # Only one heat map per image, so no need to loop through masks
    mask = Image.fromarray(heat_map["heat_map"][0])

    overlay = Image.new("RGBA", mask.size)
    odraw = ImageDraw.Draw(overlay)
    odraw.bitmap(
        (0, 0), mask, fill=(255, 0, 0, round(alpha * 255))
    )  # fill=(R, G, B, Alpha)
    combined = Image.alpha_composite(image.convert("RGBA"), overlay.resize(image.size))

    return combined.convert("RGB")
