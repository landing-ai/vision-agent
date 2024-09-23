"""Utility functions for image processing."""

import base64
import io
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as ImageType

from vision_agent.utils import extract_frames_from_video

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


def rle_decode_array(rle: Dict[str, List[int]]) -> np.ndarray:
    r"""Decode a run-length encoded mask. Returns numpy array, 1 - mask, 0 - background.

    Parameters:
        rle: The run-length encoded mask.
    """
    size = rle["size"]
    counts = rle["counts"]

    total_elements = size[0] * size[1]
    flattened_mask = np.zeros(total_elements, dtype=np.uint8)

    current_pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            flattened_mask[current_pos : current_pos + count] = 1
        current_pos += count

    binary_mask = flattened_mask.reshape(size, order="F")
    return binary_mask


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


def numpy_to_bytes(image: np.ndarray) -> bytes:
    pil_image = Image.fromarray(image).convert("RGB")
    image_buffer = io.BytesIO()
    pil_image.save(image_buffer, format="PNG")
    buffer_bytes = image_buffer.getvalue()
    image_buffer.close()
    return buffer_bytes


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


def encode_image_bytes(image: bytes, resize: Optional[int] = None) -> str:
    if resize is not None:
        image_pil = Image.open(io.BytesIO(image)).convert("RGB")
        if image_pil.size[0] > resize or image_pil.size[1] > resize:
            image_pil.thumbnail((resize, resize))
    else:
        image_pil = Image.open(io.BytesIO(image)).convert("RGB")
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_image


def encode_media(media: Union[str, Path], resize: Optional[int] = None) -> str:
    if isinstance(media, str) and media.startswith(("http", "https")):
        # for mp4 video url, we assume there is a same url but ends with png
        # vision-agent-ui will upload this png when uploading the video
        if media.endswith((".mp4", "mov")) and media.find("vision-agent-dev.s3") != -1:
            return media[:-4] + ".png"
        return media

    # if media is already a base64 encoded image return
    if isinstance(media, str) and media.startswith("data:image/"):
        return media

    extension = "png"
    extension = Path(media).suffix
    if extension.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".mp4",
        ".mov",
    }:
        raise ValueError(f"Unsupported image extension: {extension}")

    image_bytes = b""
    if extension.lower() in {".mp4", ".mov"}:
        frames = extract_frames_from_video(str(media), fps=1)
        image = frames[len(frames) // 2]
        buffer = io.BytesIO()
        if resize is not None:
            image_pil = Image.fromarray(image[0]).convert("RGB")
            if image_pil.size[0] > resize or image_pil.size[1] > resize:
                image_pil.thumbnail((resize, resize))
        else:
            image_pil = Image.fromarray(image[0]).convert("RGB")
        image_pil.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    else:
        image_bytes = open(media, "rb").read()
    return encode_image_bytes(image_bytes, resize=resize)


def denormalize_bbox(
    bbox: List[Union[int, float]], image_size: Tuple[int, ...]
) -> List[float]:
    r"""DeNormalize the bounding box coordinates so that they are in absolute values."""

    if len(bbox) != 4:
        raise ValueError("Bounding box must be of length 4.")

    arr = np.array(bbox)
    if np.all((arr[:2] >= 0) & (arr[:2] <= 1)):
        x1, y1, x2, y2 = bbox
        x1 = round(x1 * image_size[1])
        y1 = round(y1 * image_size[0])
        x2 = round(x2 * image_size[1])
        y2 = round(y2 * image_size[0])
        return [x1, y1, x2, y2]
    else:
        return bbox


def convert_quad_box_to_bbox(quad_box: List[Union[int, float]]) -> List[float]:
    r"""Convert a quadrilateral bounding box to a rectangular bounding box.

    Parameters:
        quad_box: the quadrilateral bounding box

    Returns:
        The rectangular bounding box
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = quad_box
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return [x_min, y_min, x_max, y_max]


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
        heat_map: the heatmap to overlay
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
