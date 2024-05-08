import inspect
import io
import logging
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

from vision_agent.tools.tool_utils import _send_inference_request
from vision_agent.utils import extract_frames_from_video
from vision_agent.utils.image_utils import convert_to_b64, normalize_bbox, rle_decode

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
_API_KEY = "land_sk_WVYwP00xA3iXely2vuar6YUDZ3MJT9yLX6oW5noUkwICzYLiDV"
_OCR_URL = "https://app.landing.ai/ocr/v1/detect-text"
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


def grounding_dino(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.20,
    iou_threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """'grounding_dino' is a tool that can detect and count objects given a text prompt
    such as category names or referring expressions. It returns a list and count of
    bounding boxes, label names and associated probability scores.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults
            to 0.20.
        iou_threshold (float, optional): The threshold for the Intersection over Union
            (IoU). Defaults to 0.75.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
        bounding box of the detected objects with normalized coordinates
        (x1, y1, x2, y2).

    Example
    -------
    >>> grounding_dino("car. dinosaur", image)
    [
        {'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]},
        {'score': 0.98, 'label': 'car', 'bbox': [0.2, 0.21, 0.45, 0.5},
    ]
    """
    image_size = image.shape[:2]
    image_b64 = convert_to_b64(Image.fromarray(image))
    request_data = {
        "prompt": prompt,
        "image": image_b64,
        "tool": "visual_grounding",
        "kwargs": {"box_threshold": box_threshold, "iou_threshold": iou_threshold},
    }
    data: Dict[str, Any] = _send_inference_request(request_data, "tools")
    return_data = []
    for i in range(len(data["bboxes"])):
        return_data.append(
            {
                "score": round(data["scores"][i], 2),
                "label": data["labels"][i],
                "bbox": normalize_bbox(data["bboxes"][i], image_size),
            }
        )
    return return_data


def grounding_sam(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.20,
    iou_threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """'grounding_sam' is a tool that can detect and segment objects given a text
    prompt such as category names or referring expressions. It returns a list of
    bounding boxes, label names and masks file names and associated probability scores.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults
            to 0.20.
        iou_threshold (float, optional): The threshold for the Intersection over Union
            (IoU). Defaults to 0.75.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
        bounding box, and mask of the detected objects with normalized coordinates
        (x1, y1, x2, y2).

    Example
    -------
    >>> grounding_sam("car. dinosaur", image)
    [
        {
            'score': 0.99,
            'label': 'dinosaur',
            'bbox': [0.1, 0.11, 0.35, 0.4],
            'mask': array([[0, 0, 0, ..., 0, 0, 0],
                [0, 0, 0, ..., 0, 0, 0],
                ...,
                [0, 0, 0, ..., 0, 0, 0],
                [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
        },
    ]
    """
    image_size = image.shape[:2]
    image_b64 = convert_to_b64(Image.fromarray(image))
    request_data = {
        "prompt": prompt,
        "image": image_b64,
        "tool": "visual_grounding_segment",
        "kwargs": {"box_threshold": box_threshold, "iou_threshold": iou_threshold},
    }
    data: Dict[str, Any] = _send_inference_request(request_data, "tools")
    return_data = []
    for i in range(len(data["bboxes"])):
        return_data.append(
            {
                "score": round(data["scores"][i], 2),
                "label": data["labels"][i],
                "bbox": normalize_bbox(data["bboxes"][i], image_size),
                "mask": rle_decode(mask_rle=data["masks"][i], shape=data["mask_shape"]),
            }
        )
    return return_data


def extract_frames(
    video_uri: Union[str, Path], fps: float = 0.5
) -> List[Tuple[np.ndarray, float]]:
    """'extract_frames' extracts frames from a video, returns a list of tuples (frame,
    timestamp), where timestamp is the relative time in seconds where the frame was
    captured. The frame is a local image file path.

    Parameters:
        video_uri (Union[str, Path]): The path to the video file.
        fps (float, optional): The frame rate per second to extract the frames. Defaults
            to 0.5.

    Returns:
        List[Tuple[np.ndarray, float]]: A list of tuples containing the extracted frame
        and the timestamp in seconds.

    Example
    -------
    >>> extract_frames("path/to/video.mp4")
    [(frame1, 0.0), (frame2, 0.5), ...]
    """

    return extract_frames_from_video(str(video_uri), fps)


def ocr(image: np.ndarray) -> List[Dict[str, Any]]:
    """'ocr' extracts text from an image. It returns a list of detected text, bounding
    boxes, and confidence scores.

    Parameters:
        image (np.ndarray): The image to extract text from.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the detected text, bbox,
        and confidence score.

    Example
    -------
    >>> ocr(image)
    [
        {'label': 'some text', 'bbox': [0.1, 0.11, 0.35, 0.4], 'score': 0.99},
    ]
    """

    pil_image = Image.fromarray(image).convert("RGB")
    image_size = pil_image.size[::-1]
    image_buffer = io.BytesIO()
    pil_image.save(image_buffer, format="PNG")
    buffer_bytes = image_buffer.getvalue()
    image_buffer.close()

    res = requests.post(
        _OCR_URL,
        files={"images": buffer_bytes},
        data={"language": "en"},
        headers={"contentType": "multipart/form-data", "apikey": _API_KEY},
    )

    if res.status_code != 200:
        raise ValueError(f"OCR request failed with status code {res.status_code}")

    data = res.json()
    output = []
    for det in data[0]:
        label = det["text"]
        box = [
            det["location"][0]["x"],
            det["location"][0]["y"],
            det["location"][2]["x"],
            det["location"][2]["y"],
        ]
        box = normalize_bbox(box, image_size)
        output.append({"label": label, "bbox": box, "score": round(det["score"], 2)})

    return output


# Utility and visualization functions


def load_image(image_path: str) -> np.ndarray:
    """'load_image' is a utility function that loads an image from the given path.

    Parameters:
        image_path (str): The path to the image.

    Returns:
        np.ndarray: The image as a NumPy array.

    Example
    -------
    >>> load_image("path/to/image.jpg")
    """

    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def save_image(image: np.ndarray) -> str:
    """'save_image' is a utility function that saves an image as a temporary file.

    Parameters:
        image (np.ndarray): The image to save.

    Returns:
        str: The path to the saved image.

    Example
    -------
    >>> save_image(image)
    "/tmp/tmpabc123.png"
    """

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image.save(f, "PNG")
    return f.name


def overlay_bounding_boxes(
    image: np.ndarray, bboxes: List[Dict[str, Any]]
) -> np.ndarray:
    """'display_bounding_boxes' is a utility function that displays bounding boxes on
    an image.

    Parameters:
        image (np.ndarray): The image to display the bounding boxes on.
        bboxes (List[Dict[str, Any]]): A list of dictionaries containing the bounding
            boxes.

    Returns:
        np.ndarray: The image with the bounding boxes, labels and scores displayed.

    Example
    -------
    >>> image_with_bboxes = display_bounding_boxes(
        image, [{'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]}],
    )
    """
    pil_image = Image.fromarray(image.astype(np.uint8))

    if len(set([box["label"] for box in bboxes])) > len(COLORS):
        _LOGGER.warning(
            "Number of unique labels exceeds the number of available colors. Some labels may have the same color."
        )

    color = {
        label: COLORS[i % len(COLORS)]
        for i, label in enumerate(set([box["label"] for box in bboxes]))
    }

    width, height = pil_image.size
    fontsize = max(12, int(min(width, height) / 40))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(
        str(resources.files("vision_agent.fonts").joinpath("default_font_ch_en.ttf")),
        fontsize,
    )

    for elt in bboxes:
        label = elt["label"]
        box = elt["bbox"]
        scores = elt["score"]

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
    return np.array(pil_image.convert("RGB"))


def overlay_segmentation_masks(
    image: np.ndarray, masks: List[Dict[str, Any]]
) -> np.ndarray:
    """'display_segmentation_masks' is a utility function that displays segmentation
    masks.

    Parameters:
        image (np.ndarray): The image to display the masks on.
        masks (List[Dict[str, Any]]): A list of dictionaries containing the masks.

    Returns:
        np.ndarray: The image with the masks displayed.

    Example
    -------
    >>> image_with_masks = display_segmentation_masks(
        image,
        [{
            'score': 0.99,
            'label': 'dinosaur',
            'mask': array([[0, 0, 0, ..., 0, 0, 0],
                [0, 0, 0, ..., 0, 0, 0],
                ...,
                [0, 0, 0, ..., 0, 0, 0],
                [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
        }],
    )
    """
    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGBA")

    if len(set([mask["label"] for mask in masks])) > len(COLORS):
        _LOGGER.warning(
            "Number of unique labels exceeds the number of available colors. Some labels may have the same color."
        )

    color = {
        label: COLORS[i % len(COLORS)]
        for i, label in enumerate(set([mask["label"] for mask in masks]))
    }

    for elt in masks:
        mask = elt["mask"]
        label = elt["label"]
        np_mask = np.zeros((pil_image.size[1], pil_image.size[0], 4))
        np_mask[mask > 0, :] = color[label] + (255 * 0.5,)
        mask_img = Image.fromarray(np_mask.astype(np.uint8))
        pil_image = Image.alpha_composite(pil_image, mask_img)
    return np.array(pil_image.convert("RGB"))


def get_tool_documentation(funcs: List[Callable[..., Any]]) -> str:
    docstrings = ""
    for func in funcs:
        docstrings += f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}\n\n"

    return docstrings


def get_tool_descriptions(funcs: List[Callable[..., Any]]) -> str:
    descriptions = ""
    for func in funcs:
        description = func.__doc__
        if description is None:
            description = ""

        description = (
            description[: description.find("Parameters:")].replace("\n", " ").strip()
        )
        description = " ".join(description.split())
        descriptions += f"- {func.__name__}{inspect.signature(func)}: {description}\n"
    return descriptions


def get_tools_df(funcs: List[Callable[..., Any]]) -> pd.DataFrame:
    data: Dict[str, List[str]] = {"desc": [], "doc": []}

    for func in funcs:
        desc = func.__doc__
        if desc is None:
            desc = ""
        desc = desc[: desc.find("Parameters:")].replace("\n", " ").strip()
        desc = " ".join(desc.split())

        doc = f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}"
        data["desc"].append(desc)
        data["doc"].append(doc)

    return pd.DataFrame(data)  # type: ignore


TOOLS = [
    grounding_dino,
    grounding_sam,
    extract_frames,
    ocr,
    load_image,
    save_image,
    overlay_bounding_boxes,
    overlay_segmentation_masks,
]
TOOLS_DF = get_tools_df(TOOLS)  # type: ignore
TOOL_DESCRIPTIONS = get_tool_descriptions(TOOLS)  # type: ignore
TOOL_DOCSTRING = get_tool_documentation(TOOLS)  # type: ignore
UTILITIES_DOCSTRING = get_tool_documentation(
    [load_image, save_image, overlay_bounding_boxes]
)
