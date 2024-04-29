import inspect
import tempfile
from importlib import resources
from typing import Any, Callable, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision_agent.image_utils import convert_to_b64, normalize_bbox
from vision_agent.tools.tool_utils import _send_inference_request

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


def grounding_dino(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.20,
    iou_threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """'grounding_dino' is a tool that can detect arbitrary objects with inputs such as
    category names or referring expressions.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults to 0.20.
        iou_threshold (float, optional): The threshold for the Intersection over Union (IoU). Defaults to 0.75.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
        bounding box of the detected objects with normalized coordinates.

    Example
    -------
    >>> grounding_dino("car. dinosaur", image)
    [{'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]}]
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


def display_bounding_boxes(
    image: np.ndarray, bboxes: List[Dict[str, Any]]
) -> np.ndarray:
    """'display_bounding_boxes' is a utility function that displays bounding boxes on an image.

    Parameters:
        image (np.ndarray): The image to display the bounding boxes on.
        bboxes (List[Dict[str, Any]]): A list of dictionaries containing the bounding boxes.

    Returns:
        np.ndarray: The image with the bounding boxes displayed.

    Example
    -------
    >>> image_with_bboxes = display_bounding_boxes(image, [{'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]}])
    """
    pil_image = Image.fromarray(image.astype(np.uint8))

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


def get_tool_documentation(funcs: List[Callable]) -> str:
    docstrings = ""
    for func in funcs:
        docstrings += f"{func.__name__}: {inspect.signature(func)}\n{func.__doc__}\n\n"

    return docstrings


TOOLS_DOCSTRING = get_tool_documentation([load_image, grounding_dino])
UTILITIES_DOCSTRING = get_tool_documentation(
    [load_image, save_image, display_bounding_boxes]
)
