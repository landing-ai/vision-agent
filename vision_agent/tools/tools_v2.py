import inspect
import io
import json
import logging
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance  # type: ignore

from vision_agent.tools.tool_utils import _send_inference_request
from vision_agent.utils import extract_frames_from_video
from vision_agent.utils.image_utils import (
    convert_to_b64,
    normalize_bbox,
    rle_decode,
    b64_to_pil,
    get_image_size,
    denormalize_bbox,
)

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
    iou_threshold: float = 0.20,
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
            (IoU). Defaults to 0.20.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
        bounding box of the detected objects with normalized coordinates
        (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left and
        xmax and ymax are the coordinates of the bottom-right of the bounding box.

    Example
    -------
    >>> grounding_dino("car. dinosaur", image)
    [
        {'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]},
        {'score': 0.98, 'label': 'car', 'bbox': [0.2, 0.21, 0.45, 0.5},
    ]
    """
    image_size = image.shape[:2]
    image_b64 = convert_to_b64(image)
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
    iou_threshold: float = 0.20,
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
            (IoU). Defaults to 0.20.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
        bounding box, and mask of the detected objects with normalized coordinates
        (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left and
        xmax and ymax are the coordinates of the bottom-right of the bounding box.
        The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
        the background.

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
    image_b64 = convert_to_b64(image)
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


def zero_shot_counting(image: np.ndarray) -> Dict[str, Any]:
    """'zero_shot_counting' is a tool that counts the dominant foreground object given an image and no other information about the content.
    It returns only the count of the objects in the image.

    Parameters:
        image (np.ndarray): The image that contains lot of instances of a single object

    Returns:
        Dict[str, Any]: A dictionary containing the key 'count' and the count as a value. E.g. {count: 12}.

    Example
    -------
    >>> zero_shot_counting(image)
    {'count': 45},

    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "zero_shot_counting",
    }
    resp_data = _send_inference_request(data, "tools")
    resp_data["heat_map"] = np.array(b64_to_pil(resp_data["heat_map"][0]))
    return resp_data


def visual_prompt_counting(
    image: np.ndarray, visual_prompt: Dict[str, List[float]]
) -> Dict[str, Any]:
    """'visual_prompt_counting' is a tool that counts the dominant foreground object given an image and a visual prompt which is a bounding box describing the object.
    It returns only the count of the objects in the image.

    Parameters:
        image (np.ndarray): The image that contains lot of instances of a single object

    Returns:
        Dict[str, Any]: A dictionary containing the key 'count' and the count as a value. E.g. {count: 12}.

    Example
    -------
    >>> visual_prompt_counting(image, {"bbox": [0.1, 0.1, 0.4, 0.42]})
    {'count': 45},

    """

    image_size = get_image_size(image)
    bbox = visual_prompt["bbox"]
    bbox_str = ", ".join(map(str, denormalize_bbox(bbox, image_size)))
    image_b64 = convert_to_b64(image)

    data = {
        "image": image_b64,
        "prompt": bbox_str,
        "tool": "few_shot_counting",
    }
    resp_data = _send_inference_request(data, "tools")
    resp_data["heat_map"] = np.array(b64_to_pil(resp_data["heat_map"][0]))
    return resp_data


def image_question_answering(image: np.ndarray, prompt: str) -> str:
    """'image_question_answering_' is a tool that can answer questions about the visual contents of an image given a question and an image.
    It returns an answer to the question

    Parameters:
        image (np.ndarray): The reference image used for the question
        prompt (str): The question about the image

    Returns:
        str: A string which is the answer to the given prompt. E.g. {'text': 'This image contains a cat sitting on a table with a bowl of milk.'}.

    Example
    -------
    >>> image_question_answering(image, 'What is the cat doing ?')
    'drinking milk'

    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "prompt": prompt,
        "tool": "image_question_answering",
    }

    answer = _send_inference_request(data, "tools")
    return answer["text"][0]  # type: ignore


def clip(image: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """'clip' is a tool that can classify an image given a list of input classes or tags.
    It returns the same list of the input classes along with their probability scores based on image content.

    Parameters:
        image (np.ndarray): The image to classify or tag
        classes (List[str]): The list of classes or tags that is associated with the image

    Returns:
        Dict[str, Any]: A dictionary containing the labels and scores. One dictionary contains a list of given labels and other a list of scores.

    Example
    -------
    >>> clip(image, ['dog', 'cat', 'bird'])
    {"labels": ["dog", "cat", "bird"], "scores": [0.68, 0.30, 0.02]},

    """

    image_b64 = convert_to_b64(image)
    data = {
        "prompt": ",".join(classes),
        "image": image_b64,
        "tool": "closed_set_image_classification",
    }
    resp_data = _send_inference_request(data, "tools")
    resp_data["scores"] = [round(prob, 4) for prob in resp_data["scores"]]
    return resp_data


def image_caption(image: np.ndarray) -> str:
    """'image_caption' is a tool that can caption an image based on its contents.
    It returns a text describing the image.

    Parameters:
        image (np.ndarray): The image to caption

    Returns:
       str: A string which is the caption for the given image.

    Example
    -------
    >>> image_caption(image)
    'This image contains a cat sitting on a table with a bowl of milk.'

    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "image_captioning",
    }

    answer = _send_inference_request(data, "tools")
    return answer["text"][0]  # type: ignore


def closest_mask_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """'closest_mask_distance' calculates the closest distance between two masks.

    Parameters:
        mask1 (np.ndarray): The first mask.
        mask2 (np.ndarray): The second mask.

    Returns:
        float: The closest distance between the two masks.

    Example
    -------
    >>> closest_mask_distance(mask1, mask2)
    0.5
    """

    mask1 = np.clip(mask1, 0, 1)
    mask2 = np.clip(mask2, 0, 1)
    mask1_points = np.transpose(np.nonzero(mask1))
    mask2_points = np.transpose(np.nonzero(mask2))
    dist_matrix = distance.cdist(mask1_points, mask2_points, "euclidean")
    return cast(float, np.min(dist_matrix))


def closest_box_distance(box1: List[float], box2: List[float]) -> float:
    """'closest_box_distance' calculates the closest distance between two bounding boxes.

    Parameters:
        box1 (List[float]): The first bounding box.
        box2 (List[float]): The second bounding box.

    Returns:
        float: The closest distance between the two bounding boxes.

    Example
    -------
    >>> closest_box_distance([100, 100, 200, 200], [300, 300, 400, 400])
    141.42
    """

    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    horizontal_distance = np.max([0, x21 - x12, x11 - x22])
    vertical_distance = np.max([0, y21 - y12, y11 - y22])
    return cast(float, np.sqrt(horizontal_distance**2 + vertical_distance**2))


# Utility and visualization functions


def save_json(data: Any, file_path: str) -> None:
    """'save_json' is a utility function that saves data as a JSON file. It is helpful
    for saving data that contains NumPy arrays which are not JSON serializable.

    Parameters:
        data (Any): The data to save.
        file_path (str): The path to save the JSON file.

    Example
    -------
    >>> save_json(data, "path/to/file.json")
    """

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any):  # type: ignore
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return json.JSONEncoder.default(self, obj)

    with open(file_path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)


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
    clip,
    zero_shot_counting,
    visual_prompt_counting,
    image_question_answering,
    image_caption,
    closest_mask_distance,
    closest_box_distance,
    save_json,
    load_image,
    save_image,
    overlay_bounding_boxes,
    overlay_segmentation_masks,
]
TOOLS_DF = get_tools_df(TOOLS)  # type: ignore
TOOL_DESCRIPTIONS = get_tool_descriptions(TOOLS)  # type: ignore
TOOL_DOCSTRING = get_tool_documentation(TOOLS)  # type: ignore
UTILITIES_DOCSTRING = get_tool_documentation(
    [save_json, load_image, save_image, overlay_bounding_boxes]
)
