import inspect
import io
import json
import logging
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import pandas as pd
import requests
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

from vision_agent.tools.tool_utils import _send_inference_request
from vision_agent.utils import extract_frames_from_video
from vision_agent.utils.execute import FileSerializer, MimeType
from vision_agent.utils.image_utils import (
    b64_to_pil,
    convert_to_b64,
    denormalize_bbox,
    get_image_size,
    normalize_bbox,
    rle_decode,
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
    """'grounding_dino' is a tool that can detect and count multiple objects given a text
    prompt such as category names or referring expressions. The categories in text prompt
    are separated by commas or periods. It returns a list of bounding boxes with
    normalized coordinates, label names and associated probability scores.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults
            to 0.20.
        iou_threshold (float, optional): The threshold for the Intersection over Union
            (IoU). Defaults to 0.20.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

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
    """'grounding_sam' is a tool that can detect and segment multiple objects given a
    text prompt such as category names or referring expressions. The categories in text
    prompt are separated by commas or periods. It returns a list of bounding boxes,
    label names, mask file names and associated probability scores.

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
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
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
    captured. The frame is a numpy array.

    Parameters:
        video_uri (Union[str, Path]): The path to the video file.
        fps (float, optional): The frame rate per second to extract the frames. Defaults
            to 0.5.

    Returns:
        List[Tuple[np.ndarray, float]]: A list of tuples containing the extracted frame
            as a numpy array and the timestamp in seconds.

    Example
    -------
        >>> extract_frames("path/to/video.mp4")
        [(frame1, 0.0), (frame2, 0.5), ...]
    """

    return extract_frames_from_video(str(video_uri), fps)


def ocr(image: np.ndarray) -> List[Dict[str, Any]]:
    """'ocr' extracts text from an image. It returns a list of detected text, bounding
    boxes with normalized coordinates, and confidence scores. The results are sorted
    from top-left to bottom right.

    Parameters:
        image (np.ndarray): The image to extract text from.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the detected text, bbox
            with nornmalized coordinates, and confidence score.

    Example
    -------
        >>> ocr(image)
        [
            {'label': 'hello world', 'bbox': [0.1, 0.11, 0.35, 0.4], 'score': 0.99},
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

    ocr_results = sorted(output, key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return ocr_results


def zero_shot_counting(image: np.ndarray) -> Dict[str, Any]:
    """'zero_shot_counting' is a tool that counts the dominant foreground object given
    an image and no other information about the content. It returns only the count of
    the objects in the image.

    Parameters:
        image (np.ndarray): The image that contains lot of instances of a single object

    Returns:
        Dict[str, Any]: A dictionary containing the key 'count' and the count as a
            value. E.g. {count: 12}.

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
    """'visual_prompt_counting' is a tool that counts the dominant foreground object
    given an image and a visual prompt which is a bounding box describing the object.
    It returns only the count of the objects in the image.

    Parameters:
        image (np.ndarray): The image that contains lot of instances of a single object

    Returns:
        Dict[str, Any]: A dictionary containing the key 'count' and the count as a
            value. E.g. {count: 12}.

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
    """'image_question_answering_' is a tool that can answer questions about the visual
    contents of an image given a question and an image. It returns an answer to the
    question

    Parameters:
        image (np.ndarray): The reference image used for the question
        prompt (str): The question about the image

    Returns:
        str: A string which is the answer to the given prompt. E.g. {'text': 'This
            image contains a cat sitting on a table with a bowl of milk.'}.

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
    """'clip' is a tool that can classify an image or a cropped detection given a list
    of input classes or tags. It returns the same list of the input classes along with
    their probability scores based on image content.

    Parameters:
        image (np.ndarray): The image to classify or tag
        classes (List[str]): The list of classes or tags that is associated with the image

    Returns:
        Dict[str, Any]: A dictionary containing the labels and scores. One dictionary
            contains a list of given labels and other a list of scores.

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
    """'image_caption' is a tool that can caption an image based on its contents. It
    returns a text describing the image.

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
    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour1 = max(contours1, key=cv2.contourArea)
    largest_contour2 = max(contours2, key=cv2.contourArea)
    polygon1 = cv2.approxPolyDP(largest_contour1, 1.0, True)
    polygon2 = cv2.approxPolyDP(largest_contour2, 1.0, True)
    min_distance = np.inf

    small_polygon, larger_contour = (
        (polygon1, largest_contour2)
        if len(largest_contour1) < len(largest_contour2)
        else (polygon2, largest_contour1)
    )

    # For each point in the first polygon
    for point in small_polygon:
        # Calculate the distance to the second polygon, -1 is to invert result as point inside the polygon is positive

        distance = (
            cv2.pointPolygonTest(
                larger_contour, (point[0, 0].item(), point[0, 1].item()), True
            )
            * -1
        )

        # If the distance is negative, the point is inside the polygon, so the distance is 0
        if distance < 0:
            continue
        else:
            # Update the minimum distance if the point is outside the polygon
            min_distance = min(min_distance, distance)

    return min_distance if min_distance != np.inf else 0.0


def closest_box_distance(
    box1: List[float], box2: List[float], image_size: Tuple[int, int]
) -> float:
    """'closest_box_distance' calculates the closest distance between two bounding boxes.

    Parameters:
        box1 (List[float]): The first bounding box.
        box2 (List[float]): The second bounding box.
        image_size (Tuple[int, int]): The size of the image given as (height, width).

    Returns:
        float: The closest distance between the two bounding boxes.

    Example
    -------
        >>> closest_box_distance([100, 100, 200, 200], [300, 300, 400, 400])
        141.42
    """

    x11, y11, x12, y12 = denormalize_bbox(box1, image_size)
    x21, y21, x22, y22 = denormalize_bbox(box2, image_size)

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
    """'load_image' is a utility function that loads an image from the given file path string.

    Parameters:
        image_path (str): The path to the image.

    Returns:
        np.ndarray: The image as a NumPy array.

    Example
    -------
        >>> load_image("path/to/image.jpg")
    """
    # NOTE: sometimes the generated code pass in a NumPy array
    if isinstance(image_path, np.ndarray):
        return image_path
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def save_image(image: np.ndarray, file_path: str) -> None:
    """'save_image' is a utility function that saves an image to a file path.

    Parameters:
        image (np.ndarray): The image to save.
        file_path (str): The path to save the image file.

    Example
    -------
        >>> save_image(image)
    """
    from IPython.display import display

    pil_image = Image.fromarray(image.astype(np.uint8))
    display(pil_image)
    pil_image.save(file_path)


def save_video(
    frames: List[np.ndarray], output_video_path: Optional[str] = None, fps: float = 4
) -> str:
    """'save_video' is a utility function that saves a list of frames as a mp4 video file on disk.

    Parameters:
        frames (list[np.ndarray]): A list of frames to save.
        output_video_path (str): The path to save the video file. If not provided, a temporary file will be created.
        fps (float): The number of frames composes a second in the video.

    Returns:
        str: The path to the saved video file.

    Example
    -------
        >>> save_video(frames)
        "/tmp/tmpvideo123.mp4"
    """
    if fps <= 0:
        _LOGGER.warning(f"Invalid fps value: {fps}. Setting fps to 4 (default value).")
        fps = 4
    with ImageSequenceClip(frames, fps=fps) as video:
        if output_video_path:
            f = open(output_video_path, "wb")
        else:
            f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)  # type: ignore
        video.write_videofile(f.name, codec="libx264")
        f.close()
        _save_video_to_result(f.name)
        return f.name


def _save_video_to_result(video_uri: str) -> None:
    """Saves a video into the result of the code execution (as an intermediate output)."""
    from IPython.display import display

    serializer = FileSerializer(video_uri)
    display(
        {
            MimeType.VIDEO_MP4_B64: serializer.base64(),
            MimeType.TEXT_PLAIN: str(serializer),
        },
        raw=True,
    )


def overlay_bounding_boxes(
    image: np.ndarray, bboxes: List[Dict[str, Any]]
) -> np.ndarray:
    """'overlay_bounding_boxes' is a utility function that displays bounding boxes on
    an image.

    Parameters:
        image (np.ndarray): The image to display the bounding boxes on.
        bboxes (List[Dict[str, Any]]): A list of dictionaries containing the bounding
            boxes.

    Returns:
        np.ndarray: The image with the bounding boxes, labels and scores displayed.

    Example
    -------
        >>> image_with_bboxes = overlay_bounding_boxes(
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
    bboxes = sorted(bboxes, key=lambda x: x["label"], reverse=True)

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

        # denormalize the box if it is normalized
        box = denormalize_bbox(box, (height, width))

        draw.rectangle(box, outline=color[label], width=4)
        text = f"{label}: {scores:.2f}"
        text_box = draw.textbbox((box[0], box[1]), text=text, font=font)
        draw.rectangle((box[0], box[1], text_box[2], text_box[3]), fill=color[label])
        draw.text((box[0], box[1]), text, fill="black", font=font)
    return np.array(pil_image)


def overlay_segmentation_masks(
    image: np.ndarray, masks: List[Dict[str, Any]]
) -> np.ndarray:
    """'overlay_segmentation_masks' is a utility function that displays segmentation
    masks.

    Parameters:
        image (np.ndarray): The image to display the masks on.
        masks (List[Dict[str, Any]]): A list of dictionaries containing the masks.

    Returns:
        np.ndarray: The image with the masks displayed.

    Example
    -------
        >>> image_with_masks = overlay_segmentation_masks(
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
    masks = sorted(masks, key=lambda x: x["label"], reverse=True)

    for elt in masks:
        mask = elt["mask"]
        label = elt["label"]
        np_mask = np.zeros((pil_image.size[1], pil_image.size[0], 4))
        np_mask[mask > 0, :] = color[label] + (255 * 0.5,)
        mask_img = Image.fromarray(np_mask.astype(np.uint8))
        pil_image = Image.alpha_composite(pil_image, mask_img)
    return np.array(pil_image)


def overlay_heat_map(
    image: np.ndarray, heat_map: Dict[str, Any], alpha: float = 0.8
) -> np.ndarray:
    """'overlay_heat_map' is a utility function that displays a heat map on an image.

    Parameters:
        image (np.ndarray): The image to display the heat map on.
        heat_map (Dict[str, Any]): A dictionary containing the heat map under the key
            'heat_map'.
        alpha (float, optional): The transparency of the overlay. Defaults to 0.8.

    Returns:
        np.ndarray: The image with the heat map displayed.

    Example
    -------
        >>> image_with_heat_map = overlay_heat_map(
            image,
            {
                'heat_map': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 125, 125, 125]], dtype=uint8),
            },
        )
    """
    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

    if "heat_map" not in heat_map or len(heat_map["heat_map"]) == 0:
        return image

    pil_image = pil_image.convert("L")
    mask = Image.fromarray(heat_map["heat_map"])
    mask = mask.resize(pil_image.size)

    overlay = Image.new("RGBA", mask.size)
    odraw = ImageDraw.Draw(overlay)
    odraw.bitmap((0, 0), mask, fill=(255, 0, 0, round(alpha * 255)))
    combined = Image.alpha_composite(
        pil_image.convert("RGBA"), overlay.resize(pil_image.size)
    )
    return np.array(combined)


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

        if "Parameters:" in description:
            description = (
                description[: description.find("Parameters:")]
                .replace("\n", " ")
                .strip()
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
    save_video,
    overlay_bounding_boxes,
    overlay_segmentation_masks,
    overlay_heat_map,
]
TOOLS_DF = get_tools_df(TOOLS)  # type: ignore
TOOL_DESCRIPTIONS = get_tool_descriptions(TOOLS)  # type: ignore
TOOL_DOCSTRING = get_tool_documentation(TOOLS)  # type: ignore
UTILITIES_DOCSTRING = get_tool_documentation(
    [
        save_json,
        load_image,
        save_image,
        save_video,
        overlay_bounding_boxes,
        overlay_segmentation_masks,
        overlay_heat_map,
    ]
)
