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
from pillow_heif import register_heif_opener  # type: ignore
from pytube import YouTube  # type: ignore

from vision_agent.tools.tool_utils import send_inference_request
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

register_heif_opener()

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
    model_size: str = "large",
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
        model_size (str, optional): The size of the model to use.

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
    if model_size not in ["large", "tiny"]:
        raise ValueError("model_size must be either 'large' or 'tiny'")
    request_data = {
        "prompt": prompt,
        "image": image_b64,
        "tool": (
            "visual_grounding" if model_size == "large" else "visual_grounding_tiny"
        ),
        "kwargs": {"box_threshold": box_threshold, "iou_threshold": iou_threshold},
    }
    data: Dict[str, Any] = send_inference_request(request_data, "tools")
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


def owl_v2(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.10,
    iou_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """'owl_v2' is a tool that can detect and count multiple objects given a text
    prompt such as category names or referring expressions. The categories in text prompt
    are separated by commas. It returns a list of bounding boxes with
    normalized coordinates, label names and associated probability scores.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults
            to 0.10.
        iou_threshold (float, optional): The threshold for the Intersection over Union
            (IoU). Defaults to 0.10.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> owl_v2("car. dinosaur", image)
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
        "tool": "open_vocab_detection",
        "kwargs": {"box_threshold": box_threshold, "iou_threshold": iou_threshold},
    }
    data: Dict[str, Any] = send_inference_request(request_data, "tools")
    return_data = []
    for i in range(len(data["bboxes"])):
        return_data.append(
            {
                "score": round(data["scores"][i], 2),
                "label": data["labels"][i].strip(),
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
    """'grounding_sam' is a tool that can segment multiple objects given a
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
    data: Dict[str, Any] = send_inference_request(request_data, "tools")
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
    """'extract_frames' extracts frames from a video which can be a file path or youtube
    link, returns a list of tuples (frame, timestamp), where timestamp is the relative
    time in seconds where the frame was captured. The frame is a numpy array.

    Parameters:
        video_uri (Union[str, Path]): The path to the video file or youtube link
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

    if str(video_uri).startswith(
        (
            "http://www.youtube.com/",
            "https://www.youtube.com/",
            "http://youtu.be/",
            "https://youtu.be/",
        )
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            yt = YouTube(str(video_uri))
            # Download the highest resolution video
            video = (
                yt.streams.filter(progressive=True, file_extension="mp4")
                .order_by("resolution")
                .desc()
                .first()
            )
            if not video:
                raise Exception("No suitable video stream found")
            video_file_path = video.download(output_path=temp_dir)

            return extract_frames_from_video(video_file_path, fps)

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


def loca_zero_shot_counting(image: np.ndarray) -> Dict[str, Any]:
    """'loca_zero_shot_counting' is a tool that counts the dominant foreground object given
    an image and no other information about the content. It returns only the count of
    the objects in the image.

    Parameters:
        image (np.ndarray): The image that contains lot of instances of a single object

    Returns:
        Dict[str, Any]: A dictionary containing the key 'count' and the count as a
            value. E.g. {count: 12}.

    Example
    -------
        >>> loca_zero_shot_counting(image)
        {'count': 45},
    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "zero_shot_counting",
    }
    resp_data = send_inference_request(data, "tools")
    resp_data["heat_map"] = np.array(b64_to_pil(resp_data["heat_map"][0]))
    return resp_data


def loca_visual_prompt_counting(
    image: np.ndarray, visual_prompt: Dict[str, List[float]]
) -> Dict[str, Any]:
    """'loca_visual_prompt_counting' is a tool that counts the dominant foreground object
    given an image and a visual prompt which is a bounding box describing the object.
    It returns only the count of the objects in the image.

    Parameters:
        image (np.ndarray): The image that contains lot of instances of a single object

    Returns:
        Dict[str, Any]: A dictionary containing the key 'count' and the count as a
            value. E.g. {count: 12}.

    Example
    -------
        >>> loca_visual_prompt_counting(image, {"bbox": [0.1, 0.1, 0.4, 0.42]})
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
    resp_data = send_inference_request(data, "tools")
    resp_data["heat_map"] = np.array(b64_to_pil(resp_data["heat_map"][0]))
    return resp_data


def florencev2_roberta_vqa(prompt: str, image: np.ndarray) -> str:
    """'florencev2_roberta_vqa' is a tool that takes an image and analyzes
    its contents, generates detailed captions and then tries to answer the given
    question using the generated context. It returns text as an answer to the question.

    Parameters:
        prompt (str): The question about the image
        image (np.ndarray): The reference image used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> florencev2_roberta_vqa('What is the top left animal in this image ?', image)
        'white tiger'
    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "prompt": prompt,
        "tool": "image_question_answering_with_context",
    }

    answer = send_inference_request(data, "tools")
    return answer["text"][0]  # type: ignore


def git_vqa_v2(prompt: str, image: np.ndarray) -> str:
    """'git_vqa_v2' is a tool that can answer questions about the visual
    contents of an image given a question and an image. It returns an answer to the
    question

    Parameters:
        prompt (str): The question about the image
        image (np.ndarray): The reference image used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> git_vqa_v2('What is the cat doing ?', image)
        'drinking milk'
    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "prompt": prompt,
        "tool": "image_question_answering",
    }

    answer = send_inference_request(data, "tools")
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
    resp_data = send_inference_request(data, "tools")
    resp_data["scores"] = [round(prob, 4) for prob in resp_data["scores"]]
    return resp_data


def vit_image_classification(image: np.ndarray) -> Dict[str, Any]:
    """'vit_image_classification' is a tool that can classify an image. It returns a
    list of classes and their probability scores based on image content.

    Parameters:
        image (np.ndarray): The image to classify or tag

    Returns:
        Dict[str, Any]: A dictionary containing the labels and scores. One dictionary
            contains a list of labels and other a list of scores.

    Example
    -------
        >>> vit_image_classification(image)
        {"labels": ["leopard", "lemur, otter", "bird"], "scores": [0.68, 0.30, 0.02]},
    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "image_classification",
    }
    resp_data = send_inference_request(data, "tools")
    resp_data["scores"] = [round(prob, 4) for prob in resp_data["scores"]]
    return resp_data


def vit_nsfw_classification(image: np.ndarray) -> Dict[str, Any]:
    """'vit_nsfw_classification' is a tool that can classify an image as 'nsfw' or 'normal'.
    It returns the predicted label and their probability scores based on image content.

    Parameters:
        image (np.ndarray): The image to classify or tag

    Returns:
        Dict[str, Any]: A dictionary containing the labels and scores. One dictionary
            contains a list of labels and other a list of scores.

    Example
    -------
        >>> vit_nsfw_classification(image)
        {"labels": "normal", "scores": 0.68},
    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "nsfw_image_classification",
    }
    resp_data = send_inference_request(data, "tools")
    resp_data["scores"] = round(resp_data["scores"], 4)
    return resp_data


def blip_image_caption(image: np.ndarray) -> str:
    """'blip_image_caption' is a tool that can caption an image based on its contents. It
    returns a text describing the image.

    Parameters:
        image (np.ndarray): The image to caption

    Returns:
       str: A string which is the caption for the given image.

    Example
    -------
        >>> blip_image_caption(image)
        'This image contains a cat sitting on a table with a bowl of milk.'
    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "image_captioning",
    }

    answer = send_inference_request(data, "tools")
    return answer["text"][0]  # type: ignore


def florencev2_image_caption(image: np.ndarray, detail_caption: bool = True) -> str:
    """'florencev2_image_caption' is a tool that can caption or describe an image based
    on its contents. It returns a text describing the image.

    Parameters:
        image (np.ndarray): The image to caption
        detail_caption (bool): If True, the caption will be as detailed as possible else
            the caption will be a brief description.

    Returns:
       str: A string which is the caption for the given image.

    Example
    -------
        >>> florencev2_image_caption(image, False)
        'This image contains a cat sitting on a table with a bowl of milk.'
    """
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "florence2_image_captioning",
        "detail_caption": detail_caption,
    }

    answer = send_inference_request(data, "tools")
    return answer["text"][0]  # type: ignore


def florencev2_object_detection(image: np.ndarray) -> List[Dict[str, Any]]:
    """'florencev2_object_detection' is a tool that can detect common objects in an
    image without any text prompt or thresholding. It returns a list of detected objects
    as labels and their location as bounding boxes.

    Parameters:
        image (np.ndarray): The image to used to detect objects

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box. The scores are always 1.0 and cannot be thresholded

    Example
    -------
        >>> florencev2_object_detection(image)
        [
            {'score': 1.0, 'label': 'window', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 1.0, 'label': 'car', 'bbox': [0.2, 0.21, 0.45, 0.5},
            {'score': 1.0, 'label': 'person', 'bbox': [0.34, 0.21, 0.85, 0.5},
        ]
    """
    image_size = image.shape[:2]
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "object_detection",
    }

    answer = send_inference_request(data, "tools")
    return_data = []
    for i in range(len(answer["bboxes"])):
        return_data.append(
            {
                "score": round(answer["scores"][i], 2),
                "label": answer["labels"][i],
                "bbox": normalize_bbox(answer["bboxes"][i], image_size),
            }
        )
    return return_data


def detr_segmentation(image: np.ndarray) -> List[Dict[str, Any]]:
    """'detr_segmentation' is a tool that can segment common objects in an
    image without any text prompt. It returns a list of detected objects
    as labels, their regions as masks and their scores.

    Parameters:
        image (np.ndarray): The image used to segment things and objects

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label
            and mask of the detected objects. The mask is binary 2D numpy array where 1
            indicates the object and 0 indicates the background.

    Example
    -------
        >>> detr_segmentation(image)
        [
            {
                'score': 0.45,
                'label': 'window',
                'mask': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            },
            {
                'score': 0.70,
                'label': 'bird',
                'mask': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            },
        ]
    """
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "panoptic_segmentation",
    }

    answer = send_inference_request(data, "tools")
    return_data = []

    for i in range(len(answer["scores"])):
        return_data.append(
            {
                "score": round(answer["scores"][i], 2),
                "label": answer["labels"][i],
                "mask": rle_decode(
                    mask_rle=answer["masks"][i], shape=answer["mask_shape"][0]
                ),
            }
        )
    return return_data


def depth_anything_v2(image: np.ndarray) -> np.ndarray:
    """'depth_anything_v2' is a tool that runs depth_anythingv2 model to generate a
    depth image from a given RGB image. The returned depth image is monochrome and
    represents depth values as pixel intesities with pixel values ranging from 0 to 255.

    Parameters:
        image (np.ndarray): The image to used to generate depth image

    Returns:
        np.ndarray: A grayscale depth image with pixel values ranging from 0 to 255.

    Example
    -------
        >>> depth_anything_v2(image)
        array([[0, 0, 0, ..., 0, 0, 0],
                [0, 20, 24, ..., 0, 100, 103],
                ...,
                [10, 11, 15, ..., 202, 202, 205],
                [10, 10, 10, ..., 200, 200, 200]], dtype=uint8),
    """
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "generate_depth",
    }

    answer = send_inference_request(data, "tools")
    return_data = np.array(b64_to_pil(answer["masks"][0]).convert("L"))
    return return_data


def generate_soft_edge_image(image: np.ndarray) -> np.ndarray:
    """'generate_soft_edge_image' is a tool that runs Holistically Nested edge detection
    to generate a soft edge image (HED) from a given RGB image. The returned image is
    monochrome and represents object boundaries as soft white edges on black background

    Parameters:
        image (np.ndarray): The image to used to generate soft edge image

    Returns:
        np.ndarray: A soft edge image with pixel values ranging from 0 to 255.

    Example
    -------
        >>> generate_soft_edge_image(image)
        array([[0, 0, 0, ..., 0, 0, 0],
                [0, 20, 24, ..., 0, 100, 103],
                ...,
                [10, 11, 15, ..., 202, 202, 205],
                [10, 10, 10, ..., 200, 200, 200]], dtype=uint8),
    """
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "generate_hed",
    }

    answer = send_inference_request(data, "tools")
    return_data = np.array(b64_to_pil(answer["masks"][0]).convert("L"))
    return return_data


def dpt_hybrid_midas(image: np.ndarray) -> np.ndarray:
    """'dpt_hybrid_midas' is a tool that generates a normal mapped from a given RGB
    image. The returned RGB image is texture mapped image of the surface normals and the
    RGB values represent the surface normals in the x, y, z directions.

    Parameters:
        image (np.ndarray): The image to used to generate normal image

    Returns:
        np.ndarray: A mapped normal image with RGB pixel values indicating surface
        normals in x, y, z directions.

    Example
    -------
        >>> dpt_hybrid_midas(image)
        array([[0, 0, 0, ..., 0, 0, 0],
                [0, 20, 24, ..., 0, 100, 103],
                ...,
                [10, 11, 15, ..., 202, 202, 205],
                [10, 10, 10, ..., 200, 200, 200]], dtype=uint8),
    """
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "generate_normal",
    }

    answer = send_inference_request(data, "tools")
    return_data = np.array(b64_to_pil(answer["masks"][0]).convert("RGB"))
    return return_data


def generate_pose_image(image: np.ndarray) -> np.ndarray:
    """'generate_pose_image' is a tool that generates a open pose bone/stick image from
    a given RGB image. The returned bone image is RGB with the pose amd keypoints colored
    and background as black.

    Parameters:
        image (np.ndarray): The image to used to generate pose image

    Returns:
        np.ndarray: A bone or pose image indicating the pose and keypoints

    Example
    -------
        >>> generate_pose_image(image)
        array([[0, 0, 0, ..., 0, 0, 0],
                [0, 20, 24, ..., 0, 100, 103],
                ...,
                [10, 11, 15, ..., 202, 202, 205],
                [10, 10, 10, ..., 200, 200, 200]], dtype=uint8),
    """
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "generate_pose",
    }

    answer = send_inference_request(data, "tools")
    return_data = np.array(b64_to_pil(answer["masks"][0]).convert("RGB"))
    return return_data


def template_match(
    image: np.ndarray, template_image: np.ndarray
) -> List[Dict[str, Any]]:
    """'template_match' is a tool that can detect all instances of a template in
    a given image. It returns the locations of the detected template, a corresponding
    similarity score of the same

    Parameters:
        image (np.ndarray): The image used for searching the template
        template_image (np.ndarray): The template image or crop to search in the image

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score and
            bounding box of the detected template with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> template_match(image, template)
        [
            {'score': 0.79, 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.38, 'bbox': [0.2, 0.21, 0.45, 0.5},
        ]
    """
    image_size = image.shape[:2]
    image_b64 = convert_to_b64(image)
    template_image_b64 = convert_to_b64(template_image)
    data = {
        "image": image_b64,
        "template": template_image_b64,
        "tool": "template_match",
    }

    answer = send_inference_request(data, "tools")
    return_data = []
    for i in range(len(answer["bboxes"])):
        return_data.append(
            {
                "score": round(answer["scores"][i], 2),
                "bbox": normalize_bbox(answer["bboxes"][i], image_size),
            }
        )
    return return_data


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

    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
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
    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

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
    owl_v2,
    grounding_sam,
    extract_frames,
    ocr,
    clip,
    vit_image_classification,
    vit_nsfw_classification,
    loca_zero_shot_counting,
    loca_visual_prompt_counting,
    florencev2_roberta_vqa,
    florencev2_image_caption,
    florencev2_object_detection,
    detr_segmentation,
    depth_anything_v2,
    generate_soft_edge_image,
    dpt_hybrid_midas,
    generate_pose_image,
    closest_mask_distance,
    closest_box_distance,
    save_json,
    load_image,
    save_image,
    save_video,
    overlay_bounding_boxes,
    overlay_segmentation_masks,
    overlay_heat_map,
    template_match,
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
