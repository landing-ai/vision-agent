import io
import json
import logging
import os
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from uuid import UUID

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from pillow_heif import register_heif_opener  # type: ignore
from pytube import YouTube  # type: ignore

from vision_agent.clients.landing_public_api import LandingPublicAPI
from vision_agent.lmm.lmm import AnthropicLMM, OpenAILMM
from vision_agent.tools.tool_utils import (
    add_bboxes_from_masks,
    filter_bboxes_by_threshold,
    get_tool_descriptions,
    get_tool_documentation,
    get_tools_df,
    get_tools_info,
    nms,
    send_inference_request,
    send_task_inference_request,
    single_nms,
)
from vision_agent.tools.tools_types import JobStatus, ODResponseData
from vision_agent.utils.exceptions import FineTuneModelIsNotReady
from vision_agent.utils.execute import FileSerializer, MimeType
from vision_agent.utils.image_utils import (
    b64_to_pil,
    convert_quad_box_to_bbox,
    convert_to_b64,
    denormalize_bbox,
    encode_image_bytes,
    get_image_size,
    normalize_bbox,
    numpy_to_bytes,
    rle_decode,
    rle_decode_array,
)
from vision_agent.utils.video import (
    extract_frames_from_video,
    frames_to_bytes,
    video_writer,
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
        "function_name": "grounding_dino",
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


def owl_v2_image(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.10,
    fine_tune_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """'owl_v2_image' is a tool that can detect and count multiple objects given a text
    prompt such as category names or referring expressions on images. The categories in
    text prompt are separated by commas. It returns a list of bounding boxes with
    normalized coordinates, label names and associated probability scores.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults
            to 0.10.
        fine_tune_id (Optional[str]): If you have a fine-tuned model, you can pass the
            fine-tuned model ID here to use it.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> owl_v2_image("car, dinosaur", image)
        [
            {'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.98, 'label': 'car', 'bbox': [0.2, 0.21, 0.45, 0.5},
        ]
    """

    image_size = image.shape[:2]
    if image_size[0] < 1 or image_size[1] < 1:
        return []

    buffer_bytes = numpy_to_bytes(image)
    files = [("image", buffer_bytes)]
    payload = {
        "prompts": [s.strip() for s in prompt.split(",")],
        "confidence": box_threshold,
        "model": "owlv2",
    }
    metadata = {"function_name": "owl_v2_image"}

    if fine_tune_id is not None:
        landing_api = LandingPublicAPI()
        status = landing_api.check_fine_tuning_job(UUID(fine_tune_id))
        if status is not JobStatus.SUCCEEDED:
            raise FineTuneModelIsNotReady(
                f"Fine-tuned model {fine_tune_id} is not ready yet"
            )

        # we can only execute fine-tuned models with florence2
        payload = {
            "prompts": payload["prompts"],
            "jobId": fine_tune_id,
            "model": "florence2",
        }

    detections = send_task_inference_request(
        payload,
        "text-to-object-detection",
        files=files,
        metadata=metadata,
    )

    # get the first frame
    bboxes = detections[0]
    bboxes_formatted = [
        ODResponseData(
            label=bbox["label"],
            bbox=normalize_bbox(bbox["bounding_box"], image_size),
            score=round(bbox["score"], 2),
        )
        for bbox in bboxes
    ]
    return [bbox.model_dump() for bbox in bboxes_formatted]


def owl_v2_video(
    prompt: str,
    frames: List[np.ndarray],
    box_threshold: float = 0.10,
    fine_tune_id: Optional[str] = None,
) -> List[List[Dict[str, Any]]]:
    """'owl_v2_video' will run owl_v2 on each frame of a video. It can detect multiple
    objects independently per frame given a text prompt such as a category name or
    referring expression but does not track objects across frames. The categories in
    text prompt are separated by commas. It returns a list of lists where each inner
    list contains the score, label, and bounding box of the detections for that frame.

    Parameters:
        prompt (str): The prompt to ground to the video.
        frames (List[np.ndarray]): The list of frames to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults
            to 0.30.
        fine_tune_id (Optional[str]): If you have a fine-tuned model, you can pass the
            fine-tuned model ID here to use it.

    Returns:
        List[List[Dict[str, Any]]]: A list of lists of dictionaries containing the
            score, label, and bounding box of the detected objects with normalized
            coordinates between 0 and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the
            coordinates of the top-left and xmax and ymax are the coordinates of the
            bottom-right of the bounding box.

    Example
    -------
        >>> owl_v2_video("car, dinosaur", frames)
        [
            [
                {'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]},
                {'score': 0.98, 'label': 'car', 'bbox': [0.2, 0.21, 0.45, 0.5},
            ],
            ...
        ]
    """
    if len(frames) == 0 or not isinstance(frames, List):
        raise ValueError("Must provide a list of numpy arrays for frames")

    image_size = frames[0].shape[:2]
    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]
    payload = {
        "prompts": [s.strip() for s in prompt.split(",")],
        "confidence": box_threshold,
        "model": "owlv2",
    }
    metadata = {"function_name": "owl_v2_video"}

    if fine_tune_id is not None:
        landing_api = LandingPublicAPI()
        status = landing_api.check_fine_tuning_job(UUID(fine_tune_id))
        if status is not JobStatus.SUCCEEDED:
            raise FineTuneModelIsNotReady(
                f"Fine-tuned model {fine_tune_id} is not ready yet"
            )

        # we can only execute fine-tuned models with florence2
        payload = {
            "prompts": payload["prompts"],
            "jobId": fine_tune_id,
            "model": "florence2",
        }

    detections = send_task_inference_request(
        payload,
        "text-to-object-detection",
        files=files,
        metadata=metadata,
    )

    bboxes_formatted = []
    for frame_data in detections:
        bboxes_formatted_per_frame = [
            ODResponseData(
                label=bbox["label"],
                bbox=normalize_bbox(bbox["bounding_box"], image_size),
                score=round(bbox["score"], 2),
            )
            for bbox in frame_data
        ]
        bboxes_formatted.append(bboxes_formatted_per_frame)
    return [[bbox.model_dump() for bbox in frame] for frame in bboxes_formatted]


def grounding_sam(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.20,
    iou_threshold: float = 0.20,
) -> List[Dict[str, Any]]:
    """'grounding_sam' is a tool that can segment multiple objects given a text prompt
    such as category names or referring expressions. The categories in text prompt are
    separated by commas or periods. It returns a list of bounding boxes, label names,
    mask file names and associated probability scores.

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
        "function_name": "grounding_sam",
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


def florence2_sam2_image(
    prompt: str, image: np.ndarray, fine_tune_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """'florence2_sam2_image' is a tool that can segment multiple objects given a text
    prompt such as category names or referring expressions. The categories in the text
    prompt are separated by commas. It returns a list of bounding boxes, label names,
    mask file names and associated probability scores of 1.0.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to ground the prompt to.
        fine_tune_id (Optional[str]): If you have a fine-tuned model, you can pass the
            fine-tuned model ID here to use it.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> florence2_sam2_image("car, dinosaur", image)
        [
            {
                'score': 1.0,
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
    if image.shape[0] < 1 or image.shape[1] < 1:
        return []

    buffer_bytes = numpy_to_bytes(image)
    files = [("image", buffer_bytes)]
    payload = {
        "prompt": prompt,
        "model": "florence2sam2",
    }
    metadata = {"function_name": "florence2_sam2_image"}

    if fine_tune_id is not None:
        landing_api = LandingPublicAPI()
        status = landing_api.check_fine_tuning_job(UUID(fine_tune_id))
        if status is not JobStatus.SUCCEEDED:
            raise FineTuneModelIsNotReady(
                f"Fine-tuned model {fine_tune_id} is not ready yet"
            )

        payload["jobId"] = fine_tune_id

    detections = send_task_inference_request(
        payload,
        "text-to-instance-segmentation",
        files=files,
        metadata=metadata,
    )

    # get the first frame
    frame = detections[0]
    return_data = []
    for detection in frame:
        mask = rle_decode_array(detection["mask"])
        label = detection["label"]
        bbox = normalize_bbox(detection["bounding_box"], detection["mask"]["size"])
        return_data.append({"label": label, "bbox": bbox, "mask": mask, "score": 1.0})
    return return_data


def florence2_sam2_video_tracking(
    prompt: str,
    frames: List[np.ndarray],
    chunk_length: Optional[int] = 10,
    fine_tune_id: Optional[str] = None,
) -> List[List[Dict[str, Any]]]:
    """'florence2_sam2_video_tracking' is a tool that can segment and track multiple
    entities in a video given a text prompt such as category names or referring
    expressions. You can optionally separate the categories in the text with commas. It
    can find new objects every 'chunk_length' frames and is useful for tracking and
    counting without duplicating counts and always outputs scores of 1.0.

    Parameters:
        prompt (str): The prompt to ground to the video.
        frames (List[np.ndarray]): The list of frames to ground the prompt to.
        chunk_length (Optional[int]): The number of frames to re-run florence2 to find
            new objects.
        fine_tune_id (Optional[str]): If you have a fine-tuned model, you can pass the
            fine-tuned model ID here to use it.

    Returns:
        List[List[Dict[str, Any]]]: A list of list of dictionaries containing the
        label,segment mask and bounding boxes. The outer list represents each frame and
        the inner list is the entities per frame. The label contains the object ID
        followed by the label name. The objects are only identified in the first framed
        and tracked throughout the video.

    Example
    -------
        >>> florence2_sam2_video("car, dinosaur", frames)
        [
            [
                {
                    'label': '0: dinosaur',
                    'bbox': [0.1, 0.11, 0.35, 0.4],
                    'mask': array([[0, 0, 0, ..., 0, 0, 0],
                        [0, 0, 0, ..., 0, 0, 0],
                        ...,
                        [0, 0, 0, ..., 0, 0, 0],
                        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
                },
            ],
            ...
        ]
    """
    if len(frames) == 0 or not isinstance(frames, List):
        raise ValueError("Must provide a list of numpy arrays for frames")

    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]
    payload = {
        "prompt": prompt,
        "model": "florence2sam2",
    }
    metadata = {"function_name": "florence2_sam2_video_tracking"}

    if chunk_length is not None:
        payload["chunk_length_frames"] = chunk_length  # type: ignore

    if fine_tune_id is not None:
        landing_api = LandingPublicAPI()
        status = landing_api.check_fine_tuning_job(UUID(fine_tune_id))
        if status is not JobStatus.SUCCEEDED:
            raise FineTuneModelIsNotReady(
                f"Fine-tuned model {fine_tune_id} is not ready yet"
            )

        payload["jobId"] = fine_tune_id

    detections = send_task_inference_request(
        payload,
        "text-to-instance-segmentation",
        files=files,
        metadata=metadata,
    )

    return_data = []
    for frame in detections:
        return_frame_data = []
        for detection in frame:
            mask = rle_decode_array(detection["mask"])
            label = str(detection["id"]) + ": " + detection["label"]
            return_frame_data.append({"label": label, "mask": mask, "score": 1.0})
        return_data.append(return_frame_data)
    return_data = add_bboxes_from_masks(return_data)
    return nms(return_data, iou_threshold=0.95)


def ocr(image: np.ndarray) -> List[Dict[str, Any]]:
    """'ocr' extracts text from an image. It returns a list of detected text, bounding
    boxes with normalized coordinates, and confidence scores. The results are sorted
    from top-left to bottom right.

    Parameters:
        image (np.ndarray): The image to extract text from.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the detected text, bbox
            with normalized coordinates, and confidence score.

    Example
    -------
        >>> ocr(image)
        [
            {'label': 'hello world', 'bbox': [0.1, 0.11, 0.35, 0.4], 'score': 0.99},
        ]
    """

    pil_image = Image.fromarray(image).convert("RGB")
    image_size = pil_image.size[::-1]
    if image_size[0] < 1 or image_size[1] < 1:
        return []
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
            value, e.g. {count: 12} and a heat map for visualization purposes.

    Example
    -------
        >>> loca_zero_shot_counting(image)
        {'count': 83,
        'heat_map': array([[ 0,  0,  0, ...,  0,  0,  0],
            [ 0,  0,  0, ...,  0,  0,  0],
            [ 0,  0,  0, ...,  0,  0,  1],
            ...,
            [ 0,  0,  0, ..., 30, 35, 41],
            [ 0,  0,  0, ..., 41, 47, 53],
            [ 0,  0,  0, ..., 53, 59, 64]], dtype=uint8)}
    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "function_name": "loca_zero_shot_counting",
    }
    resp_data: dict[str, Any] = send_inference_request(data, "loca", v2=True)
    resp_data["heat_map"] = np.array(resp_data["heat_map"][0]).astype(np.uint8)
    return resp_data


def loca_visual_prompt_counting(
    image: np.ndarray, visual_prompt: Dict[str, List[float]]
) -> Dict[str, Any]:
    """'loca_visual_prompt_counting' is a tool that counts the dominant foreground object
    given an image and a visual prompt which is a bounding box describing the object.
    It returns only the count of the objects in the image.

    Parameters:
        image (np.ndarray): The image that contains lot of instances of a single object
            visual_prompt (Dict[str, List[float]]): Bounding box of the object in
            format [xmin, ymin, xmax, ymax]. Only 1 bounding box can be provided.

    Returns:
        Dict[str, Any]: A dictionary containing the key 'count' and the count as a
            value, e.g. {count: 12} and a heat map for visualization purposes.

    Example
    -------
        >>> loca_visual_prompt_counting(image, {"bbox": [0.1, 0.1, 0.4, 0.42]})
        {'count': 83,
        'heat_map': array([[ 0,  0,  0, ...,  0,  0,  0],
            [ 0,  0,  0, ...,  0,  0,  0],
            [ 0,  0,  0, ...,  0,  0,  1],
            ...,
            [ 0,  0,  0, ..., 30, 35, 41],
            [ 0,  0,  0, ..., 41, 47, 53],
            [ 0,  0,  0, ..., 53, 59, 64]], dtype=uint8)}
    """

    image_size = get_image_size(image)
    bbox = visual_prompt["bbox"]
    image_b64 = convert_to_b64(image)

    data = {
        "image": image_b64,
        "bbox": list(map(int, denormalize_bbox(bbox, image_size))),
        "function_name": "loca_visual_prompt_counting",
    }
    resp_data: dict[str, Any] = send_inference_request(data, "loca", v2=True)
    resp_data["heat_map"] = np.array(resp_data["heat_map"][0]).astype(np.uint8)
    return resp_data


def countgd_counting(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.23,
) -> List[Dict[str, Any]]:
    """'countgd_counting' is a tool that can detect multiple instances of an object
    given a text prompt. It is particularly useful when trying to detect and count a
    large number of objects. It returns a list of bounding boxes with normalized
    coordinates, label names and associated confidence scores.

    Parameters:
        prompt (str): The object that needs to be counted.
        image (np.ndarray): The image that contains multiple instances of the object.
        box_threshold (float, optional): The threshold for detection. Defaults
            to 0.23.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> countgd_counting("flower", image)
        [
            {'score': 0.49, 'label': 'flower', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.68, 'label': 'flower', 'bbox': [0.2, 0.21, 0.45, 0.5},
            {'score': 0.78, 'label': 'flower', 'bbox': [0.3, 0.35, 0.48, 0.52},
            {'score': 0.98, 'label': 'flower', 'bbox': [0.44, 0.24, 0.49, 0.58},
        ]
    """
    image_size = image.shape[:2]
    if image_size[0] < 1 or image_size[1] < 1:
        return []

    buffer_bytes = numpy_to_bytes(image)
    files = [("image", buffer_bytes)]
    payload = {
        "prompts": [prompt.replace(", ", ". ")],
        "confidence": box_threshold,  # still not being used in the API
        "model": "countgd",
    }
    metadata = {"function_name": "countgd_counting"}

    detections = send_task_inference_request(
        payload, "text-to-object-detection", files=files, metadata=metadata
    )

    # get the first frame
    bboxes = detections[0]
    bboxes_formatted = [
        ODResponseData(
            label=bbox["label"],
            bbox=normalize_bbox(bbox["bounding_box"], image_size),
            score=round(bbox["score"], 2),
        )
        for bbox in bboxes
    ]
    # TODO: remove this once we start to use the confidence on countgd
    filtered_bboxes = filter_bboxes_by_threshold(bboxes_formatted, box_threshold)
    return_data = [bbox.model_dump() for bbox in filtered_bboxes]
    return single_nms(return_data, iou_threshold=0.80)


def countgd_example_based_counting(
    visual_prompts: List[List[float]],
    image: np.ndarray,
    box_threshold: float = 0.23,
) -> List[Dict[str, Any]]:
    """'countgd_example_based_counting' is a tool that can precisely count multiple
    instances of an object given few visual example prompts. It returns a list of bounding
    boxes with normalized coordinates, label names and associated confidence scores.

    Parameters:
        visual_prompts (List[List[float]]): Bounding boxes of the object in format
            [xmin, ymin, xmax, ymax]. Upto 3 bounding boxes can be provided. image
            (np.ndarray): The image that contains multiple instances of the object.
            box_threshold (float, optional): The threshold for detection. Defaults to
            0.23.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> countgd_example_based_counting(
            visual_prompts=[[0.1, 0.1, 0.4, 0.42], [0.2, 0.3, 0.25, 0.35]],
            image=image
        )
        [
            {'score': 0.49, 'label': 'object', 'bounding_box': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.68, 'label': 'object', 'bounding_box': [0.2, 0.21, 0.45, 0.5},
            {'score': 0.78, 'label': 'object', 'bounding_box': [0.3, 0.35, 0.48, 0.52},
            {'score': 0.98, 'label': 'object', 'bounding_box': [0.44, 0.24, 0.49, 0.58},
        ]
    """
    image_size = image.shape[:2]
    if image_size[0] < 1 or image_size[1] < 1:
        return []

    buffer_bytes = numpy_to_bytes(image)
    files = [("image", buffer_bytes)]
    visual_prompts = [
        denormalize_bbox(bbox, image.shape[:2]) for bbox in visual_prompts
    ]
    payload = {"visual_prompts": json.dumps(visual_prompts), "model": "countgd"}
    metadata = {"function_name": "countgd_example_based_counting"}

    detections = send_task_inference_request(
        payload, "visual-prompts-to-object-detection", files=files, metadata=metadata
    )

    # get the first frame
    bboxes_per_frame = detections[0]
    bboxes_formatted = [
        ODResponseData(
            label=bbox["label"],
            bbox=normalize_bbox(bbox["bounding_box"], image_size),
            score=round(bbox["score"], 2),
        )
        for bbox in bboxes_per_frame
    ]
    filtered_bboxes = filter_bboxes_by_threshold(bboxes_formatted, box_threshold)
    return [bbox.model_dump() for bbox in filtered_bboxes]


def florence2_roberta_vqa(prompt: str, image: np.ndarray) -> str:
    """'florence2_roberta_vqa' is a tool that takes an image and analyzes
    its contents, generates detailed captions and then tries to answer the given
    question using the generated context. It returns text as an answer to the question.

    Parameters:
        prompt (str): The question about the image
        image (np.ndarray): The reference image used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> florence2_roberta_vqa('What is the top left animal in this image?', image)
        'white tiger'
    """

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "question": prompt,
        "function_name": "florence2_roberta_vqa",
    }

    answer = send_inference_request(data, "florence2-qa", v2=True)
    return answer  # type: ignore


def ixc25_image_vqa(prompt: str, image: np.ndarray) -> str:
    """'ixc25_image_vqa' is a tool that can answer any questions about arbitrary images
    including regular images or images of documents or presentations. It returns text
    as an answer to the question.

    Parameters:
        prompt (str): The question about the image
        image (np.ndarray): The reference image used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> ixc25_image_vqa('What is the cat doing?', image)
        'drinking milk'
    """
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"Image is empty, image shape: {image.shape}")

    buffer_bytes = numpy_to_bytes(image)
    files = [("image", buffer_bytes)]
    payload = {
        "prompt": prompt,
        "function_name": "ixc25_image_vqa",
    }
    data: Dict[str, Any] = send_inference_request(
        payload, "internlm-xcomposer2", files=files, v2=True
    )
    return cast(str, data["answer"])


def qwen2_vl_images_vqa(prompt: str, images: List[np.ndarray]) -> str:
    """'qwen2_vl_images_vqa' is a tool that can answer any questions about arbitrary
    images including regular images or images of documents or presentations. It can be
    very useful for document QA or OCR text extraction. It returns text as an answer to
    the question.

    Parameters:
        prompt (str): The question about the document image
        images (List[np.ndarray]): The reference images used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> qwen2_vl_images_vqa('Give a summary of the document', images)
        'The document talks about the history of the United States of America and its...'
    """
    if isinstance(images, np.ndarray):
        images = [images]

    for image in images:
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError(f"Image is empty, image shape: {image.shape}")

    files = [("images", numpy_to_bytes(image)) for image in images]
    payload = {
        "prompt": prompt,
        "model": "qwen2vl",
        "function_name": "qwen2_vl_images_vqa",
    }
    data: Dict[str, Any] = send_inference_request(
        payload, "image-to-text", files=files, v2=True
    )
    return cast(str, data)


def claude35_text_extraction(image: np.ndarray) -> str:
    """'claude35_text_extraction' is a tool that can extract text from an image. It
    returns the extracted text as a string and can be used as an alternative to OCR if
    you do not need to know the exact bounding box of the text.

    Parameters:
        image (np.ndarray): The image to extract text from.

    Returns:
        str: The extracted text from the image.
    """

    lmm = AnthropicLMM()
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_b64 = "data:image/png;base64," + encode_image_bytes(image_bytes)
    text = lmm.generate(
        "Extract and return any text you see in this image and nothing else. If you do not read any text respond with an empty string.",
        [image_b64],
    )
    return cast(str, text)


def ixc25_video_vqa(prompt: str, frames: List[np.ndarray]) -> str:
    """'ixc25_video_vqa' is a tool that can answer any questions about arbitrary videos
    including regular videos or videos of documents or presentations. It returns text
    as an answer to the question.

    Parameters:
        prompt (str): The question about the video
        frames (List[np.ndarray]): The reference frames used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> ixc25_video_vqa('Which football player made the goal?', frames)
        'Lionel Messi'
    """

    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]
    payload = {
        "prompt": prompt,
        "function_name": "ixc25_video_vqa",
    }
    data: Dict[str, Any] = send_inference_request(
        payload, "internlm-xcomposer2", files=files, v2=True
    )
    return cast(str, data["answer"])


def qwen2_vl_video_vqa(prompt: str, frames: List[np.ndarray]) -> str:
    """'qwen2_vl_video_vqa' is a tool that can answer any questions about arbitrary videos
    including regular videos or videos of documents or presentations. It returns text
    as an answer to the question.

    Parameters:
        prompt (str): The question about the video
        frames (List[np.ndarray]): The reference frames used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> qwen2_vl_video_vqa('Which football player made the goal?', frames)
        'Lionel Messi'
    """

    if len(frames) == 0 or not isinstance(frames, List):
        raise ValueError("Must provide a list of numpy arrays for frames")

    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]
    payload = {
        "prompt": prompt,
        "model": "qwen2vl",
        "function_name": "qwen2_vl_video_vqa",
    }
    data: Dict[str, Any] = send_inference_request(
        payload, "image-to-text", files=files, v2=True
    )
    return cast(str, data)


def gpt4o_image_vqa(prompt: str, image: np.ndarray) -> str:
    """'gpt4o_image_vqa' is a tool that can answer any questions about arbitrary images
    including regular images or images of documents or presentations. It returns text
    as an answer to the question.

    Parameters:
        prompt (str): The question about the image
        image (np.ndarray): The reference image used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> gpt4o_image_vqa('What is the cat doing?', image)
        'drinking milk'
    """

    lmm = OpenAILMM()
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_b64 = "data:image/png;base64," + encode_image_bytes(image_bytes)
    resp = lmm.generate(prompt, [image_b64])
    return cast(str, resp)


def gpt4o_video_vqa(prompt: str, frames: List[np.ndarray]) -> str:
    """'gpt4o_video_vqa' is a tool that can answer any questions about arbitrary videos
    including regular videos or videos of documents or presentations. It returns text
    as an answer to the question.

    Parameters:
        prompt (str): The question about the video
        frames (List[np.ndarray]): The reference frames used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> gpt4o_video_vqa('Which football player made the goal?', frames)
        'Lionel Messi'
    """

    lmm = OpenAILMM()

    if len(frames) > 10:
        step = len(frames) / 10
        frames = [frames[int(i * step)] for i in range(10)]

    frames_b64 = []
    for frame in frames:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = "data:image/png;base64," + encode_image_bytes(image_bytes)
        frames_b64.append(image_b64)

    resp = lmm.generate(prompt, frames_b64)
    return cast(str, resp)


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
        "function_name": "git_vqa_v2",
    }

    answer = send_inference_request(data, "tools")
    return answer["text"][0]  # type: ignore


def video_temporal_localization(
    prompt: str,
    frames: List[np.ndarray],
    model: str = "qwen2vl",
    chunk_length_frames: Optional[int] = 2,
) -> List[float]:
    """'video_temporal_localization' will run qwen2vl on each chunk_length_frames
    value selected for the video. It can detect multiple objects independently per
    chunk_length_frames given a text prompt such as a referring expression
    but does not track objects across frames.
    It returns a list of floats with a value of 1.0 if the objects are found in a given
    chunk_length_frames of the video.

    Parameters:
        prompt (str): The question about the video
        frames (List[np.ndarray]): The reference frames used for the question
        model (str): The model to use for the inference. Valid values are
            'qwen2vl', 'gpt4o', 'internlm-xcomposer'
        chunk_length_frames (Optional[int]): length of each chunk in frames

    Returns:
        List[float]: A list of floats with a value of 1.0 if the objects to be found
            are present in the chunk_length_frames of the video.

    Example
    -------
        >>> video_temporal_localization('Did a goal happened?', frames)
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    """

    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "function_name": "video_temporal_localization",
    }
    if chunk_length_frames is not None:
        payload["chunk_length_frames"] = chunk_length_frames

    data = send_inference_request(
        payload, "video-temporal-localization", files=files, v2=True
    )
    return [cast(float, value) for value in data]


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

    if image.shape[0] < 1 or image.shape[1] < 1:
        return {"labels": [], "scores": []}

    image_b64 = convert_to_b64(image)
    data = {
        "prompt": ",".join(classes),
        "image": image_b64,
        "tool": "closed_set_image_classification",
        "function_name": "clip",
    }
    resp_data: dict[str, Any] = send_inference_request(data, "tools")
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
    if image.shape[0] < 1 or image.shape[1] < 1:
        return {"labels": [], "scores": []}

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "image_classification",
        "function_name": "vit_image_classification",
    }
    resp_data: dict[str, Any] = send_inference_request(data, "tools")
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
        {"label": "normal", "scores": 0.68},
    """
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"Image is empty, image shape: {image.shape}")

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "function_name": "vit_nsfw_classification",
    }
    resp_data: dict[str, Any] = send_inference_request(
        data, "nsfw-classification", v2=True
    )
    resp_data["score"] = round(resp_data["score"], 4)
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
        "function_name": "blip_image_caption",
    }

    answer = send_inference_request(data, "tools")
    return answer["text"][0]  # type: ignore


def florence2_image_caption(image: np.ndarray, detail_caption: bool = True) -> str:
    """'florence2_image_caption' is a tool that can caption or describe an image based
    on its contents. It returns a text describing the image.

    Parameters:
        image (np.ndarray): The image to caption
        detail_caption (bool): If True, the caption will be as detailed as possible else
            the caption will be a brief description.

    Returns:
       str: A string which is the caption for the given image.

    Example
    -------
        >>> florence2_image_caption(image, False)
        'This image contains a cat sitting on a table with a bowl of milk.'
    """
    image_b64 = convert_to_b64(image)
    task = "<MORE_DETAILED_CAPTION>" if detail_caption else "<DETAILED_CAPTION>"
    data = {
        "image": image_b64,
        "task": task,
        "function_name": "florence2_image_caption",
    }

    answer = send_inference_request(data, "florence2", v2=True)
    return answer[task]  # type: ignore


def florence2_phrase_grounding(
    prompt: str, image: np.ndarray, fine_tune_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """'florence2_phrase_grounding' is a tool that can detect multiple
    objects given a text prompt which can be object names or caption. You
    can optionally separate the object names in the text with commas. It returns a list
    of bounding boxes with normalized coordinates, label names and associated
    probability scores of 1.0.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to used to detect objects
        fine_tune_id (Optional[str]): If you have a fine-tuned model, you can pass the
            fine-tuned model ID here to use it.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box. The scores are always 1.0 and cannot be thresholded

    Example
    -------
        >>> florence2_phrase_grounding('person looking at a coyote', image)
        [
            {'score': 1.0, 'label': 'person', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 1.0, 'label': 'coyote', 'bbox': [0.34, 0.21, 0.85, 0.5},
        ]
    """
    image_size = image.shape[:2]
    if image_size[0] < 1 or image_size[1] < 1:
        return []

    buffer_bytes = numpy_to_bytes(image)
    files = [("image", buffer_bytes)]
    payload = {
        "prompts": [s.strip() for s in prompt.split(",")],
        "model": "florence2",
    }
    metadata = {"function_name": "florence2_phrase_grounding"}

    if fine_tune_id is not None:
        landing_api = LandingPublicAPI()
        status = landing_api.check_fine_tuning_job(UUID(fine_tune_id))
        if status is not JobStatus.SUCCEEDED:
            raise FineTuneModelIsNotReady(
                f"Fine-tuned model {fine_tune_id} is not ready yet"
            )

        payload["jobId"] = fine_tune_id

    detections = send_task_inference_request(
        payload,
        "text-to-object-detection",
        files=files,
        metadata=metadata,
    )

    # get the first frame
    bboxes = detections[0]
    bboxes_formatted = [
        ODResponseData(
            label=bbox["label"],
            bbox=normalize_bbox(bbox["bounding_box"], image_size),
            score=round(bbox["score"], 2),
        )
        for bbox in bboxes
    ]

    return [bbox.model_dump() for bbox in bboxes_formatted]


def florence2_phrase_grounding_video(
    prompt: str, frames: List[np.ndarray], fine_tune_id: Optional[str] = None
) -> List[List[Dict[str, Any]]]:
    """'florence2_phrase_grounding_video' will run florence2 on each frame of a video.
    It can detect multiple objects given a text prompt which can be object names or
    caption. You can optionally separate the object names in the text with commas.
    It returns a list of lists where each inner list contains bounding boxes with
    normalized coordinates, label names and associated probability scores of 1.0.

    Parameters:
        prompt (str): The prompt to ground to the video.
        frames (List[np.ndarray]): The list of frames to detect objects.
        fine_tune_id (Optional[str]): If you have a fine-tuned model, you can pass the
            fine-tuned model ID here to use it.

    Returns:
        List[List[Dict[str, Any]]]: A list of lists of dictionaries containing the score,
            label, and bounding box of the detected objects with normalized coordinates
            between 0 and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates
            of the top-left and xmax and ymax are the coordinates of the bottom-right of
            the bounding box. The scores are always 1.0 and cannot be thresholded.

    Example
    -------
        >>> florence2_phrase_grounding_video('person looking at a coyote', frames)
        [
            [
                {'score': 1.0, 'label': 'person', 'bbox': [0.1, 0.11, 0.35, 0.4]},
                {'score': 1.0, 'label': 'coyote', 'bbox': [0.34, 0.21, 0.85, 0.5},
            ],
            ...
        ]
    """
    if len(frames) == 0:
        raise ValueError("No frames provided")

    image_size = frames[0].shape[:2]
    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]
    payload = {
        "prompts": [s.strip() for s in prompt.split(",")],
        "model": "florence2",
    }
    metadata = {"function_name": "florence2_phrase_grounding_video"}

    if fine_tune_id is not None:
        landing_api = LandingPublicAPI()
        status = landing_api.check_fine_tuning_job(UUID(fine_tune_id))
        if status is not JobStatus.SUCCEEDED:
            raise FineTuneModelIsNotReady(
                f"Fine-tuned model {fine_tune_id} is not ready yet"
            )

        payload["jobId"] = fine_tune_id

    detections = send_task_inference_request(
        payload,
        "text-to-object-detection",
        files=files,
        metadata=metadata,
    )

    bboxes_formatted = []
    for frame_data in detections:
        bboxes_formatted_per_frame = [
            ODResponseData(
                label=bbox["label"],
                bbox=normalize_bbox(bbox["bounding_box"], image_size),
                score=round(bbox["score"], 2),
            )
            for bbox in frame_data
        ]
        bboxes_formatted.append(bboxes_formatted_per_frame)
    return [[bbox.model_dump() for bbox in frame] for frame in bboxes_formatted]


def florence2_ocr(image: np.ndarray) -> List[Dict[str, Any]]:
    """'florence2_ocr' is a tool that can detect text and text regions in an image.
    Each text region contains one line of text. It returns a list of detected text,
    the text region as a bounding box with normalized coordinates, and confidence
    scores. The results are sorted from top-left to bottom right.

    Parameters:
        image (np.ndarray): The image to extract text from.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the detected text, bbox
            with normalized coordinates, and confidence score.

    Example
    -------
        >>> florence2_ocr(image)
        [
            {'label': 'hello world', 'bbox': [0.1, 0.11, 0.35, 0.4], 'score': 0.99},
        ]
    """

    image_size = image.shape[:2]
    if image_size[0] < 1 or image_size[1] < 1:
        return []
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "task": "<OCR_WITH_REGION>",
        "function_name": "florence2_ocr",
    }

    detections = send_inference_request(data, "florence2", v2=True)
    detections = detections["<OCR_WITH_REGION>"]
    return_data = []
    for i in range(len(detections["quad_boxes"])):
        return_data.append(
            {
                "label": detections["labels"][i],
                "bbox": normalize_bbox(
                    convert_quad_box_to_bbox(detections["quad_boxes"][i]), image_size
                ),
                "score": 1.0,
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
    if image.shape[0] < 1 or image.shape[1] < 1:
        return []
    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "tool": "panoptic_segmentation",
        "function_name": "detr_segmentation",
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
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"Image is empty, image shape: {image.shape}")

    image_b64 = convert_to_b64(image)
    data = {
        "image": image_b64,
        "function_name": "depth_anything_v2",
    }

    depth_map = send_inference_request(data, "depth-anything-v2", v2=True)
    depth_map_np = np.array(depth_map["map"])
    depth_map_np = (depth_map_np - depth_map_np.min()) / (
        depth_map_np.max() - depth_map_np.min()
    )
    depth_map_np = (255 * depth_map_np).astype(np.uint8)
    return depth_map_np


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
        "function_name": "generate_soft_edge_image",
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
        "function_name": "dpt_hybrid_midas",
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
        "function_name": "generate_pose_image",
    }

    pos_img = send_inference_request(data, "pose-detector", v2=True)
    return_data = np.array(b64_to_pil(pos_img["data"]).convert("RGB"))
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
        "function_name": "template_match",
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


def flux_image_inpainting(
    prompt: str,
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """'flux_image_inpainting' performs image inpainting to fill the masked regions,
    given by mask, in the image, given image based on the text prompt and surrounding image context.
    It can be used to edit regions of an image according to the prompt given.

    Parameters:
        prompt (str): A detailed text description guiding what should be generated
            in the masked area. More detailed and specific prompts typically yield better results.
        image (np.ndarray): The source image to be inpainted.
            The image will serve as the base context for the inpainting process.
        mask (np.ndarray): A binary mask image with 0's and 1's,
            where 1 indicates areas to be inpainted and 0 indicates areas to be preserved.

    Returns:
        np.ndarray: The generated image(s) as a numpy array in RGB format with values
            ranging from 0 to 255.

    -------
    Example:
        >>> # Generate inpainting
        >>> result = flux_image_inpainting(
        ...     prompt="a modern black leather sofa with white pillows",
        ...     image=image,
        ...     mask=mask,
        ... )
        >>> save_image(result, "inpainted_room.png")
    """

    min_dim = 8

    if any(dim < min_dim for dim in image.shape[:2] + mask.shape[:2]):
        raise ValueError(f"Image and mask must be at least {min_dim}x{min_dim} pixels")

    max_size = (512, 512)

    if image.shape[0] > max_size[0] or image.shape[1] > max_size[1]:
        scaling_factor = min(max_size[0] / image.shape[0], max_size[1] / image.shape[1])
        new_size = (
            int(image.shape[1] * scaling_factor),
            int(image.shape[0] * scaling_factor),
        )
        new_size = ((new_size[0] // 8) * 8, (new_size[1] // 8) * 8)
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

    elif image.shape[0] % 8 != 0 or image.shape[1] % 8 != 0:
        new_size = ((image.shape[1] // 8) * 8, (image.shape[0] // 8) * 8)
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

    if np.array_equal(mask, mask.astype(bool).astype(int)):
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    else:
        raise ValueError("Mask should contain only binary values (0 or 1)")

    image_file = numpy_to_bytes(image)
    mask_file = numpy_to_bytes(mask)

    files = [
        ("image", image_file),
        ("mask_image", mask_file),
    ]

    payload = {
        "prompt": prompt,
        "task": "inpainting",
        "height": image.shape[0],
        "width": image.shape[1],
        "strength": 0.99,
        "guidance_scale": 18,
        "num_inference_steps": 20,
        "seed": None,
    }

    response = send_inference_request(
        payload=payload,
        endpoint_name="flux1",
        files=files,
        v2=True,
        metadata_payload={"function_name": "flux_image_inpainting"},
    )

    output_image = np.array(b64_to_pil(response[0]).convert("RGB"))
    return output_image


def siglip_classification(image: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    """'siglip_classification' is a tool that can classify an image or a cropped detection given a list
    of input labels or tags. It returns the same list of the input labels along with
    their probability scores based on image content.

    Parameters:
        image (np.ndarray): The image to classify or tag
        labels (List[str]): The list of labels or tags that is associated with the image

    Returns:
        Dict[str, Any]: A dictionary containing the labels and scores. One dictionary
            contains a list of given labels and other a list of scores.

    Example
    -------
        >>> siglip_classification(image, ['dog', 'cat', 'bird'])
        {"labels": ["dog", "cat", "bird"], "scores": [0.68, 0.30, 0.02]},
    """

    if image.shape[0] < 1 or image.shape[1] < 1:
        return {"labels": [], "scores": []}

    image_file = numpy_to_bytes(image)

    files = [("image", image_file)]

    payload = {
        "model": "siglip",
        "labels": labels,
    }

    response: dict[str, Any] = send_inference_request(
        payload=payload,
        endpoint_name="classification",
        files=files,
        v2=True,
        metadata_payload={"function_name": "siglip_classification"},
    )

    return response


# Utility and visualization functions


def extract_frames_and_timestamps(
    video_uri: Union[str, Path], fps: float = 1
) -> List[Dict[str, Union[np.ndarray, float]]]:
    """'extract_frames_and_timestamps' extracts frames and timestamps from a video
    which can be a file path, url or youtube link, returns a list of dictionaries
    with keys "frame" and "timestamp" where "frame" is a numpy array and "timestamp" is
    the relative time in seconds where the frame was captured. The frame is a numpy
    array.

    Parameters:
        video_uri (Union[str, Path]): The path to the video file, url or youtube link
        fps (float, optional): The frame rate per second to extract the frames. Defaults
            to 1.

    Returns:
        List[Dict[str, Union[np.ndarray, float]]]: A list of dictionaries containing the
            extracted frame as a numpy array and the timestamp in seconds.

    Example
    -------
        >>> extract_frames("path/to/video.mp4")
        [{"frame": np.ndarray, "timestamp": 0.0}, ...]
    """
    if isinstance(fps, str):
        # fps could be a string when it's passed in from a web endpoint deployment
        fps = float(fps)

    def reformat(
        frames_and_timestamps: List[Tuple[np.ndarray, float]],
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        return [
            {"frame": frame, "timestamp": timestamp}
            for frame, timestamp in frames_and_timestamps
        ]

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

            return reformat(extract_frames_from_video(video_file_path, fps))
    elif str(video_uri).startswith(("http", "https")):
        _, image_suffix = os.path.splitext(video_uri)
        with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix) as tmp_file:
            # Download the video and save it to the temporary file
            with urllib.request.urlopen(str(video_uri)) as response:
                tmp_file.write(response.read())
            return reformat(extract_frames_from_video(tmp_file.name, fps))

    return reformat(extract_frames_from_video(str(video_uri), fps))


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

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_image(image_path: str) -> np.ndarray:
    """'load_image' is a utility function that loads an image from the given file path string or an URL.

    Parameters:
        image_path (str): The path or URL to the image.

    Returns:
        np.ndarray: The image as a NumPy array.

    Example
    -------
        >>> load_image("path/to/image.jpg")
    """
    # NOTE: sometimes the generated code pass in a NumPy array
    if isinstance(image_path, np.ndarray):
        return image_path
    if image_path.startswith(("http", "https")):
        _, image_suffix = os.path.splitext(image_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix) as tmp_file:
            # Download the image and save it to the temporary file
            with urllib.request.urlopen(image_path) as response:
                tmp_file.write(response.read())
            image_path = tmp_file.name
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
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    from IPython.display import display

    if not isinstance(image, np.ndarray) or (
        image.shape[0] == 0 and image.shape[1] == 0
    ):
        raise ValueError("The image is not a valid NumPy array with shape (H, W, C)")

    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    display(pil_image)
    pil_image.save(file_path)


def save_video(
    frames: List[np.ndarray], output_video_path: Optional[str] = None, fps: float = 1
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
    if isinstance(fps, str):
        # fps could be a string when it's passed in from a web endpoint deployment
        fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be greater than 0 got {fps}")

    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("Frames must be a list of NumPy arrays")

    for frame in frames:
        if not isinstance(frame, np.ndarray) or (
            frame.shape[0] == 0 and frame.shape[1] == 0
        ):
            raise ValueError("A frame is not a valid NumPy array with shape (H, W, C)")

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ).name
    else:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    output_video_path = video_writer(frames, fps, output_video_path)
    _save_video_to_result(output_video_path)
    return output_video_path


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
    medias: Union[np.ndarray, List[np.ndarray]],
    bboxes: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
) -> Union[np.ndarray, List[np.ndarray]]:
    """'overlay_bounding_boxes' is a utility function that displays bounding boxes on
    an image.

    Parameters:
        medias (Union[np.ndarray, List[np.ndarra]]): The image or frames to display the
            bounding boxes on.
        bboxes (Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]): A list of
            dictionaries or a list of list of dictionaries containing the bounding
            boxes.

    Returns:
        np.ndarray: The image with the bounding boxes, labels and scores displayed.

    Example
    -------
        >>> image_with_bboxes = overlay_bounding_boxes(
            image, [{'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]}],
        )
    """

    medias_int: List[np.ndarray] = (
        [medias] if isinstance(medias, np.ndarray) else medias
    )
    if len(bboxes) == 0:
        bbox_int: List[List[Dict[str, Any]]] = [[] for _ in medias_int]
    else:
        if isinstance(bboxes[0], dict):
            bbox_int = [cast(List[Dict[str, Any]], bboxes)]
        else:
            bbox_int = cast(List[List[Dict[str, Any]]], bboxes)

    labels = set([bb["label"] for b in bbox_int for bb in b])

    if len(labels) > len(COLORS):
        _LOGGER.warning(
            "Number of unique labels exceeds the number of available colors. Some labels may have the same color."
        )

    color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(labels)}

    frame_out = []
    for i, frame in enumerate(medias_int):
        pil_image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")

        bboxes = bbox_int[i]
        bboxes = sorted(bboxes, key=lambda x: x["label"], reverse=True)

        # if more than 50 boxes use small boxes to indicate objects else use regular boxes
        if len(bboxes) > 50:
            pil_image = _plot_counting(pil_image, bboxes, color)
        else:
            width, height = pil_image.size
            fontsize = max(12, int(min(width, height) / 40))
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(
                str(
                    resources.files("vision_agent.fonts").joinpath(
                        "default_font_ch_en.ttf"
                    )
                ),
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
                draw.rectangle(
                    (box[0], box[1], text_box[2], text_box[3]), fill=color[label]
                )
                draw.text((box[0], box[1]), text, fill="black", font=font)

        frame_out.append(np.array(pil_image))
    return_frame = frame_out[0] if len(frame_out) == 1 else frame_out

    return return_frame  # type: ignore


def _get_text_coords_from_mask(
    mask: np.ndarray, v_gap: int = 10, h_gap: int = 10
) -> Tuple[int, int]:
    mask = mask.astype(np.uint8)
    if np.sum(mask) == 0:
        return (0, 0)

    rows, cols = np.nonzero(mask)
    top = rows.min()
    bottom = rows.max()
    left = cols.min()
    right = cols.max()

    if top - v_gap < 0:
        if bottom + v_gap > mask.shape[0]:
            top = top
        else:
            top = bottom + v_gap
    else:
        top = top - v_gap

    return left + (right - left) // 2 - h_gap, top


def overlay_segmentation_masks(
    medias: Union[np.ndarray, List[np.ndarray]],
    masks: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    draw_label: bool = True,
    secondary_label_key: str = "tracking_label",
) -> Union[np.ndarray, List[np.ndarray]]:
    """'overlay_segmentation_masks' is a utility function that displays segmentation
    masks.

    Parameters:
        medias (Union[np.ndarray, List[np.ndarray]]): The image or frames to display
            the masks on.
        masks (Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]): A list of
            dictionaries or a list of list of dictionaries containing the masks, labels
            and scores.
        draw_label (bool, optional): If True, the labels will be displayed on the image.
        secondary_label_key (str, optional): The key to use for the secondary
            tracking label which is needed in videos to display tracking information.

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
    if not masks:
        return medias

    medias_int: List[np.ndarray] = (
        [medias] if isinstance(medias, np.ndarray) else medias
    )
    masks_int = [masks] if isinstance(masks[0], dict) else masks
    masks_int = cast(List[List[Dict[str, Any]]], masks_int)

    labels = set()
    for mask_i in masks_int:
        for mask_j in mask_i:
            labels.add(mask_j["label"])
    color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(labels)}

    width, height = Image.fromarray(medias_int[0]).size
    fontsize = max(12, int(min(width, height) / 40))
    font = ImageFont.truetype(
        str(resources.files("vision_agent.fonts").joinpath("default_font_ch_en.ttf")),
        fontsize,
    )

    frame_out = []
    for i, frame in enumerate(medias_int):
        pil_image = Image.fromarray(frame.astype(np.uint8)).convert("RGBA")
        for elt in masks_int[i]:
            mask = elt["mask"]
            label = elt["label"]
            tracking_lbl = elt.get(secondary_label_key, None)
            np_mask = np.zeros((pil_image.size[1], pil_image.size[0], 4))
            np_mask[mask > 0, :] = color[label] + (255 * 0.5,)
            mask_img = Image.fromarray(np_mask.astype(np.uint8))
            pil_image = Image.alpha_composite(pil_image, mask_img)

            if draw_label:
                draw = ImageDraw.Draw(pil_image)
                text = tracking_lbl if tracking_lbl else label
                text_box = draw.textbbox((0, 0), text=text, font=font)
                x, y = _get_text_coords_from_mask(
                    mask,
                    v_gap=(text_box[3] - text_box[1]) + 10,
                    h_gap=(text_box[2] - text_box[0]) // 2,
                )
                if x != 0 and y != 0:
                    text_box = draw.textbbox((x, y), text=text, font=font)
                    draw.rectangle((x, y, text_box[2], text_box[3]), fill=color[label])
                    draw.text((x, y), text, fill="black", font=font)
        frame_out.append(np.array(pil_image))
    return_frame = frame_out[0] if len(frame_out) == 1 else frame_out

    return return_frame  # type: ignore


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


def _plot_counting(
    image: Image.Image,
    bboxes: List[Dict[str, Any]],
    colors: Dict[str, Tuple[int, int, int]],
) -> Image.Image:
    width, height = image.size
    fontsize = max(12, int(min(width, height) / 40))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        str(resources.files("vision_agent.fonts").joinpath("default_font_ch_en.ttf")),
        fontsize,
    )
    for i, elt in enumerate(bboxes, 1):
        label = f"{i}"
        box = elt["bbox"]

        # denormalize the box if it is normalized
        box = denormalize_bbox(box, (height, width))
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        text_box = draw.textbbox(
            (cx, cy), text=label, font=font, align="center", anchor="mm"
        )

        # Calculate the offset to center the text within the bounding box
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x0 = cx - text_width / 2
        text_y0 = cy - text_height / 2
        text_x1 = cx + text_width / 2
        text_y1 = cy + text_height / 2

        # Draw the rectangle encapsulating the text
        draw.rectangle((text_x0, text_y0, text_x1, text_y1), fill=colors[elt["label"]])

        # Draw the text at the center of the bounding box
        draw.text(
            (text_x0, text_y0),
            label,
            fill="black",
            font=font,
            anchor="lt",
        )

    return image


FUNCTION_TOOLS = [
    owl_v2_image,
    owl_v2_video,
    ocr,
    vit_image_classification,
    vit_nsfw_classification,
    countgd_counting,
    florence2_ocr,
    florence2_sam2_image,
    florence2_sam2_video_tracking,
    florence2_phrase_grounding,
    claude35_text_extraction,
    detr_segmentation,
    depth_anything_v2,
    generate_pose_image,
    closest_mask_distance,
    closest_box_distance,
    qwen2_vl_images_vqa,
    qwen2_vl_video_vqa,
    video_temporal_localization,
    flux_image_inpainting,
    siglip_classification,
]

UTIL_TOOLS = [
    extract_frames_and_timestamps,
    save_json,
    load_image,
    save_image,
    save_video,
    overlay_bounding_boxes,
    overlay_segmentation_masks,
    overlay_heat_map,
]

TOOLS = FUNCTION_TOOLS + UTIL_TOOLS

TOOLS_DF = get_tools_df(TOOLS)  # type: ignore
TOOL_DESCRIPTIONS = get_tool_descriptions(TOOLS)  # type: ignore
TOOL_DOCSTRING = get_tool_documentation(TOOLS)  # type: ignore
TOOLS_INFO = get_tools_info(FUNCTION_TOOLS)  # type: ignore
UTILITIES_DOCSTRING = get_tool_documentation(UTIL_TOOLS)  # type: ignore
