import io
import json
import logging
import os
import tempfile
import urllib.request
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union, cast
from warnings import warn
import time

import cv2
import numpy as np
import pandas as pd
import requests
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from pillow_heif import register_heif_opener  # type: ignore
import yt_dlp  # type: ignore
import pymupdf  # type: ignore
from google import genai  # type: ignore
from google.genai import types  # type: ignore

from vision_agent.lmm.lmm import LMM, AnthropicLMM, OpenAILMM
from vision_agent.utils.execute import FileSerializer, MimeType
from vision_agent.utils.image_utils import (
    b64_to_pil,
    convert_quad_box_to_bbox,
    convert_to_b64,
    denormalize_bbox,
    encode_image_bytes,
    normalize_bbox,
    numpy_to_bytes,
    rle_decode,
    rle_decode_array,
)
from vision_agent.utils.tools import (
    ToolCallTrace,
    add_bboxes_from_masks,
    nms,
    send_inference_request,
    send_task_inference_request,
    should_report_tool_traces,
    single_nms,
)
from vision_agent.utils.tools_doc import get_tool_descriptions as _get_tool_descriptions
from vision_agent.utils.tools_doc import (
    get_tool_documentation as _get_tool_documentation,
)
from vision_agent.utils.tools_doc import get_tools_df as _get_tools_df
from vision_agent.utils.tools_doc import get_tools_info as _get_tools_info
from vision_agent.utils.video import (
    extract_frames_from_video,
    frames_to_bytes,
    video_writer,
)
from vision_agent.utils.video_tracking import (
    ODModels,
    merge_segments,
    post_process,
    process_segment,
    split_frames_into_segments,
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


def _display_tool_trace(
    function_name: str,
    request: Dict[str, Any],
    response: Any,
    files: Union[List[Tuple[str, bytes]], str],
) -> None:
    # Sends data through IPython's display function so front-end can show them. We use
    # a function here instead of a decarator becuase we do not want to re-calculate data
    # such as video bytes, which can be slow. Since this is calculated inside the
    # function we can't capture it with a decarator without adding it as a return value
    # which would change the function signature and affect the agent.
    if not should_report_tool_traces():
        return

    files_in_b64: List[Tuple[str, str]]
    if isinstance(files, str):
        files_in_b64 = [("images", files)]
    else:
        files_in_b64 = [(file[0], b64encode(file[1]).decode("utf-8")) for file in files]

    request["function_name"] = function_name
    tool_call_trace = ToolCallTrace(
        endpoint_url="",
        type="tool_func_call",
        request=request,
        response={"data": response},
        error=None,
        files=files_in_b64,
    )
    display({MimeType.APPLICATION_JSON: tool_call_trace.model_dump()}, raw=True)


def _sam2(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, ...],
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    if image_bytes is None:
        image_bytes = numpy_to_bytes(image)

    files = [("images", image_bytes)]
    payload = {
        "model": "sam2",
        "bboxes": json.dumps(
            [
                {
                    "labels": [d["label"] for d in detections],
                    "bboxes": [
                        denormalize_bbox(d["bbox"], image_size) for d in detections
                    ],
                }
            ]
        ),
    }

    metadata = {"function_name": "sam2"}
    pred_detections = send_task_inference_request(
        payload, "sam2", files=files, metadata=metadata
    )
    frame = pred_detections[0]
    return_data = []
    display_data = []
    for inp_detection, detection in zip(detections, frame):
        mask = rle_decode_array(detection["mask"])
        label = detection["label"]
        bbox = normalize_bbox(detection["bounding_box"], detection["mask"]["size"])
        return_data.append(
            {
                "label": label,
                "bbox": bbox,
                "mask": mask,
                "score": inp_detection["score"],
            }
        )
        display_data.append(
            {
                "label": label,
                "bbox": detection["bounding_box"],
                "mask": detection["mask"],
                "score": inp_detection["score"],
            }
        )
    return {"files": files, "return_data": return_data, "display_data": display_data}


def sam2(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """'sam2' is a tool that can segment multiple objects given an input bounding box,
    label and score. It returns a set of masks along with the corresponding bounding
    boxes and labels.

    Parameters:
        image (np.ndarray): The image that contains multiple instances of the object.
        detections (List[Dict[str, Any]]): A list of dictionaries containing the score,
            label, and bounding box of the detected objects with normalized coordinates
            between 0 and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates
            of the top-left and xmax and ymax are the coordinates of the bottom-right of
            the bounding box.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> sam2(image, [
                {'score': 0.49, 'label': 'flower', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            ])
        [
            {
                'score': 0.49,
                'label': 'flower',
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
    ret = _sam2(image, detections, image_size)
    _display_tool_trace(
        sam2.__name__,
        {"detections": detections},
        ret["display_data"],
        ret["files"],
    )

    return ret["return_data"]  # type: ignore


def od_sam2_video_tracking(
    od_model: ODModels,
    prompt: str,
    frames: List[np.ndarray],
    box_threshold: float = 0.30,
    chunk_length: Optional[int] = 50,
    deployment_id: Optional[str] = None,
) -> Dict[str, Any]:
    chunk_length = 50 if chunk_length is None else chunk_length
    segment_size = chunk_length
    # Number of overlapping frames between segments
    overlap = 1
    # chunk_length needs to be segment_size + 1 or else on the last segment it will
    # run the OD model again and merging will not work
    chunk_length = chunk_length + 1

    if len(frames) == 0 or not isinstance(frames, List):
        return {"files": [], "return_data": [], "display_data": []}

    image_size = frames[0].shape[:2]

    # Split frames into segments with overlap
    segments = split_frames_into_segments(frames, segment_size, overlap)

    def _apply_object_detection(  # inner method to avoid circular importing issues.
        od_model: ODModels,
        prompt: str,
        segment_index: int,
        frame_number: int,
        deployment_id: str,
        segment_frames: list,
    ) -> tuple:
        """
        Applies the specified object detection model to the given image.

        Args:
            od_model: The object detection model to use.
            prompt: The prompt for the object detection model.
            segment_index: The index of the current segment.
            frame_number: The number of the current frame.
            deployment_id: Optional The Model deployment ID.
            segment_frames: List of frames for the current segment.

        Returns:
            A tuple containing the object detection results and the name of the function used.
        """

        if od_model == ODModels.COUNTGD:
            segment_results = countgd_object_detection(
                prompt=prompt,
                image=segment_frames[frame_number],
                box_threshold=box_threshold,
            )
            function_name = "countgd_object_detection"

        elif od_model == ODModels.OWLV2:
            segment_results = owlv2_object_detection(
                prompt=prompt,
                image=segment_frames[frame_number],
                box_threshold=box_threshold,
            )
            function_name = "owlv2_object_detection"

        elif od_model == ODModels.FLORENCE2:
            segment_results = florence2_object_detection(
                prompt=prompt,
                image=segment_frames[frame_number],
            )
            function_name = "florence2_object_detection"

        elif od_model == ODModels.AGENTIC:
            segment_results = agentic_object_detection(
                prompt=prompt,
                image=segment_frames[frame_number],
            )
            function_name = "agentic_object_detection"

        elif od_model == ODModels.CUSTOM:
            segment_results = custom_object_detection(
                deployment_id=deployment_id,
                image=segment_frames[frame_number],
                box_threshold=box_threshold,
            )
            function_name = "custom_object_detection"
        elif od_model == ODModels.GLEE:
            segment_results = glee_object_detection(
                prompt=prompt,
                image=segment_frames[frame_number],
                box_threshold=box_threshold,
            )
            function_name = "glee_object_detection"

        else:
            raise NotImplementedError(
                f"Object detection model '{od_model}' is not implemented."
            )

        return segment_results, function_name

    # Process each segment and collect detections
    detections_per_segment: List[Any] = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_segment,
                segment_frames=segment,
                od_model=od_model,
                prompt=prompt,
                deployment_id=deployment_id,
                chunk_length=chunk_length,
                image_size=image_size,
                segment_index=segment_index,
                object_detection_tool=_apply_object_detection,
            ): segment_index
            for segment_index, segment in enumerate(segments)
        }

        for future in as_completed(futures):
            segment_index = futures[future]
            detections_per_segment.append((segment_index, future.result()))

    detections_per_segment = [
        x[1] for x in sorted(detections_per_segment, key=lambda x: x[0])
    ]

    merged_detections = merge_segments(detections_per_segment)
    post_processed = post_process(merged_detections, image_size)

    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]

    return {
        "files": files,
        "return_data": post_processed["return_data"],
        "display_data": post_processed["display_data"],
    }


# Owl V2 Tools


def _owlv2_object_detection(
    prompt: str,
    image: np.ndarray,
    box_threshold: float,
    image_size: Tuple[int, ...],
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    if image_bytes is None:
        image_bytes = numpy_to_bytes(image)

    files = [("image", image_bytes)]
    payload = {
        "prompts": [s.strip() for s in prompt.split(",")],
        "confidence": box_threshold,
        "model": "owlv2",
    }
    metadata = {"function_name": "owlv2_object_detection"}

    detections = send_task_inference_request(
        payload,
        "text-to-object-detection",
        files=files,
        metadata=metadata,
    )

    # get the first frame
    bboxes = detections[0]
    bboxes_formatted = [
        {
            "label": bbox["label"],
            "bbox": normalize_bbox(bbox["bounding_box"], image_size),
            "score": round(bbox["score"], 2),
        }
        for bbox in bboxes
    ]
    display_data = [
        {
            "label": bbox["label"],
            "bbox": bbox["bounding_box"],
            "score": round(bbox["score"], 2),
        }
        for bbox in bboxes
    ]
    return {
        "files": files,
        "return_data": bboxes_formatted,
        "display_data": display_data,
    }


def owlv2_object_detection(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """'owlv2_object_detection' is a tool that can detect and count multiple objects
    given a text prompt such as category names or referring expressions on images. The
    categories in text prompt are separated by commas. It returns a list of bounding
    boxes with normalized coordinates, label names and associated probability scores.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (np.ndarray): The image to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults
            to 0.10.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> owlv2_object_detection("car, dinosaur", image)
        [
            {'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.98, 'label': 'car', 'bbox': [0.2, 0.21, 0.45, 0.5},
        ]
    """

    image_size = image.shape[:2]
    if image_size[0] < 1 or image_size[1] < 1:
        return []

    ret = _owlv2_object_detection(prompt, image, box_threshold, image_size)

    _display_tool_trace(
        owlv2_object_detection.__name__,
        {
            "prompts": prompt,
            "confidence": box_threshold,
        },
        ret["display_data"],
        ret["files"],
    )
    return ret["return_data"]  # type: ignore


def owlv2_sam2_instance_segmentation(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """'owlv2_sam2_instance_segmentation' is a tool that can detect and count multiple
    instances of objects given a text prompt such as category names or referring
    expressions on images. The categories in text prompt are separated by commas. It
    returns a list of bounding boxes with normalized coordinates, label names, masks
    and associated probability scores.

    Parameters:
        prompt (str): The object that needs to be counted.
        image (np.ndarray): The image that contains multiple instances of the object.
        box_threshold (float, optional): The threshold for detection. Defaults
            to 0.10.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> owlv2_sam2_instance_segmentation("flower", image)
        [
            {
                'score': 0.49,
                'label': 'flower',
                'bbox': [0.1, 0.11, 0.35, 0.4],
                'mask': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            },
        ]
    """

    od_ret = _owlv2_object_detection(prompt, image, box_threshold, image.shape[:2])
    seg_ret = _sam2(
        image, od_ret["return_data"], image.shape[:2], image_bytes=od_ret["files"][0][1]
    )

    _display_tool_trace(
        owlv2_sam2_instance_segmentation.__name__,
        {
            "prompts": prompt,
            "confidence": box_threshold,
        },
        seg_ret["display_data"],
        seg_ret["files"],
    )

    return seg_ret["return_data"]  # type: ignore


def owlv2_sam2_video_tracking(
    prompt: str,
    frames: List[np.ndarray],
    box_threshold: float = 0.10,
    chunk_length: Optional[int] = 25,
) -> List[List[Dict[str, Any]]]:
    """'owlv2_sam2_video_tracking' is a tool that can track and segment multiple
    objects in a video given a text prompt such as category names or referring
    expressions. The categories in the text prompt are separated by commas. It returns
    a list of bounding boxes, label names, masks and associated probability scores and
    is useful for tracking and counting without duplicating counts.

    Parameters:
        prompt (str): The prompt to ground to the image.
        frames (List[np.ndarray]): The list of frames to ground the prompt to.
        box_threshold (float, optional): The threshold for the box detection. Defaults
            to 0.10.
        chunk_length (Optional[int]): The number of frames to re-run owlv2 to find
            new objects.

    Returns:
        List[List[Dict[str, Any]]]: A list of list of dictionaries containing the
            label, segmentation mask and bounding boxes. The outer list represents each
            frame and the inner list is the entities per frame. The detected objects
            have normalized coordinates between 0 and 1 (xmin, ymin, xmax, ymax). xmin
            and ymin are the coordinates of the top-left and xmax and ymax are the
            coordinates of the bottom-right of the bounding box. The mask is binary 2D
            numpy array where 1 indicates the object and 0 indicates the background.
            The label names are prefixed with their ID represent the total count.

    Example
    -------
        >>> owlv2_sam2_video_tracking("car, dinosaur", frames)
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

    ret = od_sam2_video_tracking(
        ODModels.OWLV2,
        prompt=prompt,
        frames=frames,
        box_threshold=box_threshold,
        chunk_length=chunk_length,
    )
    _display_tool_trace(
        owlv2_sam2_video_tracking.__name__,
        {"prompt": prompt, "chunk_length": chunk_length},
        ret["display_data"],
        ret["files"],
    )
    return ret["return_data"]  # type: ignore


# Florence2 Tools


def florence2_object_detection(
    prompt: str,
    image: np.ndarray,
) -> List[Dict[str, Any]]:
    """'florence2_object_detection' is a tool that can detect multiple objects given a
    text prompt which can be object names or caption. You can optionally separate the
    object names in the text with commas. It returns a list of bounding boxes with
    normalized coordinates, label names and associated confidence scores of 1.0.

    Parameters:
        prompt (str): The prompt to ground to the image. Use exclusive categories that
            do not overlap such as 'person, car' and NOT 'person, athlete'.
        image (np.ndarray): The image to used to detect objects

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box. The scores are always 1.0 and cannot be thresholded

    Example
    -------
        >>> florence2_object_detection('person looking at a coyote', image)
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
    metadata = {"function_name": "florence2_object_detection"}

    detections = send_task_inference_request(
        payload,
        "text-to-object-detection",
        files=files,
        metadata=metadata,
    )

    # get the first frame
    bboxes = detections[0]
    bboxes_formatted = [
        {
            "label": bbox["label"],
            "bbox": normalize_bbox(bbox["bounding_box"], image_size),
            "score": round(bbox["score"], 2),
        }
        for bbox in bboxes
    ]

    _display_tool_trace(
        florence2_object_detection.__name__,
        payload,
        detections[0],
        files,
    )
    return [bbox for bbox in bboxes_formatted]


def florence2_sam2_instance_segmentation(
    prompt: str,
    image: np.ndarray,
) -> List[Dict[str, Any]]:
    """'florence2_sam2_instance_segmentation' is a tool that can segment multiple
    objects given a text prompt such as category names or referring expressions. The
    categories in the text prompt are separated by commas. It returns a list of
    bounding boxes, label names, mask file names and associated probability scores of
    1.0.

    Parameters:
        prompt (str): The prompt to ground to the image. Use exclusive categories that
            do not overlap such as 'person, car' and NOT 'person, athlete'.
        image (np.ndarray): The image to ground the prompt to.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> florence2_sam2_instance_segmentation("car, dinosaur", image)
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
    metadata = {"function_name": "florence2_sam2_instance_segmentation"}

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

    _display_tool_trace(
        florence2_sam2_instance_segmentation.__name__,
        payload,
        detections[0],
        files,
    )
    return return_data


def florence2_sam2_video_tracking(
    prompt: str,
    frames: List[np.ndarray],
    chunk_length: Optional[int] = 25,
) -> List[List[Dict[str, Any]]]:
    """'florence2_sam2_video_tracking' is a tool that can track and segment multiple
    objects in a video given a text prompt such as category names or referring
    expressions. The categories in the text prompt are separated by commas. It returns
    a list of bounding boxes, label names, masks and associated probability scores and
    is useful for tracking and counting without duplicating counts.

    Parameters:
        prompt (str): The prompt to ground to the image. Use exclusive categories that
            do not overlap such as 'person, car' and NOT 'person, athlete'.
        frames (List[np.ndarray]): The list of frames to ground the prompt to.
        chunk_length (Optional[int]): The number of frames to re-run florence2 to find
            new objects.

    Returns:
        List[List[Dict[str, Any]]]: A list of list of dictionaries containing the
            label, segmentation mask and bounding boxes. The outer list represents each
            frame and the inner list is the entities per frame. The detected objects
            have normalized coordinates between 0 and 1 (xmin, ymin, xmax, ymax). xmin
            and ymin are the coordinates of the top-left and xmax and ymax are the
            coordinates of the bottom-right of the bounding box. The mask is binary 2D
            numpy array where 1 indicates the object and 0 indicates the background.
            The label names are prefixed with their ID represent the total count.

    Example
    -------
        >>> florence2_sam2_video_tracking("car, dinosaur", frames)
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
            return_frame_data.append(
                {"label": label, "mask": mask, "score": 1.0, "rle": detection["mask"]}
            )
        return_data.append(return_frame_data)
    return_data = add_bboxes_from_masks(return_data)
    return_data = nms(return_data, iou_threshold=0.95)

    _display_tool_trace(
        florence2_sam2_video_tracking.__name__,
        payload,
        [
            [
                {
                    "label": e["label"],
                    "score": e["score"],
                    "bbox": denormalize_bbox(e["bbox"], frames[0].shape[:2]),
                    "mask": e["rle"],
                }
                for e in lst
            ]
            for lst in return_data
        ],
        files,
    )
    # We save the RLE for display purposes, re-calculting RLE can get very expensive.
    # Deleted here because we are returning the numpy masks instead
    for frame in return_data:
        for obj in frame:
            del obj["rle"]
    return return_data


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
    _display_tool_trace(
        florence2_ocr.__name__,
        {},
        detections,
        image_b64,
    )
    return return_data


# CountGD Tools


def _countgd_object_detection(
    prompt: str,
    image: np.ndarray,
    box_threshold: float,
    image_size: Tuple[int, ...],
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    if image_bytes is None:
        image_bytes = numpy_to_bytes(image)

    files = [("image", image_bytes)]
    prompts = [p.strip() for p in prompt.split(", ")]

    def _run_countgd(prompt: str) -> List[Dict[str, Any]]:
        payload = {
            "prompts": [prompt],
            "confidence": box_threshold,  # still not being used in the API
            "model": "countgd",
        }
        metadata = {"function_name": "countgd_object_detection"}

        detections = send_task_inference_request(
            payload, "text-to-object-detection", files=files, metadata=metadata
        )
        # get the first frame
        return detections[0]  # type: ignore

    bboxes = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_run_countgd, prompt) for prompt in prompts]
        for future in as_completed(futures):
            bboxes.extend(future.result())

    return_data = [
        {
            "label": bbox["label"],
            "bbox": normalize_bbox(bbox["bounding_box"], image_size),
            "score": round(bbox["score"], 2),
        }
        for bbox in bboxes
    ]

    return_data = single_nms(return_data, iou_threshold=0.80)
    display_data = [
        {
            "label": e["label"],
            "score": e["score"],
            "bbox": denormalize_bbox(e["bbox"], image_size),
        }
        for e in return_data
    ]
    return {"files": files, "return_data": return_data, "display_data": display_data}


def countgd_object_detection(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.23,
) -> List[Dict[str, Any]]:
    """'countgd_object_detection' is a tool that can detect multiple instances of an
    object given a text prompt. It is particularly useful when trying to detect and
    count a large number of objects. You can optionally separate object names in the
    prompt with commas. It returns a list of bounding boxes with normalized
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
        >>> countgd_object_detection("flower", image)
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

    ret = _countgd_object_detection(prompt, image, box_threshold, image_size)
    _display_tool_trace(
        countgd_object_detection.__name__,
        {
            "prompts": prompt,
            "confidence": box_threshold,
        },
        ret["display_data"],
        ret["files"],
    )
    return ret["return_data"]  # type: ignore


def countgd_sam2_instance_segmentation(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.23,
) -> List[Dict[str, Any]]:
    """'countgd_sam2_instance_segmentation' is a tool that can detect multiple
    instances of an object given a text prompt. It is particularly useful when trying
    to detect and count a large number of objects. You can optionally separate object
    names in the prompt with commas. It returns a list of bounding boxes with
    normalized coordinates, label names, masks associated confidence scores.

    Parameters:
        prompt (str): The object that needs to be counted.
        image (np.ndarray): The image that contains multiple instances of the object.
        box_threshold (float, optional): The threshold for detection. Defaults
            to 0.23.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> countgd_sam2_instance_segmentation("flower", image)
        [
            {
                'score': 0.49,
                'label': 'flower',
                'bbox': [0.1, 0.11, 0.35, 0.4],
                'mask': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            },
        ]
    """

    od_ret = _countgd_object_detection(prompt, image, box_threshold, image.shape[:2])
    seg_ret = _sam2(
        image, od_ret["return_data"], image.shape[:2], image_bytes=od_ret["files"][0][1]
    )

    _display_tool_trace(
        countgd_sam2_instance_segmentation.__name__,
        {
            "prompts": prompt,
            "confidence": box_threshold,
        },
        seg_ret["display_data"],
        seg_ret["files"],
    )

    return seg_ret["return_data"]  # type: ignore


def countgd_sam2_video_tracking(
    prompt: str,
    frames: List[np.ndarray],
    box_threshold: float = 0.23,
    chunk_length: Optional[int] = 25,
) -> List[List[Dict[str, Any]]]:
    """'countgd_sam2_video_tracking' is a tool that can track and segment multiple
    objects in a video given a text prompt such as category names or referring
    expressions. The categories in the text prompt are separated by commas. It returns
    a list of bounding boxes, label names, masks and associated probability scores and
    is useful for tracking and counting without duplicating counts.

    Parameters:
        prompt (str): The prompt to ground to the image.
        frames (List[np.ndarray]): The list of frames to ground the prompt to.
        box_threshold (float, optional): The threshold for detection. Defaults
            to 0.23.
        chunk_length (Optional[int]): The number of frames to re-run countgd to find
            new objects.

    Returns:
        List[List[Dict[str, Any]]]: A list of list of dictionaries containing the
            label, segmentation mask and bounding boxes. The outer list represents each
            frame and the inner list is the entities per frame. The detected objects
            have normalized coordinates between 0 and 1 (xmin, ymin, xmax, ymax). xmin
            and ymin are the coordinates of the top-left and xmax and ymax are the
            coordinates of the bottom-right of the bounding box. The mask is binary 2D
            numpy array where 1 indicates the object and 0 indicates the background.
            The label names are prefixed with their ID represent the total count.

    Example
    -------
        >>> countgd_sam2_video_tracking("car, dinosaur", frames)
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

    ret = od_sam2_video_tracking(
        ODModels.COUNTGD,
        prompt=prompt,
        frames=frames,
        box_threshold=box_threshold,
        chunk_length=chunk_length,
    )
    _display_tool_trace(
        countgd_sam2_video_tracking.__name__,
        {},
        ret["display_data"],
        ret["files"],
    )
    return ret["return_data"]  # type: ignore


def _countgd_visual_object_detection(
    visual_prompts: List[List[float]],
    image: np.ndarray,
    box_threshold: float = 0.23,
) -> Dict[str, Any]:
    image_size = image.shape[:2]

    buffer_bytes = numpy_to_bytes(image)
    files = [("image", buffer_bytes)]
    visual_prompts = [
        denormalize_bbox(bbox, image.shape[:2]) for bbox in visual_prompts
    ]
    payload = {
        "visual_prompts": json.dumps(visual_prompts),
        "model": "countgd",
        "confidence": box_threshold,
    }
    metadata = {"function_name": "countgd_visual_object_detection"}

    detections = send_task_inference_request(
        payload, "visual-prompts-to-object-detection", files=files, metadata=metadata
    )

    # get the first frame
    bboxes = detections[0]
    bboxes_formatted = [
        {
            "label": bbox["label"],
            "bbox": normalize_bbox(bbox["bounding_box"], image_size),
            "score": round(bbox["score"], 2),
        }
        for bbox in bboxes
    ]
    display_data = [
        {
            "label": bbox["label"],
            "bbox": bbox["bounding_box"],
            "score": bbox["score"],
        }
        for bbox in bboxes
    ]
    return {
        "files": files,
        "return_data": bboxes_formatted,
        "display_data": display_data,
    }


def countgd_visual_object_detection(
    visual_prompts: List[List[float]],
    image: np.ndarray,
    box_threshold: float = 0.23,
) -> List[Dict[str, Any]]:
    """'countgd_visual_object_detection' is a tool that can detect multiple instances
    of an object given a visual prompt. It is particularly useful when trying to detect
    and count a large number of objects. You can optionally separate object names in
    the prompt with commas. It returns a list of bounding boxes with normalized
    coordinates, label names and associated confidence scores.

    Parameters:
        visual_prompts (List[List[float]]): Bounding boxes of the object in format
            [xmin, ymin, xmax, ymax] with normalized coordinates. Up to 3 bounding
            boxes can be provided.
        image (np.ndarray): The image that contains multiple instances of the object.
        box_threshold (float, optional): The threshold for detection. Defaults to 0.23.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> countgd_visual_object_detection(
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

    od_ret = _countgd_visual_object_detection(visual_prompts, image, box_threshold)

    _display_tool_trace(
        countgd_visual_object_detection.__name__,
        {},
        od_ret["display_data"],
        od_ret["files"],
    )

    return od_ret["return_data"]  # type: ignore


def countgd_sam2_visual_instance_segmentation(
    visual_prompts: List[List[float]],
    image: np.ndarray,
    box_threshold: float = 0.23,
) -> List[Dict[str, Any]]:
    """'countgd_sam2_visual_instance_segmentation' is a tool that can precisely count
    multiple instances of an object given few visual example prompts. It returns a list
    of bounding boxes, label names, masks and associated probability scores.

    Parameters:
        visual_prompts (List[List[float]]): Bounding boxes of the object in format
            [xmin, ymin, xmax, ymax] with normalized coordinates. Up to 3 bounding
            boxes can be provided.
        image (np.ndarray): The image that contains multiple instances of the object.
        box_threshold (float, optional): The threshold for detection. Defaults to 0.23.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> countgd_sam2_visual_instance_segmentation(
            visual_prompts=[[0.1, 0.1, 0.4, 0.42], [0.2, 0.3, 0.25, 0.35]],
            image=image
        )
        [
            {
                'score': 0.49,
                'label': 'object',
                'bbox': [0.1, 0.11, 0.35, 0.4],
                'mask': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            },
        ]
    """

    od_ret = _countgd_visual_object_detection(visual_prompts, image, box_threshold)
    seg_ret = _sam2(
        image, od_ret["return_data"], image.shape[:2], image_bytes=od_ret["files"][0][1]
    )
    _display_tool_trace(
        countgd_sam2_visual_instance_segmentation.__name__,
        {},
        seg_ret["display_data"],
        seg_ret["files"],
    )
    return seg_ret["return_data"]  # type: ignore


# Custom Models


def custom_object_detection(
    deployment_id: str,
    image: np.ndarray,
    box_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """'custom_object_detection' is a tool that can detect instances of an
    object given a deployment_id of a previously finetuned object detection model.
    It is particularly useful when trying to detect objects that are not well detected by generalist models.
    It returns a list of bounding boxes with normalized
    coordinates, label names and associated confidence scores.

    Parameters:
        deployment_id (str): The id of the finetuned model.
        image (np.ndarray): The image that contains instances of the object.
        box_threshold (float, optional): The threshold for detection. Defaults
            to 0.1.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> custom_object_detection("abcd1234-5678efg", image)
        [
            {'score': 0.49, 'label': 'flower', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.68, 'label': 'flower', 'bbox': [0.2, 0.21, 0.45, 0.5]},
            {'score': 0.78, 'label': 'flower', 'bbox': [0.3, 0.35, 0.48, 0.52]},
            {'score': 0.98, 'label': 'flower', 'bbox': [0.44, 0.24, 0.49, 0.58]},
        ]
    """
    image_size = image.shape[:2]
    if image_size[0] < 1 or image_size[1] < 1:
        return []

    files = [("image", numpy_to_bytes(image))]
    payload = {
        "deployment_id": deployment_id,
        "confidence": box_threshold,
    }
    detections: List[List[Dict[str, Any]]] = send_inference_request(
        payload, "custom-object-detection", files=files, v2=True
    )

    bboxes = detections[0]
    bboxes_formatted = [
        {
            "label": bbox["label"],
            "bbox": normalize_bbox(bbox["bounding_box"], image_size),
            "score": bbox["score"],
        }
        for bbox in bboxes
    ]
    display_data = [
        {
            "label": bbox["label"],
            "bbox": bbox["bounding_box"],
            "score": bbox["score"],
        }
        for bbox in bboxes
    ]

    _display_tool_trace(
        custom_object_detection.__name__,
        payload,
        display_data,
        files,
    )
    return bboxes_formatted


def custom_od_sam2_video_tracking(
    deployment_id: str,
    frames: List[np.ndarray],
    chunk_length: Optional[int] = 25,
) -> List[List[Dict[str, Any]]]:
    """'custom_od_sam2_video_tracking' is a tool that can segment multiple objects given a
    custom model with predefined category names.
    It returns a list of bounding boxes, label names,
    mask file names and associated probability scores.

    Parameters:
        deployment_id (str): The id of the deployed custom model.
        image (np.ndarray): The image to ground the prompt to.
        chunk_length (Optional[int]): The number of frames to re-run florence2 to find
            new objects.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> custom_od_sam2_video_tracking("abcd1234-5678efg", frames)
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

    ret = od_sam2_video_tracking(
        ODModels.CUSTOM,
        prompt="",
        frames=frames,
        chunk_length=chunk_length,
        deployment_id=deployment_id,
    )
    _display_tool_trace(
        custom_od_sam2_video_tracking.__name__,
        {},
        ret["display_data"],
        ret["files"],
    )
    return ret["return_data"]  # type: ignore


# Agentic OD Tools


def _agentic_object_detection(
    prompt: str,
    image: np.ndarray,
    image_size: Tuple[int, ...],
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    if image_bytes is None:
        image_bytes = numpy_to_bytes(image)

    files = [("image", image_bytes)]
    payload = {
        "prompts": [s.strip() for s in prompt.split(",")],
        "model": "agentic",
    }
    metadata = {"function_name": "agentic_object_detection"}

    detections = send_task_inference_request(
        payload,
        "text-to-object-detection",
        files=files,
        metadata=metadata,
    )

    # get the first frame
    bboxes = detections[0]
    bboxes_formatted = [
        {
            "label": bbox["label"],
            "bbox": normalize_bbox(bbox["bounding_box"], image_size),
            "score": bbox["score"],
        }
        for bbox in bboxes
    ]
    display_data = [
        {
            "label": bbox["label"],
            "bbox": bbox["bounding_box"],
            "score": bbox["score"],
        }
        for bbox in bboxes
    ]
    return {
        "files": files,
        "return_data": bboxes_formatted,
        "display_data": display_data,
    }


def agentic_object_detection(
    prompt: str,
    image: np.ndarray,
) -> List[Dict[str, Any]]:
    """'agentic_object_detection' is a tool that can detect multiple objects given a
    text prompt such as object names or referring expressions on images. It's
    particularly good at detecting specific objects given detailed descriptive prompts
    but runs slower so not ideal for high counts. It returns a list of bounding boxes
    with normalized coordinates, label names and associated confidence score of 1.0.

    Parameters:
        prompt (str): The prompt to ground to the image, only supports a single prompt
            with no commas or periods.
        image (np.ndarray): The image to ground the prompt to.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> agentic_object_detection("a red car", image)
        [
            {'score': 0.99, 'label': 'a red car', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.98, 'label': 'a red car', 'bbox': [0.2, 0.21, 0.45, 0.5},
        ]
    """

    image_size = image.shape[:2]
    if image_size[0] < 1 or image_size[1] < 1:
        return []

    ret = _agentic_object_detection(prompt, image, image_size)

    _display_tool_trace(
        agentic_object_detection.__name__,
        {"prompts": prompt},
        ret["display_data"],
        ret["files"],
    )
    return ret["return_data"]  # type: ignore


def agentic_sam2_instance_segmentation(
    prompt: str, image: np.ndarray
) -> List[Dict[str, Any]]:
    """'agentic_sam2_instance_segmentation' is a tool that can detect multiple
    instances given a text prompt such as object names or referring expressions on
    images. It's particularly good at detecting specific objects given detailed
    descriptive prompts but runs slower so not ideal for high counts. It returns a list
    of bounding boxes with normalized coordinates, label names, masks and associated
    confidence score of 1.0.

    Parameters:
        prompt (str): The object that needs to be counted, only supports a single
            prompt with no commas or periods.
        image (np.ndarray): The image that contains multiple instances of the object.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> agentic_sam2_instance_segmentation("a large blue flower", image)
        [
            {
                'score': 0.49,
                'label': 'a large blue flower',
                'bbox': [0.1, 0.11, 0.35, 0.4],
                'mask': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            },
        ]
    """

    od_ret = _agentic_object_detection(prompt, image, image.shape[:2])
    seg_ret = _sam2(
        image, od_ret["return_data"], image.shape[:2], image_bytes=od_ret["files"][0][1]
    )

    _display_tool_trace(
        agentic_sam2_instance_segmentation.__name__,
        {
            "prompts": prompt,
        },
        seg_ret["display_data"],
        seg_ret["files"],
    )

    return seg_ret["return_data"]  # type: ignore


def agentic_sam2_video_tracking(
    prompt: str,
    frames: List[np.ndarray],
    chunk_length: Optional[int] = 25,
) -> List[List[Dict[str, Any]]]:
    """'agentic_sam2_video_tracking' is a tool that can track and segment multiple
    objects in a video given a text prompt such as object names or referring
    expressions. It's particularly good at detecting specific objects given detailed
    descriptive prompts but runs slower so not ideal for high counts. It returns a list
    of bounding boxes, label names, masks and associated confidence score of 1.0 and is
    useful for tracking and counting without duplicating counts.

    Parameters:
        prompt (str): The prompt to ground to the image, only supports a single prompt
            with  no commas or periods.
        frames (List[np.ndarray]): The list of frames to ground the prompt to.
        chunk_length (Optional[int]): The number of frames to re-run agentic object detection to
            to find new objects.

    Returns:
        List[List[Dict[str, Any]]]: A list of list of dictionaries containing the
            label, segmentation mask and bounding boxes. The outer list represents each
            frame and the inner list is the entities per frame. The detected objects
            have normalized coordinates between 0 and 1 (xmin, ymin, xmax, ymax). xmin
            and ymin are the coordinates of the top-left and xmax and ymax are the
            coordinates of the bottom-right of the bounding box. The mask is binary 2D
            numpy array where 1 indicates the object and 0 indicates the background.
            The label names are prefixed with their ID represent the total count.

    Example
    -------
        >>> agentic_sam2_video_tracking("a runner with yellow shoes", frames)
        [
            [
                {
                    'label': '0: a runner with yellow shoes',
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

    ret = od_sam2_video_tracking(
        ODModels.AGENTIC,
        prompt=prompt,
        frames=frames,
        chunk_length=chunk_length,
    )
    _display_tool_trace(
        agentic_sam2_video_tracking.__name__,
        {},
        ret["display_data"],
        ret["files"],
    )
    return ret["return_data"]  # type: ignore


# GLEE Tools


def _glee_object_detection(
    prompt: str,
    image: np.ndarray,
    box_threshold: float,
    image_size: Tuple[int, ...],
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    if image_bytes is None:
        image_bytes = numpy_to_bytes(image)

    files = [("image", image_bytes)]
    payload = {
        "prompts": [s.strip() for s in prompt.split(",")],
        "confidence": box_threshold,
        "model": "glee",
    }
    metadata = {"function_name": "glee"}
    detections = send_task_inference_request(
        payload,
        "text-to-object-detection",
        files=files,
        metadata=metadata,
    )
    # get the first frame
    bboxes = detections[0]
    bboxes_formatted = [
        {
            "label": bbox["label"],
            "bbox": normalize_bbox(bbox["bounding_box"], image_size),
            "score": round(bbox["score"], 2),
        }
        for bbox in bboxes
    ]
    display_data = [
        {
            "label": bbox["label"],
            "bbox": bbox["bounding_box"],
            "score": round(bbox["score"], 2),
        }
        for bbox in bboxes
    ]
    return {
        "files": files,
        "return_data": bboxes_formatted,
        "display_data": display_data,
    }


def glee_object_detection(
    prompt: str,
    image: np.ndarray,
    box_threshold: float = 0.23,
) -> List[Dict[str, Any]]:
    """'glee_object_detection' is a tool that can detect multiple objects given a
    text prompt such as object names or referring expressions on images. It's
    particularly good at detecting specific objects given detailed descriptive prompts.
    It returns a list of bounding boxes with normalized coordinates, label names and
    associated probability scores.

    Parameters:
        prompt (str): The prompt to ground to the image, only supports a single prompt
            with no commas or periods.
        image (np.ndarray): The image to ground the prompt to.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> glee_object_detection("person holding a box", image)
        [
            {'score': 0.99, 'label': 'person holding a box', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.98, 'label': 'person holding a box', 'bbox': [0.2, 0.21, 0.45, 0.5},
        ]
    """

    od_ret = _glee_object_detection(prompt, image, box_threshold, image.shape[:2])
    _display_tool_trace(
        glee_object_detection.__name__,
        {"prompts": prompt, "confidence": box_threshold},
        od_ret["display_data"],
        od_ret["files"],
    )
    return od_ret["return_data"]  # type: ignore


def glee_sam2_instance_segmentation(
    prompt: str, image: np.ndarray, box_threshold: float = 0.23
) -> List[Dict[str, Any]]:
    """'glee_sam2_instance_segmentation' is a tool that can detect multiple
    instances given a text prompt such as object names or referring expressions on
    images. It's particularly good at detecting specific objects given detailed
    descriptive prompts. It returns a list of bounding boxes with normalized
    coordinates, label names, masks and associated probability scores.

    Parameters:
        prompt (str): The object that needs to be counted, only supports a single
            prompt with no commas or periods.
        image (np.ndarray): The image that contains multiple instances of the object.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> glee_sam2_instance_segmentation("a large blue flower", image)
        [
            {
                'score': 0.49,
                'label': 'a large blue flower',
                'bbox': [0.1, 0.11, 0.35, 0.4],
                'mask': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            },
        ]
    """
    od_ret = _glee_object_detection(prompt, image, box_threshold, image.shape[:2])
    seg_ret = _sam2(
        image, od_ret["return_data"], image.shape[:2], image_bytes=od_ret["files"][0][1]
    )

    _display_tool_trace(
        glee_sam2_instance_segmentation.__name__,
        {
            "prompts": prompt,
            "confidence": box_threshold,
        },
        seg_ret["display_data"],
        seg_ret["files"],
    )

    return seg_ret["return_data"]  # type: ignore


def glee_sam2_video_tracking(
    prompt: str,
    frames: List[np.ndarray],
    box_threshold: float = 0.23,
    chunk_length: Optional[int] = 25,
) -> List[List[Dict[str, Any]]]:
    """'glee_sam2_video_tracking' is a tool that can track and segment multiple
    objects in a video given a text prompt such as object names or referring
    expressions. It's particularly good at detecting specific objects given detailed
    descriptive prompts and returns a list of bounding boxes, label names, masks and
    associated probability scores and is useful for tracking and counting without
    duplicating counts.

    Parameters:
        prompt (str): The prompt to ground to the image, only supports a single prompt
            with  no commas or periods.
        frames (List[np.ndarray]): The list of frames to ground the prompt to.
        chunk_length (Optional[int]): The number of frames to re-run agentic object detection to
            to find new objects.

    Returns:
        List[List[Dict[str, Any]]]: A list of list of dictionaries containing the
            label, segmentation mask and bounding boxes. The outer list represents each
            frame and the inner list is the entities per frame. The detected objects
            have normalized coordinates between 0 and 1 (xmin, ymin, xmax, ymax). xmin
            and ymin are the coordinates of the top-left and xmax and ymax are the
            coordinates of the bottom-right of the bounding box. The mask is binary 2D
            numpy array where 1 indicates the object and 0 indicates the background.
            The label names are prefixed with their ID represent the total count.

    Example
    -------
        >>> glee_sam2_video_tracking("a runner with yellow shoes", frames)
        [
            [
                {
                    'label': '0: a runner with yellow shoes',
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
    ret = od_sam2_video_tracking(
        ODModels.GLEE,
        prompt=prompt,
        frames=frames,
        box_threshold=box_threshold,
        chunk_length=chunk_length,
    )
    _display_tool_trace(
        glee_sam2_video_tracking.__name__,
        {"prompt": prompt, "chunk_length": chunk_length},
        ret["display_data"],
        ret["files"],
    )
    return ret["return_data"]  # type: ignore


# Qwen2 and 2.5 VL Tool


def qwen25_vl_images_vqa(prompt: str, images: List[np.ndarray]) -> str:
    """'qwen25_vl_images_vqa' is a tool that can answer any questions about arbitrary
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
        >>> qwen25_vl_images_vqa('Give a summary of the document', images)
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
        "model": "qwen25vl",
        "function_name": "qwen25_vl_images_vqa",
    }
    data: Dict[str, Any] = send_inference_request(
        payload, "image-to-text", files=files, v2=True
    )
    _display_tool_trace(
        qwen25_vl_images_vqa.__name__,
        payload,
        cast(str, data),
        files,
    )
    return cast(str, data)


def qwen25_vl_video_vqa(prompt: str, frames: List[np.ndarray]) -> str:
    """'qwen25_vl_video_vqa' is a tool that can answer any questions about arbitrary videos
    including regular videos or videos of documents or presentations. It returns text
    as an answer to the question.

    Parameters:
        prompt (str): The question about the video
        frames (List[np.ndarray]): The reference frames used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> qwen25_vl_video_vqa('Which football player made the goal?', frames)
        'Lionel Messi'
    """

    if len(frames) == 0 or not isinstance(frames, list):
        raise ValueError("Must provide a list of numpy arrays for frames")

    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]
    payload = {
        "prompt": prompt,
        "model": "qwen25vl",
        "function_name": "qwen25_vl_video_vqa",
    }
    data: Dict[str, Any] = send_inference_request(
        payload, "image-to-text", files=files, v2=True
    )
    _display_tool_trace(
        qwen25_vl_video_vqa.__name__,
        payload,
        cast(str, data),
        files,
    )
    return cast(str, data)


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
    _display_tool_trace(
        qwen2_vl_images_vqa.__name__,
        payload,
        cast(str, data),
        files,
    )
    return cast(str, data)


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
    _display_tool_trace(
        qwen2_vl_video_vqa.__name__,
        payload,
        cast(str, data),
        files,
    )
    return cast(str, data)


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

    _display_tool_trace(
        ocr.__name__,
        {},
        data,
        cast(List[Tuple[str, bytes]], [("image", buffer_bytes)]),
    )
    return sorted(output, key=lambda x: (x["bbox"][1], x["bbox"][0]))


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


def document_extraction(image: np.ndarray) -> Dict[str, Any]:
    """'document_extraction' is a tool that can extract structured information out of
    documents with different layouts. It returns the extracted data in a structured
    hierarchical format containing text, tables, pictures, charts, and other
    information.

    Parameters:
        image (np.ndarray): The document image to analyze

    Returns:
        Dict[str, Any]: A dictionary containing the extracted information.

    Example
    -------
        >>> document_analysis(image)
        {'pages':
            [{'bbox': [0, 0, 1.0, 1.0],
                    'chunks': [{'bbox': [0.8, 0.1, 1.0, 0.2],
                                'label': 'page_header',
                                'order': 75
                                'caption': 'Annual Report 2024',
                                'summary': 'This annual report summarizes ...' },
                               {'bbox': [0.2, 0.9, 0.9, 1.0],
                                'label': 'table',
                                'order': 1119,
                                'caption': [{'Column 1': 'Value 1', 'Column 2': 'Value 2'},
                                'summary': 'This table illustrates a trend of ...'},
                    ],
    """
    warning = (
        "This function is deprecated. For document extraction please use the agentic-doc python package on "
        "https://pypi.org/project/agentic-doc/ or the agentic_document_extraction function."
    )
    warn(warning, DeprecationWarning, stacklevel=2)

    image_file = numpy_to_bytes(image)

    files = [("image", image_file)]

    payload = {
        "model": "document-analysis",
    }

    data: Dict[str, Any] = send_inference_request(
        payload=payload,
        endpoint_name="document-analysis",
        files=files,
        v2=True,
        metadata_payload={"function_name": "document_analysis"},
    )

    # don't display normalized bboxes
    _display_tool_trace(
        document_extraction.__name__,
        payload,
        data,
        files,
    )

    def normalize(data: Any) -> Dict[str, Any]:
        if isinstance(data, Dict):
            if "bbox" in data:
                data["bbox"] = normalize_bbox(data["bbox"], image.shape[:2])
            for key in data:
                data[key] = normalize(data[key])
        elif isinstance(data, List):
            for i in range(len(data)):
                data[i] = normalize(data[i])
        return data  # type: ignore

    data = normalize(data)

    return data


def agentic_document_extraction(image: np.ndarray) -> Dict[str, Any]:
    """'agentic_document_extraction' is a tool that can extract structured information
    out of documents with different layouts. It returns the extracted data in a
    structured hierarchical format containing text, tables, figures, charts, and other
    information.

    Parameters:
        image (np.ndarray): The document image to analyze

    Returns:
        Dict[str, Any]: A dictionary containing the extracted information.

    Example
    -------
        >>> agentic_document_analysis(image)
        {
            "markdown": "# Document title ## Document subtitle This is a sample document.",
            "chunks": [
                {
                    "text": "# Document title",
                    "grounding": [
                        {
                            "box": [0.06125, 0.019355758266818696, 0.17375, 0.03290478905359179],
                            "page": 0
                        }
                    ],
                    "chunk_type": "page_header",
                    "chunk_id": "622e0374-c50e-4960-a013-650138b42528"
                },
            ...
            ]
        }

    Notes
    ----
    For more document analysis features, please use the agentic-doc python package at
    https://github.com/landing-ai/agentic-doc
    """

    image_file = numpy_to_bytes(image)

    files = [("image", image_file)]

    payload = {
        "model": "agentic-document-analysis",
    }

    data: Dict[str, Any] = send_inference_request(
        payload=payload,
        endpoint_name="agentic-document-analysis",
        files=files,
        v2=True,
        metadata_payload={"function_name": "agentic_document_analysis"},
    )

    # don't display normalized bboxes
    _display_tool_trace(
        agentic_document_extraction.__name__,
        payload,
        data,
        files,
    )

    def transform_boxes(data: Dict[str, Any]) -> Dict[str, Any]:
        for chunk in data["chunks"]:
            for grounding in chunk["grounding"]:
                box = grounding["box"]
                grounding["box"] = [box["l"], box["t"], box["r"], box["b"]]
        return data

    data = transform_boxes(data)

    return data


def document_qa(
    prompt: str,
    image: np.ndarray,
) -> str:
    """'document_qa' is a tool that can answer any questions about arbitrary documents,
    presentations, or tables. It's very useful for document QA tasks, you can ask it a
    specific question or ask it to return a JSON object answering multiple questions
    about the document.

    Parameters:
        prompt (str): The question to be answered about the document image.
        image (np.ndarray): The document image to analyze.

    Returns:
        str: The answer to the question based on the document's context.

    Example
    -------
        >>> document_qa(image, question)
        'The answer to the question ...'
    """

    image_file = numpy_to_bytes(image)

    files = [("image", image_file)]

    payload = {
        "model": "agentic-document-analysis",
    }

    data: Dict[str, Any] = send_inference_request(
        payload=payload,
        endpoint_name="agentic-document-analysis",
        files=files,
        v2=True,
        metadata_payload={"function_name": "document_qa"},
    )

    def transform_boxes(data: Dict[str, Any]) -> Dict[str, Any]:
        for chunk in data["chunks"]:
            for grounding in chunk["grounding"]:
                box = grounding["box"]
                grounding["box"] = [box["l"], box["t"], box["r"], box["b"]]
        return data

    data = transform_boxes(data)

    prompt = f"""
Document Context:
{data}\n
Question: {prompt}\n
Answer the question directly using only the information from the document, do not answer with any additional text besides the answer. If the answer is not definitively contained in the document, say "I cannot find the answer in the provided document."
    """

    lmm = AnthropicLMM()
    llm_output = lmm.generate(prompt=prompt)
    llm_output = cast(str, llm_output)

    _display_tool_trace(
        document_qa.__name__,
        payload,
        llm_output,
        files,
    )

    return llm_output


def _sample(frames: List[np.ndarray], sample_size: int) -> List[np.ndarray]:
    sample_indices = np.linspace(0, len(frames) - 1, sample_size, dtype=int)
    sampled_frames = []

    for i, frame in enumerate(frames):
        if i in sample_indices:
            sampled_frames.append(frame)
        if len(sampled_frames) >= sample_size:
            break
    return sampled_frames


def _lmm_activity_recognition(
    lmm: LMM,
    segment: List[np.ndarray],
    prompt: str,
) -> List[float]:
    frames = _sample(segment, 10)
    media = []
    for frame in frames:
        buffer = io.BytesIO()
        image_pil = Image.fromarray(frame)
        if image_pil.size[0] > 768:
            image_pil.thumbnail((768, 768))
        image_pil.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = "data:image/png;base64," + encode_image_bytes(image_bytes)
        media.append(image_b64)

    response = cast(str, lmm.generate(prompt, media))
    if "yes" in response.lower():
        return [1.0] * len(segment)
    return [0.0] * len(segment)


def _qwenvl_activity_recognition(
    segment: List[np.ndarray], prompt: str, model_name: str = "qwen2vl"
) -> List[float]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "model": model_name,
        "function_name": f"{model_name}_vl_video_vqa",
    }
    segment_buffer_bytes = [("video", frames_to_bytes(segment))]
    response = send_inference_request(
        payload, "image-to-text", files=segment_buffer_bytes, v2=True
    )
    if "yes" in response.lower():
        return [1.0] * len(segment)
    return [0.0] * len(segment)


def activity_recognition(
    prompt: str,
    frames: List[np.ndarray],
    model: str = "qwen25vl",
    chunk_length_frames: int = 10,
) -> List[float]:
    """'activity_recognition' is a tool that can recognize activities in a video given a
    text prompt. It can be used to identify where specific activities or actions
    happen in a video and returns a list of 0s and 1s to indicate the activity.

    Parameters:
        prompt (str): The event you want to identify, should be phrased as a question,
            for example, "Did a goal happen?".
        frames (List[np.ndarray]): The reference frames used for the question
        model (str): The model to use for the inference. Valid values are
            'claude-35', 'gpt-4o', 'qwen2vl'.
        chunk_length_frames (int): length of each chunk in frames

    Returns:
        List[float]: A list of floats with a value of 1.0 if the activity is detected in
            the chunk_length_frames of the video.

    Example
    -------
        >>> activity_recognition('Did a goal happened?', frames)
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    """

    buffer_bytes = frames_to_bytes(frames)
    files = [("video", buffer_bytes)]

    segments = split_frames_into_segments(
        frames, segment_size=chunk_length_frames, overlap=0
    )

    prompt = (
        f"{prompt} Please respond with a 'yes' or 'no' based on the frames provided."
    )

    if model == "claude-35":

        def _apply_activity_recognition(segment: List[np.ndarray]) -> List[float]:
            return _lmm_activity_recognition(AnthropicLMM(), segment, prompt)

    elif model == "gpt-4o":

        def _apply_activity_recognition(segment: List[np.ndarray]) -> List[float]:
            return _lmm_activity_recognition(OpenAILMM(), segment, prompt)

    elif model == "qwen2vl":

        def _apply_activity_recognition(segment: List[np.ndarray]) -> List[float]:
            return _qwenvl_activity_recognition(segment, prompt, model_name="qwen2vl")

    elif model == "qwen25vl":

        def _apply_activity_recognition(segment: List[np.ndarray]) -> List[float]:
            return _qwenvl_activity_recognition(segment, prompt, model_name="qwen25vl")

    else:
        raise ValueError(f"Invalid model: {model}")

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_apply_activity_recognition, segment): segment_index
            for segment_index, segment in enumerate(segments)
        }

        return_value_tuples = []
        for future in as_completed(futures):
            segment_index = futures[future]
            return_value_tuples.append((segment_index, future.result()))
    return_values = [x[1] for x in sorted(return_value_tuples, key=lambda x: x[0])]
    return_values_flattened = cast(List[float], [e for o in return_values for e in o])

    _display_tool_trace(
        activity_recognition.__name__,
        {"prompt": prompt, "model": model},
        return_values,
        files,
    )
    return return_values_flattened


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
    _display_tool_trace(
        vit_image_classification.__name__,
        data,
        resp_data,
        image_b64,
    )
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
    _display_tool_trace(
        vit_nsfw_classification.__name__,
        data,
        resp_data,
        image_b64,
    )
    return resp_data


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
    _display_tool_trace(
        detr_segmentation.__name__,
        {},
        return_data,
        image_b64,
    )
    return return_data


def depth_anything_v2(image: np.ndarray) -> np.ndarray:
    """'depth_anything_v2' is a tool that runs depth anything v2 model to generate a
    depth image from a given RGB image. The returned depth image is monochrome and
    represents depth values as pixel intensities with pixel values ranging from 0 to 255.

    Parameters:
        image (np.ndarray): The image to used to generate depth image

    Returns:
        np.ndarray: A grayscale depth image with pixel values ranging from 0 to 255
            where high values represent closer objects and low values further.

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
    _display_tool_trace(
        depth_anything_v2.__name__,
        {},
        depth_map,
        image_b64,
    )
    return depth_map_np


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
    _display_tool_trace(
        generate_pose_image.__name__,
        {},
        pos_img,
        image_b64,
    )
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
                "label": "match",
                "score": round(answer["scores"][i], 2),
                "bbox": normalize_bbox(answer["bboxes"][i], image_size),
            }
        )
    _display_tool_trace(
        template_match.__name__,
        {"template_image": template_image_b64},
        return_data,
        image_b64,
    )
    return return_data


def flux_image_inpainting(
    prompt: str,
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """'flux_image_inpainting' performs image inpainting to fill the masked regions,
    given by mask, in the image, given image based on the text prompt and surrounding
    image context. It can be used to edit regions of an image according to the prompt
    given.

    Parameters:
        prompt (str): A detailed text description guiding what should be generated
            in the masked area. More detailed and specific prompts typically yield
            better results.
        image (np.ndarray): The source image to be inpainted. The image will serve as
            the base context for the inpainting process.
        mask (np.ndarray): A binary mask image with 0's and 1's, where 1 indicates
            areas to be inpainted and 0 indicates areas to be preserved.

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
    _display_tool_trace(
        flux_image_inpainting.__name__,
        payload,
        output_image,
        files,
    )
    return output_image


def gemini_image_generation(
    prompt: str,
    image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """'gemini_image_generation' performs either image inpainting given an image and text prompt, or image generation given a prompt.
    It can be used to edit parts of an image or the entire image according to the prompt given.

    Parameters:
        prompt (str): A detailed text description guiding what should be generated
            in the image. More detailed and specific prompts typically yield
            better results.
        image (np.ndarray, optional): The source image to be inpainted. The image will serve as
            the base context for the inpainting process.

    Returns:
        np.ndarray: The generated image(s) as a numpy array in RGB format with values
            ranging from 0 to 255.

    -------
    Example:
        >>> # Generate inpainting
        >>> result = gemini_image_generation(
        ...     prompt="a modern black leather sofa with white pillows",
        ...     image=image,
        ... )
        >>> save_image(result, "inpainted_room.png")
    """
    client = genai.Client()
    files = []
    image_file = None

    def try_generate_content(
        input_prompt: types.Content, num_retries: int = 3
    ) -> Optional[bytes]:
        """Try to generate content with multiple attempts."""
        for attempt in range(num_retries):
            try:
                resp = client.models.generate_content(
                    model="gemini-2.0-flash-exp-image-generation",
                    contents=input_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["Text", "Image"]
                    ),
                )

                if (
                    not resp.candidates
                    or not resp.candidates[0].content
                    or not resp.candidates[0].content.parts
                    or not resp.candidates[0].content.parts[0].inline_data
                    or not resp.candidates[0].content.parts[0].inline_data.data
                ):
                    _LOGGER.warning(f"Attempt {attempt + 1}: No candidates returned")
                    time.sleep(5)
                    continue
                else:
                    return (
                        resp.candidates[0].content.parts[0].inline_data.data
                        if isinstance(
                            resp.candidates[0].content.parts[0].inline_data.data, bytes
                        )
                        else None
                    )

            except genai.errors.ClientError as e:
                _LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(5)

        return None

    if image is not None:
        # Resize if needed
        max_size = (512, 512)
        if image.shape[0] > max_size[0] or image.shape[1] > max_size[1]:
            scaling_factor = min(
                max_size[0] / image.shape[0], max_size[1] / image.shape[1]
            )
            new_size = (
                int(image.shape[1] * scaling_factor),
                int(image.shape[0] * scaling_factor),
            )
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_file = numpy_to_bytes(image)
        files = [("image", image_file)]

        input_prompt = types.Content(
            parts=[
                types.Part(
                    text="I want you to edit this image given this prompt: " + prompt
                ),
                types.Part(inline_data={"mime_type": "image/png", "data": image_file}),
            ]
        )

    else:
        input_prompt = types.Content(parts=[types.Part(text=prompt)])

    # Try to generate content
    output_image_bytes = try_generate_content(input_prompt)

    # Handle fallback if all attempts failed
    if output_image_bytes is None:
        if image is not None:
            _LOGGER.warning("Returning original image after all retries failed.")
            return image
        else:
            try:
                _LOGGER.warning("All retries failed; prompting for fresh generation.")
                time.sleep(10)
                output_image_bytes = try_generate_content(
                    types.Content(parts=[types.Part(text="Generate an image.")]),
                    num_retries=1,
                )

            except Exception as e:
                raise ValueError(f"Fallback generation failed: {str(e)}")

    # Convert bytes to image
    if output_image_bytes is not None:
        output_image_temp = io.BytesIO(output_image_bytes)
        output_image_pil = Image.open(output_image_temp)
        final_image = np.array(output_image_pil)
    else:
        raise ValueError("Fallback generation failed")

    _display_tool_trace(
        gemini_image_generation.__name__,
        {
            "prompt": prompt,
            "model": "gemini-2.0-flash-exp-image-generation",
        },
        final_image,
        files,
    )

    return final_image


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

    _display_tool_trace(
        siglip_classification.__name__,
        payload,
        response,
        files,
    )
    return response


def minimum_distance(
    det1: Dict[str, Any], det2: Dict[str, Any], image_size: Tuple[int, int]
) -> float:
    """'minimum_distance' calculates the minimum distance between two detections which
    can include bounding boxes and or masks. This will return the closest distance
    between the objects, not the distance between the centers of the objects.

    Parameters:
        det1 (Dict[str, Any]): The first detection of boxes or masks.
        det2 (Dict[str, Any]): The second detection of boxes or masks.
        image_size (Tuple[int, int]): The size of the image given as (height, width).

    Returns:
        float: The closest distance between the two detections.

    Example
    -------
        >>> closest_distance(det1, det2, image_size)
        141.42
    """

    if "mask" in det1 and "mask" in det2:
        return closest_mask_distance(det1["mask"], det2["mask"])
    elif "bbox" in det1 and "bbox" in det2:
        return closest_box_distance(det1["bbox"], det2["bbox"], image_size)
    else:
        raise ValueError("Both detections must have either bbox or mask")


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


def extract_frames_and_timestamps(
    video_uri: Union[str, Path], fps: float = 5
) -> List[Dict[str, Union[np.ndarray, float]]]:
    """'extract_frames_and_timestamps' extracts frames and timestamps from a video
    which can be a file path, url or youtube link, returns a list of dictionaries
    with keys "frame" and "timestamp" where "frame" is a numpy array and "timestamp" is
    the relative time in seconds where the frame was captured. The frame is a numpy
    array.

    Parameters:
        video_uri (Union[str, Path]): The path to the video file, url or youtube link
        fps (float, optional): The frame rate per second to extract the frames. Defaults
            to 5.

    Returns:
        List[Dict[str, Union[np.ndarray, float]]]: A list of dictionaries containing the
            extracted frame as a numpy array and the timestamp in seconds.

    Example
    -------
        >>> extract_frames("path/to/video.mp4")
        [{"frame": np.ndarray, "timestamp": 0.0}, ...]
    """
    if isinstance(fps, str):
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
            ydl_opts = {
                "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(str(video_uri), download=True)
                video_file_path = ydl.prepare_filename(info)

            return reformat(extract_frames_from_video(video_file_path, fps))

    elif str(video_uri).startswith(("http", "https")):
        _, image_suffix = os.path.splitext(video_uri)
        with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix) as tmp_file:
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
    if not isinstance(image, np.ndarray) or (
        image.shape[0] == 0 and image.shape[1] == 0
    ):
        raise ValueError("The image is not a valid NumPy array with shape (H, W, C)")

    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if should_report_tool_traces():
        from IPython.display import display

        display(pil_image)

    pil_image.save(file_path)


def load_pdf(pdf_path: str) -> List[np.ndarray]:
    """'load_pdf' is a utility function that loads a PDF from the given file path string and converts each page to an image.

    Parameters:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[np.ndarray]: A list of images as NumPy arrays, one for each page of the PDF.

    Example
    -------
        >>> load_pdf("path/to/document.pdf")
    """

    # Handle URL case
    if pdf_path.startswith(("http", "https")):
        _, pdf_suffix = os.path.splitext(pdf_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=pdf_suffix) as tmp_file:
            # Download the PDF and save it to the temporary file
            with urllib.request.urlopen(pdf_path) as response:
                tmp_file.write(response.read())
            pdf_path = tmp_file.name

    # Open the PDF
    doc = pymupdf.open(pdf_path)
    images = []

    # Convert each page to an image
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Render page to an image
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))

        # Convert to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Convert to numpy array
        images.append(np.array(img))

    # Close the document
    doc.close()

    # Clean up temporary file if it was a URL
    if pdf_path.startswith(("http", "https")):
        os.unlink(pdf_path)

    return images


def save_video(
    frames: List[np.ndarray], output_video_path: Optional[str] = None, fps: float = 5
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

    output_file: IO[bytes]
    if output_video_path is None:
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    else:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_video_path, "wb")

    with output_file as file:
        video_writer(frames, fps, file=file)
    _save_video_to_result(output_file.name)
    return output_file.name


def _save_video_to_result(video_uri: str) -> None:
    """Saves a video into the result of the code execution (as an intermediate output)."""
    if not should_report_tool_traces():
        return

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
    an image. It will draw a box around the detected object with the label and score.

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

    use_tracking_label = False
    if all([":" in label for label in labels]):
        unique_labels = set([label.split(":")[1].strip() for label in labels])
        use_tracking_label = True
        colors = {
            label: COLORS[i % len(COLORS)] for i, label in enumerate(unique_labels)
        }
    else:
        colors = {label: COLORS[i % len(COLORS)] for i, label in enumerate(labels)}

    frame_out = []
    for i, frame in enumerate(medias_int):
        pil_image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")

        bboxes = bbox_int[i]
        bboxes = sorted(bboxes, key=lambda x: x["label"], reverse=True)

        # if more than 50 boxes use small boxes to indicate objects else use regular boxes
        if len(bboxes) > 50:
            pil_image = _plot_counting(pil_image, bboxes, colors, use_tracking_label)
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
                if use_tracking_label:
                    color = colors[elt["label"].split(":")[1].strip()]
                else:
                    color = colors[elt["label"]]
                label = elt["label"]
                box = elt["bbox"]
                scores = elt["score"]

                # denormalize the box if it is normalized
                box = denormalize_bbox(box, (height, width))
                draw.rectangle(box, outline=color, width=4)
                text = f"{label}: {scores:.2f}"
                text_box = draw.textbbox((box[0], box[1]), text=text, font=font)
                draw.rectangle((box[0], box[1], text_box[2], text_box[3]), fill=color)
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
    masks. It will overlay a colored mask on the detected object with the label.

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

    use_tracking_label = False
    if all([":" in label for label in labels]):
        use_tracking_label = True
        unique_labels = set([label.split(":")[1].strip() for label in labels])
        colors = {
            label: COLORS[i % len(COLORS)] for i, label in enumerate(unique_labels)
        }
    else:
        colors = {label: COLORS[i % len(COLORS)] for i, label in enumerate(labels)}

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
            if use_tracking_label:
                color = colors[elt["label"].split(":")[1].strip()]
            else:
                color = colors[elt["label"]]
            label = elt["label"]
            tracking_lbl = elt.get(secondary_label_key, None)

            # Create semi-transparent mask overlay
            np_mask = np.zeros((pil_image.size[1], pil_image.size[0], 4))
            np_mask[mask > 0, :] = color + (255 * 0.7,)
            mask_img = Image.fromarray(np_mask.astype(np.uint8))
            pil_image = Image.alpha_composite(pil_image, mask_img)

            # Draw contour border
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            border_mask = np.zeros(
                (pil_image.size[1], pil_image.size[0], 4), dtype=np.uint8
            )
            cv2.drawContours(border_mask, contours, -1, color + (255,), 8)
            border_img = Image.fromarray(border_mask)
            pil_image = Image.alpha_composite(pil_image, border_img)

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
                    draw.rectangle((x, y, text_box[2], text_box[3]), fill=color)
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
    use_tracking_label: bool = False,
) -> Image.Image:
    width, height = image.size
    fontsize = max(12, int(min(width, height) / 40))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        str(resources.files("vision_agent.fonts").joinpath("default_font_ch_en.ttf")),
        fontsize,
    )
    for i, elt in enumerate(bboxes, 1):
        if use_tracking_label:
            label = elt["label"].split(":")[0]
            color = colors[elt["label"].split(":")[1].strip()]
        else:
            label = f"{i}"
            color = colors[elt["label"]]
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
        draw.rectangle((text_x0, text_y0, text_x1, text_y1), fill=color)

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
    glee_object_detection,
    glee_sam2_instance_segmentation,
    glee_sam2_video_tracking,
    countgd_object_detection,
    countgd_sam2_instance_segmentation,
    countgd_sam2_video_tracking,
    florence2_ocr,
    florence2_object_detection,
    florence2_sam2_instance_segmentation,
    florence2_sam2_video_tracking,
    claude35_text_extraction,
    agentic_document_extraction,
    document_qa,
    ocr,
    qwen25_vl_images_vqa,
    qwen25_vl_video_vqa,
    activity_recognition,
    depth_anything_v2,
    generate_pose_image,
    vit_nsfw_classification,
    flux_image_inpainting,
    siglip_classification,
    minimum_distance,
]

UTIL_TOOLS = [
    extract_frames_and_timestamps,
    save_json,
    load_image,
    save_image,
    save_video,
    overlay_bounding_boxes,
    overlay_segmentation_masks,
]

TOOLS = FUNCTION_TOOLS + UTIL_TOOLS


def get_tools() -> List[Callable]:
    return TOOLS  # type: ignore


def get_tools_info() -> Dict[str, str]:
    return _get_tools_info(FUNCTION_TOOLS)  # type: ignore


def get_tools_df() -> pd.DataFrame:
    return _get_tools_df(TOOLS)  # type: ignore


def get_tools_descriptions() -> str:
    return _get_tool_descriptions(TOOLS)  # type: ignore


def get_tools_docstring() -> str:
    return _get_tool_documentation(TOOLS)  # type: ignore


def get_utilties_docstring() -> str:
    return _get_tool_documentation(UTIL_TOOLS)  # type: ignore
