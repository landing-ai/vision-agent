import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from vision_agent.tools.tool_utils import (
    add_bboxes_from_masks,
    nms,
    send_task_inference_request,
)
from vision_agent.utils.image_utils import denormalize_bbox, rle_decode_array
from vision_agent.utils.video import frames_to_bytes


class ODModels(str, Enum):
    COUNTGD = "countgd"
    FLORENCE2 = "florence2"
    OWLV2 = "owlv2"


def split_frames_into_segments(
    frames: List[np.ndarray], segment_size: int = 50, overlap: int = 1
) -> List[List[np.ndarray]]:
    """
    Splits the list of frames into segments with a specified size and overlap.

    Args:
        frames (List[np.ndarray]): List of video frames.
        segment_size (int, optional): Number of frames per segment. Defaults to 50.
        overlap (int, optional): Number of overlapping frames between segments. Defaults to 1.

    Returns:
        List[List[np.ndarray]]: List of frame segments.
    """
    segments = []
    start = 0
    segment_count = 0
    while start < len(frames):
        end = start + segment_size
        if end > len(frames):
            end = len(frames)
        if start != 0:
            # Include the last frame of the previous segment
            segment = frames[start - overlap : end]
        else:
            segment = frames[start:end]
        segments.append(segment)
        start += segment_size
        segment_count += 1
    return segments


def process_segment(
    segment_frames: List[np.ndarray],
    od_model: ODModels,
    prompt: str,
    fine_tune_id: Optional[str],
    chunk_length: Optional[int],
    image_size: Tuple[int, ...],
    segment_index: int,
    object_detection_tool: Callable,
) -> Any:
    """
    Processes a segment of frames with the specified object detection model.

    Args:
        segment_frames (List[np.ndarray]): Frames in the segment.
        od_model (ODModels): Object detection model to use.
        prompt (str): Prompt for the model.
        fine_tune_id (Optional[str]): Fine-tune model ID.
        chunk_length (Optional[int]): Chunk length for processing.
        image_size (Tuple[int, int]): Size of the images.
        segment_index (int): Index of the segment.
        object_detection_tool (Callable): Object detection tool to use.

    Returns:
       Any: Detections for the segment.
    """
    segment_results: List[Optional[List[Dict[str, Any]]]] = [None] * len(segment_frames)

    if chunk_length is None:
        step = 1
    elif chunk_length <= 0:
        raise ValueError("chunk_length must be a positive integer or None.")
    else:
        step = chunk_length

    function_name = ""

    for idx in range(0, len(segment_frames), step):
        frame_number = idx
        segment_results[idx], function_name = object_detection_tool(
            od_model, prompt, segment_index, frame_number, fine_tune_id, segment_frames
        )

    transformed_detections = transform_detections(
        segment_results, image_size, segment_index
    )

    buffer_bytes = frames_to_bytes(segment_frames)
    files = [("video", buffer_bytes)]
    payload = {
        "bboxes": json.dumps(transformed_detections),
        "chunk_length_frames": chunk_length,
    }
    metadata = {"function_name": function_name}

    segment_detections = send_task_inference_request(
        payload,
        "sam2",
        files=files,
        metadata=metadata,
    )

    return segment_detections


def transform_detections(
    input_list: List[Optional[List[Dict[str, Any]]]],
    image_size: Tuple[int, ...],
    segment_index: int,
) -> List[Optional[Dict[str, Any]]]:
    """
    Transforms raw detections into a standardized format.

    Args:
        input_list (List[Optional[List[Dict[str, Any]]]]): Raw detections.
        image_size (Tuple[int, int]): Size of the images.
        segment_index (int): Index of the segment.

    Returns:
        List[Optional[Dict[str, Any]]]: Transformed detections.
    """
    output_list: List[Optional[Dict[str, Any]]] = []
    for frame_idx, frame in enumerate(input_list):
        if frame is not None:
            labels = [detection["label"] for detection in frame]
            bboxes = [
                denormalize_bbox(detection["bbox"], image_size) for detection in frame
            ]

            output_list.append(
                {
                    "labels": labels,
                    "bboxes": bboxes,
                }
            )
        else:
            output_list.append(None)
    return output_list


def _calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))

    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union

    return iou


def _match_by_iou(
    first_param: List[Dict],
    second_param: List[Dict],
    iou_threshold: float = 0.8,
) -> Tuple[List[Dict], Dict[int, int]]:
    max_id = max((item["id"] for item in first_param), default=0)

    matched_new_item_indices = set()
    id_mapping = {}

    for new_index, new_item in enumerate(second_param):
        matched_id = None

        for existing_item in first_param:
            iou = _calculate_mask_iou(
                existing_item["decoded_mask"], new_item["decoded_mask"]
            )
            if iou > iou_threshold:
                matched_id = existing_item["id"]
                matched_new_item_indices.add(new_index)
                id_mapping[new_item["id"]] = matched_id
                break

        if matched_id:
            new_item["id"] = matched_id
        else:
            max_id += 1
            id_mapping[new_item["id"]] = max_id
            new_item["id"] = max_id

    unmatched_items = [
        item for i, item in enumerate(second_param) if i not in matched_new_item_indices
    ]
    combined_list = first_param + unmatched_items

    return combined_list, id_mapping


def _update_ids(detections: List[Dict], id_mapping: Dict[int, int]) -> None:
    for inner_list in detections:
        for detection in inner_list:
            if detection["id"] in id_mapping:
                detection["id"] = id_mapping[detection["id"]]
            else:
                max_new_id = max(id_mapping.values(), default=0)
                detection["id"] = max_new_id + 1
                id_mapping[detection["id"]] = detection["id"]


def _convert_to_2d(detections_per_segment: List[Any]) -> List[Any]:
    result = []
    for i, segment in enumerate(detections_per_segment):
        if i == 0:
            result.extend(segment)
        else:
            result.extend(segment[1:])
    return result


def merge_segments(detections_per_segment: List[Any]) -> List[Any]:
    """
    Merges detections from all segments into a unified result.

    Args:
        detections_per_segment (List[Any]): List of detections per segment.

    Returns:
        List[Any]: Merged detections.
    """
    for segment in detections_per_segment:
        for detection in segment:
            for item in detection:
                item["decoded_mask"] = rle_decode_array(item["mask"])

    for segment_idx in range(len(detections_per_segment) - 1):
        combined_detection, id_mapping = _match_by_iou(
            detections_per_segment[segment_idx][-1],
            detections_per_segment[segment_idx + 1][0],
        )
        _update_ids(detections_per_segment[segment_idx + 1], id_mapping)

    merged_result = _convert_to_2d(detections_per_segment)

    return merged_result


def post_process(
    merged_detections: List[Any],
    image_size: Tuple[int, ...],
) -> Dict[str, Any]:
    """
    Performs post-processing on merged detections, including NMS and preparing display data.

    Args:
        merged_detections (List[Any]): Merged detections from all segments.
        image_size (Tuple[int, int]): Size of the images.

    Returns:
        Dict[str, Any]: Post-processed data including return_data and display_data.
    """
    return_data = []
    for frame_idx, frame in enumerate(merged_detections):
        return_frame_data = []
        for detection in frame:
            label = f"{detection['id']}: {detection['label']}"
            return_frame_data.append(
                {
                    "label": label,
                    "mask": detection["decoded_mask"],
                    "rle": detection["mask"],
                    "score": 1.0,
                }
            )
            del detection["decoded_mask"]
        return_data.append(return_frame_data)

    return_data = add_bboxes_from_masks(return_data)
    return_data = nms(return_data, iou_threshold=0.95)

    # We save the RLE for display purposes, re-calculting RLE can get very expensive.
    # Deleted here because we are returning the numpy masks instead
    display_data = []
    for frame in return_data:
        display_frame_data = []
        for obj in frame:
            display_frame_data.append(
                {
                    "label": obj["label"],
                    "bbox": denormalize_bbox(obj["bbox"], image_size),
                    "mask": obj["rle"],
                    "score": obj["score"],
                }
            )
            del obj["rle"]
        display_data.append(display_frame_data)

    return {"return_data": return_data, "display_data": display_data}
