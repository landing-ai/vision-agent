import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vision_agent.tools.tool_utils import (
    add_bboxes_from_masks,
    nms,
    send_task_inference_request,
)
from vision_agent.utils.image_utils import denormalize_bbox, rle_decode_array
from vision_agent.utils.video import frames_to_bytes

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ODModels(Enum):
    COUNTGD = "countgd"
    OWLV2 = "owlv2"
    FLORENCE2 = "florence2"


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
    _LOGGER.debug(
        "Splitting frames into segments of %d frames with %d frame overlap.",
        segment_size,
        overlap,
    )
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
            _LOGGER.debug(
                "Segment %d: frames %d to %d (including overlap).",
                segment_count + 1,
                start - overlap,
                end - 1,
            )
        else:
            segment = frames[start:end]
            _LOGGER.debug(
                "Segment %d: frames %d to %d.",
                segment_count + 1,
                start,
                end - 1,
            )
        segments.append(segment)
        start += segment_size
        segment_count += 1
    _LOGGER.debug("Total segments created: %d.", segment_count)
    return segments


def process_segment(
    segment_frames: List[np.ndarray],
    od_model: ODModels,
    prompt: str,
    fine_tune_id: Optional[str],
    chunk_length: Optional[int],
    image_size: Tuple[int, int],
    segment_index: int,
    object_detection_tool,
) -> Optional[List[Dict[str, Any]]]:
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

    Returns:
        Optional[List[Dict[str, Any]]]: Detections for the segment.
    """
    _LOGGER.debug(
        "Processing segment %d with %d frames.",
        segment_index + 1,
        len(segment_frames),
    )
    segment_results: List[Optional[List[Dict[str, Any]]]] = [None] * len(segment_frames)

    if chunk_length is None:
        step = 1
        _LOGGER.debug("Chunk length is None. Processing every frame.")
    elif chunk_length <= 0:
        _LOGGER.debug("Invalid chunk_length: %d. Raising ValueError.", chunk_length)
        raise ValueError("chunk_length must be a positive integer or None.")
    else:
        step = chunk_length
        _LOGGER.debug("Processing frames with step size: %d.", step)

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
        "chunk_length": chunk_length,
    }
    metadata = {"function_name": function_name}
    _LOGGER.debug(
        "Segment %d: Sending inference request with payload size %d bytes.",
        segment_index + 1,
        len(buffer_bytes),
    )

    segment_detections = send_task_inference_request(
        payload,
        "sam2",
        files=files,
        metadata=metadata,
    )
    _LOGGER.debug("Segment %d: Inference request completed.", segment_index + 1)

    return segment_detections


def transform_detections(
    input_list: List[Optional[List[Dict[str, Any]]]],
    image_size: Tuple[int, int],
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
    _LOGGER.debug("Transforming detections for segment %d.", segment_index + 1)
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
    _LOGGER.debug("Calculating IoU between two masks.")
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))

    iou = intersection / union if union != 0 else 0
    _LOGGER.debug("Calculated IoU: %.4f", iou)
    return iou


def _match_by_iou(
    first_param: List[Dict],
    second_param: List[Dict],
    iou_threshold: float = 0.8,
) -> Tuple[List[Dict], Dict[int, int]]:

    _LOGGER.debug(
        "Matching items between two lists with IoU threshold %.2f.",
        iou_threshold,
    )

    max_id = max((item["id"] for item in first_param), default=0)
    _LOGGER.debug("Max ID in first_param: %d", max_id)

    for item in first_param:
        item["decoded_mask"] = rle_decode_array(item["mask"])
    for item in second_param:
        item["decoded_mask"] = rle_decode_array(item["mask"])

    matched_new_item_indices = set()
    id_mapping = {}

    for new_index, new_item in enumerate(second_param):
        _LOGGER.warning("Processing new item with ID %d.", new_item["id"])
        matched_id = None

        for existing_item in first_param:
            iou = _calculate_mask_iou(
                existing_item["decoded_mask"], new_item["decoded_mask"]
            )
            if iou > iou_threshold:
                matched_id = existing_item["id"]
                matched_new_item_indices.add(new_index)
                id_mapping[new_item["id"]] = matched_id
                _LOGGER.warning(
                    "Matched new item ID %d with existing item ID %d (IoU: %.4f).",
                    new_item["id"],
                    matched_id,
                    iou,
                )
                break

        if matched_id:
            new_item["id"] = matched_id
        else:
            max_id += 1
            id_mapping[new_item["id"]] = max_id
            new_item["id"] = max_id
            _LOGGER.warning("Assigned new ID %d to unmatched item.", max_id)

    unmatched_items = [
        item for i, item in enumerate(second_param) if i not in matched_new_item_indices
    ]
    combined_list = first_param + unmatched_items

    _LOGGER.debug("Combined list size after matching: %d", len(combined_list))
    return combined_list, id_mapping


def _update_ids(detections: List[Dict], id_mapping: Dict[int, int]):
    _LOGGER.warning("Updating IDs in detections using the ID mapping.")
    for detection in detections:
        if detection["id"] in id_mapping:
            detection["id"] = id_mapping[detection["id"]]
        else:
            max_new_id = max(id_mapping.values(), default=0)
            detection["id"] = max_new_id + 1
            id_mapping[detection["id"]] = detection["id"]
            _LOGGER.warning("Assigned new sequential ID %d.", detection["id"])


def _convert_to_2d(detections_per_segment: List[Any]) -> List[Any]:
    _LOGGER.debug("Converting detections per segment into a 2D list.")
    result = []
    for i, segment in enumerate(detections_per_segment):
        if i == 0:
            result.extend(segment)
        else:
            result.extend(segment[1:])
    _LOGGER.debug("Converted list size: %d", len(result))
    return result


def merge_segments(
    detections_per_segment: List[Any], frames: List[np.ndarray]
) -> List[Any]:
    """
    Merges detections from all segments into a unified result.

    Args:
        detections_per_segment (List[Any]): List of detections per segment.
        frames (List[np.ndarray]): List of all frames.

    Returns:
        List[Any]: Merged detections.
    """
    _LOGGER.debug(
        "Starting merge_segments with %d segments.", len(detections_per_segment)
    )

    for idx in range(len(detections_per_segment) - 1):
        _LOGGER.debug("Processing segment pair: %d and %d.", idx, idx + 1)

        combined_detection, id_mapping = _match_by_iou(
            detections_per_segment[idx][-1], detections_per_segment[idx + 1][0]
        )
        _update_ids(detections_per_segment[idx + 1], id_mapping)

    merged_result = _convert_to_2d(detections_per_segment)
    _LOGGER.debug(
        "Finished merging segments. Total items in result: %d", len(merged_result)
    )

    return merged_result


def post_process(
    merged_detections: List[Any], frames: List[np.ndarray], image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Performs post-processing on merged detections, including NMS and preparing display data.

    Args:
        merged_detections (List[Any]): Merged detections from all segments.
        frames (List[np.ndarray]): List of all frames.
        image_size (Tuple[int, int]): Size of the images.

    Returns:
        Dict[str, Any]: Post-processed data including return_data and display_data.
    """
    return_data = []
    for frame_idx, frame in enumerate(merged_detections):
        return_frame_data = []
        for detection in frame:
            mask = rle_decode_array(detection["mask"])
            label = f"{detection['id']}: {detection['label']}"
            return_frame_data.append(
                {"label": label, "mask": mask, "score": 1.0, "rle": detection["mask"]}
            )
        return_data.append(return_frame_data)

    return_data = add_bboxes_from_masks(return_data)
    return_data = nms(return_data, iou_threshold=0.95)
    return {
        "return_data": return_data,
        "display_data": [],  # Assuming display_data is handled elsewhere
    }
