import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore

from vision_agent.utils.image_utils import denormalize_bbox, rle_decode_array
from vision_agent.utils.tools import add_bboxes_from_masks, send_task_inference_request
from vision_agent.utils.video import frames_to_bytes


class ODModels(str, Enum):
    COUNTGD = "countgd"
    FLORENCE2 = "florence2"
    OWLV2 = "owlv2"
    AGENTIC = "agentic"
    GLEE = "glee"
    CUSTOM = "custom"


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
    deployment_id: Optional[str],
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
        deployment_id (Optional[str]): The model deployment ID.
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
            deployment_id=deployment_id,
            frame_number=frame_number,
            od_model=od_model,
            prompt=prompt,
            segment_frames=segment_frames,
            segment_index=segment_index,
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

    segment_detections = join_scores(transformed_detections, segment_detections)
    return segment_detections


def join_scores(
    transformed_detections: List[Optional[Dict[str, Any]]],
    segment_detections: List[List[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    # The scores should really be returned from the SAM2 endpoint so we don't have to
    # try and match them.
    for detection in transformed_detections:
        if detection is not None:
            for i in range(len(detection["scores"])):
                id_to_score = {}
                if len(segment_detections) > 0:
                    # This assumes none of the initial boxes are filtered out by SAM2
                    # so we have a 1:1 mapping between the initial boxes and the SAM2 boxes
                    for j, segment_detection in enumerate(segment_detections[0]):
                        id_to_score[segment_detection["id"]] = detection["scores"][j]

                # after we've created the id_to_score, assign the scores. Some of the
                # boxes could have been removed in subsequent frames, hence the mapping
                # is needed
                for t in range(len(segment_detections)):
                    for segment_detection in segment_detections[t]:
                        if segment_detection["id"] in id_to_score:
                            segment_detection["score"] = id_to_score[
                                segment_detection["id"]
                            ]
                        else:
                            # if we can't find the score, set it to 1.0 so it doesn't
                            # get filtered out
                            segment_detection["score"] = 1.0

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
            scores = [detection["score"] for detection in frame]

            output_list.append(
                {
                    "labels": labels,
                    "bboxes": bboxes,
                    "scores": scores,
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
    max_id: int,
    iou_threshold: float = 0.05,
) -> Tuple[Dict[int, int], int]:
    max_first_id = max((item["id"] for item in first_param), default=0)
    max_second_id = max((item["id"] for item in second_param), default=0)

    cost_matrix = np.ones((max_first_id + 1, max_second_id + 1))
    for first_item in first_param:
        for second_item in second_param:
            iou = _calculate_mask_iou(
                first_item["decoded_mask"], second_item["decoded_mask"]
            )
            cost_matrix[first_item["id"], second_item["id"]] = 1 - iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    id_mapping = {second_id: first_id for first_id, second_id in zip(row_ind, col_ind)}
    first_id_to_label = {item["id"]: item["label"] for item in first_param}

    cleaned_mapping = {}
    for elt in second_param:
        second_id = elt["id"]
        # if the id is not in the mapping, give it a new id
        if second_id not in id_mapping:
            max_id += 1
            cleaned_mapping[second_id] = max_id
        else:
            first_id = id_mapping[second_id]
            iou = 1 - cost_matrix[first_id, second_id]
            # only map if the iou is above the threshold and the labels match
            if iou > iou_threshold and first_id_to_label[first_id] == elt["label"]:
                cleaned_mapping[second_id] = first_id
            else:
                max_id += 1
                cleaned_mapping[second_id] = max_id

    return cleaned_mapping, max_id


def merge_segments(detections_per_segment: List[Any], overlap: int = 1) -> List[Any]:
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

    merged_result = detections_per_segment[0]
    max_id = max((item["id"] for item in merged_result[-1]), default=0)
    for segment_idx in range(len(detections_per_segment) - 1):
        id_mapping, max_id = _match_by_iou(
            detections_per_segment[segment_idx][-1],
            detections_per_segment[segment_idx + 1][0],
            max_id,
        )
        for frame in detections_per_segment[segment_idx + 1][overlap:]:
            for detection in frame:
                detection["id"] = id_mapping[detection["id"]]
        merged_result.extend(detections_per_segment[segment_idx + 1][overlap:])

    return merged_result  # type: ignore


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
    label_remapping = {}
    for _, frame in enumerate(merged_detections):
        return_frame_data = []
        for detection in frame:
            label = detection["label"]
            id = detection["id"]

            # Remap label IDs so for each label the IDs restart at 1. This makes it
            # easier to count the number of instances per label.
            if label not in label_remapping:
                label_remapping[label] = {"max": 1, "remap": {id: 1}}
            elif label in label_remapping and id not in label_remapping[label]["remap"]:  # type: ignore
                max_id = label_remapping[label]["max"]
                max_id += 1  # type: ignore
                label_remapping[label]["remap"][id] = max_id  # type: ignore
                label_remapping[label]["max"] = max_id

            new_id = label_remapping[label]["remap"][id]  # type: ignore

            label = f"{new_id}: {detection['label']}"
            return_frame_data.append(
                {
                    "label": label,
                    "mask": detection["decoded_mask"],
                    "rle": detection["mask"],
                    "score": detection["score"],
                }
            )
            del detection["decoded_mask"]
        return_data.append(return_frame_data)

    return_data = add_bboxes_from_masks(return_data)

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
