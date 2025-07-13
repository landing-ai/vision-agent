import base64
import copy
import io
from typing import Dict, List, Optional, Tuple, Union, cast

import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImageType

from vision_agent.utils.image_utils import (
    denormalize_bbox,
    normalize_bbox,
    numpy_to_bytes,
    rle_decode_array,
)
from vision_agent.utils.tools import send_inference_request


def maybe_denormalize_bbox(
    bbox: List[Union[int, float]], image_size: Tuple[int, ...]
) -> List[float]:
    if all([0 <= c <= 1 for c in bbox]):
        return denormalize_bbox(bbox, image_size)
    return bbox


def maybe_normalize_bbox(
    bbox: List[Union[int, float]], image_size: Tuple[int, ...]
) -> List[float]:
    if any([1 <= c for c in bbox]):
        return normalize_bbox(bbox, image_size)
    return bbox


def instance_segmentation(
    prompt: str, image: np.ndarray, threshold: float = 0.23, nms_threshold: float = 0.5
) -> List[Dict[str, Union[str, float, List[float], np.ndarray]]]:
    image_bytes = numpy_to_bytes(image)
    files = [("image", image_bytes)]
    data = {"prompts": [prompt], "threshold": threshold, "nms_threshold": nms_threshold}
    results = send_inference_request(
        data,
        "glee",
        files=files,
        v2=True,
    )
    results = results[0]
    results_formatted = [
        {
            "label": elt["label"],
            "score": elt["score"],
            "bbox": normalize_bbox(elt["bounding_box"], image.shape[:2]),
            "mask": np.array(rle_decode_array(elt["mask"])),
        }
        for elt in results
    ]
    return results_formatted


def ocr(image: np.ndarray) -> List[Dict[str, Union[str, float, List[float]]]]:
    image_bytes = numpy_to_bytes(image)
    files = [("image", image_bytes)]
    results = send_inference_request(
        {},
        "paddle-ocr",
        files=files,
        v2=True,
    )
    results_formatted = [
        {
            "label": elt["label"],
            "score": elt["score"],
            "bbox": normalize_bbox(elt["bbox"], image.shape[:2]),
        }
        for elt in results
    ]
    return results_formatted


def depth_estimation(image: np.ndarray) -> np.ndarray:
    shape = image.shape[:2]
    image_bytes = numpy_to_bytes(image)
    files = [("image", image_bytes)]
    results = send_inference_request(
        {},
        "depth-pro",
        files=files,
        v2=True,
    )
    depth = np.frombuffer(base64.b64decode(results["depth"]), dtype=np.float32).reshape(
        shape
    )
    return depth


def visualize_bounding_boxes(
    image: np.ndarray, bounding_boxes: List[Dict[str, Union[str, float, List[float]]]]
) -> np.ndarray:
    image = image.copy()
    image_size = image.shape[:2]
    bounding_boxes = copy.deepcopy(bounding_boxes)

    for bbox in bounding_boxes:
        bbox["bbox"] = maybe_denormalize_bbox(
            cast(List[float], bbox["bbox"]), image_size
        )
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox["bbox"]  # type: ignore
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return image


def visualize_segmentation_masks(
    image: np.ndarray,
    segmentation_masks: List[Dict[str, Union[str, float, np.ndarray]]],
) -> np.ndarray:
    alpha = 0.5
    overlay = image.copy()
    color_mask = np.zeros_like(image)
    color_mask[:, :] = (0, 100, 255)
    for elt in segmentation_masks:
        mask = cast(np.ndarray, elt["mask"])
        overlay[mask == 1] = (1 - alpha) * overlay[mask == 1] + alpha * color_mask[
            mask == 1
        ]

        # draw outline on the mask so it doesn't just think the color of the object changed
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def get_crops(
    image: np.ndarray, bounding_boxes: List[Dict[str, Union[str, float, List[float]]]]
) -> List[np.ndarray]:
    image = image.copy()
    bounding_boxes = copy.deepcopy(bounding_boxes)

    for bbox in bounding_boxes:
        bbox["bbox"] = maybe_denormalize_bbox(
            cast(List[float], bbox["bbox"]), image.shape[:2]
        )
    crops = []
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox["bbox"]  # type: ignore
        crops.append(image[int(y1) : int(y2), int(x1) : int(x2)])
    return crops


def rotate_90(image: np.ndarray, k: int = 1) -> np.ndarray:
    return np.rot90(image, k=k, axes=(0, 1))


def iou(
    pred1: Union[List[float], np.ndarray], pred2: Union[List[float], np.ndarray]
) -> float:
    if isinstance(pred1, list) and isinstance(pred2, list):
        x1, y1, x2, y2 = pred1
        x1_, y1_, x2_, y2_ = pred2
        intersection = max(0, min(x2, x2_) - max(x1, x1_)) * max(
            0, min(y2, y2_) - max(y1, y1_)
        )
        union = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersection
        return intersection / union
    elif isinstance(pred1, np.ndarray) and isinstance(pred2, np.ndarray):
        pred1 = np.clip(pred1, 0, 1)
        pred2 = np.clip(pred2, 0, 1)
        intersection = np.sum(pred1 * pred2)
        union = np.sum(pred1) + np.sum(pred2) - intersection
        return intersection / union
    raise ValueError("Unsupported input types for IoU calculation.")


def display_image(
    image: Union[np.ndarray, PILImageType, matplotlib.figure.Figure, str],
) -> None:
    display_img: Optional[PILImageType] = None
    if isinstance(image, np.ndarray):
        display_img = Image.fromarray(image)
    elif isinstance(image, matplotlib.figure.Figure):
        # Render the figure to a BytesIO buffer
        buf = io.BytesIO()
        image.savefig(buf, format="png")
        buf.seek(0)
        # Load the buffer as a PIL Image
        display_img = Image.open(buf)
        plt.close(image)  # type: ignore
    elif isinstance(image, PILImageType):
        display_img = image  # Already a PIL Image
    elif isinstance(image, str):
        display_img = Image.open(image)

    if display_img is not None:
        plt.imshow(display_img)  # type: ignore
        plt.axis("off")  # type: ignore
        plt.show()
    else:
        # Handle cases where image type is not supported or conversion failed
        print("Unsupported image type or conversion failed.")
