import io
from collections import defaultdict
from typing import Any, Callable, Dict, List, Union

import numpy as np
from PIL import Image

from vision_agent.lmm import AnthropicLMM
from vision_agent.utils.image_utils import encode_image_bytes


def claude35_vqa(prompt: str, medias: List[np.ndarray]) -> str:
    lmm = AnthropicLMM()
    all_media_b64 = []
    for media in medias:
        buffer = io.BytesIO()
        Image.fromarray(media).save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = "data:image/png;base64," + encode_image_bytes(image_bytes)
        all_media_b64.append(image_b64)
    return lmm.generate(prompt all_media_b64)


# def run_tool_with_optimal_threshold(
#     tool: Callable[[Union[np.ndarray, List[np.ndarray]]], List[Dict[str, Any]]],
#     args: Dict[str, Any],
#     expected_objects: List[Dict[str, Any]],
# ) -> Dict[str, Any]:
#     if tool.__name__ not in {"owl_v2_image", "owl_v2_video", "countgd_counting", "countgd_example_based_counting"}:
#         detections = tool(*args)
#         return {"detections": detections, "threshold": None}
#     detections = tool(*args, box_threshold=0.05)

#     # Group detections by label
#     detections_per_label = defaultdict(list)
#     for det in detections:
#         label = det['label']
#         detections_per_label[label].append(det)

#     if isinstance(detections[0], dict):
#         detections = [detections]

#     thresholds = {}
#     for label, count in expected_objects.items():
#         if label not in detections_per_label:
#             thresholds[label] = 0
#             continue

#         # Sort detections by confidence
#         detections_per_label[label].sort(key=lambda x: x["score"], reverse=True)

#         # Find optimal threshold
#         for i in range(1, len(detections_per_label[label])):
#             tp = 0
#             fp = 0
#             for det in detections_per_label[label]:
#                 if det["score"] >= detections_per_label[label][i]["score"]:
#                     if det['label'] == label:
#                         tp += 1
#                     else:
#                         fp += 1
#             if tp == count:
#                 thresholds[label] = detections_per_label[label][i]["score"]
#                 break

#     for detection in detections:




def pick_tool_for_task(prompt: str, images: List[np.ndarray]) -> str:
    pass


def create_new_tool(prompt: str, images: List[np.ndarray]) -> Callable:
    pass
