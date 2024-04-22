import logging
import tempfile
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import requests
from PIL import Image
from PIL.Image import Image as ImageType

from vision_agent.image_utils import convert_to_b64, get_image_size
from vision_agent.tools.video import extract_frames_from_video
from vision_agent.type_defs import LandingaiAPIKey

_LOGGER = logging.getLogger(__name__)
_LND_API_KEY = LandingaiAPIKey().api_key
_LND_API_URL = "https://api.dev.landing.ai/v1/agent"


def normalize_bbox(
    bbox: List[Union[int, float]], image_size: Tuple[int, ...]
) -> List[float]:
    r"""Normalize the bounding box coordinates to be between 0 and 1."""
    x1, y1, x2, y2 = bbox
    x1 = round(x1 / image_size[1], 2)
    y1 = round(y1 / image_size[0], 2)
    x2 = round(x2 / image_size[1], 2)
    y2 = round(y2 / image_size[0], 2)
    return [x1, y1, x2, y2]


def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    r"""Decode a run-length encoded mask. Returns numpy array, 1 - mask, 0 - background.

    Parameters:
        mask_rle: Run-length as string formated (start length)
        shape: The (height, width) of array to return
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


class Tool(ABC):
    name: str
    description: str
    usage: Dict


class NoOp(Tool):
    name = "noop_"
    description = "'noop_' is a no-op tool that does nothing if you do not want answer the question directly and not use a tool."
    usage = {
        "required_parameters": [],
        "examples": [
            {
                "scenario": "If you do not want to use a tool.",
                "parameters": {},
            }
        ],
    }

    def __call__(self) -> None:
        return None


class CLIP(Tool):
    r"""CLIP is a tool that can classify or tag any image given a set if input classes
    or tags.

    Example
    -------
        >>> import vision_agent as va
        >>> clip = va.tools.CLIP()
        >>> clip("red line, yellow dot", "ct_scan1.jpg"))
        [{"labels": ["red line", "yellow dot"], "scores": [0.98, 0.02]}]
    """

    name = "clip_"
    description = "'clip_' is a tool that can classify any image given a set of input names or tags. It returns a list of the input names along with their probability scores."
    usage = {
        "required_parameters": [
            {"name": "prompt", "type": "str"},
            {"name": "image", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Can you classify this image as a cat? Image name: cat.jpg",
                "parameters": {"prompt": "cat", "image": "cat.jpg"},
            },
            {
                "scenario": "Can you tag this photograph with cat or dog? Image name: cat_dog.jpg",
                "parameters": {"prompt": "cat, dog", "image": "cat_dog.jpg"},
            },
            {
                "scenario": "Can you build me a classifier that classifies red shirts, green shirts and other? Image name: shirts.jpg",
                "parameters": {
                    "prompt": "red shirt, green shirt, other",
                    "image": "shirts.jpg",
                },
            },
        ],
    }

    # TODO: Add support for input multiple images, which aligns with the output type.
    def __call__(self, prompt: str, image: Union[str, ImageType]) -> Dict:
        """Invoke the CLIP model.

        Parameters:
            prompt: a string includes a list of classes or tags to classify the image.
            image: the input image to classify.

        Returns:
            A list of dictionaries containing the labels and scores. Each dictionary contains the classification result for an image. E.g. [{"labels": ["red line", "yellow dot"], "scores": [0.98, 0.02]}]
        """
        image_b64 = convert_to_b64(image)
        data = {
            "prompt": prompt,
            "image": image_b64,
            "tool": "closed_set_image_classification",
        }
        resp_data = _send_inference_request(data, "tools")
        resp_data["scores"] = [round(prob, 4) for prob in resp_data["scores"]]
        return resp_data


class ImageCaption(Tool):
    r"""ImageCaption is a tool that can caption an image based on its contents
    or tags.

    Example
    -------
        >>> import vision_agent as va
        >>> caption = va.tools.ImageCaption()
        >>> caption("image1.jpg")
        {'text': ['a box of orange and white socks']}
    """

    name = "image_caption_"
    description = "'image_caption_' is a tool that can caption an image based on its contents or tags. It returns a text describing the image"
    usage = {
        "required_parameters": [
            {"name": "image", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Can you describe this image ? Image name: cat.jpg",
                "parameters": {"image": "cat.jpg"},
            },
            {
                "scenario": "Can you caption this image with their main contents ? Image name: cat_dog.jpg",
                "parameters": {"image": "cat_dog.jpg"},
            },
            {
                "scenario": "Can you build me a image captioning tool ? Image name: shirts.jpg",
                "parameters": {
                    "image": "shirts.jpg",
                },
            },
        ],
    }

    # TODO: Add support for input multiple images, which aligns with the output type.
    def __call__(self, image: Union[str, ImageType]) -> Dict:
        """Invoke the Image captioning model.

        Parameters:
            image: the input image to caption.

        Returns:
            A list of dictionaries containing the labels and scores. Each dictionary contains the classification result for an image. E.g. [{"labels": ["red line", "yellow dot"], "scores": [0.98, 0.02]}]
        """
        image_b64 = convert_to_b64(image)
        data = {
            "image": image_b64,
            "tool": "image_captioning",
        }
        return _send_inference_request(data, "tools")


class GroundingDINO(Tool):
    r"""Grounding DINO is a tool that can detect arbitrary objects with inputs such as
    category names or referring expressions.

    Example
    -------
        >>> import vision_agent as va
        >>> t = va.tools.GroundingDINO()
        >>> t("red line. yellow dot", "ct_scan1.jpg")
        [{'labels': ['red line', 'yellow dot'],
        'bboxes': [[0.38, 0.15, 0.59, 0.7], [0.48, 0.25, 0.69, 0.71]],
        'scores': [0.98, 0.02]}]
    """

    name = "grounding_dino_"
    description = "'grounding_dino_' is a tool that can detect arbitrary objects with inputs such as category names or referring expressions. It returns a list of bounding boxes, label names and associated probability scores."
    usage = {
        "required_parameters": [
            {"name": "prompt", "type": "str"},
            {"name": "image", "type": "str"},
        ],
        "optional_parameters": [
            {"name": "box_threshold", "type": "float"},
            {"name": "iou_threshold", "type": "float"},
        ],
        "examples": [
            {
                "scenario": "Can you build me a car detector?",
                "parameters": {"prompt": "car", "image": ""},
            },
            {
                "scenario": "Can you detect the person on the left and right? Image name: person.jpg",
                "parameters": {
                    "prompt": "left person. right person",
                    "image": "person.jpg",
                },
            },
            {
                "scenario": "Detect the red shirts and green shirst. Image name: shirts.jpg",
                "parameters": {
                    "prompt": "red shirt. green shirt",
                    "image": "shirts.jpg",
                    "box_threshold": 0.20,
                    "iou_threshold": 0.75,
                },
            },
        ],
    }

    # TODO: Add support for input multiple images, which aligns with the output type.
    def __call__(
        self,
        prompt: str,
        image: Union[str, Path, ImageType],
        box_threshold: float = 0.20,
        iou_threshold: float = 0.75,
    ) -> Dict:
        """Invoke the Grounding DINO model.

        Parameters:
            prompt: one or multiple class names to detect. The classes should be separated by a period if there are multiple classes. E.g. "big dog . small cat"
            image: the input image to run against.
            box_threshold: the threshold to filter out the bounding boxes with low scores.
            iou_threshold: the threshold for intersection over union used in nms algorithm. It will suppress the boxes which have iou greater than this threshold.

        Returns:
            A dictionary containing the labels, scores, and bboxes, which is the detection result for the input image.
        """
        image_size = get_image_size(image)
        image_b64 = convert_to_b64(image)
        request_data = {
            "prompt": prompt,
            "image": image_b64,
            "tool": "visual_grounding",
            "kwargs": {"box_threshold": box_threshold, "iou_threshold": iou_threshold},
        }
        data: Dict[str, Any] = _send_inference_request(request_data, "tools")
        if "bboxes" in data:
            data["bboxes"] = [normalize_bbox(box, image_size) for box in data["bboxes"]]
        if "scores" in data:
            data["scores"] = [round(score, 2) for score in data["scores"]]
        if "labels" in data:
            data["labels"] = list(data["labels"])
        data["size"] = (image_size[1], image_size[0])
        return data


class GroundingSAM(Tool):
    r"""Grounding SAM is a tool that can detect and segment arbitrary objects with
    inputs such as category names or referring expressions.

    Example
    -------
        >>> import vision_agent as va
        >>> t = va.tools.GroundingSAM()
        >>> t("red line, yellow dot", "ct_scan1.jpg"])
        [{'labels': ['yellow dot', 'red line'],
        'bboxes': [[0.38, 0.15, 0.59, 0.7], [0.48, 0.25, 0.69, 0.71]],
        'masks': [array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)},
        array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [1, 1, 1, ..., 1, 1, 1],
           [1, 1, 1, ..., 1, 1, 1]], dtype=uint8)]}]
    """

    name = "grounding_sam_"
    description = "'grounding_sam_' is a tool that can detect arbitrary objects with inputs such as category names or referring expressions. It returns a list of bounding boxes, label names and masks file names and associated probability scores."
    usage = {
        "required_parameters": [
            {"name": "prompt", "type": "str"},
            {"name": "image", "type": "str"},
        ],
        "optional_parameters": [
            {"name": "box_threshold", "type": "float"},
            {"name": "iou_threshold", "type": "float"},
        ],
        "examples": [
            {
                "scenario": "Can you build me a car segmentor?",
                "parameters": {"prompt": "car", "image": ""},
            },
            {
                "scenario": "Can you segment the person on the left and right? Image name: person.jpg",
                "parameters": {
                    "prompt": "left person. right person",
                    "image": "person.jpg",
                },
            },
            {
                "scenario": "Can you build me a tool that segments red shirts and green shirts? Image name: shirts.jpg",
                "parameters": {
                    "prompt": "red shirt, green shirt",
                    "image": "shirts.jpg",
                    "box_threshold": 0.20,
                    "iou_threshold": 0.75,
                },
            },
        ],
    }

    # TODO: Add support for input multiple images, which aligns with the output type.
    def __call__(
        self,
        prompt: str,
        image: Union[str, ImageType],
        box_threshold: float = 0.2,
        iou_threshold: float = 0.75,
    ) -> Dict:
        """Invoke the Grounding SAM model.

        Parameters:
            prompt: a list of classes to segment.
            image: the input image to segment.
            box_threshold: the threshold to filter out the bounding boxes with low scores.
            iou_threshold: the threshold for intersection over union used in nms algorithm. It will suppress the boxes which have iou greater than this threshold.

        Returns:
            A dictionary containing the labels, scores, bboxes and masks for the input image.
        """
        image_size = get_image_size(image)
        image_b64 = convert_to_b64(image)
        request_data = {
            "prompt": prompt,
            "image": image_b64,
            "tool": "visual_grounding_segment",
            "kwargs": {"box_threshold": box_threshold, "iou_threshold": iou_threshold},
        }
        data: Dict[str, Any] = _send_inference_request(request_data, "tools")
        if "bboxes" in data:
            data["bboxes"] = [normalize_bbox(box, image_size) for box in data["bboxes"]]
        if "masks" in data:
            data["masks"] = [
                rle_decode(mask_rle=mask, shape=data["mask_shape"])
                for mask in data["masks"]
            ]
        data.pop("mask_shape", None)
        return data


class DINOv(Tool):
    r"""DINOv is a tool that can detect and segment similar objects with the given input masks.

    Example
    -------
        >>> import vision_agent as va
        >>> t = va.tools.DINOv()
        >>> t(prompt=[{"mask":"balloon_mask.jpg", "image": "balloon.jpg"}], image="balloon.jpg"])
        [{'scores': [0.512, 0.212],
        'masks': [array([[0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)},
        array([[0, 0, 0, ..., 0, 0, 0],
           ...,
           [1, 1, 1, ..., 1, 1, 1]], dtype=uint8)]}]
    """

    name = "dinov_"
    description = "'dinov_' is a tool that can detect and segment similar objects given a reference segmentation mask."
    usage = {
        "required_parameters": [
            {"name": "prompt", "type": "List[Dict[str, str]]"},
            {"name": "image", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Can you find all the balloons in this image that is similar to the provided masked area? Image name: input.jpg Reference image: balloon.jpg Reference mask: balloon_mask.jpg",
                "parameters": {
                    "prompt": [
                        {"mask": "balloon_mask.jpg", "image": "balloon.jpg"},
                    ],
                    "image": "input.jpg",
                },
            },
            {
                "scenario": "Detect all the objects in this image that are similar to the provided mask. Image name: original.jpg Reference image: mask.png Reference mask: background.png",
                "parameters": {
                    "prompt": [
                        {"mask": "mask.png", "image": "background.png"},
                    ],
                    "image": "original.jpg",
                },
            },
        ],
    }

    def __call__(
        self, prompt: List[Dict[str, str]], image: Union[str, ImageType]
    ) -> Dict:
        """Invoke the DINOv model.

        Parameters:
            prompt: a list of visual prompts in the form of {'mask': 'MASK_FILE_PATH', 'image': 'IMAGE_FILE_PATH'}.
            image: the input image to segment.

        Returns:
            A dictionary of the below keys: 'scores', 'masks' and 'mask_shape', which stores a list of detected segmentation masks and its scores.
        """
        image_b64 = convert_to_b64(image)
        for p in prompt:
            p["mask"] = convert_to_b64(p["mask"])
            p["image"] = convert_to_b64(p["image"])
        request_data = {
            "prompt": prompt,
            "image": image_b64,
            "tool": "dinov",
        }
        data: Dict[str, Any] = _send_inference_request(request_data, "dinov")
        if "bboxes" in data:
            data["bboxes"] = [
                normalize_bbox(box, data["mask_shape"]) for box in data["bboxes"]
            ]
        if "masks" in data:
            data["masks"] = [
                rle_decode(mask_rle=mask, shape=data["mask_shape"])
                for mask in data["masks"]
            ]
        data["labels"] = ["visual prompt" for _ in range(len(data["masks"]))]
        return data


class AgentDINOv(DINOv):
    def __call__(
        self,
        prompt: List[Dict[str, str]],
        image: Union[str, ImageType],
    ) -> Dict:
        rets = super().__call__(prompt, image)
        mask_files = []
        for mask in rets["masks"]:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                file_name = Path(tmp.name).with_suffix(".mask.png")
                Image.fromarray(mask * 255).save(file_name)
                mask_files.append(str(file_name))
        rets["masks"] = mask_files
        return rets


class AgentGroundingSAM(GroundingSAM):
    r"""AgentGroundingSAM is the same as GroundingSAM but it saves the masks as files
    returns the file name. This makes it easier for agents to use.
    """

    def __call__(
        self,
        prompt: str,
        image: Union[str, ImageType],
        box_threshold: float = 0.2,
        iou_threshold: float = 0.75,
    ) -> Dict:
        rets = super().__call__(prompt, image, box_threshold, iou_threshold)
        mask_files = []
        for mask in rets["masks"]:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                file_name = Path(tmp.name).with_suffix(".mask.png")
                Image.fromarray(mask * 255).save(file_name)
                mask_files.append(str(file_name))
        rets["masks"] = mask_files
        return rets


class Crop(Tool):
    r"""Crop crops an image given a bounding box and returns a file name of the cropped image."""

    name = "crop_"
    description = "'crop_' crops an image given a bounding box and returns a file name of the cropped image. It returns a file with the cropped image."
    usage = {
        "required_parameters": [
            {"name": "bbox", "type": "List[float]"},
            {"name": "image", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Can you crop the image to the bounding box [0.1, 0.1, 0.9, 0.9]? Image name: image.jpg",
                "parameters": {"bbox": [0.1, 0.1, 0.9, 0.9], "image": "image.jpg"},
            },
            {
                "scenario": "Cut out the image to the bounding box [0.2, 0.2, 0.8, 0.8]. Image name: car.jpg",
                "parameters": {"bbox": [0.2, 0.2, 0.8, 0.8], "image": "car.jpg"},
            },
        ],
    }

    def __call__(self, bbox: List[float], image: Union[str, Path]) -> Dict:
        pil_image = Image.open(image)
        width, height = pil_image.size
        bbox = [
            int(bbox[0] * width),
            int(bbox[1] * height),
            int(bbox[2] * width),
            int(bbox[3] * height),
        ]
        cropped_image = pil_image.crop(bbox)  # type: ignore
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cropped_image.save(tmp.name)

        return {"image": tmp.name}


class BboxArea(Tool):
    r"""BboxArea returns the area of the bounding box in pixels normalized to 2 decimal places."""

    name = "bbox_area_"
    description = "'bbox_area_' returns the area of the bounding box in pixels normalized to 2 decimal places."
    usage = {
        "required_parameters": [{"name": "bboxes", "type": "List[int]"}],
        "examples": [
            {
                "scenario": "If you want to calculate the area of the bounding box [0.2, 0.21, 0.34, 0.42]",
                "parameters": {"bboxes": [0.2, 0.21, 0.34, 0.42]},
            }
        ],
    }

    def __call__(self, bboxes: List[Dict]) -> List[Dict]:
        areas = []
        for elt in bboxes:
            height, width = elt["size"]
            for label, bbox in zip(elt["labels"], elt["bboxes"]):
                x1, y1, x2, y2 = bbox
                areas.append(
                    {
                        "area": round((x2 - x1) * (y2 - y1) * width * height, 2),
                        "label": label,
                    }
                )
        return areas


class SegArea(Tool):
    r"""SegArea returns the area of the segmentation mask in pixels normalized to 2 decimal places."""

    name = "seg_area_"
    description = "'seg_area_' returns the area of the segmentation mask in pixels normalized to 2 decimal places."
    usage = {
        "required_parameters": [{"name": "masks", "type": "str"}],
        "examples": [
            {
                "scenario": "If you want to calculate the area of the segmentation mask, pass the masks file name.",
                "parameters": {"masks": "mask_file.jpg"},
            },
        ],
    }

    def __call__(self, masks: Union[str, Path]) -> float:
        pil_mask = Image.open(str(masks))
        np_mask = np.array(pil_mask)
        np_mask = np.clip(np_mask, 0, 1)
        return cast(float, round(np.sum(np_mask), 2))


class BboxIoU(Tool):
    name = "bbox_iou_"
    description = "'bbox_iou_' returns the intersection over union of two bounding boxes. This is a good tool for determining if two objects are overlapping."
    usage = {
        "required_parameters": [
            {"name": "bbox1", "type": "List[int]"},
            {"name": "bbox2", "type": "List[int]"},
        ],
        "examples": [
            {
                "scenario": "If you want to calculate the intersection over union of the bounding boxes [0.2, 0.21, 0.34, 0.42] and [0.3, 0.31, 0.44, 0.52]",
                "parameters": {
                    "bbox1": [0.2, 0.21, 0.34, 0.42],
                    "bbox2": [0.3, 0.31, 0.44, 0.52],
                },
            }
        ],
    }

    def __call__(self, bbox1: List[int], bbox2: List[int]) -> float:
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        xA = max(x1, x3)
        yA = max(y1, y3)
        xB = min(x2, x4)
        yB = min(y2, y4)
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxa_area = (x2 - x1) * (y2 - y1)
        boxb_area = (x4 - x3) * (y4 - y3)
        iou = inter_area / float(boxa_area + boxb_area - inter_area)
        return round(iou, 2)


class SegIoU(Tool):
    name = "seg_iou_"
    description = "'seg_iou_' returns the intersection over union of two segmentation masks given their segmentation mask files."
    usage = {
        "required_parameters": [
            {"name": "mask1", "type": "str"},
            {"name": "mask2", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "If you want to calculate the intersection over union of the segmentation masks for mask_file1.jpg and mask_file2.jpg",
                "parameters": {"mask1": "mask_file1.png", "mask2": "mask_file2.png"},
            }
        ],
    }

    def __call__(self, mask1: Union[str, Path], mask2: Union[str, Path]) -> float:
        pil_mask1 = Image.open(str(mask1))
        pil_mask2 = Image.open(str(mask2))
        np_mask1 = np.clip(np.array(pil_mask1), 0, 1)
        np_mask2 = np.clip(np.array(pil_mask2), 0, 1)
        intersection = np.logical_and(np_mask1, np_mask2)
        union = np.logical_or(np_mask1, np_mask2)
        iou = np.sum(intersection) / np.sum(union)
        return cast(float, round(iou, 2))


class BboxContains(Tool):
    name = "bbox_contains_"
    description = "Given two bounding boxes, a target bounding box and a region bounding box, 'bbox_contains_' returns the intersection of the two bounding boxes over the target bounding box, reflects the percentage area of the target bounding box overlaps with the region bounding box. This is a good tool for determining if the region object contains the target object."
    usage = {
        "required_parameters": [
            {"name": "target", "type": "List[int]"},
            {"name": "target_class", "type": "str"},
            {"name": "region", "type": "List[int]"},
            {"name": "region_class", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Determine if the dog on the couch, bounding box of the dog: [0.2, 0.21, 0.34, 0.42], bounding box of the couch: [0.3, 0.31, 0.44, 0.52]",
                "parameters": {
                    "target": [0.2, 0.21, 0.34, 0.42],
                    "target_class": "dog",
                    "region": [0.3, 0.31, 0.44, 0.52],
                    "region_class": "couch",
                },
            },
            {
                "scenario": "Check if the kid is in the pool? bounding box of the kid: [0.2, 0.21, 0.34, 0.42], bounding box of the pool: [0.3, 0.31, 0.44, 0.52]",
                "parameters": {
                    "target": [0.2, 0.21, 0.34, 0.42],
                    "target_class": "kid",
                    "region": [0.3, 0.31, 0.44, 0.52],
                    "region_class": "pool",
                },
            },
        ],
    }

    def __call__(
        self, target: List[int], target_class: str, region: List[int], region_class: str
    ) -> Dict[str, Union[str, float]]:
        x1, y1, x2, y2 = target
        x3, y3, x4, y4 = region
        xA = max(x1, x3)
        yA = max(y1, y3)
        xB = min(x2, x4)
        yB = min(y2, y4)
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxa_area = (x2 - x1) * (y2 - y1)
        iou = inter_area / float(boxa_area)
        area = round(iou, 2)
        return {
            "target_class": target_class,
            "region_class": region_class,
            "intersection": area,
        }


class BoxDistance(Tool):
    name = "box_distance_"
    description = (
        "'box_distance_' returns the minimum distance between two bounding boxes."
    )
    usage = {
        "required_parameters": [
            {"name": "bbox1", "type": "List[int]"},
            {"name": "bbox2", "type": "List[int]"},
        ],
        "examples": [
            {
                "scenario": "If you want to calculate the distance between the bounding boxes [0.2, 0.21, 0.34, 0.42] and [0.3, 0.31, 0.44, 0.52]",
                "parameters": {
                    "bbox1": [0.2, 0.21, 0.34, 0.42],
                    "bbox2": [0.3, 0.31, 0.44, 0.52],
                },
            }
        ],
    }

    def __call__(self, bbox1: List[int], bbox2: List[int]) -> float:
        x11, y11, x12, y12 = bbox1
        x21, y21, x22, y22 = bbox2

        horizontal_dist = np.max([0, x21 - x12, x11 - x22])
        vertical_dist = np.max([0, y21 - y12, y11 - y22])

        return cast(float, round(np.sqrt(horizontal_dist**2 + vertical_dist**2), 2))


class ExtractFrames(Tool):
    r"""Extract frames from a video."""

    name = "extract_frames_"
    description = "'extract_frames_' extracts frames from a video, returns a list of tuples (frame, timestamp), where timestamp is the relative time in seconds where the frame was captured. The frame is a local image file path."
    usage = {
        "required_parameters": [{"name": "video_uri", "type": "str"}],
        "examples": [
            {
                "scenario": "Can you extract the frames from this video? Video: www.foobar.com/video?name=test.mp4",
                "parameters": {"video_uri": "www.foobar.com/video?name=test.mp4"},
            },
            {
                "scenario": "Can you extract the images from this video file? Video path: tests/data/test.mp4",
                "parameters": {"video_uri": "tests/data/test.mp4"},
            },
        ],
    }

    def __call__(self, video_uri: str) -> List[Tuple[str, float]]:
        """Extract frames from a video.


        Parameters:
            video_uri: the path to the video file or a url points to the video data

        Returns:
            a list of tuples containing the extracted frame and the timestamp in seconds. E.g. [(path_to_frame1, 0.0), (path_to_frame2, 0.5), ...]. The timestamp is the time in seconds from the start of the video. E.g. 12.125 means 12.125 seconds from the start of the video. The frames are sorted by the timestamp in ascending order.
        """
        frames = extract_frames_from_video(video_uri)
        result = []
        _LOGGER.info(
            f"Extracted {len(frames)} frames from video {video_uri}. Temporarily saving them as images to disk for downstream tasks."
        )
        for frame, ts in frames:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                file_name = Path(tmp.name).with_suffix(".frame.png")
                Image.fromarray(frame).save(file_name)
            result.append((str(file_name), ts))
        return result


class Calculator(Tool):
    r"""Calculator is a tool that can perform basic arithmetic operations."""

    name = "calculator_"
    description = (
        "'calculator_' is a tool that can perform basic arithmetic operations."
    )
    usage = {
        "required_parameters": [{"name": "equation", "type": "str"}],
        "examples": [
            {
                "scenario": "If you want to calculate (2 * 3) + 4",
                "parameters": {"equation": "2 + 4"},
            },
            {
                "scenario": "If you want to calculate (4 + 2.5) / 2.1",
                "parameters": {"equation": "(4 + 2.5) / 2.1"},
            },
        ],
    }

    def __call__(self, equation: str) -> float:
        return cast(float, round(eval(equation), 2))


TOOLS = {
    i: {"name": c.name, "description": c.description, "usage": c.usage, "class": c}
    for i, c in enumerate(
        [
            NoOp,
            CLIP,
            ImageCaption,
            GroundingDINO,
            AgentGroundingSAM,
            AgentDINOv,
            ExtractFrames,
            Crop,
            BboxArea,
            SegArea,
            BboxIoU,
            SegIoU,
            BboxContains,
            BoxDistance,
            Calculator,
        ]
    )
    if (hasattr(c, "name") and hasattr(c, "description") and hasattr(c, "usage"))
}


def _send_inference_request(
    payload: Dict[str, Any], endpoint_name: str
) -> Dict[str, Any]:
    res = requests.post(
        f"{_LND_API_URL}/model/{endpoint_name}",
        headers={
            "Content-Type": "application/json",
            "apikey": _LND_API_KEY,
        },
        json=payload,
    )
    if res.status_code != 200:
        _LOGGER.error(f"Request failed: {res.text}")
        raise ValueError(f"Request failed: {res.text}")
    return res.json()["data"]  # type: ignore
