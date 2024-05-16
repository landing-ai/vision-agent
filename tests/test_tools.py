import skimage as ski
from PIL import Image

from vision_agent.tools.tools import CLIP, GroundingDINO, GroundingSAM, ImageCaption
from vision_agent.tools.tools_v2 import (
    clip,
    zero_shot_counting,
    visual_prompt_counting,
    image_question_answering,
    ocr,
)


def test_grounding_dino():
    img = Image.fromarray(ski.data.coins())
    result = GroundingDINO()(
        prompt="coin",
        image=img,
    )
    assert result["labels"] == ["coin"] * 24
    assert len(result["bboxes"]) == 24
    assert len(result["scores"]) == 24


def test_grounding_sam():
    img = Image.fromarray(ski.data.coins())
    result = GroundingSAM()(
        prompt="coin",
        image=img,
    )
    assert result["labels"] == ["coin"] * 24
    assert len(result["bboxes"]) == 24
    assert len(result["scores"]) == 24
    assert len(result["masks"]) == 24


def test_clip():
    img = ski.data.coins()
    result = clip(
        classes=["coins", "notes"],
        image=img,
    )
    assert result["scores"] == [0.99, 0.01]


def test_image_caption() -> None:
    img = Image.fromarray(ski.data.coins())
    result = ImageCaption()(image=img)
    assert result["text"]


def test_zero_shot_counting() -> None:
    img = Image.fromarray(ski.data.coins())
    result = zero_shot_counting(
        image=img,
    )
    assert result["count"] == 24


def test_visual_prompt_counting() -> None:
    img = Image.fromarray(ski.data.checkerboard())
    result = visual_prompt_counting(
        visual_prompt={"bbox": [0.125, 0, 0.25, 0.125]},
        image=img,
    )
    assert result["count"] == 32


def test_image_question_answering() -> None:
    img = Image.fromarray(ski.data.rocket())
    result = image_question_answering(
        prompt="Is the scene captured during day or night ?",
        image=img,
    )
    assert result == "night"


def test_ocr() -> None:
    img = Image.fromarray(ski.data.page())
    result = ocr(
        image=img,
    )
    assert result[0]["label"] == "Region-based segmentation"
