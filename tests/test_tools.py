import skimage as ski
from PIL import Image

from vision_agent.tools.tools import CLIP, GroundingDINO, GroundingSAM, ImageCaption


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
    img = Image.fromarray(ski.data.coins())
    result = CLIP()(
        prompt="coins",
        image=img,
    )
    assert result["scores"] == [1.0]


def test_image_caption() -> None:
    img = Image.fromarray(ski.data.coins())
    result = ImageCaption()(image=img)
    assert result["text"]
