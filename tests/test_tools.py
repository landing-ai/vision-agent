import numpy as np
import skimage as ski

from vision_agent.tools import (
    clip,
    closest_mask_distance,
    grounding_dino,
    grounding_sam,
    blip_image_caption,
    git_vqa_v2,
    ocr,
    loca_visual_prompt_counting,
    loca_zero_shot_counting,
    vit_nsfw_classification,
    vit_image_classification,
    owl_v2,
)


def test_grounding_dino():
    img = ski.data.coins()
    result = grounding_dino(
        prompt="coin",
        image=img,
    )
    assert len(result) == 24
    assert [res["label"] for res in result] == ["coin"] * 24


def test_grounding_dino_tiny():
    img = ski.data.coins()
    result = grounding_dino(prompt="coin", image=img, model_size="tiny")
    assert len(result) == 24
    assert [res["label"] for res in result] == ["coin"] * 24


def test_owl():
    img = ski.data.coins()
    result = owl_v2(prompt="coin", image=img, box_threshold=0.15)
    assert len(result) == 25
    assert [res["label"] for res in result] == ["coin"] * 25


def test_grounding_sam():
    img = ski.data.coins()
    result = grounding_sam(
        prompt="coin",
        image=img,
    )
    assert len(result) == 24
    assert [res["label"] for res in result] == ["coin"] * 24
    assert len([res["mask"] for res in result]) == 24


def test_clip():
    img = ski.data.coins()
    result = clip(
        classes=["coins", "notes"],
        image=img,
    )
    assert result["scores"] == [0.9999, 0.0001]


def test_vit_classification():
    img = ski.data.coins()
    result = vit_image_classification(
        image=img,
    )
    assert "typewriter keyboard" in result["labels"]


def test_nsfw_classification():
    img = ski.data.coins()
    result = vit_nsfw_classification(
        image=img,
    )
    assert result["labels"] == "normal"


def test_image_caption() -> None:
    img = ski.data.rocket()
    result = blip_image_caption(
        image=img,
    )
    assert result.strip() == "a rocket on a stand"


def test_loca_zero_shot_counting() -> None:
    img = ski.data.coins()
    result = loca_zero_shot_counting(
        image=img,
    )
    assert result["count"] == 21


def test_loca_visual_prompt_counting() -> None:
    img = ski.data.coins()
    result = loca_visual_prompt_counting(
        visual_prompt={"bbox": [85, 106, 122, 145]},
        image=img,
    )
    assert result["count"] == 25


def test_git_vqa_v2() -> None:
    img = ski.data.rocket()
    result = git_vqa_v2(
        prompt="Is the scene captured during day or night ?",
        image=img,
    )
    assert result.strip() == "night"


def test_ocr() -> None:
    img = ski.data.page()
    result = ocr(
        image=img,
    )
    assert any("Region-based segmentation" in res["label"] for res in result)


def test_mask_distance():
    # Create two binary masks
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[:10, :10] = 1  # Top left
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[-10:, -10:] = 1  # Bottom right

    # Calculate the distance between the masks
    distance = closest_mask_distance(mask1, mask2)
    print(f"Distance between the masks: {distance}")

    # Check the result
    assert np.isclose(
        distance,
        np.sqrt(2) * 81,
        atol=1e-2,
    ), f"Expected {np.sqrt(2) * 81}, got {distance}"
