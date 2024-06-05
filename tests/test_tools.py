import numpy as np
import skimage as ski

from vision_agent.tools import (
    clip,
    closest_mask_distance,
    grounding_dino,
    grounding_sam,
    image_caption,
    image_question_answering,
    ocr,
    visual_prompt_counting,
    zero_shot_counting,
)


def test_grounding_dino():
    img = ski.data.coins()
    result = grounding_dino(
        prompt="coin",
        image=img,
    )
    assert len(result) == 24
    assert [res["label"] for res in result] == ["coin"] * 24


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


def test_image_caption() -> None:
    img = ski.data.rocket()
    result = image_caption(
        image=img,
    )
    assert result.strip() == "a rocket on a stand"


def test_zero_shot_counting() -> None:
    img = ski.data.coins()
    result = zero_shot_counting(
        image=img,
    )
    assert result["count"] == 21


def test_visual_prompt_counting() -> None:
    img = ski.data.coins()
    result = visual_prompt_counting(
        visual_prompt={"bbox": [85, 106, 122, 145]},
        image=img,
    )
    assert result["count"] == 25


def test_image_question_answering() -> None:
    img = ski.data.rocket()
    result = image_question_answering(
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
