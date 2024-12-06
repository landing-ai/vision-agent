import numpy as np
import skimage as ski
from PIL import Image

from vision_agent.tools import (
    closest_mask_distance,
    countgd_object_detection,
    countgd_sam2_object_detection,
    countgd_example_based_counting,
    depth_anything_v2,
    detr_segmentation,
    florence2_ocr,
    florence2_phrase_grounding,
    florence2_phrase_grounding_video,
    florence2_sam2_image,
    florence2_sam2_video_tracking,
    flux_image_inpainting,
    generate_pose_image,
    ocr,
    owl_v2_image,
    owl_v2_video,
    qwen2_vl_images_vqa,
    qwen2_vl_video_vqa,
    siglip_classification,
    template_match,
    video_temporal_localization,
    vit_image_classification,
    vit_nsfw_classification,
)

FINE_TUNE_ID = "65ebba4a-88b7-419f-9046-0750e30250da"


def test_owl_v2_image():
    img = ski.data.coins()
    result = owl_v2_image(
        prompt="coin",
        image=img,
    )
    assert 24 <= len(result) <= 26
    assert [res["label"] for res in result] == ["coin"] * len(result)
    assert all([all([0 <= x <= 1 for x in obj["bbox"]]) for obj in result])


def test_owl_v2_image_empty():
    result = owl_v2_image(
        prompt="coin",
        image=np.zeros((0, 0, 3)).astype(np.uint8),
    )
    assert result == []


def test_owl_v2_fine_tune_id():
    img = ski.data.coins()
    result = owl_v2_image(
        prompt="coin",
        image=img,
        fine_tune_id=FINE_TUNE_ID,
    )
    # this calls a fine-tuned florence2 model which is going to be worse at this task
    assert 13 <= len(result) <= 26
    assert [res["label"] for res in result] == ["coin"] * len(result)
    assert all([all([0 <= x <= 1 for x in obj["bbox"]]) for obj in result])


def test_owl_v2_video():
    frames = [
        np.array(Image.fromarray(ski.data.coins()).convert("RGB")) for _ in range(10)
    ]
    result = owl_v2_video(
        prompt="coin",
        frames=frames,
    )

    assert len(result) == 10
    assert 24 <= len([res["label"] for res in result[0]]) <= 26
    assert all([all([0 <= x <= 1 for x in obj["bbox"]]) for obj in result[0]])


def test_owl_v2_video_fine_tune_id():
    frames = [
        np.array(Image.fromarray(ski.data.coins()).convert("RGB")) for _ in range(10)
    ]
    # this calls a fine-tuned florence2 model which is going to be worse at this task
    result = owl_v2_video(
        prompt="coin",
        frames=frames,
        fine_tune_id=FINE_TUNE_ID,
    )

    assert len(result) == 10
    assert 12 <= len([res["label"] for res in result[0]]) <= 26
    assert all([all([0 <= x <= 1 for x in obj["bbox"]]) for obj in result[0]])


def test_florence2_phrase_grounding():
    img = ski.data.coins()
    result = florence2_phrase_grounding(
        image=img,
        prompt="coin",
    )

    assert 18 <= len(result) <= 24
    assert [res["label"] for res in result] == ["coin"] * len(result)
    assert all([all([0 <= x <= 1 for x in obj["bbox"]]) for obj in result])


def test_florence2_phrase_grounding_empty():
    result = florence2_phrase_grounding(
        image=np.zeros((0, 0, 3)).astype(np.uint8),
        prompt="coin",
    )
    assert result == []


def test_florence2_phrase_grounding_fine_tune_id():
    img = ski.data.coins()
    result = florence2_phrase_grounding(
        prompt="coin",
        image=img,
        fine_tune_id=FINE_TUNE_ID,
    )
    # this calls a fine-tuned florence2 model which is going to be worse at this task
    assert 13 <= len(result) <= 26
    assert [res["label"] for res in result] == ["coin"] * len(result)
    assert all([all([0 <= x <= 1 for x in obj["bbox"]]) for obj in result])


def test_florence2_phrase_grounding_video():
    frames = [
        np.array(Image.fromarray(ski.data.coins()).convert("RGB")) for _ in range(10)
    ]
    result = florence2_phrase_grounding_video(
        prompt="coin",
        frames=frames,
    )
    assert len(result) == 10
    assert 2 <= len([res["label"] for res in result[0]]) <= 26
    assert all([all([0 <= x <= 1 for x in obj["bbox"]]) for obj in result[0]])


def test_florence2_phrase_grounding_video_fine_tune_id():
    frames = [
        np.array(Image.fromarray(ski.data.coins()).convert("RGB")) for _ in range(10)
    ]
    # this calls a fine-tuned florence2 model which is going to be worse at this task
    result = florence2_phrase_grounding_video(
        prompt="coin",
        frames=frames,
        fine_tune_id=FINE_TUNE_ID,
    )
    assert len(result) == 10
    assert 12 <= len([res["label"] for res in result[0]]) <= 26
    assert all([all([0 <= x <= 1 for x in obj["bbox"]]) for obj in result[0]])


def test_template_match():
    img = ski.data.coins()
    result = template_match(
        image=img,
        template_image=img[32:76, 20:68],
    )
    assert len(result) == 2


def test_florence2_sam2_image():
    img = ski.data.coins()
    result = florence2_sam2_image(
        prompt="coin",
        image=img,
    )
    assert len(result) == 24
    assert [res["label"] for res in result] == ["coin"] * 24
    assert len([res["mask"] for res in result]) == 24


def test_florence2_sam2_image_fine_tune_id():
    img = ski.data.coins()
    result = florence2_sam2_image(
        prompt="coin",
        image=img,
        fine_tune_id=FINE_TUNE_ID,
    )
    # this calls a fine-tuned florence2 model which is going to be worse at this task
    assert 13 <= len(result) <= 26
    assert [res["label"] for res in result] == ["coin"] * len(result)
    assert len([res["mask"] for res in result]) == len(result)


def test_florence2_sam2_image_empty():
    result = florence2_sam2_image(
        prompt="coin",
        image=np.zeros((0, 0, 3)).astype(np.uint8),
    )
    assert result == []


def test_florence2_sam2_video():
    frames = [
        np.array(Image.fromarray(ski.data.coins()).convert("RGB")) for _ in range(10)
    ]
    result = florence2_sam2_video_tracking(
        prompt="coin",
        frames=frames,
    )
    assert len(result) == 10
    assert len([res["label"] for res in result[0]]) == 24
    assert len([res["mask"] for res in result[0]]) == 24


def test_florence2_sam2_video_fine_tune_id():
    frames = [
        np.array(Image.fromarray(ski.data.coins()).convert("RGB")) for _ in range(10)
    ]
    # this calls a fine-tuned florence2 model which is going to be worse at this task
    result = florence2_sam2_video_tracking(
        prompt="coin",
        frames=frames,
        fine_tune_id=FINE_TUNE_ID,
    )
    assert len(result) == 10
    assert 15 <= len([res["label"] for res in result[0]]) <= 24
    assert 15 <= len([res["mask"] for res in result[0]]) <= 24


def test_detr_segmentation():
    img = ski.data.coins()
    result = detr_segmentation(
        image=img,
    )
    assert len(result) == 1
    assert [res["label"] for res in result] == ["pizza"]
    assert len([res["mask"] for res in result]) == 1


def test_detr_segmentation_empty():
    result = detr_segmentation(
        image=np.zeros((0, 0, 3)).astype(np.uint8),
    )
    assert result == []


def test_vit_classification():
    img = ski.data.coins()
    result = vit_image_classification(
        image=img,
    )
    assert "typewriter keyboard" in result["labels"]


def test_vit_classification_empty():
    result = vit_image_classification(
        image=np.zeros((0, 0, 3)).astype(np.uint8),
    )
    assert result["labels"] == []
    assert result["scores"] == []


def test_nsfw_classification():
    img = ski.data.coins()
    result = vit_nsfw_classification(
        image=img,
    )
    assert result["label"] == "normal"


def test_qwen2_vl_images_vqa():
    img = ski.data.page()
    result = qwen2_vl_images_vqa(
        prompt="What is the document about?",
        images=[img],
    )
    assert len(result) > 0


def test_qwen2_vl_video_vqa():
    frames = [
        np.array(Image.fromarray(ski.data.cat()).convert("RGB")) for _ in range(10)
    ]
    result = qwen2_vl_video_vqa(
        prompt="What animal is in this video?",
        frames=frames,
    )
    assert "cat" in result.strip()


def test_video_temporal_localization():
    frames = [
        np.array(Image.fromarray(ski.data.cat()).convert("RGB")) for _ in range(10)
    ]
    result = video_temporal_localization(
        prompt="Is it there a cat in this video?",
        frames=frames,
        model="qwen2vl",
    )
    assert len(result) == 5


def test_ocr():
    img = ski.data.page()
    result = ocr(
        image=img,
    )
    assert any("Region-based segmentation" in res["label"] for res in result)


def test_ocr_empty():
    result = ocr(
        image=np.zeros((0, 0, 3)).astype(np.uint8),
    )
    assert result == []


def test_florence2_ocr():
    img = ski.data.page()
    result = florence2_ocr(
        image=img,
    )
    assert any("Region-based segmentation" in res["label"] for res in result)


def test_florence2_ocr_empty():
    result = florence2_ocr(
        image=np.zeros((0, 0, 3)).astype(np.uint8),
    )
    assert result == []


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


def test_generate_depth():
    img = ski.data.coins()
    result = depth_anything_v2(
        image=img,
    )

    assert result.shape == img.shape


def test_generate_pose():
    img = ski.data.coins()
    result = generate_pose_image(
        image=img,
    )
    import cv2

    cv2.imwrite("imag.png", result)
    assert result.shape == img.shape + (3,)


def test_countgd_sam2_object_detection():
    img = ski.data.coins()
    result = countgd_sam2_object_detection(image=img, prompt="coin")
    assert len(result) == 24
    assert "mask" in result[0]
    assert [res["label"] for res in result] == ["coin"] * 24


def test_countgd_object_detection():
    img = ski.data.coins()
    result = countgd_object_detection(image=img, prompt="coin")
    assert len(result) == 24
    assert [res["label"] for res in result] == ["coin"] * 24


def test_countgd_object_detection_empty():
    result = countgd_object_detection(
        prompt="coin",
        image=np.zeros((0, 0, 3)).astype(np.uint8),
    )
    assert result == []


def test_countgd_example_based_counting():
    img = ski.data.coins()
    result = countgd_example_based_counting(
        visual_prompts=[[85, 106, 122, 145]],
        image=img,
    )
    assert len(result) == 24
    assert [res["label"] for res in result] == ["object"] * 24


def test_countgd_example_based_counting_empty():
    result = countgd_example_based_counting(
        visual_prompts=[[85, 106, 122, 145]],
        image=np.zeros((0, 0, 3)).astype(np.uint8),
    )
    assert result == []


def test_flux_image_inpainting():
    mask_image = np.zeros((32, 32), dtype=np.uint8)
    mask_image[:4, :4] = 1
    image = np.zeros((32, 32), dtype=np.uint8)

    result = flux_image_inpainting(
        prompt="horse",
        image=image,
        mask=mask_image,
    )

    assert result.shape[0] == 32
    assert result.shape[1] == 32
    assert result.shape[0] == image.shape[0]
    assert result.shape[1] == image.shape[1]


def test_siglip_classification():
    img = ski.data.cat()
    labels = ["cat", "dog", "bird"]

    result = siglip_classification(
        image=img,
        labels=labels,
    )

    assert len(result["scores"]) == 3
    assert len(result["labels"]) == 3
    assert result["labels"][0] == "cat"
    assert result["labels"][1] == "dog"
    assert result["labels"][2] == "bird"
    assert result["scores"][0] > result["scores"][1]
    assert result["scores"][0] > result["scores"][2]


def test_flux_image_inpainting_resizing_not_multiple_8():
    mask_image = np.zeros((37, 37), dtype=np.uint8)
    mask_image[:4, :4] = 1
    image = np.zeros((37, 37), dtype=np.uint8)

    result = flux_image_inpainting(
        prompt="horse",
        image=image,
        mask=mask_image,
    )

    assert result.shape[0] == 32
    assert result.shape[1] == 32
    assert result.shape[0] != image.shape[0]
    assert result.shape[1] != image.shape[1]


def test_flux_image_inpainting_resizing_big_image():
    mask_image = np.zeros((1200, 500), dtype=np.uint8)
    mask_image[:100, :100] = 1
    image = np.zeros((1200, 500), dtype=np.uint8)

    result = flux_image_inpainting(
        prompt="horse",
        image=image,
        mask=mask_image,
    )

    assert result.shape[0] == 512
    assert result.shape[1] == 208
