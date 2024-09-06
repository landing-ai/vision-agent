import skimage as ski

from vision_agent.tools import countgd_counting, countgd_example_based_counting


def test_countgd_counting() -> None:
    img = ski.data.coins()
    result = countgd_counting(image=img, prompt="coin")
    assert len(result) == 24


def test_countgd_example_based_counting() -> None:
    img = ski.data.coins()
    result = countgd_example_based_counting(
        visual_prompts=[[85, 106, 122, 145]],
        image=img,
    )
    assert len(result) == 24
