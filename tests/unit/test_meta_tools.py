from pathlib import Path

from vision_agent.tools.meta_tools import (
    Artifacts,
    check_and_load_image,
)


def test_check_and_load_image_none():
    assert check_and_load_image("print('Hello, World!')") == []


def test_check_and_load_image_one():
    assert check_and_load_image("view_media_artifact(artifacts, 'image.jpg')") == [
        "image.jpg"
    ]


def test_check_and_load_image_two():
    code = "view_media_artifact(artifacts, 'image1.jpg')\nview_media_artifact(artifacts, 'image2.jpg')"
    assert check_and_load_image(code) == ["image1.jpg", "image2.jpg"]
