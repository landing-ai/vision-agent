import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from vision_agent.tools.tools import (
    load_image,
    overlay_bounding_boxes,
    save_image,
    save_video,
)


def test_load_image_from_file_path(tmp_path: Path):
    image_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    image_path = tmp_path / "test.jpg"
    cv2.imwrite(image_path, image_array)

    image = load_image(str(image_path))
    assert isinstance(image, np.ndarray)
    assert image.shape == (480, 640, 3)


def test_load_image_from_url():
    image_url = "https://upload.wikimedia.org/wikipedia/en/a/a9/Example.jpg"
    image = load_image(image_url)
    assert isinstance(image, np.ndarray)
    assert image.shape == (300, 300, 3)


def test_load_image_from_base64():
    image_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_bytes = cv2.imencode(".jpg", image_array)[1].tobytes()
    image_base64 = base64.b64encode(img_bytes).decode("utf-8")
    image_base64 = f"data:image/jpeg;base64,{image_base64}"
    image = load_image(image_base64)
    assert isinstance(image, np.ndarray)
    assert image.shape == (480, 640, 3)


def test_load_image_from_numpy_array():
    image_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    image = load_image(image_array)
    assert isinstance(image, np.ndarray)
    assert np.array_equal(image, image_array)


def test_saves_frames_without_output_path():
    frames = [
        np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(10)
    ]
    output_path = save_video(frames)
    assert Path(output_path).exists()
    os.remove(output_path)


def test_saves_frames_with_output_path():
    frames = [
        np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(10)
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_output_path = Path(tmp_dir) / "output.mp4"
        output_path = save_video(frames, str(video_output_path))

        assert output_path == str(video_output_path)
        assert Path(output_path).exists()


def test_save_null_image():
    image = None
    try:
        save_image(image, "tmp.jpg")
    except ValueError as e:
        assert str(e) == "The image is not a valid NumPy array with shape (H, W, C)"


def test_save_empty_image():
    image = np.zeros((0, 0, 3), dtype=np.uint8)
    try:
        save_image(image, "tmp.jpg")
    except ValueError as e:
        assert str(e) == "The image is not a valid NumPy array with shape (H, W, C)"


def test_save_null_video():
    frames = None
    try:
        save_video(frames, "tmp.mp4")
    except ValueError as e:
        assert str(e) == "Frames must be a list of NumPy arrays"


def test_save_empty_list():
    frames = []
    try:
        save_video(frames, "tmp.mp4")
    except ValueError as e:
        assert str(e) == "Frames must be a list of NumPy arrays"


def test_save_invalid_frame():
    frames = [np.zeros((0, 0, 3), dtype=np.uint8)]
    try:
        save_video(frames, "tmp.mp4")
    except ValueError as e:
        assert str(e) == "A frame is not a valid NumPy array with shape (H, W, C)"


def test_overlay_bounding_boxes_with_empty_bboxes_single_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bboxes = []
    output = overlay_bounding_boxes(image, bboxes)
    assert np.array_equal(image, output)


def test_overlay_bounding_boxes_with_empty_bboxes_multiple_images():
    image1 = np.zeros((480, 640, 3), dtype=np.uint8)
    image2 = np.zeros((400, 600, 3), dtype=np.uint8)
    bboxes = []
    output1, output2 = overlay_bounding_boxes([image1, image2], bboxes)
    assert np.array_equal(image1, output1)
    assert np.array_equal(image2, output2)
