import os
import tempfile
from pathlib import Path

import numpy as np

from vision_agent.tools.tools import overlay_bounding_boxes, save_image, save_video


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
