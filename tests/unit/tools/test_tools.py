import os
import tempfile
from pathlib import Path

import numpy as np

from vision_agent.tools.tools import save_image, save_video


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
