import tempfile
from typing import Optional

import cv2
import numpy as np

from vision_agent.utils.video import extract_frames_from_video


def test_extract_frames_from_video():
    video_path = _create_video(duration=2)
    # there are 48 frames at 24 fps in this video file
    res = extract_frames_from_video(video_path, fps=24)
    assert len(res) == 48

    res = extract_frames_from_video(video_path, fps=2)
    assert len(res) == 4

    res = extract_frames_from_video(video_path, fps=1)
    assert len(res) == 2


def test_extract_frames_from_invalid_uri():
    uri = "https://www.youtube.com/watch?v=HjGJvNRkuqY&ab_channel=TheSAHDStudio"
    res = extract_frames_from_video(uri, 1.0)
    assert len(res) == 0


def test_extract_frames_with_illegal_fps():
    video_path = _create_video(duration=1)
    res = extract_frames_from_video(video_path, -1.0)
    assert len(res) == 1

    res = extract_frames_from_video(video_path, None)
    assert len(res) == 1

    res = extract_frames_from_video(video_path, 0.0)
    assert len(res) == 1


def test_extract_frames_with_input_video_has_no_fps():
    video_path = _create_video(fps_video_prop=None)
    res = extract_frames_from_video(video_path, 1.0)
    assert len(res) == 0


def _create_video(
    *, duration: int = 3, fps: int = 24, fps_video_prop: Optional[int] = 24
) -> str:
    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        video_path = temp_video.name
    # Set video properties
    width, height = 640, 480
    # Create a VideoWriter object without setting FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps_video_prop, (width, height))
    # Generate and write random frames
    for _ in range(duration * fps):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path
