import logging
import tempfile
from functools import lru_cache
from typing import IO, List, Optional, Tuple

import av  # type: ignore
import cv2
import numpy as np

_LOGGER = logging.getLogger(__name__)
# The maximum length of the clip to extract frames from, in seconds

_DEFAULT_VIDEO_FPS = 24
_DEFAULT_INPUT_FPS = 1.0


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    new_width = width - (width % 2)
    new_height = height - (height % 2)
    return cv2.resize(frame, (new_width, new_height))


def video_writer(
    frames: List[np.ndarray],
    fps: float = _DEFAULT_INPUT_FPS,
    file: Optional[IO[bytes]] = None,
) -> str:
    if isinstance(fps, str):
        # fps could be a string when it's passed in from a web endpoint deployment
        fps = float(fps)
    if file is None:
        file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with av.open(file, "w") as container:
        stream = container.add_stream("h264", rate=fps)
        height, width = frames[0].shape[:2]
        stream.height = height - (height % 2)
        stream.width = width - (width % 2)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "10"}
        for frame in frames:
            # Remove the alpha channel (convert RGBA to RGB)
            frame_rgb = frame[:, :, :3]
            # Resize the frame to make dimensions divisible by 2
            frame_rgb = _resize_frame(frame_rgb)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    return file.name


def frames_to_bytes(
    frames: List[np.ndarray], fps: float = _DEFAULT_INPUT_FPS, file_ext: str = ".mp4"
) -> bytes:
    r"""Convert a list of frames to a video file encoded into a byte string.

    Parameters:
        frames: the list of frames
        fps: the frames per second of the video
        file_ext: the file extension of the video file
    """
    if isinstance(fps, str):
        # fps could be a string when it's passed in from a web endpoint deployment
        fps = float(fps)
    with tempfile.NamedTemporaryFile(delete=True, suffix=file_ext) as f:
        video_writer(frames, fps, f)
        f.seek(0)
        buffer_bytes = f.read()
    return buffer_bytes


def rescale(frame: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    new_h, new_w = h, w
    if new_h > max_size[0]:
        new_h = max_size[0]
        new_w = int(w * new_h / h)
    if new_w > max_size[1]:
        new_w = max_size[1]
        new_h = int(h * new_w / w)
    if h != new_h or w != new_w:
        frame = cv2.resize(frame, (new_w, new_h))
    return frame


# WARNING: This cache is a little dangerous because if the underlying video
# contents change but the filename remains the same it will return the old file contents.
# For vision agent it's unlikely to change the file contents while keeping the
# same file name and the time savings are very large.
@lru_cache(maxsize=8)
def extract_frames_from_video(
    video_uri: str, fps: float = _DEFAULT_INPUT_FPS
) -> List[Tuple[np.ndarray, float]]:
    """Extract frames from a video along with the timestamp in seconds.

    Parameters:
        video_uri (str): the path to the video file or a video file url
        fps (float): the frame rate per second to extract the frames

    Returns:
        a list of tuples containing the extracted frame and the timestamp in seconds.
            E.g. [(frame1, 0.0), (frame2, 0.5), ...]. The timestamp is the time in seconds
            from the start of the video. E.g. 12.125 means 12.125 seconds from the start of
            the video. The frames are sorted by the timestamp in ascending order.
    """
    if isinstance(fps, str):
        # fps could be a string when it's passed in from a web endpoint deployment
        fps = float(fps)

    cap = cv2.VideoCapture(video_uri)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        _LOGGER.warning(
            f"Input video, {video_uri}, has no fps, using the default value {_DEFAULT_VIDEO_FPS}"
        )
        orig_fps = _DEFAULT_VIDEO_FPS
    if not fps or fps <= 0:
        _LOGGER.warning(
            f"Input fps, {fps}, is illegal, using the default value: {_DEFAULT_INPUT_FPS}"
        )
        fps = _DEFAULT_INPUT_FPS
    orig_frame_time = 1 / orig_fps
    targ_frame_time = 1 / fps
    frames: List[Tuple[np.ndarray, float]] = []
    i = 0
    elapsed_time = 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time += orig_frame_time
        # This is to prevent float point precision loss issue, which can cause
        # the elapsed time to be slightly less than the target frame time, which
        # causes the last frame to be skipped
        elapsed_time = round(elapsed_time, 8)
        if elapsed_time >= targ_frame_time:
            frame = rescale(frame, (1024, 1024))
            frames.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), i / orig_fps))
            elapsed_time -= targ_frame_time

        i += 1
    cap.release()
    _LOGGER.info(f"Extracted {len(frames)} frames from {video_uri}")
    return frames
