import base64
import logging
import tempfile
from functools import lru_cache
from typing import List, Optional, Tuple

import av  # type: ignore
import cv2
import numpy as np

_LOGGER = logging.getLogger(__name__)
# The maximum length of the clip to extract frames from, in seconds


def play_video(video_base64: str) -> None:
    """Play a video file"""
    video_data = base64.b64decode(video_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_data)
        temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            _LOGGER.error("Error: Could not open video.")
            return

        # Display the first frame and wait for any key press to start the video
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Video Player", frame)
            _LOGGER.info(f"Press any key to start playing the video: {temp_video_path}")
            cv2.waitKey(0)  # Wait for any key press

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Video Player", frame)
            # Press 'q' to exit the video
            if cv2.waitKey(200) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    new_width = width - (width % 2)
    new_height = height - (height % 2)
    return cv2.resize(frame, (new_width, new_height))


def video_writer(
    frames: List[np.ndarray], fps: float = 1.0, filename: Optional[str] = None
) -> str:
    if filename is None:
        filename = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    container = av.open(filename, mode="w")
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
    container.close()
    return filename


def frames_to_bytes(
    frames: List[np.ndarray], fps: float = 1.0, file_ext: str = ".mp4"
) -> bytes:
    r"""Convert a list of frames to a video file encoded into a byte string.

    Parameters:
        frames: the list of frames
        fps: the frames per second of the video
        file_ext: the file extension of the video file
    """
    with tempfile.NamedTemporaryFile(delete=True, suffix=file_ext) as temp_file:
        video_writer(frames, fps, temp_file.name)

        with open(temp_file.name, "rb") as f:
            buffer_bytes = f.read()
    return buffer_bytes


# WARNING: this cache is cache is a little dangerous because if the underlying video
# contents change but the filename remains the same it will return the old file contents
# but for vision agent it's unlikely to change the file contents while keeping the
# same file name and the time savings are very large.
@lru_cache(maxsize=8)
def extract_frames_from_video(
    video_uri: str, fps: float = 1.0
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

    cap = cv2.VideoCapture(video_uri)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
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
        if elapsed_time >= targ_frame_time:
            frames.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), i / orig_fps))
            elapsed_time -= targ_frame_time

        i += 1
    cap.release()
    return frames
