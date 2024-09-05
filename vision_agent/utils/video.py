import base64
import logging
import tempfile
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np
from decord import VideoReader

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


@lru_cache(maxsize=8)
def extract_frames_from_video(
    video_uri: str, fps: float = 1.0
) -> List[Tuple[np.ndarray, float]]:
    """Extract frames from a video

    Parameters:
        video_uri (str): the path to the video file or a video file url
        fps (float): the frame rate per second to extract the frames

    Returns:
        a list of tuples containing the extracted frame and the timestamp in seconds.
            E.g. [(frame1, 0.0), (frame2, 0.5), ...]. The timestamp is the time in seconds
            from the start of the video. E.g. 12.125 means 12.125 seconds from the start of
            the video. The frames are sorted by the timestamp in ascending order.
    """
    vr = VideoReader(video_uri)
    orig_fps = vr.get_avg_fps()
    if fps > orig_fps:
        fps = orig_fps

    s = orig_fps / fps
    samples = [(int(i * s), int(i * s) / orig_fps) for i in range(int(len(vr) / s))]
    frames = vr.get_batch([s[0] for s in samples]).asnumpy()
    return [(frames[i, :, :, :], samples[i][1]) for i in range(len(samples))]
