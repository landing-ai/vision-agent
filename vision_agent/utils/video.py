import base64
import logging
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, cast

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

_LOGGER = logging.getLogger(__name__)
# The maximum length of the clip to extract frames from, in seconds
_CLIP_LENGTH = 30.0


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


def extract_frames_from_video(
    video_uri: str, fps: float = 0.5, motion_detection_threshold: float = 0.0
) -> List[Tuple[np.ndarray, float]]:
    """Extract frames from a video

    Parameters:
        video_uri: the path to the video file or a video file url
        fps: the frame rate per second to extract the frames
        motion_detection_threshold: The threshold to detect motion between
            changes/frames. A value between 0-1, which represents the percentage change
            required for the frames to be considered in motion. For example, a lower
            value means more frames will be extracted. A non-positive value will disable
            motion detection and extract all frames.

    Returns:
        a list of tuples containing the extracted frame and the timestamp in seconds.
            E.g. [(frame1, 0.0), (frame2, 0.5), ...]. The timestamp is the time in seconds
            from the start of the video. E.g. 12.125 means 12.125 seconds from the start of
            the video. The frames are sorted by the timestamp in ascending order.
    """
    with VideoFileClip(video_uri) as video:
        video_duration: float = video.duration
        num_workers = os.cpu_count()
        clip_length: float = min(video_duration, _CLIP_LENGTH)
        start_times = list(range(0, math.ceil(video_duration), math.ceil(clip_length)))
        assert start_times, f"No frames to extract from the input video: {video_uri}"
        segment_args = [
            {
                "video_uri": video_uri,
                "start": start,
                "end": (
                    start + clip_length if i < len(start_times) - 1 else video_duration
                ),
                "fps": fps,
                "motion_detection_threshold": motion_detection_threshold,
            }
            for i, start in enumerate(start_times)
        ]
        if (
            cast(float, segment_args[-1]["end"])
            - cast(float, segment_args[-1]["start"])
            < 1
        ):
            # If the last segment is less than 1s, merge it with the previous segment
            # This is to avoid the failure of the last segment extraction
            assert (
                len(segment_args) > 1
            ), "Development bug - Expect at least 2 segments."
            segment_args[-2]["end"] = video_duration
            segment_args.pop(-1)
        _LOGGER.info(
            f"""Created {len(segment_args)} segments from the input video  {video_uri} of length {video.duration}s, with clip size: {clip_length}s and {num_workers} workers.
            Segments: {segment_args}
            """
        )
        frames = []
        with tqdm(total=len(segment_args)) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_extract_frames_by_clip, **kwargs)  # type: ignore
                    for kwargs in segment_args
                ]
                for future in as_completed(futures):
                    result = future.result()
                    frames.extend(result)
                    pbar.update(1)
        frames.sort(key=lambda x: x[1])
        _LOGGER.info(f"Extracted {len(frames)} frames from video {video_uri}")
        return frames


def _extract_frames_by_clip(
    video_uri: str,
    start: int = 0,
    end: float = -1,
    fps: int = 2,
    motion_detection_threshold: float = 0.06,
) -> List[Tuple[np.ndarray, float]]:
    """Extract frames from a video clip with start and end time in seconds.

    Parameters:
        video_uri: the path to the video file or a video file url
        start: the start time (in seconds) of the clip to extract
        end: the end time (in seconds, up to millisecond level precision) of the clip to extract, if -1, extract the whole video
        fps: the frame rate to extract the frames
        motion_detection_threshold: the threshold to detect the motion between frames
    """
    with VideoFileClip(video_uri) as video:
        source_fps = video.fps
        if end <= 0:
            end = video.duration
        _LOGGER.info(
            f"Extracting frames from video {video_uri} ({video.duration}s) with start={start}s and end={end}s"
        )
        clip = video.subclip(start, end)
        processable_frames = int(clip.duration * fps)
        _LOGGER.info(
            f"Extracting frames from video clip of length {clip.duration}s with FPS={fps} and start_time={start}s. Total number of frames in clip: {processable_frames}"
        )
        frames = []
        total_count, skipped_count = 0, 0
        prev_processed_frame = None
        pbar = tqdm(
            total=processable_frames, desc=f"Extracting frames from clip {start}-{end}"
        )
        for i, frame in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
            total_count += 1
            pbar.update(1)
            if motion_detection_threshold > 0:
                curr_processed_frame = _preprocess_frame(frame)
                # Skip the frame if it is similar to the previous one
                if prev_processed_frame is not None and _similar_frame(
                    prev_processed_frame,
                    curr_processed_frame,
                    threshold=motion_detection_threshold,
                ):
                    skipped_count += 1
                    continue
                prev_processed_frame = curr_processed_frame
            ts = round(clip.reader.pos / source_fps, 3)
            frames.append((frame, ts))

        _LOGGER.info(
            f"""Finished!
                Frames extracted: {len(frames)}
                Extracted frame timestamp: {[f[1] for f in frames]}
                Total processed frames: {total_count}
                Skipped frames:  {skipped_count}
                Scan FPS: {fps}
                Clip start time: {start}s, {clip.pos}
                Clip end time: {end}s
                Clip duration: {clip.duration}s
                Clip total frames: {clip.duration * source_fps}
                Video duration: {video.duration}s
                Video FPS: {video.fps}
                Video total frames: {video.reader.nframes}"""
        )
        return frames


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(src=frame, ksize=(5, 5), sigmaX=0)
    return frame


def _similar_frame(
    prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: float
) -> bool:
    """Detect two frames are similar or not

    Parameters:
        threshold: similarity threshold, a value between 0-1, the percentage change that is considered a different frame.
    """
    # calculate difference and update previous frame TODO: don't assume the processed image is cached
    diff_frame = cv2.absdiff(src1=prev_frame, src2=curr_frame)
    # Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(
        src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY
    )[1]
    change_percentage = cv2.countNonZero(thresh_frame) / (
        curr_frame.shape[0] * curr_frame.shape[1]
    )
    _LOGGER.debug(f"Image diff: {change_percentage}")
    return change_percentage < threshold
