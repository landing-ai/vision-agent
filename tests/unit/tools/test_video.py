from vision_agent.utils.video import extract_frames_from_video


def test_extract_frames_from_video():
    # TODO: consider generating a video on the fly instead
    video_path = "tests/data/video/test.mp4"

    # there are 48 frames at 24 fps in this video file
    res = extract_frames_from_video(video_path, fps=24)
    assert len(res) == 48
