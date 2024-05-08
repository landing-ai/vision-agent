from vision_agent.utils.video import extract_frames_from_video


def test_extract_frames_from_video():
    # TODO: consider generating a video on the fly instead
    video_path = "tests/data/video/test.mp4"
    res = extract_frames_from_video(video_path)
    assert len(res) == 1
