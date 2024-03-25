from vision_agent.tools.video import extract_frames_from_video


def test_extract_frames_from_video():
    video_path = "tests/data/video/test.mp4"
    res = extract_frames_from_video(video_path)
    assert len(res) == 1
