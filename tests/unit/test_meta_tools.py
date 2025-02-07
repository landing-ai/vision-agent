from pathlib import Path

from vision_agent.tools.meta_tools import (
    Artifacts,
    check_and_load_image,
    use_extra_vision_agent_args,
)


def test_check_and_load_image_none():
    assert check_and_load_image("print('Hello, World!')") == []


def test_check_and_load_image_one():
    assert check_and_load_image("view_media_artifact(artifacts, 'image.jpg')") == [
        "image.jpg"
    ]


def test_check_and_load_image_two():
    code = "view_media_artifact(artifacts, 'image1.jpg')\nview_media_artifact(artifacts, 'image2.jpg')"
    assert check_and_load_image(code) == ["image1.jpg", "image2.jpg"]


def test_use_extra_vision_agent_args_real_case():
    code = "generate_vision_code(artifacts, 'code.py', 'write code', ['/home/user/n0xn5X6_IMG_2861%20(1).mov'])"
    expected_code = "generate_vision_code(artifacts, 'code.py', 'write code', ['/home/user/n0xn5X6_IMG_2861%20(1).mov'], test_multi_plan=True)"
    out_code = use_extra_vision_agent_args(code)
    assert out_code == expected_code

    code = "edit_vision_code(artifacts, 'code.py', ['write code 1', 'write code 2'], ['/home/user/n0xn5X6_IMG_2861%20(1).mov'])"
    expected_code = "edit_vision_code(artifacts, 'code.py', ['write code 1', 'write code 2'], ['/home/user/n0xn5X6_IMG_2861%20(1).mov'])"
    out_code = use_extra_vision_agent_args(code)
    assert out_code == expected_code


def test_use_extra_vision_args_with_custom_tools():
    code = "generate_vision_code(artifacts, 'code.py', 'write code', ['/home/user/n0xn5X6_IMG_2861%20(1).mov'])"
    expected_code = "generate_vision_code(artifacts, 'code.py', 'write code', ['/home/user/n0xn5X6_IMG_2861%20(1).mov'], test_multi_plan=True, custom_tool_names=[\"tool1\", \"tool2\"])"
    out_code = use_extra_vision_agent_args(code, custom_tool_names=["tool1", "tool2"])
    assert out_code == expected_code

    code = "edit_vision_code(artifacts, 'code.py', 'write code', ['/home/user/n0xn5X6_IMG_2861%20(1).mov'])"
    expected_code = "edit_vision_code(artifacts, 'code.py', 'write code', ['/home/user/n0xn5X6_IMG_2861%20(1).mov'], custom_tool_names=[\"tool1\", \"tool2\"])"
    out_code = use_extra_vision_agent_args(code, custom_tool_names=["tool1", "tool2"])
    assert out_code == expected_code


def test_use_extra_vision_args_with_non_ascii():
    code = "generate_vision_code(artifacts, 'code.py', 'write code', ['/home/user/n0xn5X6_IMG_2861%20(1)漢.mov'])"
    expected_code = "generate_vision_code(artifacts, 'code.py', 'write code', ['/home/user/n0xn5X6_IMG_2861%20(1)漢.mov'], test_multi_plan=True, custom_tool_names=[\"tool1\", \"tool2\"])"
    out_code = use_extra_vision_agent_args(code, custom_tool_names=["tool1", "tool2"])
    assert out_code == expected_code
