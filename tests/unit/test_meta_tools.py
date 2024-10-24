from pathlib import Path
from vision_agent.tools.meta_tools import (
    Artifacts,
    check_and_load_image,
    use_extra_vision_agent_args,
    use_object_detection_fine_tuning,
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


def test_use_object_detection_fine_tuning_none():
    artifacts = Artifacts(Path.cwd())
    code = "print('Hello, World!')"
    artifacts["code"] = code
    output = use_object_detection_fine_tuning(artifacts, "code", "123")
    assert (
        output == "[No function calls to replace with fine tuning id in artifact code]"
    )
    assert artifacts["code"] == code


def test_use_object_detection_fine_tuning():
    artifacts = Artifacts(Path.cwd())
    code = """florence2_phrase_grounding('one', image1)
owl_v2_image('two', image2)
florence2_sam2_image('three', image3)"""
    expected_code = """florence2_phrase_grounding("one", image1, "123")
owl_v2_image("two", image2, "123")
florence2_sam2_image("three", image3, "123")"""
    artifacts["code"] = code

    output = use_object_detection_fine_tuning(artifacts, "code", "123")
    assert 'florence2_phrase_grounding("one", image1, "123")' in output
    assert 'owl_v2_image("two", image2, "123")' in output
    assert 'florence2_sam2_image("three", image3, "123")' in output
    assert artifacts["code"] == expected_code


def test_use_object_detection_fine_tuning_twice():
    artifacts = Artifacts(Path.cwd())
    code = """florence2_phrase_grounding('one', image1)
owl_v2_image('two', image2)
florence2_sam2_image('three', image3)"""
    expected_code1 = """florence2_phrase_grounding("one", image1, "123")
owl_v2_image("two", image2, "123")
florence2_sam2_image("three", image3, "123")"""
    expected_code2 = """florence2_phrase_grounding("one", image1, "456")
owl_v2_image("two", image2, "456")
florence2_sam2_image("three", image3, "456")"""
    artifacts["code"] = code
    output = use_object_detection_fine_tuning(artifacts, "code", "123")
    assert 'florence2_phrase_grounding("one", image1, "123")' in output
    assert 'owl_v2_image("two", image2, "123")' in output
    assert 'florence2_sam2_image("three", image3, "123")' in output
    assert artifacts["code"] == expected_code1

    output = use_object_detection_fine_tuning(artifacts, "code", "456")
    assert 'florence2_phrase_grounding("one", image1, "456")' in output
    assert 'owl_v2_image("two", image2, "456")' in output
    assert 'florence2_sam2_image("three", image3, "456")' in output
    assert artifacts["code"] == expected_code2


def test_use_object_detection_fine_tuning_real_case():
    artifacts = Artifacts(Path.cwd())
    code = "florence2_phrase_grounding('(strange arg)', image1)"
    expected_code = 'florence2_phrase_grounding("(strange arg)", image1, "123")'
    artifacts["code"] = code
    output = use_object_detection_fine_tuning(artifacts, "code", "123")
    assert 'florence2_phrase_grounding("(strange arg)", image1, "123")' in output
    assert artifacts["code"] == expected_code


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
