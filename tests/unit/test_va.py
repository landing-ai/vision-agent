from vision_agent.utils.agent import extract_tag
from vision_agent.agent.vision_agent import _clean_response
from vision_agent.tools.meta_tools import use_extra_vision_agent_args


def parse_execution(code, test_multi_plan=True, custom_tool_names=None):
    code = extract_tag(code, "execute_python")
    if not code:
        return None
    return use_extra_vision_agent_args(code, test_multi_plan, custom_tool_names)


def test_parse_execution_zero():
    code = "print('Hello, World!')"
    assert parse_execution(code) is None


def test_parse_execution_one():
    code = "<execute_python>print('Hello, World!')</execute_python>"
    assert parse_execution(code) == "print('Hello, World!')"


def test_parse_execution_no_test_multi_plan_generate():
    code = "<execute_python>generate_vision_code(artifacts, 'code.py', 'Generate code', ['image.png'])</execute_python>"
    assert (
        parse_execution(code, False)
        == "generate_vision_code(artifacts, 'code.py', 'Generate code', ['image.png'], test_multi_plan=False)"
    )


def test_parse_execution_no_test_multi_plan_edit():
    code = "<execute_python>edit_vision_code(artifacts, 'code.py', ['Generate code'], ['image.png'])</execute_python>"
    assert (
        parse_execution(code, False)
        == "edit_vision_code(artifacts, 'code.py', ['Generate code'], ['image.png'])"
    )


def test_parse_execution_custom_tool_names_generate():
    code = "<execute_python>generate_vision_code(artifacts, 'code.py', 'Generate code', ['image.png'])</execute_python>"
    assert (
        parse_execution(code, test_multi_plan=False, custom_tool_names=["owl_v2_image"])
        == "generate_vision_code(artifacts, 'code.py', 'Generate code', ['image.png'], test_multi_plan=False, custom_tool_names=[\"owl_v2_image\"])"
    )


def test_parse_execution_custom_tool_names_edit():
    code = "<execute_python>edit_vision_code(artifacts, 'code.py', ['Generate code'], ['image.png'])</execute_python>"
    assert (
        parse_execution(code, test_multi_plan=False, custom_tool_names=["owl_v2_image"])
        == "edit_vision_code(artifacts, 'code.py', ['Generate code'], ['image.png'], custom_tool_names=[\"owl_v2_image\"])"
    )


def test_parse_execution_multiple_executes():
    code = "<execute_python>print('Hello, World!')</execute_python><execute_python>print('Hello, World!')</execute_python>"
    assert parse_execution(code) == "print('Hello, World!')\nprint('Hello, World!')"


def test_clean_response():
    response = """<thinking>Thinking...</thinking>
<response>Here is the code:</response>
<execute_python>print('Hello, World!')</execute_python>"""
    assert _clean_response(response) == response


def test_clean_response_remove_extra():
    response = """<thinking>Thinking...</thinking>
<response>Here is the code:</response>
<execute_python>print('Hello, World!')</execute_python>
<thinking>More thinking...</thinking>
<response>Response to code...</response>"""
    expected_response = """<thinking>Thinking...</thinking>
<response>Here is the code:</response>
<execute_python>print('Hello, World!')</execute_python>"""
    assert _clean_response(response) == expected_response
