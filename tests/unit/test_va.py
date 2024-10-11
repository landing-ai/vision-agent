from vision_agent.agent.vision_agent import parse_execution


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
        == "edit_vision_code(artifacts, 'code.py', ['Generate code'], ['image.png'], test_multi_plan=False)"
    )


def test_parse_execution_custom_tool_names_generate():
    code = "<execute_python>generate_vision_code(artifacts, 'code.py', 'Generate code', ['image.png'])</execute_python>"
    assert (
        parse_execution(
            code, test_multi_plan=False, custom_tool_names=["owl_v2_image"]
        )
        == "generate_vision_code(artifacts, 'code.py', 'Generate code', ['image.png'], test_multi_plan=False, custom_tool_names=['owl_v2_image'])"
    )


def test_parse_execution_custom_tool_names_edit():
    code = "<execute_python>edit_vision_code(artifacts, 'code.py', ['Generate code'], ['image.png'])</execute_python>"
    assert (
        parse_execution(
            code, test_multi_plan=False, custom_tool_names=["owl_v2_image"]
        )
        == "edit_vision_code(artifacts, 'code.py', ['Generate code'], ['image.png'], test_multi_plan=False, custom_tool_names=['owl_v2_image'])"
    )


def test_parse_execution_multiple_executes():
    code = "<execute_python>print('Hello, World!')</execute_python><execute_python>print('Hello, World!')</execute_python>"
    assert parse_execution(code) == "print('Hello, World!')\nprint('Hello, World!')"
