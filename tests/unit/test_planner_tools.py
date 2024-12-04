from vision_agent.tools.planner_tools import check_function_call, replace_box_threshold


def test_check_function_call():
    code = """
test_function('one', image1)
"""
    assert check_function_call(code, "test_function") == True
    assert check_function_call(code, "test_function2") == False


def test_check_function_call_try_catch():
    code = """
try:
    test_function('one', image1)
except Exception as e:
    pass
"""
    assert check_function_call(code, "test_function") == True
    assert check_function_call(code, "test_function2") == False


def test_replace_box_threshold():
    code = """
test_function('one', image1, box_threshold=0.1)
"""
    expected_code = """
test_function('one', image1, box_threshold=0.5)
"""
    assert replace_box_threshold(code, ["test_function"], 0.5) == expected_code


def test_replace_box_threshold_in_function():
    code = """
def test_function_outer():
    test_function('one', image1, box_threshold=0.1)
"""
    expected_code = """
def test_function_outer():
    test_function('one', image1, box_threshold=0.5)
"""
    assert replace_box_threshold(code, ["test_function"], 0.5) == expected_code


def test_replace_box_threshold_no_arg():
    code = """
test_function('one', image1)
"""
    expected_code = """
test_function('one', image1, box_threshold=0.5)
"""
    assert replace_box_threshold(code, ["test_function"], 0.5) == expected_code


def test_replace_box_threshold_no_func():
    code = """
test_function2('one', image1)
"""
    expected_code = """
test_function2('one', image1)
"""
    assert replace_box_threshold(code, ["test_function"], 0.5) == expected_code
