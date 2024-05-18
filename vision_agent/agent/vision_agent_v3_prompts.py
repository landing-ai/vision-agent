USER_REQ = """
## User Request
{user_request}
"""


PLAN = """
**Context**
{context}

**Tools Available**:
{tool_desc}

**Previous Feedback**:
{feedback}

**Instructions**:
Based on the context and tools you have available, write a plan of subtasks to achieve the user request utilizing tools when necessary. Output a list of jsons in the following format:

```json
{{
    "plan":
        [
            {{
                "instruction": str # what you should do in this task, one short phrase or sentence
            }}
        ]
}}
```
"""

CODE = """
**Role**: You are a software programmer.

**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is efficient, readable, and well-commented. Return the requested information from the function you create. Do no call your code unnecessarily, a test will be run after it is submitted.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools.tools_v2 import *`.

{docstring}

**Input Code Snippet**:
```python
# Your code here
```

**User Instructions**:
{question}

**Previous Feedback**:
{feedback}

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task.
2. **Algorithm/Method Selection**: Decide on the most efficient way.
3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode.
4. **Code Generation**: Translate your pseudocode into executable Python code.
"""

TEST = """
**Role**: As a tester, your task is to create comprehensive test cases for the provided code. These test cases should encompass Basic and Edge case scenarios to ensure the code's robustness and reliability if possible.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools.tools_v2 import *`. You do not need to test these functions, only the code provided by the user.

{docstring}

**User Instructions**:
{question}

**Input Code Snippet**:
```python
### Please decided how would you want to generate test cases. Based on incomplete code or completed version.
{code}
```

**1. Basic Test Cases**:
- **Objective**: To verify the fundamental functionality under normal conditions.

**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.

**Instructions**:
1. Implement a comprehensive set of test cases following the guidelines above and testing each of the user's instructions.
2. Ensure each test case is well-documented with comments explaining the scenario it covers.
3. Pay special attention to edge cases as they often reveal hidden bugs.
4. DO NOT use any files that are not provided by the user's instructions, your test must be run and will crash if it tries to load a non-existent file.
5. If it is a vision task, you can visualize the output to ensure it is correct.

You should format your test cases at the end of your response wrapped in ```python ``` tags like in the following example:
```python
# You can run assertions to ensure the function is working as expected
assert function(input) == expected_output, "Test case description"

# You can simply call the function to ensure it runs
function(input)

# Or you can visualize the output
output = function(input)
visualize(output)
```

**Examples**
## Prompt 1:
```python
def detect_cats_and_dogs(image_path: str) -> Dict[str, List[List[float]]]:
    \""" Detects cats and dogs in an image. Returns a dictionary with
    {{
        "cats": [[x1, y1, x2, y2], ...], "dogs": [[x1, y1, x2, y2], ...]
    }} 
    \"""
```

## Completion 1:
```python
# We can test to ensure the output has the correct structure but we cannot test the
# content of the output without knowing the image. We can test on "image.jpg" because
# it is provided by the user so we know it exists.
output = detect_cats_and_dogs("image.jpg")
assert "cats" in output, "The output should contain 'cats'
assert "dogs" in output, "The output should contain 'dogs'
```

## Prompt 2:
```python
def find_text(image_path: str, text: str) -> str:
    \""" Finds the text in the image and returns the text. \"""

## Completion 2:
```python
# Because we do not know ahead of time what text is in the image, we can only run the
# code and print the results. We can test on "image.jpg" because it is provided by the
# user so we know it exists.
found_text = find_text("image.jpg", "Hello World")
print(found_text)
```
"""


FIX_BUG = """
**Role** As a coder, your job is to find the error in the code and fix it. You are running in a notebook setting so feel free to run !pip install to install missing packages.

**Instructions**:
Please re-complete the code to fix the error message. Here is the previous version:
```python
{code}
```

When we run this test code:
```python
{tests}
```

It raises this error:
```python
{result}
```

This is previous feedback provided on the code:
{feedback}

Please fix the bug by follow the error information and return a JSON object with the following format:
{{
    "reflections": str # any thoughts you have about the bug and how you fixed it
    "code": str # the fixed code if any, else an empty string
    "test": str # the fixed test code if any, else an empty string
}}
"""


REFLECT = """
**Role**: You are a reflection agent. Your job is to look at the original user request and the code and test cases produced and determine if they meet the user's request. If they do not, you must provide feedback on how to improve the code and test cases.

**Context**:
{context}

**Plan**:
{plan}

**Code**:
{code}

**Test Cases**:
{test}

**Instructions**:
Respond in JSON format with the following structure:
{{
    "feedback": str # the feedback you would give to the coder and tester
    "success": bool # whether the code and tests meet the user request
}}
"""
