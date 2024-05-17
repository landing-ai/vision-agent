USER_REQ = """
## User Request
{user_request}
"""


PLAN = """
# Context
{context}

# Tools Available
{tool_desc}

# Task
Based on the context and tools you have available, write a plan of subtasks to achieve the user request utilizing tools when necessary. Output a list of jsons in the following format:

```json
{{
    "user_req": str # a detailed version of the user requirement
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

**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is efficient, readable, and well-commented. Return the requested information from the function you create.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools.tools_v2 import *`.
{docstring}

**Input Code Snippet**:
```python
def execute(image_path: str):
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
**Role**: As a tester, your task is to create comprehensive test cases for the incomplete `execute` function. These test cases should encompass Basic and Edge case scenarios to ensure the code's robustness and reliability.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools.tools_v2 import *`.
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
- Implement a comprehensive set of test cases following the guidelines above.
- Ensure each test case is well-documented with comments explaining the scenario it covers.
- Pay special attention to edge cases as they often reveal hidden bugs.
"""

FIX_BUG = """
Please re-complete the code to fix the error message. Here is the previous version:
```python
{code}
```

When we run this code:
```python
{tests}
```

It raises this error:
```python
{result}
```

This is previous feedback provided on the code:
{feedback}

Please fix the bug by follow the error information and only return python code. You do not need return the test cases. The re-completion code should in triple backticks format(i.e., in ```python ```).
"""

