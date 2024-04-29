PROGRAM = """
**Role**: You are a software programmer.

**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is efficient, readable, and well-commented. Return the requested information from the function you create.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task, you do not need to worry about defining them or importing them and can assume they are available to you.
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

DEBUG = """
**Role**: You are a software programmer.

**Task**: Your task is to run the `execute` function and either print the output or print a file name containing visualized output for another agent to examine. The other agent will then use your output, either the printed return value of the function or the visualized output as a file, to determine if `execute` is functioning correctly.

**Documentation**
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task, you do not need to worry about defining them or importing them and can assume they are available to you.
{docstring}

**Input Code Snippet**:
```python
### Please decided how would you want to generate test cases. Based on incomplete code or completed version.
{code}
```

**User Instructions**:
{question}

**Previous Feedback**:
{feedback}

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task.
2. **Code Execution**: Run the `execute` function with the given input from the user instructions.
3. **Output Generation**: Print the output or save it as a file for visualization utilizing the functions you have access to.
"""

VISUAL_TEST = """
**Role**: You are a machine vision expert.

**Task**: Your task is to visually inspect the output of the `execute` function and determine if the visualization of the function output looks correct given the user's instructions. If not, you can provide suggestions to improve the `execute` function to imporve it.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task, you do not need to worry about defining them or importing them and can assume they are available to you.
{docstring}


**Input Code Snippet**:
This is the code that
```python
{code}
```

**User Instructions**:
{question}

**Previous Feedback**:
{feedback}

**Instructions**:
1. **Visual Inspection**: Examine the visual output of the `execute` function.
2. **Evaluation**: Determine if the visualization is correct based on the user's instructions.
3. **Feedback**: Provide feedback on the visualization and suggest improvements if necessary.
4. **Clear Concrete Instructions**: Provide clear concrete instructions to improve the results. You can only make coding suggestions based on the either the input code snippet or the documented code provided. For example, do not say the threshold needs to be adjust, instead provide an exact value for adjusting the threshold.

Provide output in JSON format {{"finished": boolean, "feedback": "your feedback"}} where "finished" is True if the output is correct and False if not and "feedback" is your feedback.
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

TEST = """
**Role**: As a tester, your task is to create comprehensive test cases for the incomplete `execute` function. These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's robustness, reliability, and scalability.

**User Instructions**:
{question}

**Input Code Snippet**:
```python
### Please decided how would you want to generate test cases. Based on incomplete code or completed version.
{code}
```

**1. Basic Test Cases**:
- **Objective**: To verify the fundamental functionality of the `has_close_elements` function under normal conditions.

**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.

**3. Large Scale Test Cases**:
- **Objective**: To assess the function’s performance and scalability with large data samples.

**Instructions**:
- Implement a comprehensive set of test cases following the guidelines above.
- Ensure each test case is well-documented with comments explaining the scenario it covers.
- Pay special attention to edge cases as they often reveal hidden bugs.
- For large-scale tests, focus on the function's efficiency and performance under heavy loads.
"""
