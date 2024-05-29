USER_REQ_CONTEXT = """
## User Requirement
{user_requirement}
"""

USER_REQ_SUBTASK_CONTEXT = """
## User Requirement
{user_requirement}

## Current Subtask
{subtask}
"""

USER_REQ_SUBTASK_WM_CONTEXT = """
## User Requirement
{user_requirement}

## Current Subtask
{subtask}

## Previous Task
{working_memory}
"""

PLAN = """
# Context
{context}

# Current Plan
{plan}

# Tools Available
{tool_desc}

# Task:
Based on the context and the tools you have available, write a plan of subtasks to achieve the user request that adhere to the following requirements:
- For each subtask, you should provide instructions on what to do. Write detailed subtasks, ensure they are large enough to be meaningful, encompassing multiple lines of code.
- You do not need to have the agent rewrite any tool functionality you already have, you should instead instruct it to utilize one or more of those tools in each subtask.
- You can have agents either write coding tasks, to code some functionality or testing tasks to test previous functionality.
- If a current plan exists, examine each item in the plan to determine if it was successful. If there was an item that failed, i.e. 'success': False, then you should rewrite that item and all subsequent items to ensure that the rewritten plan is successful.

Output a list of jsons in the following format:

```json
{{
    "user_req": str, # "a summarized version of the user requirement"
    "plan":
        [
            {{
                "task_id": int, # "unique identifier for a task in plan, can be an ordinal"
                "dependent_task_ids": list[int], # "ids of tasks prerequisite to this task"
                "instruction": str, # "what you should do in this task, one short phrase or sentence"
                "type": str, # "the type of the task, tasks can either be 'code' for coding tasks or 'test' for testing tasks"
            }},
            ...
        ]
}}
```
"""


CODE_SYS_MSG = """You are an AI Python assistant. You need to help user to achieve their goal by implementing a function. Your code will be run in a jupyter notebook environment so don't use asyncio.run. Instead, use await if you need to call an async function. Do not use 'display' for showing images, instead use matplotlib or PIL."""


CODE = """
# Context
{context}

# Tool Info for Current Subtask
{tool_info}

# Previous Code
{code}

# Constraints
- Write a function that accomplishes the 'Current Subtask'. You are supplied code from a previous task under 'Previous Code', do not delete or change previous code unless it contains a bug or it is necessary to complete the 'Current Subtask'.
- Always prioritize using pre-defined tools or code for the same functionality from 'Tool Info' when working on 'Current Subtask'. You have access to all these tools through the `from vision_agent.tools import *` import.
- You may recieve previous trials and errors under 'Previous Task', this is code, output and reflections from previous tasks. You can use these to avoid running in to the same issues when writing your code.
- Use the `save_json` function from `vision_agent.tools` to save your output as a json file.
- Write clean, readable, and well-documented code.

# Output
While some concise thoughts are helpful, code is absolutely required. If possible, execute your defined functions in the code output. Output code in the following format:
```python
from vision_agent.tools imoprt *

# your code goes here
```
"""


DEBUG_SYS_MSG = """You are an AI Python assistant. You will be given your previous implementation code of a task, runtime error results, and a hint to change the implementation appropriately. Your code will be run in a jupyter notebook environment. Write your full implementation."""


DEBUG_EXAMPLE = '''
[previous impl]:
```python
def add(a: int, b: int) -> int:
   """Given integers a and b, return the total value of a and b."""
   return a - b
```

[previous output]
Tests failed:
assert add(1, 2) == 3 # output: -1
assert add(1, 3) == 4 # output: -2

[reflection on previous impl]:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.

[improved impl]:
def add(a: int, b: int) -> int:
   """Given integers a and b, return the total value of a and b."""
   return a + b
'''


PREV_CODE_CONTEXT = """
[previous impl]
```python
{code}
```

[previous output]
{result}
"""


PREV_CODE_CONTEXT_WITH_REFLECTION = """
[reflection on previous impl]
{reflection}

[new impl]
```python
{code}
```

[new output]
{result}

"""

# don't need [previous impl] because it will come from PREV_CODE_CONTEXT or PREV_CODE_CONTEXT_WITH_REFLECTION
DEBUG = """
[example]
Here is an example of debugging with reflection.
{debug_example}
[/example]

[context]
{context}

{previous_impl}

[instruction]
Analyze your previous code and error in [context] step by step, provide me with improved method and code. Remember to follow [context] requirement. Because you are writing code in a jupyter notebook, you can run `!pip install` to install missing packages. Output a json following the format:
```json
{{
    "reflection": str = "Reflection on previous implementation",
    "improved_impl": str = "Refined code after reflection.",
}}
```
"""


TEST = """
# Context
{context}

# Tool Info for Current Subtask
{tool_info}

# Code to Test
{code}

# Constraints
- Write code to test the functionality of the provided code according to the 'Current Subtask'. If you cannot test the code, then write code to visualize the result by calling the code.
- Always prioritize using pre-defined tools for the same functionality.
- Write clean, readable, and well-documented code.

# Output
While some concise thoughts are helpful, code is absolutely required. Always output one and only one code block in your response. Output code in the following format:
```python
your code
```
"""
