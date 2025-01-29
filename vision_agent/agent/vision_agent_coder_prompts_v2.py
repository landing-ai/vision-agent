FEEDBACK = """
## This contains code and feedback from previous runs and is used for providing context so you do not make the same mistake again.

{feedback}
"""


CODE = """
**Role**: You are an expert software programmer.

**Task**: You are given a plan by a planning agent that solves a vision problem posed by the user. You are also given code snippets that the planning agent used to solve the task. Your job is to organize the code so that it can be easily called by the user to solve the task.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`.

{docstring}

**User Instructions**:
{question}

**Plan**:
--- START PLAN ---
{plan}
--- END PLAN ---

**Instructions**:
1. Reread the plan and all code and understand the task.
2. Organize the code snippets into a single function that can be called by the user.
3. DO NOT alter the code logic and ensure you utilize all the code provided as is without changing it.
4. DO NOT create dummy input or functions, the code must be usable if the user provides new media.
5. DO NOT hardcode the output, the function must work for any media provided by the user.
6. Ensure the function is well-documented and follows the best practices and returns the expected output from the user.
7. Output your code using <code> tags:

<code>
# your code here
</code>
"""


TEST = """
**Role**: As a tester, your task is to create a simple test case for the provided code. This test case should verify the fundamental functionality under normal conditions.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`. You do not need to test these functions, only the code provided by the user.

{docstring}

**User Instructions**:
{question}

**Input Code Snippet**:
<code>
### Please decide how would you want to generate test cases. Based on incomplete code or completed version.
{code}
</code>

**Instructions**:
1. Verify the fundamental functionality under normal conditions.
2. Ensure each test case is well-documented with comments explaining the scenario it covers.
3. Your test case MUST run only on the given images which are {media}
4. Your test case MUST run only with the given values which is available in the question - {question}
5. DO NOT use any non-existent or dummy image or video files that are not provided by the user's instructions.
6. DO NOT mock any functions, you must test their functionality as is.
7. DO NOT assert the output value, run the code and assert only the output format or data structure.
8. DO NOT use try except block to handle the error, let the error be raised if the code is incorrect.
9. DO NOT import the testing function as it will available in the testing environment.
10. Print the output of the function that is being tested and ensure it is not empty.
11. Use the output of the function that is being tested as the return value of the testing function.
12. Run the testing function in the end and don't assign a variable to its output.
13. Output your test code using <code> tags:

<code>
# your test code here
</code>
"""


FIX_BUG = """
**Role** As a coder, your job is to find the error in the code and fix it. You are running in a notebook setting but do not run !pip install to install new packages.

**Task**: A previous agent has written some code and some testing code according to a plan given to it. It has introduced a bug into it's code while trying to implement the plan. You are given the plan, code, test code and error. Your job is to fix the error in the code or test code.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`.

{docstring}


**Plan**:
--- START PLAN ---
{plan}
--- END PLAN ---

**Instructions**:
Please re-complete the code to fix the error message. Here is the current version of the CODE:
<code>
{code}
</code>

When we run the TEST code:
<test>
{tests}
</test>

It raises this error, if the error is empty it means the code and tests were not run:
<error>
{result}
</error>

This is from your previous attempt to fix the bug, if it is empty no previous attempt has been made:
{debug}

Please fix the bug by correcting the error. ONLY change the code logic if it is necessary to fix the bug. Do not change the code logic for any other reason. Output your fixed code using <code> tags and fixed test using <test> tags:

<thoughts>Your thoughts here...</thoughts>
<code># your fixed code here</code>
<test># your fixed test here</test>
"""
