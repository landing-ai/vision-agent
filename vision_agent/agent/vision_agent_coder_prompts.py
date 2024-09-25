USER_REQ = """
## User Request
{user_request}
"""

FULL_TASK = """
## User Request
{user_request}

## Subtasks
{subtasks}
"""

FEEDBACK = """
## This contains code and feedback from previous runs and is used for providing context so you do not make the same mistake again.

{feedback}
"""


PLAN = """
**Context**:
{context}

**Tools Available**:
{tool_desc}

**Previous Feedback**:
{feedback}

**Instructions**:
1. Based on the context and tools you have available, create a plan of subtasks to achieve the user request.
2. For each subtask, be sure to include the tool(s) you want to use to accomplish that subtask.
3. Output three different plans each utilize a different strategy or set of tools ordering them from most likely to least likely to succeed.

Output a list of jsons in the following format:

```json
{{
    "plan1":
        {{
            "thoughts": str # your thought process for choosing this plan
            "instructions": [
                str # what you should do in this task associated with a tool
            ]
        }},
    "plan2": ...,
    "plan3": ...
}}
```
"""


TEST_PLANS = """
**Role**: You are a software programmer responsible for testing different tools.

**Task**: Your responsibility is to take a set of several plans and test the different tools for each plan.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`.

{docstring}

**Plans**:
{plans}

**Previous Attempts**:
{previous_attempts}

**Examples**:
--- EXAMPLE1 ---
plan1:
- Load the image from the provided file path 'image.jpg'.
- Use the 'owl_v2_image' tool with the prompt 'person' to detect and count the number of people in the image.
plan2:
- Load the image from the provided file path 'image.jpg'.
- Use the 'florence2_sam2_image' tool with the prompt 'person' to detect and count the number of people in the image.
- Count the number of detected objects labeled as 'person'.
plan3:
- Load the image from the provided file path 'image.jpg'.
- Use the 'countgd_counting' tool to count the dominant foreground object, which in this case is people.

```python
from vision_agent.tools import load_image, owl_v2_image, florence2_sam2_image, countgd_counting
image = load_image("image.jpg")
owl_v2_out = owl_v2_image("person", image)

f2s2_out = florence2_sam2_image("person", image)
# strip out the masks from the output becuase they don't provide useful information when printed
f2s2_out = [{{k: v for k, v in o.items() if k != "mask"}} for o in f2s2_out]

cgd_out = countgd_counting(image)

final_out = {{"owl_v2_image": owl_v2_out, "florence2_sam2_image": f2s2, "countgd_counting": cgd_out}}
print(final_out)
--- END EXAMPLE1 ---

--- EXAMPLE2 ---
plan1:
- Extract frames from 'video.mp4' at 10 FPS using the 'extract_frames_and_timestamps' tool.
- Use the 'owl_v2_video' tool with the prompt 'person' to detect where the people are in the video.
plan2:
- Extract frames from 'video.mp4' at 10 FPS using the 'extract_frames_and_timestamps' tool.
- Use the 'florence2_phrase_grounding' tool with the prompt 'person' to detect where the people are in the video.
plan3:
- Extract frames from 'video.mp4' at 10 FPS using the 'extract_frames_and_timestamps' tool.
- Use the 'florence2_sam2_video_tracking' tool with the prompt 'person' to detect where the people are in the video.


```python
import numpy as np
from vision_agent.tools import extract_frames_and_timestamps, owl_v2_video, florence2_phrase_grounding, florence2_sam2_video_tracking

# sample at 1 FPS and use the first 10 frames to reduce processing time
frames = extract_frames_and_timestamps("video.mp4", 1)
frames = [f["frame"] for f in frames][:10]

# strip arrays from the output to make it easier to read
def remove_arrays(o):
    if isinstance(o, list):
        return [remove_arrays(e) for e in o]
    elif isinstance(o, dict):
        return {{k: remove_arrays(v) for k, v in o.items()}}
    elif isinstance(o, np.ndarray):
        return "array: " + str(o.shape)
    else:
        return o

# return the counts of each label per frame to help determine the stability of the model results
def get_counts(preds):
    counts = {{}}
    for i, pred_frame in enumerate(preds):
        counts_i = {{}}
        for pred in pred_frame:
            label = pred["label"].split(":")[1] if ":" in pred["label"] else pred["label"]
            counts_i[label] = counts_i.get(label, 0) + 1
        counts[f"frame_{{i}}"] = counts_i
    return counts


# plan1
owl_v2_out = owl_v2_video("person", frames)
owl_v2_counts = get_counts(owl_v2_out)

# plan2
florence2_out = [florence2_phrase_grounding("person", f) for f in frames]
florence2_counts = get_counts(florence2_out)

# plan3
f2s2_tracking_out = florence2_sam2_video_tracking("person", frames)
remove_arrays(f2s2_tracking_out)
f2s2_counts = get_counts(f2s2_tracking_out)

final_out = {{
    "owl_v2_video": owl_v2_out,
    "florence2_phrase_grounding": florence2_out,
    "florence2_sam2_video_tracking": f2s2_out,
}}

counts = {{
    "owl_v2_video": owl_v2_counts,
    "florence2_phrase_grounding": florence2_counts,
    "florence2_sam2_video_tracking": f2s2_counts,
}}

print(final_out)
print(labels_and_scores)
print(counts)
```
--- END EXAMPLE2 ---

**Instructions**:
1. Write a program to load the media and call each tool and print it's output along with other relevant information.
2. Create a dictionary where the keys are the tool name and the values are the tool outputs. Remove numpy arrays from the printed dictionary.
3. Your test case MUST run only on the given images which are {media}
4. Print this final dictionary.
5. For video input, sample at 1 FPS and use the first 10 frames only to reduce processing time.
"""


PREVIOUS_FAILED = """
**Previous Failed Attempts**:
You previously ran this code:
```python
{code}
```

But got the following error or no stdout:
{error}
"""


PICK_PLAN = """
**Role**: You are an advanced AI model that can understand the user request and construct plans to accomplish it.

**Task**: Your responsibility is to pick the best plan from the three plans provided.

**Context**:
{context}

**Plans**:
{plans}

**Tool Output**:
{tool_output}

**Instructions**:
1. Re-read the user request, plans, tool outputs and examine the image.
2. Solve the problem yourself given the image and pick the most accurate plan that matches your solution the best.
3. Add modifications to improve the plan including: changing a tool, adding thresholds, string matching.
3. Output a JSON object with the following format:
{{
    "predicted_answer": str # the answer you would expect from the best plan
    "thoughts": str # your thought process for choosing the best plan over other plans and any modifications you made
    "best_plan": str # the best plan you have chosen, must be `plan1`, `plan2`, or `plan3`
}}
"""

CODE = """
**Role**: You are a software programmer.

**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is efficient, readable, and well-commented. Return the requested information from the function you create. Do not call your code, a test will be run after the code is submitted.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`.

{docstring}

**User Instructions**:
{question}

**Tool Tests and Outputs**:
{tool_output}

**Tool Output Thoughts**:
{plan_thoughts}

**Previous Feedback**:
{feedback}

**Examples**:
--- EXAMPLE1 ---
**User Instructions**:

## User Request
Can you write a program to check if each person is wearing a helmet? First detect all the people in the image, then detect the helmets, check whether or not a person is wearing a helmet if the helmet is on the worker. Return a dictionary with the count of people with helments and people without helmets. Media name worker_helmets.webp

## Subtasks

This plan uses the owl_v2_image tool to detect both people and helmets in a single pass, which should be efficient and accurate. We can then compare the detections to determine if each person is wearing a helmet.
-Use owl_v2_image with prompt 'person, helmet' to detect both people and helmets in the image
-Process the detections to match helmets with people based on bounding box proximity
-Count people with and without helmets based on the matching results
-Return a dictionary with the counts


**Tool Tests and Outputs**:
After examining the image, I can see 4 workers in total, with 3 wearing yellow safety helmets and 1 not wearing a helmet. Plan 1 using owl_v2_image seems to be the most accurate in detecting both people and helmets. However, it needs some modifications to improve accuracy. We should increase the confidence threshold to 0.15 to filter out the lowest confidence box, and implement logic to associate helmets with people based on their bounding box positions. Plan 2 and Plan 3 seem less reliable given the tool outputs, as they either failed to distinguish between people with and without helmets or misclassified all workers as not wearing helmets.

**Tool Output Thoughts**:
```python
...
```
----- stdout -----
Plan 1 - owl_v2_image:

[{{'label': 'helmet', 'score': 0.15, 'bbox': [0.85, 0.41, 0.87, 0.45]}}, {{'label': 'helmet', 'score': 0.3, 'bbox': [0.8, 0.43, 0.81, 0.46]}}, {{'label': 'helmet', 'score': 0.31, 'bbox': [0.85, 0.45, 0.86, 0.46]}}, {{'label': 'person', 'score': 0.31, 'bbox': [0.84, 0.45, 0.88, 0.58]}}, {{'label': 'person', 'score': 0.31, 'bbox': [0.78, 0.43, 0.82, 0.57]}}, {{'label': 'helmet', 'score': 0.33, 'bbox': [0.3, 0.65, 0.32, 0.67]}}, {{'label': 'person', 'score': 0.29, 'bbox': [0.28, 0.65, 0.36, 0.84]}}, {{'label': 'helmet', 'score': 0.29, 'bbox': [0.13, 0.82, 0.15, 0.85]}}, {{'label': 'person', 'score': 0.3, 'bbox': [0.1, 0.82, 0.24, 1.0]}}]

...

**Input Code Snippet**:
```python
from vision_agent.tools import load_image, owl_v2_image

def check_helmets(image_path):
    image = load_image(image_path)
    # Detect people and helmets, filter out the lowest confidence helmet score of 0.15
    detections = owl_v2_image("person, helmet", image, box_threshold=0.15)
    height, width = image.shape[:2]

    # Separate people and helmets
    people = [d for d in detections if d['label'] == 'person']
    helmets = [d for d in detections if d['label'] == 'helmet']

    people_with_helmets = 0
    people_without_helmets = 0

    for person in people:
        person_x = (person['bbox'][0] + person['bbox'][2]) / 2
        person_y = person['bbox'][1]  # Top of the bounding box

        helmet_found = False
        for helmet in helmets:
            helmet_x = (helmet['bbox'][0] + helmet['bbox'][2]) / 2
            helmet_y = (helmet['bbox'][1] + helmet['bbox'][3]) / 2

            # Check if the helmet is within 20 pixels of the person's head. Unnormalize
            # the coordinates so we can better compare them.
            if (abs((helmet_x - person_x) * width) < 20 and
                -5 < ((helmet_y - person_y) * height) < 20):
                helmet_found = True
                break

        if helmet_found:
            people_with_helmets += 1
        else:
            people_without_helmets += 1

    return {{
        "people_with_helmets": people_with_helmets,
        "people_without_helmets": people_without_helmets
    }}
```
--- END EXAMPLE1 ---

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task.
2. **Algorithm/Method Selection**: Decide on the most efficient method, use the tool outputs and tool thoughts to guide you.
3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode.
4. **Code Generation**: Translate your pseudocode into executable Python code.
    4.1. Take in the media path as an argument and load with either `load_image` or `extract_frames_and_timestamps`.
    4.2. Coordinates are always returned normalized from `vision_agent.tools`.
    4.3. Do not create dummy input or functions, the code must be usable if the user provides new media.
    4.4. Use unnormalized coordinates when comparing bounding boxes.
"""

TEST = """
**Role**: As a tester, your task is to create comprehensive test cases for the provided code. These test cases should encompass Basic and Edge case scenarios to ensure the code's robustness and reliability if possible.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`. You do not need to test these functions. Test only the code provided by the user.

{docstring}

**User Instructions**:
{question}

**Input Code Snippet**:
```python
### Please decided how would you want to generate test cases. Based on incomplete code or completed version.
{code}
```

**Instructions**:
1. Verify the fundamental functionality under normal conditions.
2. Ensure each test case is well-documented with comments explaining the scenario it covers.
3. DO NOT use any files that are not provided by the user's instructions, your test must be run and will crash if it tries to load a non-existent file.
4. DO NOT mock any functions, you must test their functionality as is.

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

**Examples**:
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

SIMPLE_TEST = """
**Role**: As a tester, your task is to create a simple test case for the provided code. This test case should verify the fundamental functionality under normal conditions.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`. You do not need to test these functions, only the code provided by the user.

{docstring}

**User Instructions**:
{question}

**Input Code Snippet**:
```python
### Please decide how would you want to generate test cases. Based on incomplete code or completed version.
{code}
```

**Previous Feedback**:
{feedback}

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
10. Print the output of the function that is being tested.
11. Use the output of the function that is being tested as the return value of the testing function.
12. Run the testing function in the end and don't assign a variable to its output.
"""


FIX_BUG = """
**Role** As a coder, your job is to find the error in the code and fix it. You are running in a notebook setting so you can run !pip install to install missing packages.

**Documentation**:
This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`.

{docstring}

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
```
{result}
```

This is previous feedback provided on the code:
{feedback}

Please fix the bug by correcting the error. Return the following JSON object followed by the fixed code in the below format:
```json
{{
    "reflections": str # any thoughts you have about the bug and how you fixed it
    "which_code": str # the code that was fixed, can only be 'code' or 'test'
}}
```

```python
# Your fixed code here
```
"""


REFLECT = """
**Role**: You are a reflection agent. Your job is to look at the original user request and the code produced and determine if the code satisfies the user's request. If it does not, you must provide feedback on how to improve the code. You are concerned only if the code meets the user request, not if the code is good or bad.

**Context**:
{context}

**Plan**:
{plan}

**Code**:
{code}

**Instructions**:
1. **Understand the User Request**: Read the user request and understand what the user is asking for.
2. **Review the Plan**: Check the plan to see if it is a viable approach to solving the user request.
3. **Review the Code**: Check the code to see if it solves the user request.
4. DO NOT add any reflections for test cases, these are taken care of.

Respond in JSON format with the following structure:
{{
    "feedback": str # the feedback you would give to the coder and tester
    "success": bool # whether the code and tests meet the user request
}}
"""
