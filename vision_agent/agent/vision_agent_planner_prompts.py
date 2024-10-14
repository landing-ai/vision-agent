USER_REQ = """
## User Request
{user_request}
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
4. Output a JSON object with the following format:
{{
    "predicted_answer": str # the answer you would expect from the best plan
    "thoughts": str # your thought process for choosing the best plan over other plans and any modifications you made
    "best_plan": str # the best plan you have chosen, must be `plan1`, `plan2`, or `plan3`
}}
"""

PLAN2 = """
**Role**: You are an expert planning agent that can understand the user request and search for a plan to accomplish it.

**Task**: As a planning agent you are required to understand the user's request and search for a plan to accomplish it. Use Chain-of-Thought approach to break down the problem, create a plan, and then provide a response. Esnure your response is clear, concise, andhelpful. You can use an interactive Pyton (Jupyter Notebok) environment, executing code with <execute_python>.

**Documentation**: this is the documentation for the functions you can use to accomplish the task:
{tool_desc}

**Example Planning**: Here are some examples of how you can search for a plan, in the examples the user output is denoted by USER, your output is denoted by AGENT and the observations after your code execution are denoted by OBSERVATION:
{examples}

**Current Planning**:
--- START PLANNING ---
{planning}
--- END PLANNING ---

**Instructions**:
1. Read the user request and the context provided.
2. Output <thinking> tags to indicate your thought process.
3. You can execute python code in the ipython notebook using <execute_python> tags. Only output one <execute_python> tag at a time.
4. Only output <finalize_plan> when you are done planning and want to end the planning process. DO NOT output <finalize_plan> with <execute_python> tags, only after OBSERVATION's.
5. DO NOT output OBSERVATION, these are reserved for showing you the output of your code execution.
7. You can only respond in the following format with a single <thinking>, <execute_python> or <finalize_plan> tag

<thinking>My thought process...</thinking>
<execute_python>code here</execute_python>
<finalize_plan>My final plan...</finalize_plan>
"""


EXAMPLE_PLAN1 = """
--- EXAMPLE1 ---
USER: Count the number of pedestrians in the image.

AGENT: <thinking>I need to gather some more information about the image.</thinking>
<execute_python>claude35_vqa('How many pedestrians are in the image?', 'drone.jpg')</execute_python>

OBSERVATION: [claude35_vqa output]: "From this aerial view of a busy urban street, it's difficult to clearly see or count individual pedestrians. The image shows a bird's eye view of a city intersection with multiple lanes of traffic, parked cars, sidewalks, and some green spaces. While there may be people in cars or on the sidewalks, they are not distinctly visible from this height and perspective. The focus of the image is more on the urban infrastructure, traffic patterns, and overall city layout rather than on individuals."]

AGENT: <thinking>I will subdivide the image into smaller regions with overlap and see if that helps.</thinking>
<execute_python>
image = load_image('drone.jpg')
height, width, _ = image.shape
mid_height = height // 2
mid_width = width // 2
overlap_height = int(mid_height * 0.1)
overlap_width = int(mid_width * 0.1)
top_left = image[:mid_height + overlap_height, :mid_width + overlap_width, :]
top_right = image[:mid_height + overlap_height, mid_width - overlap_width:, :]
bottom_left = image[mid_height - overlap_height:, :mid_width + overlap_width, :]
bottom_right = image[mid_height - overlap_height:, mid_width - overlap_width:, :]
regions = [top_left, top_right, bottom_left, bottom_right]
claude35_vqa('How many pedestrians are in each of these images?', regions)
</execute_python>

OBSERVATION: [claude35_vqa output]: "I'll count the visible pedestrians in each image:\n\nImage 1: There are approximately 5-6 pedestrians visible, mostly as small figures in the distance on the plaza area.\n\nImage 2: There are no clearly visible pedestrians in this image. The focus is on the urban landscape with cars, trees, and a road.\n\nImage 3: There are about 2-3 pedestrians visible as small figures, likely pedestrians crossing the street or walking in the parking area.\n\nImage 4: There are no clearly visible pedestrians in this image. The focus is on the busy street with many cars and a parking area."

AGENT: <thinking>Subdividing the image into smaller image shelps, I will use this approach and test different counting tools on the regions.</thinking>
<execute_python>
def subdivide_image(image):
    height, width, _ = image.shape
    mid_height = height // 2
    mid_width = width // 2
    overlap_height = int(mid_height * 0.1)
    overlap_width = int(mid_width * 0.1)
    top_left = image[:mid_height + overlap_height, :mid_width + overlap_width, :]
    top_right = image[:mid_height + overlap_height, mid_width - overlap_width:, :]
    bottom_left = image[mid_height - overlap_height:, :mid_width + overlap_width, :]
    bottom_right = image[mid_height - overlap_height:, mid_width - overlap_width:, :]
    return [top_left, top_right, bottom_left, bottom_right]

detector = get_tool_for_task('Detect the number of pedestrians in each region.', subdivide_image(image))
</execute_python>

OBSERVATION:
[get_tool_for_task output]: After examining the image, I can see it's an aerial view of a busy urban intersection with multiple lanes of traffic. There are numerous cars visible, each likely containing at least one person (the driver). While it's difficult to count exact individuals, I can estimate based on the number of vehicles. The countgd_counting tool returned an empty list, which is incorrect given the visible evidence in the image. This suggests the tool may have failed to detect people inside vehicles or had difficulty with the aerial perspective. Despite this tool being specifically requested, its output is clearly inaccurate for this task.

Tool Documentation:
countgd_counting(prompt: str, image: numpy.ndarray, box_threshold: float = 0.23) -> List[Dict[str, Any]]:
'countgd_counting' is a tool that can precisely count multiple instances of an
    object given a text prompt. It returns a list of bounding boxes with normalized
    coordinates, label names and associated confidence scores.

    Parameters:
        prompt (str): The object that needs to be counted.
        image (np.ndarray): The image that contains multiple instances of the object.
        box_threshold (float, optional): The threshold for detection. Defaults
            to 0.23.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
            bounding box of the detected objects with normalized coordinates between 0
            and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
            top-left and xmax and ymax are the coordinates of the bottom-right of the
            bounding box.

    Example
    -------
        >>> countgd_counting("flower", image)
        [
            {'score': 0.49, 'label': 'flower', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            {'score': 0.68, 'label': 'flower', 'bbox': [0.2, 0.21, 0.45, 0.5},
            {'score': 0.78, 'label': 'flower', 'bbox': [0.3, 0.35, 0.48, 0.52},
            {'score': 0.98, 'label': 'flower', 'bbox': [0.44, 0.24, 0.49, 0.58},
        ]

AGENT: <thinking>I need to now merge the boxes from all regions and count the unique pedestrians.</thinking>
<execute_python>
def translate_ofset(bbox, offset_x, offset_y):
    return (bbox[0] + offset_x, bbox[1] + offset_y, bbox[2] + offset_x, bbox[3] + offset_y)

def bounding_boxes_overlap(bbox1, bbox2):
    if bbox1[2] < bbox2[0] or bbox2[0] > bbox1[2]:
        return False
    if bbox1[3] < bbox2[1] or bbox2[3] > bbox1[3]:
        return False
    return True

def merge_bounding_boxes(bbox1, bbox2):
    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    x_max = max(bbox1[2], bbox2[2])
    y_max = max(bbox1[3], bbox2[3])
    return (x_min, y_min, x_max, y_max)

def merge_bounding_box_list(bboxes):
    merged_bboxes = []
    while bboxes:
        bbox = bboxes.pop()
        overlap_found = False
        for i, other_bbox in enumerate(merged_bboxes):
            if bounding_boxes_overlap(bbox, other_bbox):
                merged_bboxes[i] = merge_bounding_boxes(bbox, other_bbox)
                overlap_found = True
                break
        if not overlap_found:
            merged_bboxes.append(bbox)
    return merged_bboxes

detections = []
for region in subdivide_image(image):
    detections.extend(detector(region))
detections = merge_bounding_box_list(detections)
print(len(detections))
</execute_python>

OBSERVATION: 14

AGENT: <finalize_plan>This plan successfully detected 14 pedestrians in the image by subdividing the image into smaller regions and using the countgd_counting tool on each region. The bounding boxes were then merged to count the unique pedestrians.</finalize_plan>
--- END EXAMPLE1 ---
"""

TEST_TOOLS2 = """
**Role**: You are an expert software programming that specializes in tool testing.

**Task**: You are responsible for testing different tools on a set of images to determine the best tool for the user request.

**Tools**: This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`.
{tool_docs}

**Previous Attempts**: You previously tried to execute this code and got the following error or no output (if empty that means you have not tried to previously execute code, if it is 'EMPTY' that means you have tried but got no output):
{previous_attempts}

**User Request**:
{user_request}

**Media**:
{media}

**Examples**:
--- EXAMPLE1 ---
**User Request**:
Count the number of pedestrians across all the images.

**Media**:
["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]

```python
from vision_agent.tools import load_image, owl_v2_image, florence2_phrase_grounding, countgd_counting

owl_v2_results = []
florence2_results = []
countgd_results = []
for image_path in ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]:
    image = load_image(image_path)
    owl_v2_out = owl_v2_image("person", image)
    florence2_out = florence2_phrase_grounding("person", image)
    countgd_out = countgd_counting("person", image)
    owl_v2_results.extend(owl_v2_out)
    florence2_results.extend(florence2_out)
    countgd_results.extend(countgd_out)

final_results = {{
    "owl_v2_image": owl_v2_results,
    "florence2_phrase_grounding": florence2_results,
    "countgd_counting": countgd_results,
}}
print(final_results)
```
--- END EXAMPLE1 ---

**Instructions**:
1. Write a program to load the media and call each tool and print it's output along with other relevant information.
2. Create a dictionary where the keys are the tool name and the values are the tool outputs. Remove numpy arrays from the printed dictionary.
3. Your test case MUST run only on the given images which are {media}
4. Print this final dictionary.
"""

PICK_TOOL2 = """
**Role**: You are an expert evaluator that can understand user requests and evaluate the output of different tools.

**Task**: You are given the output of different tools for a user request along with the image. You must evaluate the output and determine the best tool for the user request.

**User Request**:
{user_request}

**Tools**: This is the documentation of all the functions that were tested.
{tool_docs}

**Testing Code and Tool Output**:
{context}


**Previous Attempt**: This is the code and output of the previous attempt, if it is empty then there was no previous attempt.
{previous_attempts}

**Instructions**:
1. Re-read the user request, plans, tool outputs and examine the image.
2. Solve the problem yourself given the image and pick the most accurate tool that matches your solution the best.
3. Output a JSON object with the following format:
{{
    "predicted_answer": str # the answer you would expect from the best plan
    "thoughts": str # your thought process for choosing the best plan over other plans and any modifications you made
    "tool": str # the name of the tool you have chosen, must be from the **Tools** section
}}
"""

FINALIZE_PLAN = """
**Role**: You are an expert AI model that can understand the user request and construct plans to accomplish it.

**Task**: You are given a chain of thoughts, python executions and observations from a planning agent as it tries to construct a plan to solve a user request. Your task is to summarize the plan it found so that another programming agnet to write a program to accomplish the user request.

**Planning**: Here is chain of thoughts, executions and observations from the planning agent:
{planning}

**Instructions**:
1. Read the chain of thoughts and python executions.
2. Summarize the plan that the planning agent found.
3. Include any relevant python code in your plan.
4. Specifically call out the tools used and the order in which they were used. Only include tools obtained from calling `get_tool_for_task`.
5. Do not include {excluded_tools} tools in your instructions.
6. Respond in the following JSON format followed by the code:
{{
    "plan": str # the plan you have summarized
    "instructions": [
        str # the instructions for each step in the plan with either specific code or a specific tool name
    ]
}}
<code>
# Code snippets here
</code>
"""

FIX_BUG = """
**Role**: You are a software programmer responsible for fixing bugs in code.

**Task**: Your responsibility is to fix the bug in the code and provide the correct output.

**Instructions**:

This is the code that was previously run:
```python
{code}
```

And this is the error message that was received:
{error}

Please fix the code by correcting the error. Output your code in the following format:
```python
# Your fixed code here
```
"""
