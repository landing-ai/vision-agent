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
3. Output a JSON object with the following format:
{{
    "predicted_answer": str # the answer you would expect from the best plan
    "thoughts": str # your thought process for choosing the best plan over other plans and any modifications you made
    "best_plan": str # the best plan you have chosen, must be `plan1`, `plan2`, or `plan3`
}}
"""

PLAN2 = """
**Role**: You are an expert planning agent that can understand the user request and discover a plan to accomplish it.

**Task**: You are given a list of tools that can be used to solve the users request. You must search through different subtasks and tools to figure out the best plan. You may also use tool evaluators to help you decide which tools to use.

**Tools**:
{tool_desc}

**Tool Evaluators**:
{tool_evaluators_desc}

**Examples**:

USER: Count the number of people in the image.

AGENT: <thinking>I need to gather some more information about the image.</thinking>
<execute_python>claude35_vqa('How many people are in the image?', 'drone.jpg')</execute_python>

OBSERVATION: "From this aerial view of a busy urban street, it's difficult to clearly see or count individual people. The image shows a bird's eye view of a city intersection with multiple lanes of traffic, parked cars, sidewalks, and some green spaces. While there may be people in cars or on the sidewalks, they are not distinctly visible from this height and perspective. The focus of the image is more on the urban infrastructure, traffic patterns, and overall city layout rather than on individuals."

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
claude35_vqa('How many people are in each of these images?', regions)
</execute_python>

OBSERVATION: "I'll count the visible people in each image:\n\nImage 1: There are approximately 5-6 people visible, mostly as small figures in the distance on the plaza area.\n\nImage 2: There are no clearly visible people in this image. The focus is on the urban landscape with cars, trees, and a road.\n\nImage 3: There are about 2-3 people visible as small figures, likely pedestrians crossing the street or walking in the parking area.\n\nImage 4: There are no clearly visible people in this image. The focus is on the busy street with many cars and a parking area."

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

counter = get_tool('Count the number of people in each region.', subdivide_image(image))
</execute_python>

OBSERVATION: "countgd_counting is the best tool for this task, it detects a total of 14 people across all regions. It returns a list of bounding boxes, for example: [{'label': 'person', 'score': 0.99, 'box': [x1, y1, x2, y2]}, ...]."

AGENT:
"""

TEST_PLANS2 = """
**Role**: You are an expert software programming that specializes in tool testing.

**Task**: You are responsible for testing different tools on a set of images to determine the best tool for the user request.

**Tools**: This is the documentation for the functions you have access to. You may call any of these functions to help you complete the task. They are available through importing `from vision_agent.tools import *`.
{tool_docs}

**Previous Attempts**:
{previous_attempts}

**User Request**:
{user_request}

**Media**:
{media}

**Examples**:
--- EXAMPLE1 ---
**User Request**:
Count the number of people across all the images.

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

final_results = {
    "owl_v2_image": owl_v2_results,
    "florence2_phrase_grounding": florence2_results,
    "countgd_counting": countgd_results,
}
print(final_results)
```
--- END EXAMPLE1 ---

**Instructions**:
1. Write a program to load the media and call each tool and print it's output along with other relevant information.
2. Create a dictionary where the keys are the tool name and the values are the tool outputs. Remove numpy arrays from the printed dictionary.
3. Your test case MUST run only on the given images which are {media}
4. Print this final dictionary.
"""

PICK_TOOL = """
**Role**: You are an expert evaluator that can understand user requests and evaluate the output of different tools.

**Task**: You are given the output of different tools for a user request along with the image. You must evaluate the output and determine the best tool for the user request.

**User Request**:
{user_request}

**Testing Code and Tool Output**:
{context}

**Instructions**:
1. Re-read the user request, plans, tool outputs and examine the image.
2. Solve the problem yourself given the image and pick the most accurate tool that matches your solution the best.
3. Return the 
"""
