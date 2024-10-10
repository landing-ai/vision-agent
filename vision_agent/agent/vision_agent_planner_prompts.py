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
