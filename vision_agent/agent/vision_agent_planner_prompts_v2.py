PLAN = """
**Role**: You are an expert planning agent that can understand the user request and search for a plan to accomplish it.

**Task**: As a planning agent you are required to understand the user's request and search for a plan to accomplish it. Use Chain-of-Thought approach to break down the problem, create a plan, and then provide a response. Esnure your response is clear, concise, and helpful. You can use an interactive Pyton (Jupyter Notebok) environment but do not !pip install packages, execute code with <execute_python>, each execution is a new cell so old code and outputs are saved.

**Documentation**: this is the documentation for the functions you can use to accomplish the task:
{tool_desc}

**Example Planning**: Here are some examples of how you can search for a plan, in the examples the user output is denoted by USER, your output is denoted by AGENT and the observations after your code execution are denoted by OBSERVATION:
{examples}

**Current Planning**: This is the plan you are currently working on
--- START CURRENT PLANNING ---
{planning}
--- END CURRENT PLANNING ---

**Instructions**:
1. Read over the user request and context provided and output <thinking> tags to indicate your thought process. You can <count> number of turns to complete the user's request.
2. You can execute python code in the ipython notebook using <execute_python> tags. Only output one <execute_python> tag at a time.
3. Only output <finalize_plan> when you are done planning and want to end the planning process. DO NOT output <finalize_plan> with <execute_python> tags, only after OBSERVATION's.
4. Only load/save files from {media_list} unless you specifically saved the file previously.
5. Ensure you always call `suggestion` and `claude35_vqa` initially and `get_tool_for_task` to get the right tool for the subtask.
6. Calling `plt.imshow` or `save_image` will display the image to you so you can check your results. If you see an image after <execute_python> it's generated from your code.
7. Be sure to print results returned for tools so you can see the output.
8. DO NOT hard code the answer into your code, it should be dynamic and work for any similar request.
9. DO NOT over index on claude35_vqa, if tool output is close to claude35_vqa's output you do not need to improve the tool output, tools are often better at things like counting and detecting small objects.
10. You can only respond in the following format with a single <thinking>, <execute_python> or <finalize_plan> tag:

<thinking>Your thought process...</thinking>
<execute_python>Your code here</execute_python>
<finalize_plan>Your final recommended plan</finalize_plan>
"""

EXAMPLE_PLAN1 = """
--- EXAMPLE1 ---
USER: Count the number of pedestrians in the image.
<count>10</count>

AGENT: <thinking>I need to gather more information, I will ask for a description of the image and a solution to the problem to work towards as well as a suggestion on how best to solve it.</thinking>
<execute_python>
image = load_image('drone.jpg')
claude35_vqa('Can you describe this image? How many pedestrians do you count in the image?', [image])
suggestion('How can I count the number of pedestrians in the image?', [image])
</execute_python>

OBSERVATION: [claude35_vqa output]
From this aerial view of a busy urban street, it's difficult to clearly see or count individual pedestrians. The image shows a bird's eye view of a city intersection with multiple lanes of traffic, parked cars, sidewalks, and some green spaces. While there may be people in cars or on the sidewalks, they are not distinctly visible from this height and perspective. The focus of the image is more on the urban infrastructure, traffic patterns, and overall city layout rather than on individuals.
[end of claude35_vqa output]

[suggestions]
[suggestion 0]
The image is very large and the items you need to detect are small.

Step 1: You should start by splitting the image into overlapping sections and runing the detection algorithm on each section:

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

get_tool_for_task('<your prompt here>', subdivide_image(image))

Step 2: Once you have the detections from each subdivided image, you will need to merge them back together to remove overlapping predictions, be sure to tranlate the offset back to the original image:

def bounding_box_match(b1: List[float], b2: List[float], iou_threshold: float = 0.1) -> bool:
    # Calculate intersection coordinates
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    # Calculate intersection area
    if x2 < x1 or y2 < y1:
        return False  # No overlap

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union area
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0

    return iou >= iou_threshold

def merge_bounding_box_list(detections):
    merged_detections = []
    for detection in detections:
        matching_box = None
        for i, other in enumerate(merged_detections):
            if bounding_box_match(detection["bbox"], other["bbox"]):
                matching_box = i
                break

        if matching_box is not None:
            # Keep the box with higher confidence score
            if detection["score"] > merged_detections[matching_box]["score"]:
                merged_detections[matching_box] = detection
        else:
            merged_detections.append(detection)

def sub_image_to_original(elt, sub_image_position, original_size):
    offset_x, offset_y = sub_image_position
    return {
        "label": elt["label"],
        "score": elt["score"],
        "bbox": [
            (elt["bbox"][0] + offset_x) / original_size[1],
            (elt["bbox"][1] + offset_y) / original_size[0],
            (elt["bbox"][2] + offset_x) / original_size[1],
            (elt["bbox"][3] + offset_y) / original_size[0],
        ],
    }

def normalized_to_unnormalized(elt, image_size):
    return {
        "label": elt["label"],
        "score": elt["score"],
        "bbox": [
            elt["bbox"][0] * image_size[1],
            elt["bbox"][1] * image_size[0],
            elt["bbox"][2] * image_size[1],
            elt["bbox"][3] * image_size[0],
        ],
    }

height, width, _ = image.shape
mid_width = width // 2
mid_height = height // 2

detection_from_subdivided_images = []
for i, sub_image in enumerate(subdivided_images):
    detections = <your detection function here>("pedestrian", sub_image)
    unnorm_detections = [
        normalized_to_unnormalized(
            detection, (sub_image.shape[0], sub_image.shape[1])
        )
        for detection in detections
    ]
    offset_x = i % 2 * (mid_width - int(mid_width * 0.1))
    offset_y = i // 2 * (mid_height - int(mid_height * 0.1))
    offset_detections = [
        sub_image_to_original(
            unnorm_detection, (offset_x, offset_y), (height, width)
        )
        for unnorm_detection in unnorm_detections
    ]
    detection_from_subdivided_images.extend(offset_detections)

detections = merge_bounding_box_list(detection_from_subdivided_images)
[end of suggestion 0]
[end of suggestion]
<count>9</count>

AGENT: <thinking>Following the suggested plan, I will subdivide the image into smaller parts and find the right tool on the smaller parts. I will also re-check with claude35_vqa on the subdivided images.</thinking>
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

images = subdivide_image(image)
claude35_vqa('How many pedestrians do you see in these images?', images)
get_tool_for_task('Count the number of pedestrians in the images', images)
</execute_python>

OBSERVATION:
[claude35_vqa_output]
In these aerial images, I can see approximately 5-6 pedestrians walking in various locations, including near the retail signage are and along the sidewalks.
[end of claude35_vqa_output]

[get_tool_for_task output]
After examining the image, I can see it's an aerial view of a busy urban intersection with multiple lanes of traffic. There are numerous cars visible, each likely containing at least one person (the driver). While it's difficult to count exact individuals, I can estimate based on the number of vehicles. The countgd_object_detection tool is the best choice for this task as it can detect and count multiple instances of an object given a text prompt. I will use this tool with the prompt 'pedistrian' to count the number of pedestrians in the image.

Tool Documentation:
countgd_object_detection(prompt: str, image: numpy.ndarray, box_threshold: float = 0.23) -> List[Dict[str, Any]]:
    'countgd_object_detection' is a tool that can detect multiple instances of an
    object given a text prompt. It is particularly useful when trying to detect and
    count a large number of objects. You can optionally separate object names in the
    prompt with commas. It returns a list of bounding boxes with normalized
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
        >>> countgd_object_detection("flower", image)
        [
            {'score': 0.49, 'label': 'flower', 'bbox': [0.1, 0.11, 0.35, 0.4]},
        ]
[end of get_tool_for_task output]
<count>8</count>

AGENT: <thinking>I need to now merge the boxes from all region and use the countgd_object_detection tool with the prompt 'pedestrian' as suggested by get_tool_for_task.</thinking>
<execute_python>
def bounding_box_match(b1: List[float], b2: List[float], iou_threshold: float = 0.1) -> bool:
    # Calculate intersection coordinates
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    # Calculate intersection area
    if x2 < x1 or y2 < y1:
        return False  # No overlap

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union area
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0

    return iou >= iou_threshold

def merge_bounding_box_list(detections):
    merged_detections = []
    for detection in detections:
        matching_box = None
        for i, other in enumerate(merged_detections):
            if bounding_box_match(detection["bbox"], other["bbox"]):
                matching_box = i
                break

        if matching_box is not None:
            # Keep the box with higher confidence score
            if detection["score"] > merged_detections[matching_box]["score"]:
                merged_detections[matching_box] = detection
        else:
            merged_detections.append(detection)

detections = []
for region in subdivide_image(image):
    detections.extend(countgd_object_detection("pedestrian", region))
detections = merge_bounding_box_list(detections)
print(len(detections))
</execute_python>

OBSERVATION: 24
<count>7</count>

AGENT: <thinking>countgd_object_detection detected 24 pedestrians in the image by subdividing the image into smaller regions, this differs from claude35_vqa's answer of 5-6 pedestrians. I will use countgd_object_detection's answer since it is specifically designed to count objects.</thinking>
<finalize_plan>This plan successfully detected 24 pedestrians in the image by subdividing the image into smaller regions and using the countgd_object_detection tool on each region. The bounding boxes were then merged to count the unique pedestrians.</finalize_plan>
--- END EXAMPLE1 ---
"""

EXAMPLE_PLAN2 = """
--- EXAMPLE2 ---
USER: Can you identify which boxes have tape on them in this video?
<count>10</count>

AGENT: <thinking>I need to gather more information, I will ask for a description of the video and a solution to the problem to work towards as well as a suggestion on how best to solve it.</thinking>
<execute_python>
frames = extract_frames_and_timestamps('video.mp4', 1)
frames  = [f["frame"] for f in frames]
claude35_vqa('Can you describe this video? How many boxes are there and do they have tape on them?', frames[:5])
suggestion('How can I identify boxes with tape on them in the video?', frames[:5])
</execute_python>

OBSERVATION: [claude35_vqa output]
In this video, there appear to be 2 cardboard boxes moving along a roller conveyor system. The boxes are brown/kraft colored and appear to have tape sealing them. You can see clear packing tape on the top and sides of the boxes.
[end of claude35_vqa output]

[suggestions]
[suggestion 0]
You are trying to track objects in a video and identify features on those objects. You need to first get a tool that can track objects in a video, and then for each object find another tool to identify the features on the object. You can use the following code to help with this task:

track_predictions = <object tracking tool>("object", video_frames)


# Step 1: go through each frame and each prediction and extract the predicted bounding boxes as crops
obj_to_info = {{}}
for frame, frame_predictions in zip(video_frames, track_predictions):
    for obj in frame_predictions:
        if obj["label"] not in obj_to_info:
            obj_to_info[obj["label"]] = []
        height, width = frame.shape[:2]
        crop = frame[
            int(obj["bbox"][1] * height) : int(obj["bbox"][3] * height),
            int(obj["bbox"][0] * width) : int(obj["bbox"][2] * width),
            :,
        ]
        # For each crop use an object detection tool, VQA tool or classification tool to identify if the object contains the features you want
        detections = <object detection tool to find features>("<your feature here>", crop)
        obj_to_info[obj["label"]].extend(detections)

print(f"{len(obj_to_info)} objects tracked")

objects_with_info = set()
for infos in obj_to_info:
    for info in info:
        if info["label"] == "<your feature here>":
            objects_with_info.add(info)
            break

print(f"{len(objects_with_info)} objects with features found")
[end of suggestion 0]
[end of suggestions]
<count>9</count>

AGENT: <thinking>Following the suggested plan, I first need to identify a tool for tracking.</thinking>
<execute_python>
get_tool_for_task('Identify and track the boxes in the video', frames[:5])
</execute_python>

OBSERVATION:
[get_tool_for_task output]
For tracking boxes moving on a conveyor belt, we need a tool that can consistently track the same box across frames without losing it or double counting. Looking at the outputs: countgd_sam2_video_tracking successfully tracks the single box across all 5 frames, maintaining consistent tracking IDs and showing the box's movement along the conveyor and using the prompt 'box'.

Tool Documentation:
def countgd_sam2_video_tracking(prompt: str, frames: List[np.ndarray], chunk_length: Optional[int] = 25) -> List[List[Dict[str, Any]]]:
    'countgd_sam2_video_tracking' is a tool that can track and segment multiple
    objects in a video given a text prompt such as category names or referring
    expressions. The categories in the text prompt are separated by commas. It returns
    a list of bounding boxes, label names, masks and associated probability scores and
    is useful for tracking and counting without duplicating counts.

    Parameters:
        prompt (str): The prompt to ground to the image.
        frames (List[np.ndarray]): The list of frames to ground the prompt to.
        chunk_length (Optional[int]): The number of frames to re-run countgd to find
            new objects.

    Returns:
        List[List[Dict[str, Any]]]: A list of list of dictionaries containing the
            label, segmentation mask and bounding boxes. The outer list represents each
            frame and the inner list is the entities per frame. The detected objects
            have normalized coordinates between 0 and 1 (xmin, ymin, xmax, ymax). xmin
            and ymin are the coordinates of the top-left and xmax and ymax are the
            coordinates of the bottom-right of the bounding box. The mask is binary 2D
            numpy array where 1 indicates the object and 0 indicates the background.
            The label names are prefixed with their ID represent the total count.

    Example
    -------
        >>> countgd_sam2_video_tracking("car, dinosaur", frames)
        [
            [
                {
                    'label': '0: dinosaur',
                    'bbox': [0.1, 0.11, 0.35, 0.4],
                    'mask': array([[0, 0, 0, ..., 0, 0, 0],
                        ...,
                        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
                },
            ],
            ...
        ]
[end of get_tool_for_task output]
<count>8</count>

AGENT: <thinking>Now that I have the tool and prompt for tracking I need to find a tool to identify if there is tape on each of the boxes.</thinking>
<execute_python>
track_predictions = countgd_sam2_video_tracking("box", frames)

# only use first 5 frames to reduce processing time
crops = []
for frame, frame_predictions in zip(frames[:5], track_predictions[:5]):
    for obj in frame_predictions:
        height, width = frame.shape[:2]
        crop = frame[
            int(obj["bbox"][1] * height) : int(obj["bbox"][3] * height),
            int(obj["bbox"][0] * width) : int(obj["bbox"][2] * width),
            :,
        ]
        crops.append(crop)

# only use first 5 crops to reduce processing time
get_tool_for_task('Identify if there is tape on the boxes', crops[:3])

OBSERVATION:
[get_tool_for_task output]
owlv2_object_detection performed best with the prompt 'tape', as it specifically detected multiple instances of tape with localized bounding boxes, which matches what's visible in the images.

'owlv2_object_detection' is a tool that can detect and count multiple objects given a
text prompt such as category names or referring expressions on images. The categories
in text prompt are separated by commas. It returns a list of bounding boxes with
normalized coordinates, label names and associated probability scores.

Parameters:
    prompt (str): The prompt to ground to the image.
    image (np.ndarray): The image to ground the prompt to.
    box_threshold (float, optional): The threshold for the box detection. Defaults
        to 0.10.
    fine_tune_id (Optional[str]): If you have a fine-tuned model, you can pass the
        fine-tuned model ID here to use it.

Returns:
    List[Dict[str, Any]]: A list of dictionaries containing the score, label, and
        bounding box of the detected objects with normalized coordinates between 0
        and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the
        top-left and xmax and ymax are the coordinates of the bottom-right of the
        bounding box.

Example
-------
    >>> owlv2_object_detection("car, dinosaur", image)
    [
        {'score': 0.99, 'label': 'dinosaur', 'bbox': [0.1, 0.11, 0.35, 0.4]},
        {'score': 0.98, 'label': 'car', 'bbox': [0.2, 0.21, 0.45, 0.5},
    ]
[end of get_tool_for_task output]
<count>7</count>

AGENT: <thinking>Now that I have the tools for tracking and identifying 'tape' on the boxes, I will use them to accomplish the user's request.</thinking>
<execute_python>
obj_to_info = {{}}
for frame, frame_predictions in zip(frames, track_predictions):
    for obj in frame_predictions:
        if obj["label"] not in obj_to_info:
            obj_to_info[obj["label"]] = []
        height, width = frame.shape[:2]
        crop = frame[
            int(obj["bbox"][1] * height) : int(obj["bbox"][3] * height),
            int(obj["bbox"][0] * width) : int(obj["bbox"][2] * width),
            :,
        ]
        detections = owlv2_object_detection("tape", crop)
        obj_to_info[obj["label"]].extend(detections)


print(f"{len(obj_to_info)} boxes tracked")
objects_with_tape = set()
for infos in obj_to_info:
    for info in info:
        if info["label"] == "tape":
            objects_with_tape.add(info)
            break

print(f"{len(objects_with_tape)} boxes with tape found")
</execute_python>

OBSERVATION:
3 boxes were tracked
2 boxes with tape found
<count>6</count>

AGENT: <thinking>After tracking the boxes and identifying tape on them, I found 3 boxes in the video and 2 of them had tape on them. I will finalize this plan.</thinking>
<finalize_plan>This plan successfully tracked the boxes in the video and identified tape on them. It found 3 boxes in the video and 2 of them had tape on them.</finalize_plan>
--- END EXAMPLE2 ---
"""

PICK_PLAN = """
**Role**: You are an expert evaluator that can understand user requests and evaluate different plans to determine the best one.

**Task**: You are given a planning chains by a planning agent complete with thoughts, python executions and observations. You are also given 3 different responses to add to the chain. Your task is to evaluate the different plans and determine the best one for the user request.

**Current Planning**:
--- START PLANNING ---
{planning}
--- END PLANNING ---

**Responses**:
--- RESPONSE1 ---
{response1}
--- END RESPONSE1 ---
--- RESPONSE2 ---
{response2}
--- END RESPONSE2 ---
--- RESPONSE3 ---
{response3}
--- END RESPONSE3 ---

**Instructions**:
1. Read over the planning chain and responses.
2. The agent can output <thinking>, <execute_python> or <finalize_plan> tags. Any code under OBSERVATION: has not been executed by the agent, only code under <execute_python> tags has been executed.
3. Rate each instruction on a scale of 1 to 10 based on how well it solves the user request and give your thoughts on why you rated it that way.
4. Return your results in the following JSON format inside <json> tags:

<json>
{{
    "response1": # int: rating for response1,
    "response1_thoughts": # str: your thoughts on response1,
    "response2": # int: rating for response2,
    "response2_thoughts": # str: your thoughts on response2,
    "response3": # int: rating for response3,
    "response3_thoughts": # str: your thoughts on response3
}}
</json>
"""

CATEGORIZE_TOOL_REQUEST = """
You are given a task: "{task}" from the user. You must extract the type of category this task belongs to, it can be one or more of the following:
- "VQA" - answering questions about an image or video, can be used for most tasks, should generally be included.
- "object detection and counting" - detecting objects or counting objects from a text prompt in an image.
- "instance segmentation" - segmenting objects in an image given a text prompt.
- "classification" - classifying objects in an image given a text prompt.
- "segmentation" - segmenting objects in an image or video given a text prompt.
- "OCR" - extracting text from an image.
- "DocQA" - answering questions about a document or extracting information from a document.
- "video object tracking" - tracking objects in a video.
- "depth and pose estimation" - estimating the depth or pose of objects in an image.
- "activity recognition" - identifying time period(s) an event occurs in a video.
- "inpainting" - filling in masked parts of an image.

Return the category or categories (comma separated) inside tags <category># your categories here</category>. If you are unsure about a task, it is better to include more categories than less.
"""

TEST_TOOLS = """
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
{examples}

**Instructions**:
1. List all the tools under **Tools** and the user request. Write a program to load the media and call the most relevant tools in parallel and print it's output along with other relevant information.
2. Create a dictionary where the keys are the tool name and the values are the tool outputs. Remove numpy arrays from the printed dictionary.
3. Your test case MUST run only on the given images which are {media}
4. For video tracking, use chunk_length=1 and at least 3 frames to ensure the best results when evaluating the tool.
5. Use mutually exclusive categories for prompts such as 'person, car' and not 'person, athlete' to avoid over counting.
6. Print this final dictionary.
7. Output your code in the following format wrapped in <code> tags:
<code>
# Your code here
</code>
"""

TEST_TOOLS_EXAMPLE1 = """
--- EXAMPLE1 ---
**User Request**:
Count the number of pedestrians across all the images.

**Media**:
["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]

<code>
from concurrent.futures import ThreadPoolExecutor, as_completed
from vision_agent.tools import load_image, owlv2_object_detection, florence2_object_detection, countgd_object_detection

# process functions in a try catch so that if it fails it doesn't cause `as_completed` to hang
def process_owlv2(image_paths):
    try:
        results = []
        for image_path in image_paths:
            image = load_image(image_path)
            results.extend(owlv2_object_detection("person", image))
    except Exception as e:
        results = f"Encountered error when executing process_owlv2: {str(e)}"
    return results

def process_florence2(image_paths):
    try:
        results = []
        for image_path in image_paths:
            image = load_image(image_path)
            results.extend(florence2_object_detection("person", image))
    except Exception as e:
        results = f"Encountered error when executing process_florence2: {str(e)}"
    return results

def process_countgd(image_paths):
    try:
        results = []
        for image_path in image_paths:
            image = load_image(image_path)
            results.extend(countgd_object_detection("person", image))
    except Exception as e:
        results = f"Encountered error when executing process_countgd: {str(e)}"
    return results

image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]

with ThreadPoolExecutor() as executor:
    futures = {{
        executor.submit(process_owlv2, image_paths): "owlv2_object_detection",
        executor.submit(process_florence2, image_paths): "florence2_phrase_grounding",
        executor.submit(process_countgd, image_paths): "countgd_object_detection",
    }}

    final_results = {{}}
    for future in as_completed(futures):
        tool_name = futures[future]
        final_results[tool_name] = future.result()

print(final_results)
</code>
--- END EXAMPLE1 ---
"""

TEST_TOOLS_EXAMPLE2 = """
--- EXAMPLE2 ---
**User Request**:
Count the number of people in the video.

**Media**:
["video.mp4"]

<code>
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from vision_agent.tools import extract_frames_and_timestamps, owlv2_sam2_video_tracking, florence2_sam2_video_tracking

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

def process_owlv2_sam2_video_tracking(frames):
    try:
        # run with chunk_length=1 to ensure best results
        results = owlv2_sam2_video_tracking("person", frames, chunk_length=1)
    except Exception as e:
        results = f"Encountered error when executing process_owlv2_sam2_video_tracking: {str(e)}"
    return results

def process_florence2_sam2_video_tracking(frames):
    try:
        # run with chunk_length=1 to ensure best results
        results = florence2_sam2_video_tracking("person", frames, chunk_length=1)
    except Exception as e:
        results = f"Encountered error when executing process_florence2_sam2: {str(e)}"
    return results


with ThreadPoolExecutor() as executor:
    futures = {{
        executor.submit(process_owlv2_sam2_video_tracking, frames): "owlv2_sam2_video_tracking",
        executor.submit(process_florence2_sam2_video_tracking, frames): "florence2_sam2_video_tracking",
    }}
    final_results = {{}}
    for future in as_completed(futures):
        tool_name = futures[future]
        final_results[tool_name] = remove_arrays(future.result())

print(final_results)
</code>
--- END EXAMPLE2 ---
"""

PICK_TOOL = """
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
2. Given the user request, try to solve the problem yourself.
3. Pick which tool output best matches your solution and the user request, DO NOT focus on other factors.
4. DO NOT modify confidence thresholds unless the tool output is completely wrong.
5. Remember for videos that in order to count objects a video some sort of tracking is needed, or else you will overcount the objects.
7. Return the following JSON format inside <json> tags using the exact tool name for best_tool:
<json>
{{
    "predicted_answer": str # the answer you would expect from the best plan
    "thoughts": str # your thought process for choosing the best tool over other tools and any modifications you madeas well as the prompt you used with the tool.
    "best_tool": str # the name of the best tool
}}
</json>
"""

FINALIZE_PLAN = """
**Task**: You are given a chain of thoughts, python executions and observations from a planning agent as it tries to construct a plan to solve a user request. Your task is to summarize the plan it found so that another programming agent to write a program to accomplish the user request.

**Documentation**: You can use these tools to help you visualize or save the output (they are imported `from vision_agent.tools import *`):
{tool_desc}

**Planning**: Here is chain of thoughts, executions and observations from the planning agent:
{planning}

**Instructions**:
1. Summarize the plan that the planning agent found.
2. Write a single function that solves the problem based on what the planner found and only returns the final solution.
3. Only use tools obtained from calling `get_tool_for_task`.
4. Do not include {excluded_tools} tools in your instructions.
5. Ensure the function is well documented and easy to understand.
6. Ensure you visualize the output with `overlay_bounding_boxes` or `overlay_segmentation_masks`, if bounding boxes or segmentaiton masks are produced, and save it to a file with `save_image` or `save_video`.
7. Use the default FPS for extracting frames from videos unless otherwise specified by the user.
8. Include the expected answer in your 'plan' so that the programming agent can properly test if it has the correct answer.
9. Respond in the following format with JSON surrounded by <json> tags and code surrounded by <code> tags:

<json>
{{
    "plan": str # the plan you have summarized
    "instructions": [
        str # the instructions for each step in the plan with either specific code or a specific tool name
    ]
}}
</json>

<code>
# Code snippets here
</code>
"""

FIX_BUG = """
**Role**: You are a software programmer responsible for fixing bugs in code.

**Task**: Your responsibility is to fix the bug in the code and provide the correct output.

**Instructions**: This is the previous chat history:
--- CHAT HISTORY ---
{chat_history}
--- END CHAT HISTORY ---

This is the code that was previously run:
<python>
{code}
</python>

And this is the error message that was received:
<error>
{error}
</error>

Please fix the code by correcting the error. Do not wrap failures in try-except statements and instead solve the root cause. Output your code in the following format wrapped in <code> tags:

<code>
# Your fixed code here
</code>
"""

CRITIQUE_PLAN = """
**Role**: You are an expert evaluator that can understand user requests and evaluate different plans to determine the best one.

**Task**: You are given a planning chain by a planning agent complete with thoughts, python executions and observations. The agent is trying to use tools to accomplish a vision task requested by a user. Your job is to evaluate the plan so far given the user request and image.


**Current Planning**:
--- START PLANNING ---
{planning}
--- END PLANNING ---

**Instructions**:
1. Read over the planning chain and examine the image and user request. Try to solve the user problem yourself and compare it against the solution the agent has come up with so far.
2. Rate the plan on a scale of 1 to 10 based on how well it solves the user request. A score of 8+ means the plan is good and the agent should continue on the same path, 5-7 means the plan is good but needs some minor modifications, less than 5 means the plan is bad and the agent should consider a different approach.
3. Provide your score and thoughts in the following format:

<score>Your score here</score>
<thoughts>Your thoughts here</thoughts>
"""
