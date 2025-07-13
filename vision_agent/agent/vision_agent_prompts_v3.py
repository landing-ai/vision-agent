TOOLS = """
load_image(image_path: str) -> np.ndarray:
    A function that loads an image from a file path and returns it as a numpy array.

instance_segmentation(prompt: str, image: np.ndarray, threshold: float = 0.23, nms_threshold: float = 0.5) -> list[dict[str, str | float | list[float] | np.ndarray]]:
    A function that takes a prompt and an image and returns a list of dictionaries.
    Each dictionary represents an object detected in the image and contains it's label,
    confidence score, bounding box coordinates and mask. The prompt can be a noun
    phrase such as 'dog' or 'person', the model generally has higher recall and lower
    precision. Do not use plural prompts only single for detecting multiple instances.
    An example of the return value:
        [{
            "label": "dog",
            "score": 0.95,
            "bbox": [0.1, 0.2, 0.3, 0.4], # normalized coordinates
            "mask": np.ndarray # binary mask
        }, ...]

ocr(image: np.ndarray) -> list[dict[str, str | float | list[float]]:
    A function that takes an image and returns the text detected in the image with
    bounding boxes. For example:
        [{
            "score": # float confidence,
            "bbox": [x1, y1, x2, y2] # normalized coordinates
            "label": "this is some text inside the bounding box",
        }, ...]

depth_estimation(image: np.ndarray) -> np.ndarray:
    A function that takes an image and returns the depth map of the image as a numpy
    array. The value represents an estimate in meters of the distance of the object
    from the camera.

visualize_bounding_boxes(image: np.ndarray, bounding_boxes: list[dict[str, str | float | list[float] | np.ndarray]]) -> np.ndarray:
    A function that takes an image and a list of bounding boxes and returns an image
    that displays the boxes on the image.

visualize_segmentation_masks(image: np.ndarray, segmentation_masks: list[dict[str, str | float | list[float] | np.ndarray]]) -> np.ndarray:
    A function that takes an image and a list of segmentation masks and returns an image
    that displays the masks on the image.

get_crops(image: np.ndarray, bounding_box: list[dict[str, str | float | list[float] | np.ndarray]]) -> list[np.ndarray]:
    A function that takes an image and a list of bounding boxes and returns the cropped
    bounding boxes.

rotate_90(image: np.ndarray, k: int) -> np.ndarray:
    Rotates the image 90 degrees k times. The function takes an image and an integer.

display_image(image: Union[np.ndarray, PILImageType, matplotlib.figure.Figure, str]) -> None:
    A function that takes an image and displays it. The image can be a numpy array, a
    PIL image, a matplotlib figure or a string (path to the image).

iou(pred1: list[float] | np.ndarray, pred2: list[float] | np.ndarray) -> float:
    A function that takes either two bounding boxes or two masks and returns the
    intersection over union. Remember to unnormalize the bounding boxes before
    calculating the iou.
"""

EXAMPLES = """
<code>
# This is a plan for trying to identify a missing object that existing detectors cannot find by taking advantage of a grid pattern positioning of the non-missing elements:

widths = [detection["bbox"][2] - detection["bbox"][0] for detection in detections]
heights = [detection["bbox"][3] - detection["bbox"][1] for detection in detections]

med_width = np.median(widths)
med_height = np.median(heights)

sorted_detections = sorted(detections, key=lambda x: x["bbox"][1])
rows = []
current_row = []
prev_y = sorted_detections[0]["bbox"][1]

for detection in sorted_detections:
    if abs(detection["bbox"][1] - prev_y) > med_height / 2:
        rows.append(current_row)
        current_row = []
    current_row.append(detection)
    prev_y = detection["bbox"][1]

if current_row:
    rows.append(current_row)
sorted_rows = [sorted(row, key=lambda x: x["bbox"][0]) for row in rows]
max_cols = max(len(row) for row in sorted_rows)
max_rows = len(sorted_rows)

column_positions = []
for col in range(max(len(row) for row in sorted_rows)):
    column = [row[col] for row in sorted_rows if col < len(row)]
    med_left = np.median([d["bbox"][0] for d in column])
    med_right = np.median([d["bbox"][2] for d in column])
    column_positions.append((med_left, med_right))

row_positions = []
for row in sorted_rows:
    med_top = np.median([d["bbox"][1] for d in row])
    med_bottom = np.median([d["bbox"][3] for d in row])
    row_positions.append((med_top, med_bottom))


def find_element(left, right, top, bottom, elements):
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    for element in elements:
        x_min, y_min, x_max, y_max = element["bbox"]
        elt_center_x = (x_min + x_max) / 2
        elt_center_y = (y_min + y_max) / 2
        if (abs(center_x - elt_center_x) < med_width / 2) and (
            abs(center_y - elt_center_y) < med_height / 2
        ):
            return element
    return

missing_elements = []
for row in range(max_rows):
    for col in range(max_cols):
        left, right = column_positions[col]
        top, bottom = row_positions[row]
        match = find_element(left, right, top, bottom, sorted_rows[row])
        if match is None:
            missing_elements.append((left, top, right, bottom))
</code>

<code>
# This a plan to find objects that are only identifiable when compared to other objects such "find the smaller object"

from sklearn.cluster import KMeans
import numpy as np

detections = instance_segmentation("object", image)

def get_area(detection):
    return (detection["bbox"][2] - detection["bbox"][0]) * (detection["bbox"][3] - detection["bbox"][1])

areas = [get_area(detection) for detection in detections]
X = np.array(areas)[:, None]

kmeans = KMeans(n_clusters=<number of clusters>).fit(X)
smallest_cluster = np.argmin(kmeans.cluster_centers_)
largest_cluster = np.argmax(kmeans.cluster_centers_)

clusters = kmeans.predict(X)
smallest_detections = [detection for detection, cluster in zip(detections, clusters) if cluster == smallest_cluster]
largest_detections = [detection for detection, cluster in zip(detections, clusters) if cluster == largest_cluster]
</code>

<code>
# This is a plan to help find object that are identified spatially relative to other 'anchor' objects, for example find the boxes to the right of the chair

# First find a model that can detect the location of the anchor objects
anchor_dets = instance_segmentation("anchor object", image)
# Then find a model that can detect the location of the relative objects
relative_dets = instance_segmentation("relative object", image)

# This will give you relative objects 'above' the anchor objects since it's the
# distance between the lower left corner of the relative object and the upper left
# corner of the anchor object. The remaining functions can be used to get the other
# relative positions.
def above_distance(box1, box2):
    return (box1["bbox"][0] - box2["bbox"][0]) ** 2 + (
        box1["bbox"][3] - box2["bbox"][1]
    ) ** 2

def below_distance(box1, box2):
    return (box1["bbox"][0] - box2["bbox"][0]) ** 2 + (
        box1["bbox"][1] - box2["bbox"][3]
    ) ** 2

def right_distance(box1, box2):
    return (box1["bbox"][0] - box2["bbox"][2]) ** 2 + (
        box1["bbox"][1] - box2["bbox"][1]
    ) ** 2

def left_distance(box1, box2):
    return (box1["bbox"][2] - box2["bbox"][0]) ** 2 + (
        box1["bbox"][1] - box2["bbox"][1]
    ) ** 2

closest_boxes = []
for anchor_det in anchor_dets:
    # You can use any of the above functions to get the relative position
    distances = [
        (relative_det, above_distance(relative_det, anchor_det))
        for relative_det in relative_dets
    ]
    # You must grab the nearest object for each of the anchors. This line will give
    # you the box directly above the anchor box (or below, left, right depending on
    # the function used)
    closest_box = min(distances, key=lambda x: x[1])[0]
    closest_boxes.append(closest_box)
</code>

<code>
# This is a plan to help you find objects according to their depth, for example find the person nearest to the camera

# First find a model to estimate the depth of the image
depth = depth_estimation(image)
# Then find a model to segment the objects in the image
masks = instance_segmentation("object", image)

for elt in masks:
    # Multiply the depth by the mask and keep track of the mean depth for the masked
    # object
    depth_mask = depth * elt["mask"]
    elt["mean_depth"] = depth_mask.mean()

# Sort the masks by mean depth in reverse, objects that are closer will have higher
# mean depth values and further objects will have lower mean depth values.
masks = sorted(masks, key=lambda x: x["mean_depth"], reverse=True)
closest_mask = masks[0]
</code>

<code>
# This plan helps you assign objects to other objects, for example "count the number of people sitting at a table"

pred = instance_segmentation("object 1, object 2", image)
objects_1 = [p for p in pred if p["label"] == "object 1"]
objects_2 = [p for p in pred if p["label"] == "object 2"]

def box_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    # Get coordinates of intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    # Calculate area of both boxes
    box1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate IoU
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0.0

# initialize assignment counts
objects_2_counts = {i: 0 for i in range(len(objects_2))}
# you can set a minimum iou threshold for assignment
iou_threshold = 0.05

# You can expand the object 2 box by a certain percentage if needed to help with the
# assignment.
for object_2 in objects_2:
    box = object_2["bbox"]
    # If your camera is at an angle you need to expand the top of the box like so:
    box = [0.9 * box[0], 0.9 * box[1], 1.1 * box[2], box[3]]
    # If the camera is top down you should expand all sides of the box like this:
    box = [0.9 * box[0], 0.9 * box[1], 1.1 * box[2], 1.1 * box[3]]

    object_2["bbox"] = box

for object_1 in objects_1:
    best_iou = 0
    best_object_2_idx = None

    for j, object_2 in enumerate(objects_2):
        iou = box_iou(object_1["bbox"], object_2["bbox"])
        if iou > best_iou and iou > iou_threshold:
            best_iou = iou
            best_object_2_idx = j

    if best_object_2_idx is not None:
        objects_2_counts[best_object_2_idx] += 1
</code>

<code>
# This plan helps you adjust the threshold to get the best results for a specific object detection model

dets = instance_segmentation("object", image)
proposed_threshold = 0.3
for d in dets:
    d["dist"] = abs(d["score"] - proposed_threshold)

nearby_dets = sorted(dets, key=lambda x: x["dist"])[:10]
unnormalized_coords = []
for d in nearby_dets:
    unnormalized_coords.append([
        d["bbox"][0] * image.shape[1],
        d["bbox"][1] * image.shape[0],
        d["bbox"][2] * image.shape[1],
        d["bbox"][3] * image.shape[0],
    ])

crops = crop_image(image, unnormalized_coords)

for i, crop in enumerate(crops):
    print(f"object {i}")
    display(crop)
</code>

<code>
# this plan helps you analyze bounding boxes and filter out possible false positives

image = load_image(image_path)
dets = instance_segmentation("object", image)
crops = get_crops(image, dets)

# you can only display the first 5 crops
for i, crop in enumerate(crops[:5]):
    print(f"object {i}, label: {dets[i]['label']}, score: {dets[i]['score']}")
    display_image(crop)

# in your final <answer> you can choose only the crops that you want to keep
</code>

<code>
# If you want to zoom into an area of the image, you can use the following plan

image = load_image(image_path)
center_crop = get_crops(image, [{"bbox": [0.25, 0.25, 0.75, 0.75]}])[0]

# or you could split the image into 4 quadrants and display them
quadrants = get_crops(image, [{"bbox": [0, 0, 0.5, 0.5]}, {"bbox": [0.5, 0, 1, 0.5]}, {"bbox": [0, 0.5, 0.5, 1]}, {"bbox": [0.5, 0.5, 1, 1]}])
</code>
"""


def get_init_prompt(
    model: str,
    turns: int,
    question: str,
    category: str,
    image_path: str,
    tool_docs: str = TOOLS,
    examples: str = EXAMPLES,
) -> str:
    if category == "localization":
        cat_text = "Return bounding box coordinates for this question."
    elif category == "text":
        cat_text = "Return only text for this question."
    elif category == "counting":
        cat_text = "Return only the number of objects for this question."
    else:
        cat_text = ""

    initial_prompt = f"""You are given a question and an image from the user. Your task is using the vision tools and multi-modal reasoning to answer the user's question in a generalized way, so the user could give you a different image and run it through your code to get the correct answer to the same question. You have {turns} turns to solve the problem, be sure to utilize the tools to their fullest extent, also reflect on the results of the tools to best solve the problem. The tools are not perfect, so you may need to look into the results, refer back to the image and refine your answer.

Here is the documentation for the tools you have access to, you can implement new utilities if needed, but DO NOT redefine these functions or make placeholder functions for them or import them, you will have access to them:
<docs>
{tool_docs}
</docs>

Here's some example pseudo code plans that you can use to help solve your problem:
<examples>
{examples}
</examples>

<instructions>
- You should approximate the answer using a symbolic approach as much as possible (e.g. using combinations of the tools and python code to get the answer) since it's more robust and generalizable. However, if you need to use your own reasoning and observation to get the answer, you can do so.
- ALWAYS PUT THE CODE YOU WANT TO EXECUTE IN <code> TAGS (IMPORTANT!)
- You can progressively refine your answer using your own reflection on both the tasks, code outputs and tools to get the best results.
- During your iteration output <code> tags to execute code which will be run in a jupyter notebook environment.
- Do not re-define the tools as an another function, you will have access to them (e.g load_image, etc)
- For object detection queries, include only the bounding box coordinates within <answer> and omit any image data.
- If one of the tool results has low confidence or you believe it's wrong, you should focus on that result manually by yourself, observe and verify it, do not just accept it.
- Within <answer>, provide only the final answer without any explanation. For instance, for a bounding box query, return in a list of bounding box, for example <answer>[[0.234375, 0.68359375, 0.33203125, 0.7356770833333334]]</answer> or <answer>3</answer> or <answer>blue</answer>. For other tasks, return a single word or number; if the task pertains to pricing, prefix the number with a dollar sign, e.g., <answer>$3.4</answer>.
- If you cannot find an answer to the question, for example you are asked to find an object and it does not exist in the image, you should return <answer>None</answer>.
- In your first turn, you should look into the requested image, observe intelligently, tell me what you see, think about the user question, and plan your next steps or analysis.
- Each turn you only visualize or display any image or crop maximum 5 images, use `display_image` or any tool that can display an image in an intelligent way. You can first analyze the result, then target few of them to display for further analysis or understanding, do not waste your turns displaying all the images or waste one turn to display many irrelevant images.
- Output in the following format:
{"<thinking>Your very verbose visual observation, Your thoughts on the next step, reflecting on the last code execution result, planning on the next steps</thinking>" if model != "claude" else ""}
<code>The python code you want to execute (image manipulation, tool calls, etc)</code>
<answer>The final answer you want to output, but compulsory to output if you are on your final turn</answer>
</instructions>

Here is the user question: {question}
Here is the image path: {image_path}
{cat_text}"""
    return initial_prompt
