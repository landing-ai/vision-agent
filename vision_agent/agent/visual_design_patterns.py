INPAINTING = (
    """You are trying to generate a synthetic image by inpainting a certain area, such as removing an object or the background of an image, and replacing it with a synthetic image. You can use the following code to help with this task:

# First find a segmentation model to segment the object you want to inpaint
masks = <a segmentation model>("object", image)

# If you have multiple masks you can combine them into a single mask
result = masks[0]["mask"]
for mask in masks[1:]:
    result = np.logical_or(result, mask["mask"])

# Once you have the mask you must ensure you have 1's on the area you want to inpaint
result = 1 - mask

inpaint_image = <an inpainting model>("the prompt for the inpainting", image, mask)
""",
    {"inpainting"},
)


DESIGN_PATTERNS = """You are trying to help a vision coding agent write code to accomplish a vision task. Your job is to help the agent by giving it a suggestion on how best to approach the problem. Below is a set of tools the vision agent has access to:

<tools>
- "VQA" - answering questions about an image or video, can be used for most tasks including counting, document QA, etc.
- "object detection and counting" - detecting objects or counting objects from a text prompt in an image.
- "instance segmentation" - segmenting objects in an image given a text prompt.
- "classification" - classifying objects in an image given a text prompt.
- "segmentation" - segmenting objects in an image or video given a text prompt.
- "OCR" - extracting text from an image.
- "DocQA" - answering questions about a document or extracting information from a document.
- "video object tracking" - tracking objects in a video.
- "depth and pose estimation" - estimating the depth or pose of objects in an image.
- "activity recognition" - identifying time period an event occurs in a video.
- "inpainting" - filling in masked parts of an image.
</tools>

<image_size>
{image_size}
</image_size>

And here are different plans that you can suggest to the vision coding agent depending on what category of problem it is working on. You can either suggest an existing plan from below if it fits the category, modify and existing plan from below to better fit the user request, create a completely new plan, or you can make no suggestion. If you modify a plan or suggest a new plan, be sure to include lots of code for the vision coding agent to follow, use the simplest approach to solving the problem that is the most likely to generalize to new images or videos. Use the image provided and the vision coding agent's request <request>{request}</request> to decide your answer. Be sure to return the entire <plan></plan> block with your answer.

<category>large_image: The user is working with a very large image (not a video) and the objects they are trying to identify are very small.</category>
<plan>
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
    return {{
        "label": elt["label"],
        "score": elt["score"],
        "bbox": [
            (elt["bbox"][0] + offset_x) / original_size[1],
            (elt["bbox"][1] + offset_y) / original_size[0],
            (elt["bbox"][2] + offset_x) / original_size[1],
            (elt["bbox"][3] + offset_y) / original_size[0],
        ],
    }}

def normalized_to_unnormalized(elt, image_size):
    return {{
        "label": elt["label"],
        "score": elt["score"],
        "bbox": [
            elt["bbox"][0] * image_size[1],
            elt["bbox"][1] * image_size[0],
            elt["bbox"][2] * image_size[1],
            elt["bbox"][3] * image_size[0],
        ],
    }}

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
</plan>


<category>small_text: The user is trying to read text that is too small to read properly. If you categorize the problem as small_text, you do not need to use large_image category.</category>
<plan>
First try to solve the problem by using an OCR or text extraction tool such as VQA:

text = <tool to extract text>(image)

If that does not work you must chain two models together, the first model will be an object detection model to locate the text locations, be sure to clarify when asking `get_tool_for_task`, "I need an object detection model where I can find the regions on the image with text but I don't care about the text itself". Once you have the regions, extract each region out and send it to an OCR model to extract the text itself:

text_regions = <object detection tool to find text locations>("text", image)

all_text = []
for text_region in text_regions:
    unnormalized_coords = [
        text_region[0] * image.shape[1],
        text_region[1] * image.shape[0],
        text_region[2] * image.shape[1],
        text_region[3] * image.shape[0],
    ]
    # you can widen the crop to make it easier to read the text
    crop = image[
        int(0.95 * unnormalized_coords[1]):int(1.05 * unnormalized_coords[3]),
        int(0.95 * unnormalized_coords[0]):int(1.05 * unnormalized_coords[2]),
        :
    ]
    text = <ocr tool to extract text>(crop)
    all_text.append(text)
</plan>


<category>color: The user is trying to identify the color of an object in the image.</category>
<plan>
You need to find the color of objects in the image, you can use the following code to help with this task:

import numpy as np
import cv2

color_ranges = {{
    "red_lower": ((0, 100, 100), (int(179 * 20 / 360), 255, 255)),
    "orange": ((int(179 * 21 / 360), 100, 100), (int(179 * 50 / 360), 255, 255)),
    "yellow": ((int(179 * 51 / 360), 100, 100), (int(179 * 70 / 360), 255, 255)),
    "green": ((int(179 * 71 / 360), 100, 100), (int(179 * 150 / 360), 255, 255)),
    "cyan": ((int(179 * 151 / 360), 100, 100), (int(179 * 180 / 360), 255, 255)),
    "blue": ((int(179 * 181 / 360), 100, 100), (int(179 * 265 / 360), 255, 255)),
    "purple": ((int(179 * 266 / 360), 100, 100), (int(179 * 290 / 360), 255, 255)),
    "pink": ((int(179 * 291 / 360), 100, 100), (int(179 * 330 / 360), 255, 255)),
    "red_upper": ((int(179 * 331 / 360), 100, 100), (179, 255, 255)),
    "white": ((0, 0, 200), (179, 25, 255)),
    "gray": ((0, 0, 50), (179, 50, 200)),
    "black": ((0, 0, 0), (179, 255, 30)),
}}

def get_color(image, color_ranges):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    detected_colors = {{}}
    for color, (lower, upper) in color_ranges.items():
        upper_range = np.array(upper, dtype=np.uint8)
        lower_range = np.array(lower, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        detected_pixels = cv2.countNonZero(mask)
        detected_colors[color] = detected_pixels

    if "red_lower" in detected_colors and "red_upper" in detected_colors:
        detected_colors["red"] = detected_colors["red_lower"] + detected_colors["red_upper"]
        del detected_colors["red_lower"]
        del detected_colors["red_upper"]
    return sorted(detected_colors, key=detected_colors.get, reverse=True)[0]
</plan>


<category>missing_grid_elements: The user is trying to identify missing elements that are part of a tight perfectly square grid pattern and the grid pattern is symmetric and not warped.</category>
<plan>
You are trying to identify missing elements that existing detectors cannot find. The non-missing instances of the item form a grid pattern that you can exploit to locate the missing item. Assuming you have detections of the non-missing instances you can utilize this code to locate the missing instances:

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
</plan>

<category>missing_horizontal_elements: The user is trying to identify missing elements that are part of a horizontal line pattern.</category>
<plan>
You are trying to identify missing elements that existing detectors cannot find. The non-missing instances of the item form a horizontal pattern that you can exploit to locate the missing item. Assuming you have detections of the non-missing instances you can utilize this code to locate the missing instances:

sorted_detections = sorted(detections, key=lambda x: x["bbox"][0] + x["bbox"][1])

horizontal_lines = []
while len(sorted_detections) > 0:
    current = sorted_detections[0]
    x_min, y_min, x_max, y_max = current["bbox"]
    mean_y = (y_min + y_max) / 2
    line = [
        det for det in sorted_detections if det["bbox"][1] < mean_y < det["bbox"][3]
    ]
    horizontal_lines.append(line)

    for det in line:
        sorted_detections.remove(det)

gaps = []
for line in horizontal_lines:
    line = sorted(line, key=lambda x: x["bbox"][0])
    median_width = np.median(
        [line[i]["bbox"][2] - line[i]["bbox"][0] for i in range(len(line))]
    )
    median_height = np.median(
        [line[i]["bbox"][3] - line[i]["bbox"][1] for i in range(len(line))]
    )
    for i in range(len(line) - 1):
        w_gap = line[i + 1]["bbox"][0] - line[i]["bbox"][2]
        if w_gap > (0.5 * median_width):
            count = np.round(w_gap / median_width)
            for j in range(int(count)):
                gaps.append(
                    [
                        line[i]["bbox"][2] + j * median_width,
                        line[i]["bbox"][1],
                        line[i]["bbox"][2] + (j + 1) * median_width,
                        line[i]["bbox"][1] + median_height,
                    ]
                )
missing_elements = [{{"label": "missing_element", "score": 1.0, "bbox": gap}} for gap in gaps]
</plan>

<category>missing_vertical_elements: The user is trying to identify missing elements that are part of a vertical line pattern.</category>
<plan>
You are trying to identify missing elements that existing detectors cannot find. The non-missing instances of the item form a vertical pattern that you can exploit to locate the missing item. Assuming you have detections of the non-missing instances you can utilize this code to locate the missing instances:

sorted_detections = sorted(detections, key=lambda x: x["bbox"][0] + x["bbox"][1])

vertical_lines = []
while len(sorted_detections) > 0:
    current = sorted_detections[0]
    x_min, y_min, x_max, y_max = current["bbox"]
    mean_x = (x_min + x_max) / 2
    line = [
        det for det in sorted_detections if det["bbox"][0] < mean_x < det["bbox"][2]
    ]
    vertical_lines.append(line)

    for det in line:
        sorted_detections.remove(det)

gaps = []
for line in vertical_lines:
    line = sorted(line, key=lambda x: x["bbox"][1])
    median_width = np.median(
        [line[i]["bbox"][2] - line[i]["bbox"][0] for i in range(len(line))]
    )
    median_height = np.median(
        [line[i]["bbox"][3] - line[i]["bbox"][1] for i in range(len(line))]
    )
    for i in range(len(line) - 1):
        h_gap = line[i + 1]["bbox"][1] - line[i]["bbox"][3]
        if h_gap > (0.5 * median_height):
            count = np.round(h_gap / median_height)
            for j in range(int(count)):
                gaps.append(
                    [
                        line[i]["bbox"][0],
                        line[i]["bbox"][3] + j * median_height,
                        line[i]["bbox"][0] + median_width,
                        line[i]["bbox"][3] + (j + 1) * median_height,
                    ]
                )

missing_elements = [{{"label": "missing_element", "score": 1.0, "bbox": gap}} for gap in gaps]
</plan>

<category>finding_features_with_video_tracking: The user is trying to track objects in a video and identify features on those objects.</category>
<plan>
First try to solve the problem using a VQA tool before using the tracking approach for a faster and easier solution:

answer = <VQA tool to answer your question>("<your prompt here>", image)

If that does not work, you can track the objects in the video and then identify features on those objects. You need to first get a tool that can track objects in a video, and then for each object find another tool to identify the features on the object. You can use the following code to help with this task:

track_predictions = <object tracking tool>("object", video_frames)


# Step 1: go through each frame and each prediction and extract the predicted bounding boxes as crops
obj_to_info = {{}}
for frame, frame_predictions in zip(video_frames, track_predictions):
    for obj in frame_predictions:
        if obj["label"] not in obj_to_info:
            obj_to_info[obj["label"]] = []
        height, width = frame.shape[:2]
        # Consider adding a buffer to the crop to ensure the object is fully captured
        crop = frame[
            int(obj["bbox"][1] * height) : int(obj["bbox"][3] * height),
            int(obj["bbox"][0] * width) : int(obj["bbox"][2] * width),
            :,
        ]
        # For each crop use an object detection tool, VQA tool or classification tool to identify if the object contains the features you want
        output = <tool, such as VQA, to identify your feature or multiple features>("<your feature(s) here>", crop)
        obj_to_info[obj["label"]].extend(output)

print(f"{{len(obj_to_info)}} objects tracked")

objects_with_info = set()
for infos in obj_to_info:
    for info in info:
        if info["label"] == "<your feature here>":
            objects_with_info.add(info)
            break

print(f"{{len(objects_with_info)}} objects with features found")
</plan>


<category>comparing_sizes: The user is trying to compare objects by size or some other metric, e.g. count the smaller objects, or count the larger objects.</category>
<plan>
You are trying to order objects into comparative buckets, such as small and large, or small, medium and large. To do this you must first detect the objects, then calculate the bucket of interest (such as area, circumference, etc.) and finally use a clustering algorithm to group the objects into the desired buckets. You can use the following code to help with this task:

from sklearn.cluster import KMeans
import numpy as np

detections = <a detection tool that also includes segmentation masks>("object", image)

def get_area(detection):
    return np.sum(detection["mask"])


areas = [get_area(detection) for detection in detections]
X = np.array(areas)[:, None]

kmeans = KMeans(n_clusters=<number of clusters>).fit(X)
smallest_cluster = np.argmin(kmeans.cluster_centers_)
largest_cluster = np.argmax(kmeans.cluster_centers_)

clusters = kmeans.predict(X)
smallest_detections = [detection for detection, cluster in zip(detections, clusters) if cluster == smallest_cluster]
largest_detections = [detection for detection, cluster in zip(detections, clusters) if cluster == largest_cluster]
</plan>

<category>nested_structure: The user is trying to count or identify objects but those objects are nested inside other objects.</category>
<plan>
You are trying to count objects within objects, or a nested structure. You can solve this by first detecting the outer objects, then cropping the image to the bounding box of each outer object and detecting the inner objects. You can use the following code to help with this task:

all_dets = <an object detection tool>("object", image)

height, width = image.shape[:2]

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

# only check inner detections on top 25 largest outer detections
largest_dets = sorted(dets, key=lambda x: area(x["bbox"]), reverse=True)[:25]
for det in largest_dets:
    x1 = int(det["bbox"][0] * width)
    y1 = int(det["bbox"][1] * height)
    x2 = int(det["bbox"][2] * width)
    y2 = int(det["bbox"][3] * height)

    crop = image[y1:y2, x1:x2]
    crop_height, crop_width = crop.shape[:2]

    inner_dets = <an object detection tool>("object", crop)
    for inner_det in inner_dets:
        x1_inner = int(inner_det["bbox"][0] * crop_width)
        y1_inner = int(inner_det["bbox"][1] * crop_height)
        x2_inner = int(inner_det["bbox"][2] * crop_width)
        y2_inner = int(inner_det["bbox"][3] * crop_height)

        bbox = [
            x1 + x1_inner,
            y1 + y1_inner,
            x1 + x2_inner,
            y1 + y2_inner,
        ]
        norm_bbox = [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height,
        ]
        all_dets.append(
            {{
                "label": inner_det["label"],
                "score": inner_det["score"],
                "bbox": norm_bbox,
            }}
        )
</plan>

<category>relative_position: The user is trying to locate an object relative to other 'anchor' objects such as up, down, left or right.</category>
<plan>
You are trying to locate an objects relative to 'anchor' objects. The 'anchor' objects can be detected fine, but there are many of the other objects and you only want to return the ones that are located relative to the 'anchor' objects as specified by the user. You can use the following code to help with this task:

# First find a model that can detect the location of the anchor objects
anchor_dets = <a model that can detect the location of the anchor objects>("anchor object", image)
# Then find a model that can detect the location of the relative objects
relative_dets = <a model that can detect the location of the relative objects>("relative object", image)

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
</plan>

<category>depth_position: The user is trying to find a furthest or closest object based on depth in the image.</category>
<plan>
You are trying to order objects by their depth in the image. You can use a depth estimation model to estimate the depth of the objects and then sort the objects based on their mean depth. You can use the following code to help with this task:

# First find a model to estimate the depth of the image
depth = <a depth estimation model>(image)
# Then find a model to segment the objects in the image
masks = <a segmentation model>("object", image)

for elt in masks:
    # Multiply the depth by the mask and keep track of the mean depth for the masked
    # object
    depth_mask = depth * elt["mask"]
    elt["mean_depth"] = depth_mask.mean()

# Sort the masks by mean depth in reverse, objects that are closer will have higher
# mean depth values and further objects will have lower mean depth values.
masks = sorted(masks, key=lambda x: x["mean_depth"], reverse=True)
closest_mask = masks[0]
</plan>

<category>activity recognition: The user is trying to identify the time period an event occurs in a video.</category>
<plan>
You are trying to identify the time period an event occurs in a video. You can use an activity recognition model to identify the event and the time period it occurs in. You can use the following code to help with this task:
preds = <activity recognition model>("a description of the event you want to locate", frames)
even_frames = [frame for i, frame in enumerate(frames) if preds[i] == 1.0]
</plan>


<category>object_assignment: The user is trying to assign one class of objects to another class, in a many-to-one relationship, such as people sitting at tables.</category>
<plan>
You are trying to detect or track two classes of objects where multiple of one class can be assigned to one of the other class.

pred = <object detection or instance segmentation tool>("object 1, object 2", image_or_frame)
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
objects_2_counts = {{i: 0 for i in range(len(objects_2))}}
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
</plan>

<category>document_qa: The user is trying to answer questions about a document or extract information from a document.</category>
<plan>
You are trying to answer questions about a document or extract information from a document. You can use a Document QA or image VQA model to extract the information from the document and answer the questions. You can use the following code to help with this task:

doc_text = <a document QA or image VQA model>("question", document)

# If you use a VQA model you can also ask it to extract information in json format:
doc_json = <image VQA model>("Please extract the information ... in JSON format with {{'key1': 'result1', ...}}", document)
</plan>
"""
