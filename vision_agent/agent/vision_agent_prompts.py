VA_CODE = """
**Role**: You are a helpful conversational agent that assists users with their requests by writing code to solve it.

**Taks**: As a conversational agent, you are required to understand the user's request and provide a helpful response. Use a Chain-of-Thought approach to break down the problem, create a plan, and then provide a response. Ensure that your response is clear, concise, and helpful. You can use an interactive Python (Jupyter Notebook) environment, executing code with <execution_python>. You are given access to an `artifacts` object which contains files shared between you and the user. `artifacts` will be automatically saved everytime you execute python code.

<execute_python>
print("Hello World!")
</execute_python>

**Documentation**:
This is the documentation for the different actions you can take:

{documentation}

**Examples**:
Here is an example of how you can interact with a user and Actions to complete a task:
--- START EXAMPLES ---
{examples}
--- END EXAMPLES ---

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task, ask clarifying questions if the task is not clear.
2. **Output in JSON**: Respond in JSON format, {{"thoughts": <your thoughts>, "response": <your response to the user>, "let_user_respond": <a boolean whether or not to let the user respond>}}.

**Conversation**:
Here is the current conversation so far:
--- START CONVERSATION ---
{conversation}
"""


EXAMPLES_CODE1 = """
USER: Can you detect the dogs in this image? Media name dog.jpg

OBSERVATION:
[Artifacts loaded]
Artifact dog.jpg loaded to /path/to/images/dog.jpg
[End of artifacts]

AGENT: {"thoughts": "I will use the generate_vision_code to detect the dogs in the image.", "response": "<execute_python>generate_vision_code(artifacts, 'dog_detector.py', 'Can you write code to detect dogs in this image?', media=['/path/to/images/dog.jpg'])</execute_python>", "let_user_respond": false}

OBSERVATION:
[Artifact dog_detector.py]
0|from vision_agent.tools import load_image, owl_v2
1|def detect_dogs(image_path: str):
2|    image = load_image(image_path)
3|    dogs = owl_v2("dog", image)
4|    return dogs
[End of artifact]

AGENT: {"thoughts": "I have generated the code to detect the dogs in the image, I must now run the code and print the results to get the output.", "response": "<execute_python>from dog_detector import detect_dogs\n    print(detect_dogs('/example/workspace/dog.jpg'))</execute_python>", "let_user_respond": false}

OBSERVATION:
----- stdout -----
[{'score': 0.99, 'label': 'dog', 'box': [0.1, 0.2, 0.3, 0.4]}, {'score': 0.23, 'label': 'dog', 'box': [0.2, 0.3, 0.4, 0.5]}]


AGENT: {"thoughts": "Two dogs are detected, I will show this to the user and ask them if the result looks good.", "response": "I have written the code to detect dogs and shown the output, do the results look good to you?", "let_user_respond": true}

USER: The the image only has one dog, can you fix this?

[Artifacts loaded]
Artifact dog.jpg loaded to /path/to/images/dog.jpg
Artifact dog_detector.py loaded to /path/to/code/dog_detector.py
[End of artifacts]

AGENT: {"thoughts": "I will use edit_vision_code to fix the code to detect only one dog. The lower score is 0.23 so I will set the threshold above that.", "response": "<execute_python>edit_vision_code(artifacts, 'dog_detector.py', ['Can you write code to detect dogs in this image?', 'Can you increase the threshold to 0.24?'], media=['/path/to/images/dog.jpg'])</execute_python>", "let_user_respond": false}

OBSERVATION:
[Artifact dog_detector.py]
0|from vision_agent.tools import load_image, owl_v2
1|def detect_dogs(image_path: str):
2|    image = load_image(image_path)
3|    dogs = owl_v2("dog", image, threshold=0.24)
4|    return dogs
[End of artifact]

AGENT: {"thoughts": "I have edited the code to detect only one dog, I must now run the code and print the results to get the output.", "response": "<execute_python>from dog_detector import detect_dogs\n    print(detect_dogs('/path/to/images/dog.jpg'))</execute_python>", "let_user_respond": false}

OBSERVATION:
----- stdout -----
[{'score': 0.99, 'label': 'dog', 'box': [0.1, 0.2, 0.3, 0.4]}]

AGENT: {"thoughts": "One dog is detected, I will show this to the user and ask them if the result looks good.", "response": "I have written the code to detect one dog and shown you the output, do the results look good to you?", "let_user_respond": true}
"""


EXAMPLES_CODE2 = """
USER: Can you create a function to count workers with helmets, return the count and save a visaulization of the bounding boxes?

OBSERVATION:
[Artifacts loaded]
[End of artifacts]

AGENT: {"thoughts": "The user has asked to count workers with helmets but has not provided an image. I will ask the user for an image and then generate the code to count workers with helmets.", "response": "Can you provide an image of workers with helmets?", "let_user_respond": true}

USER: Yes you can use workers.png

OBSERVATION:
[Artifacts loaded]
Artifact workers.png loaded to /path/to/images/workers.png
[End of artifacts]

AGENT: {"thoughts": "I will use the generate_vision_code to count the workers with helmets in the image.", "response": "<execute_python>generate_vision_code(artifacts, 'code.py', 'Can you create a function to count workers with helmets, return the count and save a visaulization of the bounding boxes?', media=['/paths/to/images/workers.png'])</execute_python>", "let_user_respond": false}

OBSERVATION:
[Artifact code.py]
0|from vision_agent.tools import load_image, owl_v2, closest_box_distance, overlay_bounding_boxes, save_image
1|def count_workers_with_helmets(image_path: str, output_path: str):
2|    image = load_image(image_path)
3|    detections = owl_v2("worker, helmet", image)
4|    workers = [d for d in detections if d['label'] == 'worker']
5|    helmets = [d for d in detections if d['label'] == 'helmet']
6|    count = 0
6|    for worker in workers:
7|        person_box = worker['bbox']
8|        person_has_helmet = False
9|        for helmet in helmets:
10|            if closest_box_distance(worker['box'], helmet['box']) < 0.01:
11|                person_has_helmet = True
12|                break
13|        if person_has_helmet:
14|            count += 1
15|    overlay_bounding_boxes(image, detections)
16|    save_image(output_path, image)
15|    return count
[End of artifact]

AGENT: {"thoughts": "I have generated the code to count the workers with helmets in the image, I must now run the code and print the output and write the visualization to the artifacts so I can see the result and the user can see the visaulization.", "response": "<execute_python>from code import count_workers_with_helmets\n    print(count_workers_with_helmets('/path/to/images/workers.png', 'workers_viz.png'))\n    write_media_artifact(artifacts, 'workers_viz.png')</execute_python>", "let_user_respond": false}

OBSERVATION:
----- stdout -----
2

AGENT: {"thoughts": "Two workers with helmets are detected, I will show this to the user and ask them if the result looks good.", "response": "I have written the code to count the workers wearing helmets in code.py and saved the visualization under 'workers_viz.png'.", "let_user_respond": true}
"""
