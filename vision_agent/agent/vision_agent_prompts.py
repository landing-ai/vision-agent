VISION_AGENT_REFLECTION = """You are an advanced reasoning agent that can improve based on self-refection. You will be given a previous reasoning trial in which you were given the user's question, the available tools that the agent has, the decomposed tasks and tools that the agent used to answer the question, the tool usage for each of the tools used and the final answer the agent provided. You may also receive an image with the visualized bounding boxes or masks with their associated labels and scores from the tools used.

Please note that:
1. You must ONLY output parsible JSON format. If the agents output was correct set "Finish" to true, else set "Finish" to false. An example output looks like:
{{"Finish": true, "Reflection": "The agent's answer was correct."}}
2. You must utilize the image with the visualized bounding boxes or masks and determine if the tools were used correctly or if the tools were used incorrectly or the wrong tools were used.
3. If the agent's answer was incorrect, you must diagnose the reason for failure and devise a new concise and concrete plan that aims to mitigate the same failure with the tools available. An example output looks like:
    {{"Finish": false, "Reflection": "I can see from the visualized bounding boxes that the agent's answer was incorrect because the grounding_dino_ tool produced false positive predictions. The agent should use the following tools with the following parameters:
        Step 1: Use 'grounding_dino_' with a 'prompt' of 'baby. bed' and a 'box_threshold' of 0.7 to reduce the false positives.
        Step 2: Use 'box_iou_' with the baby bounding box and the bed bounding box to determine if the baby is on the bed or not."}}
4. If the task cannot be completed with the existing tools or by adjusting the parameters, set "Finish" to true.

User's question: {question}

Tools available:
{tools}

Tasks and tools used:
{tool_results}

Tool's used API documentation:
{tool_usage}

Final answer:
{final_answer}

Reflection: """

TASK_DECOMPOSE = """You need to decompose a user's complex question into one or more simple subtasks and let the model execute it step by step.
This is the user's question: {question}
This is the tool list:
{tools}

Please note that:
1. If the given task is simple and the answer can be provided by executing one tool, you should only use that tool to provide the answer.
2. If the given task is complex, You should decompose this user's complex question into simple subtasks which can only be executed easily by using one single tool in the tool list.
3. You should try to decompose the complex question into least number of subtasks.
4. If one subtask needs the results from another subtask, you should write clearly. For example:
{{"Tasks": ["Convert 23 km/h to X km/min by 'divide_'", "Multiply X km/min by 45 min to get Y by 'multiply_'"]}}
5. You must ONLY output in a parsible JSON format. An example output looks like:

{{"Tasks": ["Task 1", "Task 2", ...]}}

Output: """

TASK_DECOMPOSE_DEPENDS = """You need to decompose a user's complex question into one or more simple subtasks and let the model execute it step by step.
This is the user's question: {question}

This is the tool list:
{tools}

This is a reflection from a previous failed attempt:
{reflections}

Please note that:
1. If the given task is simple and the answer can be provided by executing one tool, you should only use that tool to provide the answer.
2. If the given task is complex, You should decompose this user's complex question into simple subtasks which can only be executed easily by using one single tool in the tool list.
3. You should try to decompose the complex question into least number of subtasks.
4. If one subtask needs the results from another subtask, you should write clearly. For example:
{{"Tasks": ["Convert 23 km/h to X km/min by 'divide_'", "Multiply X km/min by 45 min to get Y by 'multiply_'"]}}
5. You must ONLY output in a parsible JSON format. An example output looks like:

{{"Tasks": ["Task 1", "Task 2", ...]}}

Output: """

CHOOSE_TOOL = """This is the user's question: {question}
These are the tools you can select to solve the question:
{tools}

Please note that:
1. You should only choose one tool from the Tool List to solve this question and it should have maximum chance of solving the question.
2. You should only choose the tool whose parameters are most relevant to the user's question and are availale as part of the question.
3. You should choose the tool whose return type is most relevant to the answer of the user's question.
4. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two example outputs look like:

Example 1: {{"ID": 1}}
Example 2: {{"ID": 2}}

Output: """

CHOOSE_TOOL_DEPENDS = """This is the user's question: {question}
These are the tools you can select to solve the question:
{tools}

This is a reflection from a previous failed attempt:
{reflections}

Please note that:
1. You should only choose one tool from the Tool List to solve this question and it should have maximum chance of solving the question.
2. You should only choose the tool whose parameters are most relevant to the user's question and are availale as part of the question.
3. You should choose the tool whose return type is most relevant to the answer of the user's question.
4. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two example outputs look like:

Example 1: {{"ID": 1}}
Example 2: {{"ID": 2}}

Output: """

CHOOSE_PARAMETER_DEPENDS = """Given a user's question and an API tool documentation, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.
Please note that:
1. The Example in the API tool documentation can help you better understand the use of the API. Pay attention to the examples which show how to parse the question and extract tool parameters such as prompts and visual inputs.
2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If there are no paremters in the required parameters and optional parameters, just leave it as {{"Parameters":{{}}}}
3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.
4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers for your reference.
5. If you need to use this API multiple times, please set "Parameters" to a list.
6. You must ONLY output in a parsible JSON format. Two example outputs look like:

Example 1: {{"Parameters":{{"input": [1,2,3]}}}}
Example 2: {{"Parameters":[{{"input": [1,2,3]}}, {{"input": [2,3,4]}}]}}

This is a reflection from a previous failed attempt:
{reflections}

These are logs of previous questions and answers:
{previous_log}

This is the current user's question: {question}
This is the API tool documentation: {tool_usage}
Output: """

ANSWER_GENERATE_DEPENDS = """You should answer the question based on the response output by the API tool.
Please note that:
1. You should try to organize the response into a natural language answer.
2. We will not show the API response to the user, thus you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.
3. If the API tool does not provide useful information in the response, please answer with your knowledge.
4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers.

This is a reflection from a previous failed attempt:
{reflections}

These are logs of previous questions and answers:
{previous_log}

This is the user's question: {question}

This is the response output by the API tool:
{call_results}

We will not show the API response to the user, thus you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.
Output: """

ANSWER_SUMMARIZE_DEPENDS = """We break down a user's complex problems into simple subtasks and provide answers to each simple subtask. You need to organize these answers to each subtask and form a self-consistent final answer to the user's question
This is the user's question: {question}

These are subtasks and their answers:
{answers}

This is a reflection from a previous failed attempt:
{reflections}

Final answer: """
