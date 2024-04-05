VISION_AGENT_REFLECTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given the user's question, the available tools that the agent has, the decomposed tasks and tools that the agent used to answer the question and the final answer the agent provided. You must determine if the agent's answer was correct or incorrect. If the agent's answer was correct, respond with Finish. If the agent's answer was incorrect, you must diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure with the tools available. Use complete sentences.

User's question: {question}

Tools available:
{tools}

Tasks and tools used:
{tool_results}

Final answer:
{final_answer}

Reflection: """

TASK_DECOMPOSE = """You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.
This is the user's question: {question}
This is tool list:
{tools}

Please note that:
1. You should only decompose this complex user's question into some simple subtasks which can be executed easily by using one single tool in the tool list.
2. If one subtask need the results from other subtask, you can should write clearly. For example:
{{"Tasks": ["Convert 23 km/h to X km/min by 'divide_'", "Multiply X km/min by 45 min to get Y by 'multiply_'"]}}
3. You must ONLY output in a parsible JSON format. An example output looks like:

{{"Tasks": ["Task 1", "Task 2", ...]}}

Output: """

TASK_DECOMPOSE_DEPENDS = """You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.
This is the user's question: {question}

This is tool list:
{tools}

This is a reflection from a previous failed attempt:
{reflections}

Please note that:
1. You should only decompose this complex user's question into some simple subtasks which can be executed easily by using one single tool in the tool list.
2. If one subtask need the results from other subtask, you can should write clearly. For example:
{{"Tasks": ["Convert 23 km/h to X km/min by 'divide_'", "Multiply X km/min by 45 min to get Y by 'multiply_'"]}}
3. You must ONLY output in a parsible JSON format. An example output looks like:

{{"Tasks": ["Task 1", "Task 2", ...]}}

Output: """

CHOOSE_TOOL = """This is the user's question: {question}
These are the tools you can select to solve the question:

{tools}

Please note that:
1. You should only chooce one tool the Tool List to solve this question.
2. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two example outputs look like:

Example 1: {{"ID": 1}}
Example 2: {{"ID": 2}}

Output: """

CHOOSE_TOOL_DEPENDS = """This is the user's question: {question}
These are the tools you can select to solve the question:

{tools}

This is a reflection from a previous failed attempt:
{reflections}

Please note that:
1. You should only chooce one tool the Tool List to solve this question.
2. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two example outputs look like:

Example 1: {{"ID": 1}}
Example 2: {{"ID": 2}}

Output: """

CHOOSE_PARAMETER_DEPENDS = """Given a user's question and a API tool documentation, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.
Please note that:
1. The Example in the API tool documentation can help you better understand the use of the API.
2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {{"Parameters":{{}}}}
3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.
4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers for your reference.
5. If you need to use this API multiple times,, please set "Parameters" to a list.
6. You must ONLY output in a parsible JSON format. Two examples output looks like:

Example 1: {{"Parameters":{{"input": [1,2,3]}}}}
Example 2: {{"Parameters":[{{"input": [1,2,3]}}, {{"input": [2,3,4]}}]}}

This is a reflection from a previous failed attempt:
{reflections}

There are logs of previous questions and answers:
{previous_log}

This is the current user's question: {question}
This is API tool documentation: {tool_usage}
Output: """

ANSWER_GENERATE_DEPENDS = """You should answer the question based on the response output by the API tool.
Please note that:
1. Try to organize the response into a natural language answer.
2. We will not show the API response to the user, thus you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.
3. If the API tool does not provide useful information in the response, please answer with your knowledge.
4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers.

This is a reflection from a previous failed attempt:
{reflections}

There are logs of previous questions and answers:
{previous_log}

This is the user's question: {question}

This is the response output by the API tool:
{call_results}

We will not show the API response to the user, thus you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.
Output: """

ANSWER_SUMMARIZE_DEPENDS = """We break down a complex user's problems into simple subtasks and provide answers to each simple subtask. You need to organize these answers to each subtask and form a self-consistent final answer to the user's question
This is the user's question: {question}

These are subtasks and their answers:
{answers}

This is a reflection from a previous failed attempt:
{reflections}
Final answer: """
