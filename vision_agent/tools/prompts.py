SYSTEM_PROMPT = "You are a helpful assistant."

GROUNDING_DINO = (
    "Grounding DINO is a tool that can detect arbitrary objects with inputs such as category names or referring expressions."
    "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n" 
    'Example 1: User Question: "Can you build me a car detector?" {{"Parameters":{{"prompt": "car"}}}}\n'
    'Example 2: User Question: "Can you detect the person on the left?" {{"Parameters":{{"prompt": "person on the left"}}\n'
    'Exmaple 3: User Question: "Can you build me a tool that detects red shirts and green shirts?" {{"Parameters":{{"prompt": "red shirt. green shirt"}}}}\n'
)

GROUNDING_SAM = (
    "Grounding SAM is a tool that can detect and segment arbitrary objects with inputs such as category names or referring expressions."
    "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
    'Example 1: User Question: "Can you build me a car segmentor?" {{"Parameters":{{"prompt": "car"}}}}\n'
    'Example 2: User Question: "Can you segment the person on the left?" {{"Parameters":{{"prompt": "person on the left"}}\n'
    'Exmaple 3: User Question: "Can you build me a tool that segments red shirts and green shirts?" {{"Parameters":{{"prompt": "red shirt. green shirt"}}}}\n'
)

CLIP = (
    "CLIP is a tool that can classify or tag any image given a set if input classes or tags."
    "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
    'Example 1: User Question: "Can you classify this image as a cat?" {{"Parameters":{{"prompt": ["cat"]}}}}\n'
    'Example 2: User Question: "Can you tag this photograph with cat or dog?" {{"Parameters":{{"prompt": ["cat", "dog"]}}}}\n'
    'Exmaple 3: User Question: "Can you build me a classifier taht classifies red shirts, green shirts and other?" {{"Parameters":{{"prompt": ["red shirt", "green shirt", "other"]}}}}\n'
)

# EasyTool prompts
CHOOSE_PARAMS = (
    "This is an API tool documentation. Given a user's question, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
    "This is the API tool documentation: {api_doc}\n"
    "Please note that: \n"
    "1. The Example in the API tool documentation can help you better understand the use of the API.\n"
    '2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {{"Parameters":{{}}}}\n'
    "3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
    '4. If you need to use this API multiple times, please set "Parameters" to a list.\n'
    "5. You must ONLY output in a parsible JSON format. Two examples output looks like:\n"
    "'''\n"
    'Example 1: {{"Parameters":{{"keyword": "Artificial Intelligence", "language": "English"}}}}\n'
    'Example 2: {{"Parameters":[{{"keyword": "Artificial Intelligence", "language": "English"}}, {{"keyword": "Machine Learning", "language": "English"}}]}}\n'
    "'''\n"
    "This is user's question: {question}\n"
    "Output:\n"
)
