SYSTEM_PROMPT = "You are a helpful assistant."

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
