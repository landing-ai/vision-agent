<div align="center">
    <img alt="vision_agent" height="200px" src="https://github.com/landing-ai/vision-agent/blob/main/assets/logo.jpg?raw=true">

# 🔍🤖 Vision Agent
[![](https://dcbadge.vercel.app/api/server/wPdN8RCYew?compact=true&style=flat)](https://discord.gg/wPdN8RCYew)
![ci_status](https://github.com/landing-ai/vision-agent/actions/workflows/ci_cd.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/vision-agent.svg)](https://badge.fury.io/py/vision-agent)
![version](https://img.shields.io/pypi/pyversions/vision-agent)
</div>

Vision Agent is a library that helps you utilize agent frameworks to generate code to
solve your vision task. Many current vision problems can easily take hours or days to
solve, you need to find the right model, figure out how to use it and program it to
accomplish the task you want. Vision Agent aims to provide an in-seconds experience by
allowing users to describe their problem in text and have the agent framework generate
code to solve the task for them. Check out our discord for updates and roadmaps!


## Web Application

Try Vision Agent live on [va.landing.ai](https://va.landing.ai/)

## Documentation

[Vision Agent Library Docs](https://landing-ai.github.io/vision-agent/)


## Getting Started
### Installation
To get started, you can install the library using pip:

```bash
pip install vision-agent
```

Ensure you have an OpenAI API key and set it as an environment variable (if you are
using Azure OpenAI please see the Azure setup section):

```bash
export OPENAI_API_KEY="your-api-key"
```

### Important Note on API Usage
Please be aware that using the API in this project requires you to have API credits (minimum of five US dollars). This is different from the OpenAI subscription used in this chatbot. If you don't have credit, further information can be found [here](https://github.com/moutasemalakkad/vision-agent/blob/57e50b59e39fde7d943d534a8bf83c5ff6281562/README.md#L215-L216)

### Vision Agent
#### Basic Usage
You can interact with the agent as you would with any LLM or LMM model:

```python
>>> from vision_agent.agent import VisionAgent
>>> agent = VisionAgent()
>>> code = agent("What percentage of the area of the jar is filled with coffee beans?", media="jar.jpg")
```

Which produces the following code:
```python
from vision_agent.tools import load_image, grounding_sam

def calculate_filled_percentage(image_path: str) -> float:
    # Step 1: Load the image
    image = load_image(image_path)

    # Step 2: Segment the jar
    jar_segments = grounding_sam(prompt="jar", image=image)

    # Step 3: Segment the coffee beans
    coffee_beans_segments = grounding_sam(prompt="coffee beans", image=image)

    # Step 4: Calculate the area of the segmented jar
    jar_area = 0
    for segment in jar_segments:
        jar_area += segment['mask'].sum()

    # Step 5: Calculate the area of the segmented coffee beans
    coffee_beans_area = 0
    for segment in coffee_beans_segments:
        coffee_beans_area += segment['mask'].sum()

    # Step 6: Compute the percentage of the jar area that is filled with coffee beans
    if jar_area == 0:
        return 0.0  # To avoid division by zero
    filled_percentage = (coffee_beans_area / jar_area) * 100

    # Step 7: Return the computed percentage
    return filled_percentage
```

To better understand how the model came up with it's answer, you can run it in debug
mode by passing in the verbose argument:

```python
>>> agent = VisionAgent(verbose=2)
```

#### Detailed Usage
You can also have it return more information by calling `chat_with_workflow`. The format
of the input is a list of dictionaries with the keys `role`, `content`, and `media`:

```python
>>> results = agent.chat_with_workflow([{"role": "user", "content": "What percentage of the area of the jar is filled with coffee beans?", "media": ["jar.jpg"]}])
>>> print(results)
{
    "code": "from vision_agent.tools import ..."
    "test": "calculate_filled_percentage('jar.jpg')",
    "test_result": "...",
    "plan": [{"code": "...", "test": "...", "plan": "..."}, ...],
    "working_memory": ...,
}
```

With this you can examine more detailed information such as the testing code, testing
results, plan or working memory it used to complete the task.

#### Multi-turn conversations
You can have multi-turn conversations with vision-agent as well, giving it feedback on
the code and having it update. You just need to add the code as a response from the
assistant:

```python
agent = va.agent.VisionAgent(verbosity=2)
conv = [
    {
        "role": "user",
        "content": "Are these workers wearing safety gear? Output only a True or False value.",
        "media": ["workers.png"],
    }
]
result = agent.chat_with_workflow(conv)
code = result["code"]
conv.append({"role": "assistant", "content": code})
conv.append(
    {
        "role": "user",
        "content": "Can you also return the number of workers wearing safety gear?",
    }
)
result = agent.chat_with_workflow(conv)
```

### Tools
There are a variety of tools for the model or the user to use. Some are executed locally
while others are hosted for you. You can also ask an LMM directly to build a tool for
you. For example:

```python
>>> import vision_agent as va
>>> llm = va.llm.OpenAILMM()
>>> detector = llm.generate_detector("Can you build a jar detector for me?")
>>> detector(va.tools.load_image("jar.jpg"))
[{"labels": ["jar",],
  "scores": [0.99],
  "bboxes": [
    [0.58, 0.2, 0.72, 0.45],
  ]
}]
```

You can also add custom tools to the agent:

```python
import vision_agent as va
import numpy as np

@va.tools.register_tool(imports=["import numpy as np"])
def custom_tool(image_path: str) -> str:
    """My custom tool documentation.

    Parameters:
        image_path (str): The path to the image.

    Returns:
        str: The result of the tool.

    Example
    -------
    >>> custom_tool("image.jpg")
    """

    return np.zeros((10, 10))
```

You need to ensure you call `@va.tools.register_tool` with any imports it might use and
ensure the documentation is in the same format above with description, `Parameters:`,
`Returns:`, and `Example\n-------`. You can find an example use case [here](examples/custom_tools/).

### Azure Setup
If you want to use Azure OpenAI models, you need to have two OpenAI model deployments:

1. OpenAI GPT-4o model
2. OpenAI text embedding model

<img width="1201" alt="Screenshot 2024-06-12 at 5 54 48 PM" src="https://github.com/landing-ai/vision-agent/assets/2736300/da125592-b01d-45bc-bc99-d48c9dcdfa32">

Then you can set the following environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
# The deployment name of your Azure OpenAI chat model
export AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT_NAME="your_gpt4o_model_deployment_name"
# The deployment name of your Azure OpenAI text embedding model
export AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME="your_embedding_model_deployment_name"
```

> NOTE: make sure your Azure model deployment have enough quota (token per minute) to support it. The default value 8000TPM is not enough.

You can then run Vision Agent using the Azure OpenAI models:

```python
import vision_agent as va
agent = va.agent.AzureVisionAgent()
```
******************************************************************************************************************************
#### To get started with API credits:

1. Visit the[OpenAI API platform](https://beta.openai.com/signup/) to sign up for an API key.
2. Follow the instructions to purchase and manage your API credits.
3. Ensure your API key is correctly configured in your project settings.

Failure to have sufficient API credits may result in limited or no functionality for the features that rely on the OpenAI API.

For more details on managing your API usage and credits, please refer to the OpenAI API documentation.
