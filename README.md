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

## Documentation

- [Vision Agent Library Docs](https://landing-ai.github.io/vision-agent/)


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

### Vision Agent
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

You can also have it return more information by calling `chat_with_workflow`:

```python
>>> results = agent.chat_with_workflow([{"role": "user", "content": "What percentage of the area of the jar is filled with coffee beans?"}], media="jar.jpg")
>>> print(results)
{
    "code": "from vision_agent.tools import ..."
    "test": "calculate_filled_percentage('jar.jpg')",
    "test_result": "...",
    "plan": [{"code": "...", "test": "...", "plan": "..."}, ...],
    "working_memory": ...,
}
```

With this you can examine more detailed information such as the etesting code, testing
results, plan or working memory it used to complete the task.

### Tools
There are a variety of tools for the model or the user to use. Some are executed locally
while others are hosted for you. You can also ask an LLM directly to build a tool for
you. For example:

```python
>>> import vision_agent as va
>>> llm = va.llm.OpenAILLM()
>>> detector = llm.generate_detector("Can you build a jar detector for me?")
>>> detector("jar.jpg")
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
If you want to use Azure OpenAI models, you can set the environment variable:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

You can then run Vision Agent using the Azure OpenAI models:

```python
import vision_agent as va
import vision_agent.tools as T
agent = va.agent.VisionAgent(
    planner=va.llm.AzureOpenAILLM(),
    coder=va.lmm.AzureOpenAILLM(),
    tester=va.lmm.AzureOpenAILLM(),
    debugger=va.lmm.AzureOpenAILLM(),
    tool_recommender=va.utils.AzureSim(T.TOOLS_DF, sim_key="desc"),
)
```
