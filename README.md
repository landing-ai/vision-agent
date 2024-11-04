<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/landing-ai/vision-agent/blob/main/assets/logo_light.svg?raw=true">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/landing-ai/vision-agent/blob/main/assets/logo_dark.svg?raw=true">
        <img alt="VisionAgent" height="200px" src="https://github.com/landing-ai/vision-agent/blob/main/assets/logo_light.svg?raw=true">
    </picture>

[![](https://dcbadge.vercel.app/api/server/wPdN8RCYew?compact=true&style=flat)](https://discord.gg/wPdN8RCYew)
![ci_status](https://github.com/landing-ai/vision-agent/actions/workflows/ci_cd.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/vision-agent.svg)](https://badge.fury.io/py/vision-agent)
![version](https://img.shields.io/pypi/pyversions/vision-agent)
</div>

VisionAgent is a library that helps you utilize agent frameworks to generate code to
solve your vision task. Many current vision problems can easily take hours or days to
solve, you need to find the right model, figure out how to use it and program it to
accomplish the task you want. VisionAgent aims to provide an in-seconds experience by
allowing users to describe their problem in text and have the agent framework generate
code to solve the task for them. Check out our discord for updates and roadmaps!

## Table of Contents
- [🚀Quick Start](#quick-start)
- [📚Documentation](#documentation)
- [🔍🤖VisionAgent](#visionagent-basic-usage)
- [🛠️Tools](#tools)
- [🤖LMMs](#lmms)
- [💻🤖VisionAgent Coder](#visionagent-coder)
- [🏗️Additional Backends](#additional-backends)

## Quick Start
### Web Application
The fastest way to test out VisionAgent is to use our web application. You can find it
[here](https://va.landing.ai/).


### Installation
To get started with the python library, you can install it using pip:

```bash
pip install vision-agent
```

Ensure you have an Anthropic key and an OpenAI API key and set in your environment
variables (if you are using Azure OpenAI please see the Azure setup section):

```bash
export ANTHROPIC_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"
```

### Basic Usage
To get started you can just import the `VisionAgent` and start chatting with it:
```python
>>> from vision_agent.agent import VisionAgent
>>> agent = VisionAgent()
>>> resp = agent("Hello")
>>> print(resp)
[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "{'thoughts': 'The user has greeted me. I will respond with a greeting and ask how I can assist them.', 'response': 'Hello! How can I assist you today?', 'let_user_respond': True}"}]
>>> resp.append({"role": "user", "content": "Can you count the number of people in this image?", "media": ["people.jpg"]})
>>> resp = agent(resp)
```

The chat messages are similar to `OpenAI`'s format with `role` and `content` keys but
in addition to those you can add `medai` which is a list of media files that can either
be images or video files.

## Documentation

[VisionAgent Library Docs](https://landing-ai.github.io/vision-agent/)

## VisionAgent Basic Usage
### Chatting and Message Formats
`VisionAgent` is an agent that can chat with you and call other tools or agents to
write vision code for you. You can interact with it like you would ChatGPT or any other
chatbot. The agent uses Clause-3.5 for it's LMM and OpenAI for embeddings for searching
for tools.

The message format is:
```json
{
    "role": "user",
    "content": "Hello",
    "media": ["image.jpg"]
}
```
Where `role` can be `user`, `assistant` or `observation` if the agent has executed a 
function and needs to observe the output. `content` is always the text message and
`media` is a list of media files that can be images or videos that you want the agent
to examine.

When the agent responds, inside it's `context` you will find the following data structure:
```json
{
    "thoughts": "The user has greeted me. I will respond with a greeting and ask how I can assist them.",
    "response": "Hello! How can I assist you today?",
    "let_user_respond": true
}
```

`thoughts` are the thoughts the agent had when processing the message, `response` is the
response it generated which could contain a python execution, and `let_user_respond` is
a boolean that tells the agent if it should wait for the user to respond before
continuing, for example it may want to execute code and look at the output before
letting the user respond.

### Chatting and Artifacts
If you run `chat_with_artifacts` you will also notice an `Artifact` object. `Artifact`'s
are a way to sync files between local and remote environments. The agent will read and
write to the artifact object, which is just a pickle object, when it wants to save or
load files.

```python
import vision_agent as va
from vision_agent.tools.meta_tools import Artifact

artifact = Artifact("artifact.pkl")
# you can store text files such as code or images in the artifact
with open("code.py", "r") as f:
    artifacts["code.py"] = f.read()
with open("image.png", "rb") as f:
    artifacts["image.png"] = f.read()

agent = va.agent.VisionAgent()
response, artifacts = agent.chat_with_artifacts(
    [
        {
            "role": "user",
            "content": "Can you write code to count the number of people in image.png",
        }
    ],
    artifacts=artifacts,
)
```

### Running the Streamlit App
To test out things quickly, sometimes it's easier to run the streamlit app locally to
chat with `VisionAgent`, you can run the following command:

```bash
pip install -r examples/chat/requirements.txt
export WORKSPACE=/path/to/your/workspace
export ZMQ_PORT=5555
streamlit run examples/chat/app.py
```
You can find more details about the streamlit app [here](examples/chat/), there are
still some concurrency issues with the streamlit app so if you find it doing weird things
clear your workspace and restart the app.

## Tools
There are a variety of tools for the model or the user to use. Some are executed locally
while others are hosted for you. You can easily access them yourself, for example if
you want to run `owl_v2_image` and visualize the output you can run:

```python
import vision_agent.tools as T
import matplotlib.pyplot as plt

image = T.load_image("dogs.jpg")
dets = T.owl_v2_image("dogs", image)
# visualize the owl_v2_ bounding boxes on the image
viz = T.overlay_bounding_boxes(image, dets)

# plot the image in matplotlib or save it
plt.imshow(viz)
plt.show()
T.save_image(viz, "viz.png")
```

Or if you want to run on video data, for example track sharks and people at 10 FPS:

```python
frames_and_ts = T.extract_frames_and_timestamps("sharks.mp4", fps=10)
# extract only the frames from frames and timestamps
frames = [f["frame"] for f in frames_and_ts]
# track the sharks and people in the frames, returns segmentation masks
track = T.florence2_sam2_video_tracking("shark, person", frames)
# plot the segmentation masks on the frames
viz = T.overlay_segmentation_masks(frames, track)
T.save_video(viz, "viz.mp4")
```

You can find all available tools in `vision_agent/tools/tools.py`, however the
`VisionAgent` will only utilizes a subset of tools that have been tested and provide
the best performance. Those can be found in the same file under the `FUNCION_TOOLS`
variable inside `tools.py`.

#### Custom Tools
If you can't find the tool you are looking for you can also add custom tools to the
agent:

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

You need to ensure you call `@va.tools.register_tool` with any imports it uses. Global
variables will not be captured by `register_tool` so you need to include them in the
function. Make sure the documentation is in the same format above with description,
`Parameters:`, `Returns:`, and `Example\n-------`. The `VisionAgent` will use your
documentation when trying to determine when to use your tool. You can find an example
use case [here](examples/custom_tools/) for adding a custom tool. Note you may need to
play around with the prompt to ensure the model picks the tool when you want it to.

Can't find the tool you need and want us to host it? Check out our
[vision-agent-tools](https://github.com/landing-ai/vision-agent-tools) repository where
we add the source code for all the tools used in `VisionAgent`.

## LMMs
All of our agents are based off of LMMs or Large Multimodal Models. We provide a thin
abstraction layer on top of the underlying provider APIs to be able to more easily
handle media.


```python
from vision_agent.lmm import AnthropicLMM

lmm = AnthropicLMM()
response = lmm("Describe this image", media=["apple.jpg"])
>>> "This is an image of an apple."
```

Or you can use the `OpenAI` chat interaface and pass it other media like videos:

```python
response = lmm(
    [
        {
            "role": "user",
            "content": "What's going on in this video?",
            "media": ["video.mp4"]
        }
    ]
)
```

## VisionAgent Coder
Underneath the hood, `VisionAgent` uses `VisionAgentCoder` to generate code to solve
vision tasks. You can use `VisionAgentCoder` directly to generate code if you want:

```python
>>> from vision_agent.agent import VisionAgentCoder
>>> agent = VisionAgentCoder()
>>> code = agent("What percentage of the area of the jar is filled with coffee beans?", media="jar.jpg")
```

Which produces the following code:
```python
from vision_agent.tools import load_image, florence2_sam2_image

def calculate_filled_percentage(image_path: str) -> float:
    # Step 1: Load the image
    image = load_image(image_path)

    # Step 2: Segment the jar
    jar_segments = florence2_sam2_image("jar", image)

    # Step 3: Segment the coffee beans
    coffee_beans_segments = florence2_sam2_image("coffee beans", image)

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
>>> agent = VisionAgentCoder(verbosity=2)
```

### Detailed Usage
You can also have it return more information by calling `generate_code`. The format
of the input is a list of dictionaries with the keys `role`, `content`, and `media`:

```python
>>> results = agent.generate_code([{"role": "user", "content": "What percentage of the area of the jar is filled with coffee beans?", "media": ["jar.jpg"]}])
>>> print(results)
{
    "code": "from vision_agent.tools import ..."
    "test": "calculate_filled_percentage('jar.jpg')",
    "test_result": "...",
    "plans": {"plan1": {"thoughts": "..."}, ...},
    "plan_thoughts": "...",
    "working_memory": ...,
}
```

With this you can examine more detailed information such as the testing code, testing
results, plan or working memory it used to complete the task.

### Multi-turn conversations
You can have multi-turn conversations with vision-agent as well, giving it feedback on
the code and having it update. You just need to add the code as a response from the
assistant:

```python
agent = va.agent.VisionAgentCoder(verbosity=2)
conv = [
    {
        "role": "user",
        "content": "Are these workers wearing safety gear? Output only a True or False value.",
        "media": ["workers.png"],
    }
]
result = agent.generate_code(conv)
code = result["code"]
conv.append({"role": "assistant", "content": code})
conv.append(
    {
        "role": "user",
        "content": "Can you also return the number of workers wearing safety gear?",
    }
)
result = agent.generate_code(conv)
```


## Additional Backends
### E2B Code Execution
If you wish to run your code on the E2B backend, make sure you have your `E2B_API_KEY`
set and then set `CODE_SANDBOX_RUNTIME=e2b` in your environment variables. This will
run all the agent generated code on the E2B backend.

### Anthropic
`AnthropicVisionAgentCoder` uses Anthropic. To get started you just need to get an
Anthropic API key and set it in your environment variables:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Because Anthropic does not support embedding models, the default embedding model used
is the OpenAI model so you will also need to set your OpenAI API key:

```bash
export OPEN_AI_API_KEY="your-api-key"
```

Usage is the same as `VisionAgentCoder`:

```python
>>> import vision_agent as va
>>> agent = va.agent.AnthropicVisionAgentCoder()
>>> agent("Count the apples in the image", media="apples.jpg")
```

### OpenAI
`OpenAIVisionAgentCoder` uses OpenAI. To get started you just need to get an OpenAI API
key and set it in your environment variables:

```bash
export OPEN_AI_API_KEY="your-api-key"
```

Usage is the same as `VisionAgentCoder`:

```python
>>> import vision_agent as va
>>> agent = va.agent.OpenAIVisionAgentCoder()
>>> agent("Count the apples in the image", media="apples.jpg")
```


### Ollama
`OllamaVisionAgentCoder` uses Ollama. To get started you must download a few models:

```bash
ollama pull llama3.1
ollama pull mxbai-embed-large
```

`llama3.1` is used for the `OllamaLMM` for `OllamaVisionAgentCoder`. Normally we would
use an actual LMM such as `llava` but `llava` cannot handle the long context lengths
required by the agent. Since `llama3.1` cannot handle images you may see some
performance degredation. `mxbai-embed-large` is the embedding model used to look up
tools. You can use it just like you would use `VisionAgentCoder`:

```python
>>> import vision_agent as va
>>> agent = va.agent.OllamaVisionAgentCoder()
>>> agent("Count the apples in the image", media="apples.jpg")
```
> WARNING: VisionAgent doesn't work well unless the underlying LMM is sufficiently powerful. Do not expect good results or even working code with smaller models like Llama 3.1 8B.

### Azure OpenAI
`AzureVisionAgentCoder` uses Azure OpenAI models. To get started follow the Azure Setup
section below. You can use it just like you would use `VisionAgentCoder`:

```python
>>> import vision_agent as va
>>> agent = va.agent.AzureVisionAgentCoder()
>>> agent("Count the apples in the image", media="apples.jpg")
```


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

You can then run VisionAgent using the Azure OpenAI models:

```python
import vision_agent as va
agent = va.agent.AzureVisionAgentCoder()
```

******************************************************************************************************************************

## Q&A

### How to get started with OpenAI API credits

1. Visit the [OpenAI API platform](https://beta.openai.com/signup/) to sign up for an API key.
2. Follow the instructions to purchase and manage your API credits.
3. Ensure your API key is correctly configured in your project settings.

Failure to have sufficient API credits may result in limited or no functionality for
the features that rely on the OpenAI API. For more details on managing your API usage
and credits, please refer to the OpenAI API documentation.
