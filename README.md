<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/landing-ai/vision-agent/blob/main/assets/logo_light.svg?raw=true">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/landing-ai/vision-agent/blob/main/assets/logo_dark.svg?raw=true">
        <img alt="VisionAgent" height="200px" src="https://github.com/landing-ai/vision-agent/blob/main/assets/logo_light.svg?raw=true">
    </picture> 
    
_Prompt with an image/video → Get runnable vision code → Build Visual AI App in minutes_


[![](https://dcbadge.vercel.app/api/server/wPdN8RCYew?compact=true&style=flat)](https://discord.gg/wPdN8RCYew)
![ci_status](https://github.com/landing-ai/vision-agent/actions/workflows/ci_cd.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/vision-agent.svg)](https://badge.fury.io/py/vision-agent)
![version](https://img.shields.io/pypi/pyversions/vision-agent)
</div>

<p align="center">
  <a href="https://va.landing.ai/agent" target="_blank"><strong>Web App</strong></a> ·
  <a href="https://discord.com/invite/RVcW3j9RgR" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://landing.ai/blog/visionagent-an-agentic-approach-for-complex-visual-reasoning" target="_blank"><strong>Architecture</strong></a> ·
  <a href="https://support.landing.ai/docs/visionagent" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://www.youtube.com/playlist?list=PLrKGAzovU85fvo22OnVtPl90mxBygIf79" target="_blank"><strong>YouTube</strong></a>
</p>

<br />

**VisionAgent** is the Visual AI pilot from LandingAI. Give it a prompt and an image, and it automatically picks the right vision models and outputs ready‑to‑run code—letting you build vision‑enabled apps in minutes.

Prefer full control? Install the library and run VisionAgent locally. Just want to dive in quickly? Use the [VisionAgent web app](https://va.landing.ai/).

## Steps to Set Up the Library  

### Get Your VisionAgent API Key
The most important step is to [signup](https://va.landing.ai/agent) and obtain your [API key](https://va.landing.ai/settings/api-key).

### Other Prerequisites
- Python version 3.9 or higher
- [Anthropic API key](#get-an-anthropic-api-key)
- [Google API key](#get-a-google-api-key)

### Why do I need Anthropic and Google API Keys?
VisionAgent uses models from Anthropic and Google to respond to prompts and generate code. 

When you run the web-based version of VisionAgent, the app uses the LandingAI API keys to access these models. 

When you run VisionAgent programmatically, the app will need to use your API keys to access the Anthropic and Google models. This ensures that any projects you run with VisionAgent aren’t limited by the rate limits in place with the LandingAI accounts, and it also prevents many users from overloading the LandingAI rate limits.

Anthropic and Google each have their own rate limits and paid tiers. Refer to their documentation and pricing to learn more.

> **_NOTE:_** In VisionAgent v1.0.2 and earlier, VisionAgent was powered by Anthropic Claude-3.5 and OpenAI o1. If using one of these VisionAgent versions, you get an OpenAI API key and set it as an environment variable.


### Get an Anthropic API Key
1. If you don’t have one yet, create an [Anthropic Console account](https://console.anthropic.com/).
2. In the Anthropic Console, go to the [API Keys](https://console.anthropic.com/settings/keys) page.
3. Generate an API key.

### Get a Google API Key
1. If you don’t have one yet, create a [Google AI Studio account](https://aistudio.google.com/).
2. In Google AI Studio, go to the [Get API Key](https://aistudio.google.com/app/apikey) page.
3. Generate an API key.


## Installation

Install with uv:
```bash
uv add vision-agent
```

Install with pip:

```bash
pip install vision-agent
```

## Quickstart: Prompt VisionAgent
Follow this quickstart to learn how to prompt VisionAgent. After learning the basics, customize your prompt and workflow to meet your needs.

1. Get your Anthropic, Google, and VisionAgent API keys.
2. [Set the Anthropic, Google, and VisionAgent API keys as environment variables](#set-api-keys-as-environment-variables).
3. [Install VisionAgent](#installation).
4. Create a folder called `quickstart`. 
5. Find an image you want to analyze and save it to the `quickstart` folder.
6. Copy the [Sample Script](#sample-script-prompt-visionagent) to a file called `source.py`. Save the file to the `quickstart` folder.  
7. Run `source.py`. 
8. VisionAgent creates a file called `generated_code.py` and saves the generated code there.  

### Set API Keys as Environment Variables
Before running VisionAgent code, you must set the Anthropic, Google, and VisionAgent API keys as environment variables. Each operating system offers different ways to do this.

Here is the code for setting the variables:
```bash
export VISION_AGENT_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-api-key" 
```
### Sample Script: Prompt VisionAgent
To use VisionAgent to generate code, use the following script as a starting point:

```python
# Import the classes you need from the VisionAgent package
from vision_agent.agent import VisionAgentCoderV2
from vision_agent.models import AgentMessage

# Enable verbose output 
agent = VisionAgentCoderV2(verbose=True)

# Add your prompt (content) and image file (media)
code_context = agent.generate_code(
    [
        AgentMessage(
            role="user",
            content="Describe the image",
            media=["friends.jpg"]
        )
    ]
)

# Write the output to a file
with open("generated_code.py", "w") as f:
    f.write(code_context.code + "\n" + code_context.test)
```
### What to Expect When You Prompt VisionAgent
When you submit a prompt, VisionAgent performs the following tasks.

1. Generates a plan for the code generation task. If verbose output is on, the numbered steps for this plan display.
2. Generates code and a test case based on the plan. 
3. Tests the generated code with the test case. If the test case fails, VisionAgent iterates on the code generation process until the test case passes.

## Example: Count Cans in an Image
Check out how to use VisionAgent in this Jupyter Notebook to learn how to count the number of cans in an image:

[Count Cans in an Image](https://github.com/landing-ai/vision-agent/blob/main/examples/notebooks/counting_cans.ipynb)

## Use Specific Tools from VisionAgent
The VisionAgent library includes a set of [tools](vision_agent/tools), which are standalone models or functions that complete specific tasks. When you prompt VisionAgent, VisionAgent selects one or more of these tools to complete the tasks outlined in your prompt.

For example, if you prompt VisionAgent to “count the number of dogs in an image”, VisionAgent might use the `florence2_object_detection` tool to detect all the dogs, and then the `countgd_object_detection` tool to count the number of detected dogs.

After installing the VisionAgent library, you can also use the tools in your own scripts. For example, if you’re writing a script to track objects in videos, you can call the `owlv2_sam2_video_tracking` function. In other words, you can use the VisionAgent tools outside of simply prompting VisionAgent. 

The tools are in the [vision_agent.tools](vision_agent/tools) API.

### Sample Script: Use Specific Tools for Images
You can call the `countgd_object_detection` function to count the number of objects in an image. 

To do this, you could run this script:
```python
# Import the VisionAgent Tools library; import Matplotlib to visualize the results
import vision_agent.tools as T
import matplotlib.pyplot as plt

# Load the image
image = T.load_image("people.png")

# Call the function to count objects in an image, and specify that you want to count people
dets = T.countgd_object_detection("person", image)

# Visualize the countgd bounding boxes on the image
viz = T.overlay_bounding_boxes(image, dets)

# Save the visualization to a file
T.save_image(viz, "people_detected.png")

# Display the visualization
plt.imshow(viz)
plt.show()

```
### Sample Script: Use Specific Tools for Videos
You can call the `countgd_sam2_video_tracking` function to track people in a video and pair it with the `extract_frames_and_timestamps` function to return the frames and timestamps in which those people appear.

To do this, you could run this script:
```python
# Import the VisionAgent Tools library
import vision_agent.tools as T

# Call the function to get the frames and timestamps
frames_and_ts = T.extract_frames_and_timestamps("people.mp4")

# Extract the frames from the frames_and_ts list
frames = [f["frame"] for f in frames_and_ts]

# Call the function to track objects, and specify that you want to track people
tracks = T.countgd_sam2_video_tracking("person", frames)

# Visualize the countgd tracking results on the frames and save the video
viz = T.overlay_segmentation_masks(frames, tracks)
T.save_video(viz, "people_detected.mp4")
```


## Use Other LLM Providers
VisionAgent uses [Anthropic Claude 3.7 Sonnet](https://www.anthropic.com/claude/sonnet) and [Gemini Flash 2.0 Experimental](https://ai.google.dev/gemini-api/docs/models/experimental-models) (`gemini-2.0-flash-exp`) to respond to prompts and generate code. We’ve found that these provide the best performance for VisionAgent and are available on the free tiers (with rate limits) from their providers.

If you prefer to use only one of these models or a different set of models, you can change the selected LLM provider in this file: `vision_agent/configs/config.py`. You must also add the provider’s API Key as an [environment variable](#set-api-keys-as-environment-variables).

For example, if you want to use **only** the Anthropic model, run this command:
```bash
cp vision_agent/configs/anthropic_config.py vision_agent/configs/config.py
```

Or, you can manually enter the model details in the `config.py` file. For example, if you want to change the planner model from Anthropic to OpenAI, you would replace this code:
```python
    planner: Type[LMM] = Field(default=AnthropicLMM)
    planner_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "claude-3-7-sonnet-20250219",
            "temperature": 0.0,
            "image_size": 768,
        }
    )
```

with this code:

```python
    planner: Type[LMM] = Field(default=OpenAILMM)
    planner_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-11-20",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )
```

## Resources
- [Discord](https://discord.com/invite/RVcW3j9RgR): Check out our community of VisionAgent users to share use cases and learn about updates.
- [VisionAgent Library Docs](https://landing-ai.github.io/vision-agent/): Learn how to use this library.
- [VisionAgent Web App Docs](https://support.landing.ai/docs/agentic-ai): Learn how to use the web-based version of VisionAgent. 
- [Video Tutorials](https://www.youtube.com/playlist?list=PLrKGAzovU85fvo22OnVtPl90mxBygIf79): Watch the latest video tutorials to see how VisionAgent is used in a variety of use cases.
