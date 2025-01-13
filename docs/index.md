## VisionAgent
VisionAgent is a library that helps you utilize agent frameworks to generate code to
solve your vision task. Check out our discord for updates and roadmaps! The fastest
way to test out VisionAgent is to use our web application which you can find [here](https://va.landing.ai/).

## Installation
```bash
pip install vision-agent
```

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

---
**NOTE**
You must have the Anthropic API key set in your environment variables to use
VisionAgent. If you don't have an Anthropic key you can use another provider like
OpenAI or Ollama.
---

## Documentation

[VisionAgent Library Docs](https://landing-ai.github.io/vision-agent/)

## Examples
### Counting cans in an image
You can run VisionAgent in a local Jupyter Notebook [Counting cans in an image](https://github.com/landing-ai/vision-agent/blob/main/examples/notebooks/counting_cans.ipynb)

### Generating code
You can use VisionAgent to generate code to count the number of people in an image:
```python
from vision_agent.agent import VisionAgentCoderV2
from vision_agent.agent.types import AgentMessage

agent = VisionAgentCoderV2(verbose=True)
code_context = agent.generate_code(
    [
        AgentMessage(
            role="user",
            content="Count the number of people in this image",
            media=["people.png"]
        )
    ]
)

with open("generated_code.py", "w") as f:
    f.write(code_context.code + "\n" + code_context.test)
```

### Using the tools directly
VisionAgent produces code that utilizes our tools. You can also use the tools directly.
For example if you wanted to detect people in an image and visualize the results:
```python
import vision_agent.tools as T
import matplotlib.pyplot as plt

image = T.load_image("people.png")
dets = T.countgd_object_detection("person", image)
# visualize the countgd bounding boxes on the image
viz = T.overlay_bounding_boxes(image, dets)

# save the visualization to a file
T.save_image(viz, "people_detected.png")

# display the visualization
plt.imshow(viz)
plt.show()
```

You can also use the tools for running on video files:
```python
import vision_agent.tools as T

frames_and_ts = T.extract_frames_and_timestamps("people.mp4")
# extract the frames from the frames_and_ts list
frames = [f["frame"] for f in frames_and_ts]

# run the countgd tracking on the frames
tracks = T.countgd_sam2_video_tracking("person", frames)
# visualize the countgd tracking results on the frames and save the video
viz = T.overlay_segmentation_masks(frames, tracks)
T.save_video(viz, "people_detected.mp4")
```

## Support
Currently `VisionAgentV2` only support Anthropic as a provider. You can checkout our
older version of `VisionAgent` which supports additional providers such as `OpenAI`,
`Ollama` and `AzureOpenAI`.

**NOTE**
VisionAgent moves fast and we are constantly updating and changing the library. If you
have any questions or need help, please reach out to us on our discord channel.
---
