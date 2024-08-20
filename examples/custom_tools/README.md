# Template Matching Custom Tool

## Example

This demo shows you how to create a custom tool for template matching that your Vision
Agent can then use to help you answer questions. To get started, you can install the
requirements by running:

```bash
pip install -r requirements.txt
```

You can then run the custom tool by running:

```bash
python run_custom_tool.py
```

Tool choice can be difficult for the agent to get, so sometimes it helps to explicitly
call out which tool you want to use. For example:

```python
import vision_agent as va

agent = va.agent.VisionAgentCoder(verbosity=2)
agent(
    "Can you use the 'template_match_' tool to find the location of pid_template.png in pid.png?",
    media="pid.png",
)
```

## Details
Because we execute code on a separate process, we need to re-register the tools inside
the new process. To do this, `register_tools` copies the source code and prepends it to
the code that is executed in the new process. But there's a catch, it cannot copy the
imports needed to run the tool code. To solve this, you must pass in the necessary
imports in the `register_tool` like so:

```python
import vision_agent as va

@va.register_tool(
    imports=["import cv2"],
)
def custom_tool(*args):
    # Your tool code here
    pass
```

This way the code executed in the new process will have the necessary imports to run.
