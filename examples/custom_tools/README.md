# Template Matching Custom Tool

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

agent = va.agent.VisionAgent(verbose=True)
agent(
    "Can you use the 'template_match_' tool to find the location of pid_template.png in pid.png?",
    image="pid.png",
    reference_data={"image": "pid_template.png"},
)
```
