## V2 Agents

This gives an overview of all the V2 agents, how they communicate, and how human-in-the-loop works.

- `vision_agent_v2` - This is the conversation agent. It can take exactly one action and one response. The actions are fixed JSON actions (meaning it does not execute code but instead returns a JSON and we execute the code on it's behalf). This is so we can control the action arguments as well as pass around the notebook code interpreter.
- `vision_agent_planner_v2` - This agent is responsible for planning. It can run for N turns, each turn it can take some code action (it can execute it's own python code and has access to `planner_tools`) to test a new potential step in the plan.
- `vision_agent_coder_v2` - This agent is responsible for the final code. It can call the planner on it's own or it can take in the final `PlanContext` returned by `vision_agent_planner_v2` and use that to write the final code and test it.

### Communication
The agents communicate through `AgentMessage`'s and return `PlanContext`'s and `CodeContext`'s for the planner and coder agent respectively.
```
_______________
|VisionAgentV2|
---------------
       |                       ____________________
       -----(AgentMessage)---> |VisionAgentCoderV2|
                               --------------------
                                         |                        ______________________
                                         -----(AgentMessage)----> |VisionAgentPlannerV2|
                                                                  ----------------------
                               ____________________                         |
                               |VisionAgentCoderV2| <----(PlanContext)-------
                               --------------------
_______________                          |
|VisionAgentV2|<-----(CodeContext)--------
---------------                    
```

#### AgentMessage and Contexts
`AgentMessage` is a basic chat message but with extended roles. The roles can be your typical `user` and `assistant` but can also be `conversation`, `planner` or `coder` where the come from `VisionAgentV2`, `VisionAgentPlannerV2` and `VisionAgentCoderV2` respectively. `conversation`, `planner` and `coder` are all types of `assistant`. `observation`'s come from responses from executing python code internally by the planner.

The `VisionAgentPlannerV2` returns `PlanContext` which contains the a finalized version of the plan, including instructions and code snippets used during planning. `VisionAgentCoderV2` will then take in that `PlanContext` and return a `CodeContext` which contains the final code and any additional information.


#### Callbacks
If you want to recieve intermediate messages you can use the `update_callback` argument in all the `V2` constructors. This will asynchronously send `AgentMessage`'s to the callback function you provide. You can see an example of how to run this in `examples/chat/app.py`

### Human-in-the-loop
Human-in-the-loop is a feature that allows the user to interact with the agents at certain points in the conversation. This is handled by using the `interaction` and `interaction_response` roles in the `AgentMessage`. You can enable this feature by passing `hil=True` to the `VisionAgentV2`, currently you can only use human-in-the-loop if you are also using the `update_callback` to collect the messages and pass them back to `VisionAgentV2`.

When the planner agent wants to interact with a human, it will return `InteractionContext` which will propogate back up to `VisionAgentV2` and then to the user. This exits the planner so it can ask for human input. If you collect the messages from `update_callback`, you will see the last `AgentMessage` has a role of `interaction` and the content will include a JSON string surrounded by `<interaction>` tags:

```
AgentMessage(
    role="interaction",
    content="<interaction>{\"prompt\": \"Should I use owl_v2_image or countgd_counting?\"}</interaction>",
    media=None,
)
```

The user can then add an additional `AgentMessage` with the role `interaction_response` and the response they want to give:

```
AgentMessage(
    role="interaction_response",
    content="{\"function_name\": \"owl_v2_image\"}",
    media=None,
)
```

You can see an example of how this works in `examples/chat/chat-app/src/components/ChatSection.tsx` under the `handleSubmit` function.


#### Human-in-the-loop Caveats
One issue with this approach is the planner is running code on a notebook and has access to all the previous executions. This means that for the internal notebook that the agents use, usually named `code_interpreter` from `CodeInterpreterFactor.new_instance`, you cannot close it or restart it. This is an issue because the notebook will close once you exit from the `VisionAgentV2` call.

To fix this you must construct a notebook outside of the chat and ensure it has `non_exiting=True`:

```python
agent = VisionAgentV2(
    verbose=True,
    update_callback=update_callback,
    hil=True,
)

code_interpreter = CodeInterpreterFactory.new_instance(non_exiting=True)
agent.chat(
    [
        AgentMessage(
            role="user",
            content="Hello",
            media=None
        )
    ],
    code_interpreter=code_interpreter
)
```

An example of this can be seen in `examples/chat/app.py`. Here the `code_intepreter` is constructed outside of the chat and passed in. This is so the notebook does not close when the chat ends or returns when asking for human feedback.
