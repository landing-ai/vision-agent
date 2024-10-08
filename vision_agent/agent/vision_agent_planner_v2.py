import copy
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import vision_agent.tools as T
import vision_agent.tools.planner_tools as pt
from vision_agent.agent import Agent
from vision_agent.agent.agent_utils import (
    extract_execution,
    extract_finalize_plan,
    extract_json,
    extract_thinking,
    print_code,
)
from vision_agent.agent.vision_agent_planner_prompts import (
    EXAMPLE_PLAN1,
    FINALIZE_PLAN,
    PLAN2,
)
from vision_agent.lmm import LMM, AnthropicLMM, Message
from vision_agent.utils.execute import (
    CodeInterpreter,
    CodeInterpreterFactory,
    Execution,
)

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)
READ_DOCSTRING = T.get_tool_documentation(
    [T.load_image, T.extract_frames_and_timestamps]
)
PLANNING_TOOLS_DOCSTRING = READ_DOCSTRING + "\n" + pt.PLANNER_DOCSTRING


def get_planning(
    chat: List[Message],
) -> str:
    chat = copy.deepcopy(chat)
    planning = ""
    for chat_i in chat:
        if chat_i["role"] == "user":
            planning += f"USER: {chat_i['content']}\n\n"
        elif chat_i["role"] == "observation":
            planning += f"OBSERVATION: {chat_i['content']}\n\n"
        elif chat_i["role"] == "assistant":
            planning += f"ASSISTANT: {chat_i['content']}\n\n"
        else:
            raise ValueError(f"Unknown role: {chat_i['role']}")

    return planning


def run_planning(
    chat: List[Message],
    model: LMM,
) -> str:
    planning = get_planning(chat)
    prompt = PLAN2.format(
        tool_desc=PLANNING_TOOLS_DOCSTRING,
        examples=EXAMPLE_PLAN1,
        planning=planning,
    )

    response = cast(str, model.generate(prompt))
    return response


def execute_code_action(
    code: str, code_interpreter: CodeInterpreter
) -> Tuple[Execution, str]:
    execution = code_interpreter.exec_cell(code)
    obs = execution.text(include_results=False).strip()
    return execution, obs


class VisionAgentPlannerV2(Agent):
    def __init__(
        self,
        planner: Optional[LMM] = None,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbosity: int = 0,
        max_steps: int = 20,
        code_sandbox_runtime: Optional[str] = None,
    ) -> None:
        self.planner = planner if planner is not None else AnthropicLMM(temperature=0.0)
        self.report_progress_callback = report_progress_callback
        self.max_steps = max_steps
        self.code_sandbox_runtime = code_sandbox_runtime
        self.verbosity = verbosity

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ) -> Union[str, List[Message]]:
        if isinstance(input, str):
            if media is not None:
                input = [{"role": "user", "content": input, "media": [media]}]
            else:
                input = [{"role": "user", "content": input}]
        plan = self.generate_plan(input)
        return str(plan)

    def generate_plan(
        self,
        chat: List[Message],
    ) -> Dict[str, Any]:
        if not chat:
            raise ValueError("Chat cannot be empty")

        with CodeInterpreterFactory.new_instance(
            self.code_sandbox_runtime
        ) as code_interpreter:
            orig_chat = copy.deepcopy(chat)
            int_chat = copy.deepcopy(chat)
            media_list = []
            for chat_i in chat:
                if "media" in chat_i:
                    for media in chat_i["media"]:
                        media = (
                            media
                            if type(media) is str
                            and media.startswith(("http", "https"))
                            else code_interpreter.upload_file(cast(str, media))
                        )
                        chat_i["content"] += f" Media name {media}"  # type: ignore
                        media_list.append(str(media))

            int_chat = cast(
                List[Message],
                [
                    (
                        {
                            "role": c["role"],
                            "content": c["content"],
                            "media": c["media"],
                        }
                        if "media" in c
                        else {"role": c["role"], "content": c["content"]}
                    )
                    for c in int_chat
                ],
            )

            step = 0
            finished = False
            while step < self.max_steps or not finished:
                response = run_planning(chat, self.planner)

                __import__("ipdb").set_trace()
                thinking = extract_thinking(response)
                code = extract_execution(response)
                finalize_plan = extract_finalize_plan(response)

                if self.verbosity > 0:
                    _LOGGER.info(f"Step {step}: {thinking}")

                int_chat.append({"role": "assistant", "content": response})
                orig_chat.append({"role": "assistant", "content": response})

                if finalize_plan is not None:
                    _LOGGER.info(f"Finalizing plan: {finalize_plan}")
                    break

                if code is not None:
                    print_code("Execute Code:", code)
                    _, obs = execute_code_action(code, code_interpreter)
                    if self.verbosity > 0:
                        _LOGGER.info(f"Code Execution: {obs}")
                    int_chat.append({"role": "observation", "content": obs})
                    orig_chat.append({"role": "observation", "content": obs})

                step += 1

            prompt = FINALIZE_PLAN.format(
                user_request=chat[-1]["content"],
                planning=get_planning(chat),
            )

            plan_str = cast(str, self.planner.generate(prompt))
            plan = extract_json(plan_str)

        return plan

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
