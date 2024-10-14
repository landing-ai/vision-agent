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
    extract_tag,
    extract_finalize_plan,
    extract_json,
    extract_thinking,
    print_code,
)
from vision_agent.agent.vision_agent_planner_prompts import (
    EXAMPLE_PLAN1,
    FINALIZE_PLAN,
    FIX_BUG,
    PLAN2,
)
from vision_agent.lmm import LMM, AnthropicLMM, Message
from vision_agent.utils.execute import (
    CodeInterpreter,
    CodeInterpreterFactory,
    Execution,
)

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
READ_DOCSTRING = T.get_tool_documentation(
    [T.load_image, T.extract_frames_and_timestamps]
)
PLANNING_TOOLS_DOCSTRING = READ_DOCSTRING + "\n" + pt.PLANNER_DOCSTRING


class DefaultPlanningImports:
    imports = [
        "import os",
        "import numpy as np",
        "from typing import *",
        "from vision_agent.tools import *",
        "from vision_agent.tools.planner_tools import claude35_vqa, get_tool_for_task",
        "from pillow_heif import register_heif_opener",
        "register_heif_opener()",
    ]

    @staticmethod
    def prepend_imports(code: str) -> str:
        return "\n".join(DefaultPlanningImports.imports) + "\n\n" + code


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


def code_safeguards(code: str) -> str:
    if "get_tool_for_task" in code:
        lines = code.split("\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if "get_tool_for_task" in line:
                break
        code = "\n".join(new_lines)
    return code


def execute_code_action(
    code: str, code_interpreter: CodeInterpreter, verbosity: int = 0
) -> Tuple[Execution, str, str]:
    execution = code_interpreter.exec_cell(DefaultPlanningImports.prepend_imports(code))
    obs = execution.text(include_results=False).strip()
    count = 1
    if verbosity > 0:
        print_code("Execute Code:", code)
        _LOGGER.info(f"Code Execution Output: {obs}")

    while not execution.success and count <= 3:
        code = FIX_BUG.format(code=code, error=obs)
        execution = code_interpreter.exec_cell(
            DefaultPlanningImports.prepend_imports(code)
        )
        obs = execution.text(include_results=False).strip()
        if verbosity > 0:
            print_code(f"Execute Code Round {count}:", code)
            _LOGGER.info(f"Code Execution Output: {obs}")
        count += 1

    if obs.startswith("----- stdout -----\n"):
        obs = obs[19:]
    if obs.endswith("\n----- stderr -----"):
        obs = obs[:-19]
    return execution, obs, code


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
        if self.verbosity >= 1:
            _LOGGER.setLevel(logging.INFO)

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
            for chat_i in int_chat:
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
                __import__("ipdb").set_trace()
                response = run_planning(int_chat, self.planner)
                if self.verbosity > 1:
                    _LOGGER.info(f"Response: {response}")

                thinking = extract_thinking(response)
                code = extract_execution(response)
                finalize_plan = extract_finalize_plan(response)

                if self.verbosity > 0:
                    _LOGGER.info(f"Step {step}: {thinking}")

                int_chat.append({"role": "assistant", "content": response})
                orig_chat.append({"role": "assistant", "content": response})

                if code is not None:
                    code = code_safeguards(code)
                    _, obs, code = execute_code_action(
                        code, code_interpreter, self.verbosity
                    )
                    int_chat.append({"role": "observation", "content": obs})
                    orig_chat.append({"role": "observation", "content": obs})

                if finalize_plan is not None:
                    _LOGGER.info(f"Finalizing plan: {finalize_plan}")
                    break

                step += 1

            prompt = FINALIZE_PLAN.format(
                planning=get_planning(int_chat),
                excluded_tools=str([t.__name__ for t in pt.PLANNER_TOOLS]),
            )

            __import__("ipdb").set_trace()
            plan_str = cast(str, self.planner.generate(prompt))
            plan = extract_json(plan_str)
            code_snippets = extract_tag("code", plan_str)
            plan["code"] = code_snippets
            if self.verbosity > 0:
                _LOGGER.info(f"Final Plan: {plan}")

        return plan

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
