import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np

import vision_agent.tools as T
import vision_agent.tools.planner_tools as pt
from vision_agent.agent import Agent
from vision_agent.agent.agent_utils import (
    extract_json,
    extract_tag,
    print_code,
    print_table,
)
from vision_agent.agent.vision_agent_planner_prompts_v2 import (
    CRITIQUE_PLAN,
    EXAMPLE_PLAN1,
    FINALIZE_PLAN,
    FIX_BUG,
    PICK_PLAN,
    PLAN,
)
from vision_agent.lmm import LMM, AnthropicLMM, Message
from vision_agent.utils.execute import (
    CodeInterpreter,
    CodeInterpreterFactory,
    Execution,
)
from vision_agent.utils.image_utils import b64_to_pil, convert_to_b64

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
UTIL_DOCSTRING = T.get_tool_documentation(
    [
        T.load_image,
        T.extract_frames_and_timestamps,
        T.save_image,
        T.save_video,
        T.overlay_bounding_boxes,
        T.overlay_segmentation_masks,
    ]
)
PLANNING_TOOLS_DOCSTRING = UTIL_DOCSTRING + "\n" + pt.PLANNER_DOCSTRING


class DefaultPlanningImports:
    imports = [
        "import os",
        "import numpy as np",
        "import cv2",
        "from typing import *",
        "from vision_agent.tools import *",
        "from vision_agent.tools.planner_tools import claude35_vqa, suggestion, get_tool_for_task",
        "from pillow_heif import register_heif_opener",
        "register_heif_opener()",
        "import matplotlib.pyplot as plt",
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
    media_list: List[str],
    model: LMM,
) -> str:
    planning = get_planning(chat)
    prompt = PLAN.format(
        tool_desc=PLANNING_TOOLS_DOCSTRING,
        examples=EXAMPLE_PLAN1,
        planning=planning,
        media_list=str(media_list),
    )

    message: Message = {"role": "user", "content": prompt}
    if chat[-1]["role"] == "observation" and "media" in chat[-1]:
        message["media"] = chat[-1]["media"]

    response = model.chat([message])
    return cast(str, response)


def run_multi_trial_planning(
    chat: List[Message],
    media_list: List[str],
    model: LMM,
) -> str:
    planning = get_planning(chat)
    prompt = PLAN.format(
        tool_desc=PLANNING_TOOLS_DOCSTRING,
        examples=EXAMPLE_PLAN1,
        planning=planning,
        media_list=str(media_list),
    )

    message: Message = {"role": "user", "content": prompt}
    if chat[-1]["role"] == "observation" and "media" in chat[-1]:
        message["media"] = chat[-1]["media"]

    responses = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(lambda: model.chat([message], temperature=1.0))
            for _ in range(3)
        ]
        for future in as_completed(futures):
            responses.append(future.result())

    prompt = PICK_PLAN.format(
        planning=planning,
        response1=responses[0],
        response2=responses[1],
        response3=responses[2],
    )
    response = cast(str, model.chat([{"role": "user", "content": prompt}]))
    json_str = extract_tag(response, "json")
    if json_str:
        json_data = extract_json(json_str)
        best = np.argmax([int(json_data[f"response{k}"]) for k in [1, 2, 3]])
        return responses[best]
    else:
        return responses[0]


def run_critic(chat: List[Message], media_list: List[str], model: LMM) -> Optional[str]:
    planning = get_planning(chat)
    prompt = CRITIQUE_PLAN.format(
        planning=planning,
    )
    message: Message = {"role": "user", "content": prompt}
    if len(media_list) > 0:
        message["media"] = media_list

    response = cast(str, model.chat([message]))
    score = extract_tag(response, "score")
    thoughts = extract_tag(response, "thoughts")
    if score is not None and thoughts is not None:
        try:
            score = float(score)
            if score < 8:
                return thoughts
        except ValueError:
            pass
    return None


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


def response_safeguards(response: str) -> str:
    if "<execute_python>" in response:
        response = response[
            : response.index("</execute_python>") + len("</execute_python>")
        ]
    return response


def fix_chat(
    chat: List[Message], code_interpreter: CodeInterpreter
) -> Tuple[List[Message], List[Message], List[str]]:
    orig_chat = copy.deepcopy(chat)
    int_chat = copy.deepcopy(chat)
    media_list = []
    for chat_i in int_chat:
        if "media" in chat_i:
            for media in chat_i["media"]:
                media = (
                    media
                    if type(media) is str and media.startswith(("http", "https"))
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
    return int_chat, orig_chat, media_list


def execute_code_action(
    code: str,
    code_interpreter: CodeInterpreter,
    chat: List[Message],
    model: LMM,
    verbosity: int = 0,
) -> Tuple[Execution, str, str]:
    if verbosity > 0:
        print_code("Code to Execute:", code)
    execution = code_interpreter.exec_cell(DefaultPlanningImports.prepend_imports(code))
    obs = execution.text(include_results=False).strip()
    if verbosity > 0:
        _LOGGER.info(f"Code Execution Output: {obs}")

    count = 1
    while not execution.success and count <= 3:
        prompt = cast(
            str, FIX_BUG.format(chat_history=get_planning(chat), code=code, error=obs)
        )
        response = cast(str, model.chat([{"role": "user", "content": prompt}]))
        new_code = extract_tag(response, "code")
        if not new_code:
            continue
        else:
            code = new_code

        execution = code_interpreter.exec_cell(
            DefaultPlanningImports.prepend_imports(code)
        )
        obs = execution.text(include_results=False).strip()
        if verbosity > 0:
            print_code(f"Fixing Bug Round {count}:", code)
            _LOGGER.info(f"Code Execution Output: {obs}")
        count += 1

    if obs.startswith("----- stdout -----\n"):
        obs = obs[19:]
    if obs.endswith("\n----- stderr -----"):
        obs = obs[:-19]
    return execution, obs, code


def capture_images_from_exec(execution: Execution) -> List[str]:
    images = []
    for result in execution.results:
        for format in result.formats():
            if format in ["png", "jpeg"]:
                # converts the image to png and then to base64
                images.append(
                    "data:image/png;base64,"
                    + convert_to_b64(b64_to_pil(result[format]))
                )
    return images


def find_and_replace_code(response: str, code: str) -> str:
    code_start = response.index("<execute_python>") + len("<execute_python>")
    code_end = response.index("</execute_python>")
    return response[:code_start] + code + response[code_end:]


class VisionAgentPlannerV2(Agent):
    def __init__(
        self,
        planner: Optional[LMM] = None,
        critic: Optional[LMM] = None,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbosity: int = 0,
        max_steps: int = 10,
        code_sandbox_runtime: Optional[str] = None,
        use_multi_trial_planning: bool = False,
        critique_steps: int = 3,
    ) -> None:
        self.planner = planner if planner is not None else AnthropicLMM(temperature=0.0)
        self.critic = critic if critic is not None else AnthropicLMM(temperature=0.0)
        self.report_progress_callback = report_progress_callback
        self.max_steps = max_steps
        self.code_sandbox_runtime = code_sandbox_runtime
        self.use_multi_trial_planning = use_multi_trial_planning
        self.critique_steps = critique_steps
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
            int_chat, orig_chat, media_list = fix_chat(chat, code_interpreter)
            critque_steps = 1
            step = 1
            finished = False
            while step < self.max_steps or not finished:
                if self.use_multi_trial_planning:
                    response = run_multi_trial_planning(
                        int_chat, media_list, self.planner
                    )
                else:
                    response = run_planning(int_chat, media_list, self.planner)
                response = response_safeguards(response)

                thinking = extract_tag(response, "thinking")
                code = extract_tag(response, "execute_python")
                finalize_plan = extract_tag(response, "finalize_plan")
                finished = finalize_plan is not None

                if self.verbosity > 0:
                    _LOGGER.info(f"Step {step}: {thinking}")
                    if finalize_plan is not None:
                        _LOGGER.info(f"Finalizing Plan: {finalize_plan}")

                if code is not None:
                    code = code_safeguards(code)
                    execution, obs, code = execute_code_action(
                        code, code_interpreter, int_chat, self.planner, self.verbosity
                    )

                    # if we had to debug the code to fix an issue, replace the old code
                    # with the fixed code in the response
                    fixed_response = find_and_replace_code(response, code)
                    int_chat.append({"role": "assistant", "content": fixed_response})
                    orig_chat.append({"role": "assistant", "content": fixed_response})

                    media_data = capture_images_from_exec(execution)
                    int_chat_elt: Message = {"role": "observation", "content": obs}
                    if media_list:
                        int_chat_elt["media"] = media_data
                    int_chat.append(int_chat_elt)
                    orig_chat.append({"role": "observation", "content": obs})
                else:
                    int_chat.append({"role": "assistant", "content": response})
                    orig_chat.append({"role": "assistant", "content": response})

                if critque_steps % self.critique_steps == 0:
                    critique = run_critic(int_chat, media_list, self.critic)
                    if critique is not None and int_chat[-1]["role"] == "observation":
                        _LOGGER.info(f"Critique: {critique}")
                        critique_str = f"\n[critique]\n{critique}\n[end of critique]"
                        int_chat[-1]["content"] += critique_str
                        orig_chat[-1]["content"] += critique_str
                        # if plan was critiqued, ensure we don't finish so we can
                        # respond to the critique
                        finished = False

                critque_steps += 1
                step += 1

            prompt = FINALIZE_PLAN.format(
                planning=get_planning(int_chat),
                excluded_tools=str([t.__name__ for t in pt.PLANNER_TOOLS]),
            )

            plan_str = cast(str, self.planner.generate(prompt))
            plan = extract_json(extract_tag(plan_str, "json"))
            code_snippets = extract_tag(plan_str, "code")
            plan["code"] = code_snippets
            if self.verbosity > 0:
                _LOGGER.info(f"Final Plan: {plan['plan']}")
                print_table(
                    "Plan", ["Instructions"], [[p] for p in plan["instructions"]]
                )
                print_code("Plan Code", code_snippets)

        return plan

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
