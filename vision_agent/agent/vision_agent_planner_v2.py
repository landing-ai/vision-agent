import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from rich.console import Console
from rich.markup import escape

import vision_agent.tools as T
import vision_agent.tools.planner_tools as pt
from vision_agent.agent import Agent
from vision_agent.agent.agent_utils import (
    add_media_to_chat,
    capture_media_from_exec,
    extract_json,
    extract_tag,
    print_code,
    print_table,
)
from vision_agent.agent.types import AgentMessage, PlanContext
from vision_agent.agent.vision_agent_planner_prompts_v2 import (
    CRITIQUE_PLAN,
    EXAMPLE_PLAN1,
    EXAMPLE_PLAN2,
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

logging.basicConfig(level=logging.INFO)
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
_CONSOLE = Console()


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
    chat: List[AgentMessage],
) -> str:
    chat = copy.deepcopy(chat)
    planning = ""
    for chat_i in chat:
        if chat_i.role == "user":
            planning += f"USER: {chat_i.content}\n\n"
        elif chat_i.role == "observation":
            planning += f"OBSERVATION: {chat_i.content}\n\n"
        elif chat_i.role == "planner":
            planning += f"AGENT: {chat_i.content}\n\n"

    return planning


def run_planning(
    chat: List[AgentMessage],
    media_list: List[Union[str, Path]],
    model: LMM,
) -> str:
    # only keep last 10 messages for planning
    planning = get_planning(chat[-10:])
    prompt = PLAN.format(
        tool_desc=PLANNING_TOOLS_DOCSTRING,
        examples=f"{EXAMPLE_PLAN1}\n{EXAMPLE_PLAN2}",
        planning=planning,
        media_list=str(media_list),
    )

    message: Message = {"role": "user", "content": prompt}
    if chat[-1].role == "observation" and chat[-1].media is not None:
        message["media"] = chat[-1].media

    response = model.chat([message])
    return cast(str, response)


def run_multi_trial_planning(
    chat: List[AgentMessage],
    media_list: List[Union[str, Path]],
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
    if chat[-1].role == "observation" and chat[-1].media is not None:
        message["media"] = chat[-1].media

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
        return cast(str, responses[best])
    else:
        return cast(str, responses[0])


def run_critic(
    chat: List[AgentMessage], media_list: List[Union[str, Path]], model: LMM
) -> Optional[str]:
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
            fscore = float(score)
            if fscore < 8:
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


def execute_code_action(
    code: str,
    code_interpreter: CodeInterpreter,
    chat: List[AgentMessage],
    model: LMM,
    verbose: bool = False,
) -> Tuple[Execution, str, str]:
    if verbose:
        print_code("Code to Execute:", code)
    execution = code_interpreter.exec_cell(DefaultPlanningImports.prepend_imports(code))
    obs = execution.text(include_results=False).strip()
    if verbose:
        _CONSOLE.print(
            f"[bold cyan]Code Execution Output:[/bold cyan] [yellow]{escape(obs)}[/yellow]"
        )

    count = 1
    while not execution.success and count <= 3:
        prompt = FIX_BUG.format(chat_history=get_planning(chat), code=code, error=obs)
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
        if verbose:
            print_code(f"Fixing Bug Round {count}:", code)
            _CONSOLE.print(
                f"[bold cyan]Code Execution Output:[/bold cyan] [yellow]{escape(obs)}[/yellow]"
            )
        count += 1

    if obs.startswith("----- stdout -----\n"):
        obs = obs[19:]
    if obs.endswith("\n----- stderr -----"):
        obs = obs[:-19]
    return execution, obs, code


def find_and_replace_code(response: str, code: str) -> str:
    code_start = response.index("<execute_python>") + len("<execute_python>")
    code_end = response.index("</execute_python>")
    return response[:code_start] + code + response[code_end:]


def maybe_run_code(
    code: Optional[str],
    response: str,
    chat: List[AgentMessage],
    media_list: List[Union[str, Path]],
    model: LMM,
    code_interpreter: CodeInterpreter,
    verbose: bool = False,
) -> List[AgentMessage]:
    return_chat: List[AgentMessage] = []
    if code is not None:
        code = code_safeguards(code)
        execution, obs, code = execute_code_action(
            code, code_interpreter, chat, model, verbose
        )

        # if we had to debug the code to fix an issue, replace the old code
        # with the fixed code in the response
        fixed_response = find_and_replace_code(response, code)
        return_chat.append(
            AgentMessage(role="planner", content=fixed_response, media=None)
        )

        media_data = capture_media_from_exec(execution)
        int_chat_elt = AgentMessage(role="observation", content=obs, media=None)
        if media_list:
            int_chat_elt.media = cast(List[Union[str, Path]], media_data)
        return_chat.append(int_chat_elt)
    else:
        return_chat.append(AgentMessage(role="planner", content=response, media=None))
    return return_chat


def create_finalize_plan(
    chat: List[AgentMessage],
    model: LMM,
    verbose: bool = False,
) -> Tuple[List[AgentMessage], PlanContext]:
    prompt = FINALIZE_PLAN.format(
        planning=get_planning(chat),
        excluded_tools=str([t.__name__ for t in pt.PLANNER_TOOLS]),
    )
    response = model.chat([{"role": "user", "content": prompt}])
    plan_str = cast(str, response)
    return_chat = [AgentMessage(role="planner", content=plan_str, media=None)]

    plan_json = extract_tag(plan_str, "json")
    plan = (
        extract_json(plan_json)
        if plan_json is not None
        else {"plan": plan_str, "instructions": [], "code": ""}
    )
    code_snippets = extract_tag(plan_str, "code")
    plan["code"] = code_snippets if code_snippets is not None else ""
    if verbose:
        _CONSOLE.print(
            f"[bold cyan]Final Plan:[/bold cyan] [magenta]{plan['plan']}[/magenta]"
        )
        print_table("Plan", ["Instructions"], [[p] for p in plan["instructions"]])
        print_code("Plan Code", plan["code"])

    return return_chat, PlanContext(**plan)


def get_steps(chat: List[AgentMessage], max_steps: int) -> int:
    for chat_elt in reversed(chat):
        if "<count>" in chat_elt.content:
            return int(extract_tag(chat_elt.content, "count"))  # type: ignore
    return max_steps


class VisionAgentPlannerV2(Agent):
    def __init__(
        self,
        planner: Optional[LMM] = None,
        critic: Optional[LMM] = None,
        max_steps: int = 10,
        use_multi_trial_planning: bool = False,
        critique_steps: int = 11,
        verbose: bool = False,
        code_sandbox_runtime: Optional[str] = None,
        update_callback: Callable[[Dict[str, Any]], None] = lambda _: None,
    ) -> None:
        self.planner = (
            planner
            if planner is not None
            else AnthropicLMM(model_name="claude-3-5-sonnet-20241022", temperature=0.0)
        )
        self.critic = (
            critic
            if critic is not None
            else AnthropicLMM(model_name="claude-3-5-sonnet-20241022", temperature=0.0)
        )
        self.max_steps = max_steps
        self.use_multi_trial_planning = use_multi_trial_planning
        self.critique_steps = critique_steps

        self.verbose = verbose
        self.code_sandbox_runtime = code_sandbox_runtime
        self.update_callback = update_callback

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ) -> Union[str, List[Message]]:
        if isinstance(input, str):
            input_msg = [
                AgentMessage(
                    role="user",
                    content=input,
                    media=([media] if media is not None else None),
                )
            ]
        else:
            input_msg = [
                AgentMessage(role=msg["role"], content=msg["content"], media=None)
                for msg in input
            ]
            input_msg[0].media = [media] if media is not None else None
        plan = self.generate_plan(input_msg)
        return plan.plan

    def generate_plan(
        self,
        chat: List[AgentMessage],
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> PlanContext:
        if not chat:
            raise ValueError("Chat cannot be empty")

        chat = copy.deepcopy(chat)
        code_interpreter = code_interpreter or CodeInterpreterFactory.new_instance(
            self.code_sandbox_runtime
        )

        with code_interpreter:
            critque_steps = 1
            finished = False
            int_chat, _, media_list = add_media_to_chat(chat, code_interpreter)

            step = get_steps(int_chat, self.max_steps)
            if "<count>" not in int_chat[-1].content and step == self.max_steps:
                int_chat[-1].content += f"\n<count>{step}</count>\n"
            while step > 0 and not finished:
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

                if self.verbose:
                    _CONSOLE.print(
                        f"[bold cyan]Step {step}:[/bold cyan] [green]{thinking}[/green]"
                    )
                    if finalize_plan is not None:
                        _CONSOLE.print(
                            f"[bold cyan]Finalizing Plan:[/bold cyan] [magenta]{finalize_plan}[/magenta]"
                        )

                updated_chat = maybe_run_code(
                    code,
                    response,
                    int_chat,
                    media_list,
                    self.planner,
                    code_interpreter,
                    self.verbose,
                )

                if critque_steps % self.critique_steps == 0:
                    critique = run_critic(int_chat, media_list, self.critic)
                    if critique is not None and int_chat[-1].role == "observation":
                        _CONSOLE.print(
                            f"[bold cyan]Critique:[/bold cyan] [red]{critique}[/red]"
                        )
                        critique_str = f"\n[critique]\n{critique}\n[end of critique]"
                        updated_chat[-1].content += critique_str
                        # if plan was critiqued, ensure we don't finish so we can
                        # respond to the critique
                        finished = False

                critque_steps += 1
                step -= 1
                updated_chat[-1].content += f"\n<count>{step}</count>\n"
                int_chat.extend(updated_chat)
                for chat_elt in updated_chat:
                    self.update_callback(chat_elt.model_dump())

            updated_chat, plan_context = create_finalize_plan(
                int_chat, self.planner, self.verbose
            )
            int_chat.extend(updated_chat)
            for chat_elt in updated_chat:
                self.update_callback(chat_elt.model_dump())

        return plan_context

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
