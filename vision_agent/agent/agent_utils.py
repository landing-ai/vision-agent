import copy
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import libcst as cst
from rich.console import Console
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table

import vision_agent.tools as T
from vision_agent.agent.types import AgentMessage, PlanContext
from vision_agent.lmm.types import Message
from vision_agent.utils.execute import CodeInterpreter, Execution
from vision_agent.utils.image_utils import b64_to_pil, convert_to_b64

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)
_CONSOLE = Console()
_MAX_TABULATE_COL_WIDTH = 80


def _extract_sub_json(json_str: str) -> Optional[Dict[str, Any]]:
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, json_str, re.DOTALL)
    if match:
        json_str = match.group()
        try:
            # remove trailing comma
            trailing_bracket_pattern = r",\s+\}"
            json_str = re.sub(trailing_bracket_pattern, "}", json_str, flags=re.DOTALL)

            json_dict = json.loads(json_str)
            return json_dict  # type: ignore
        except json.JSONDecodeError:
            return None
    return None


def _find_markdown_json(json_str: str) -> str:
    pattern = r"```json(.*?)```"
    match = re.search(pattern, json_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return json_str


def _strip_markdown_code(inp_str: str) -> str:
    pattern = r"```python.*?```"
    cleaned_str = re.sub(pattern, "", inp_str, flags=re.DOTALL)
    return cleaned_str


def extract_json(json_str: str) -> Dict[str, Any]:
    json_str_mod = json_str.replace("\n", " ").strip()
    json_str_mod = json_str_mod.replace(": True", ": true").replace(
        ": False", ": false"
    )

    # sometimes the json is in single quotes
    try:
        return json.loads(json_str_mod.replace("'", '"'))  # type: ignore
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(json_str_mod)  # type: ignore
    except json.JSONDecodeError:
        json_orig = json_str
        # don't replace quotes here or booleans since it can also introduce errors
        json_str = json_str.replace("\n", " ").strip()
        json_str = _strip_markdown_code(json_str)
        json_str = _find_markdown_json(json_str)
        json_dict = _extract_sub_json(json_str)

        if json_dict is None:
            error_msg = f"Could not extract JSON from the given str: {json_orig}"
            _LOGGER.exception(error_msg)
            raise json.JSONDecodeError(
                msg="Could not extract JSON", doc=json_orig, pos=0
            )

        return json_dict


def extract_code(code: str) -> str:
    if "\n```python" in code:
        start = "\n```python"
    elif "```python" in code:
        start = "```python"
    else:
        return code

    code = code[code.find(start) + len(start) :]
    code = code[: code.find("```")]
    if code.startswith("python\n"):
        code = code[len("python\n") :]
    return code


def extract_tag(
    content: str,
    tag: str,
) -> Optional[str]:
    inner_content = None
    remaning = content
    all_inner_content = []

    while f"<{tag}>" in remaning:
        inner_content_i = remaning[remaning.find(f"<{tag}>") + len(f"<{tag}>") :]
        if f"</{tag}>" not in inner_content_i:
            break
        inner_content_i = inner_content_i[: inner_content_i.find(f"</{tag}>")]
        remaning = remaning[remaning.find(f"</{tag}>") + len(f"</{tag}>") :]
        all_inner_content.append(inner_content_i)

    if len(all_inner_content) > 0:
        inner_content = "\n".join(all_inner_content)
    return inner_content


def remove_installs_from_code(code: str) -> str:
    pattern = r"\n!pip install.*?(\n|\Z)\n"
    code = re.sub(pattern, "", code, flags=re.DOTALL)
    return code


def format_feedback(memory: List[Dict[str, str]]) -> str:
    output_str = ""
    for i, m in enumerate(memory):
        output_str += f"### Feedback {i}:\n"
        output_str += f"Code {i}:\n```python\n{m['code']}```\n\n"
        output_str += f"Feedback {i}: {m['feedback']}\n\n"
        if "edits" in m:
            output_str += f"Edits {i}:\n{m['edits']}\n"
        output_str += "\n"

    return output_str


def format_plan_v2(plan: PlanContext) -> str:
    plan_str = plan.plan + "\n"
    plan_str += "Instructions:\n"
    for v in plan.instructions:
        plan_str += f"    - {v}\n"
    plan_str += "Code:\n"
    plan_str += plan.code
    return plan_str


def format_plans(plans: Dict[str, Any]) -> str:
    plan_str = ""
    for k, v in plans.items():
        plan_str += "\n" + f"{k}: {v['thoughts']}\n"
        plan_str += "    -" + "\n    -".join([e for e in v["instructions"]])

    return plan_str


class DefaultImports:
    """Container for default imports used in the code execution."""

    common_imports = [
        "import os",
        "import numpy as np",
        "from vision_agent.tools import *",
        "from typing import *",
        "from pillow_heif import register_heif_opener",
        "register_heif_opener()",
    ]

    @staticmethod
    def to_code_string() -> str:
        return "\n".join(DefaultImports.common_imports + T.__new_tools__)

    @staticmethod
    def prepend_imports(code: str) -> str:
        """Run this method to prepend the default imports to the code.
        NOTE: be sure to run this method after the custom tools have been registered.
        """
        return DefaultImports.to_code_string() + "\n\n" + code


def print_code(title: str, code: str, test: Optional[str] = None) -> None:
    _CONSOLE.print(title, style=Style(bgcolor="dark_orange3", bold=True))
    _CONSOLE.print("=" * 30 + " Code " + "=" * 30)
    _CONSOLE.print(
        Syntax(
            code,
            "python",
            theme="gruvbox-dark",
            line_numbers=True,
            word_wrap=True,
        )
    )
    if test:
        _CONSOLE.print("=" * 30 + " Test " + "=" * 30)
        _CONSOLE.print(Syntax(test, "python", theme="gruvbox-dark", line_numbers=True))


def print_table(title: str, columns: List[str], rows: List[List[str]]) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for col in columns:
        table.add_column(col, style="cyan", no_wrap=True)

    for i, row in enumerate(rows):
        table.add_row(*row)
        if i < len(rows) - 1:
            table.add_row(*["-" * len(col) for col in row])
    _CONSOLE.print(table)


def add_media_to_chat(
    chat: List[AgentMessage], code_interpreter: Optional[CodeInterpreter] = None
) -> Tuple[List[AgentMessage], List[AgentMessage], List[Union[str, Path]]]:
    orig_chat = copy.deepcopy(chat)
    int_chat = copy.deepcopy(chat)
    media_list: List[Union[str, Path]] = []
    for chat_i in int_chat:
        if chat_i.media is not None:
            media_list_i: List[Union[str, Path]] = []
            for media in chat_i.media:
                if isinstance(media, str) and media.startswith("data:image/"):
                    media_pil = b64_to_pil(media)
                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".png", delete=False
                    ) as temp_file:
                        media_pil.save(temp_file, format="PNG")
                        media = str(temp_file.name)
                if code_interpreter is not None:
                    media = str(code_interpreter.upload_file(media))
                media_list_i.append(media)
                # don't duplicate appending media name and only add them for user messages
                if (
                    not str(chat_i.content).endswith(f" Media name {media}")
                    and chat_i.role == "user"
                ):
                    chat_i.content += f" Media name {media}"
            chat_i.media = media_list_i if len(media_list_i) > 0 else None
            media_list.extend(media_list_i)

    int_chat = cast(
        List[AgentMessage],
        [
            (
                AgentMessage(
                    role=c.role,
                    content=c.content,
                    media=c.media,
                )
                if c.media is not None
                else AgentMessage(role=c.role, content=c.content, media=None)
            )
            for c in int_chat
        ],
    )
    return int_chat, orig_chat, media_list


def capture_media_from_exec(execution: Execution) -> List[str]:
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


def convert_message_to_agentmessage(
    input: Union[str, List[Message]],
    media: Optional[Union[str, Path]] = None,
) -> List[AgentMessage]:
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
    return input_msg


def strip_function_calls(  # noqa: C901
    code: str, exclusions: Optional[List[str]] = None
) -> str:
    """This will strip out all code that calls functions except for functions included
    in exclusions.
    """
    if exclusions is None:
        exclusions = []

    def check_and_remove_node(node: cst.CSTNode, exclusions: List[str]) -> cst.CSTNode:
        if hasattr(node, "value") and isinstance(node.value, cst.Call):
            if (
                isinstance(node.value.func, cst.Name)
                and node.value.func.value in exclusions
            ):
                return node
            return cst.RemoveFromParent()  # type: ignore
        return node

    class StripFunctionCallsTransformer(cst.CSTTransformer):
        def __init__(self, exclusions: List[str]):
            # Store exclusions to skip removing certain function calls
            self.exclusions = exclusions
            self.in_function_or_class = False

        def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
            self.in_function_or_class = True
            return True

        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.BaseStatement:
            self.in_function_or_class = False
            return updated_node

        def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
            self.in_function_or_class = True
            return True

        def leave_ClassDef(
            self, node: cst.ClassDef, updated_node: cst.ClassDef
        ) -> cst.BaseStatement:
            self.in_function_or_class = False
            return updated_node

        def leave_Expr(
            self, original_node: cst.Expr, updated_node: cst.Expr
        ) -> cst.Expr:
            if not self.in_function_or_class:
                return cast(
                    cst.Expr, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_Assign(
            self, original_node: cst.Assign, updated_node: cst.Assign
        ) -> cst.Assign:
            if not self.in_function_or_class:
                return cast(
                    cst.Assign, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
            if not self.in_function_or_class:
                return cast(
                    cst.If, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
            if not self.in_function_or_class:
                return cast(
                    cst.For, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_While(
            self, original_node: cst.While, updated_node: cst.While
        ) -> cst.While:
            if not self.in_function_or_class:
                return cast(
                    cst.While, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_With(
            self, original_node: cst.With, updated_node: cst.With
        ) -> cst.With:
            if not self.in_function_or_class:
                return cast(
                    cst.With, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_Try(self, original_node: cst.Try, updated_node: cst.Try) -> cst.Try:
            if not self.in_function_or_class:
                return cast(
                    cst.Try, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

    tree = cst.parse_module(code)
    transformer = StripFunctionCallsTransformer(exclusions)
    modified_tree = tree.visit(transformer)
    return modified_tree.code
