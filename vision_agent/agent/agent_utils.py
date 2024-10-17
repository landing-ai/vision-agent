import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.style import Style
from rich.syntax import Syntax

import vision_agent.tools as T

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


def format_memory(memory: List[Dict[str, str]]) -> str:
    output_str = ""
    for i, m in enumerate(memory):
        output_str += f"### Feedback {i}:\n"
        output_str += f"Code {i}:\n```python\n{m['code']}```\n\n"
        output_str += f"Feedback {i}: {m['feedback']}\n\n"
        if "edits" in m:
            output_str += f"Edits {i}:\n{m['edits']}\n"
        output_str += "\n"

    return output_str


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
            DefaultImports.prepend_imports(code),
            "python",
            theme="gruvbox-dark",
            line_numbers=True,
        )
    )
    if test:
        _CONSOLE.print("=" * 30 + " Test " + "=" * 30)
        _CONSOLE.print(Syntax(test, "python", theme="gruvbox-dark", line_numbers=True))
