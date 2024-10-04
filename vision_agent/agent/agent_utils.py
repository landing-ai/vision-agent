import json
import logging
import re
import sys
from typing import Any, Dict, Optional

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)


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
    json_str_mod = json_str_mod.replace("'", '"')
    json_str_mod = json_str_mod.replace(": True", ": true").replace(
        ": False", ": false"
    )

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
            raise ValueError(error_msg)

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


def remove_installs_from_code(code: str) -> str:
    pattern = r"\n!pip install.*?(\n|\Z)\n"
    code = re.sub(pattern, "", code, flags=re.DOTALL)
    return code
