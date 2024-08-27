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
            json_dict = json.loads(json_str)
            return json_dict  # type: ignore
        except json.JSONDecodeError:
            return None
    return None


def extract_json(json_str: str) -> Dict[str, Any]:
    try:
        json_str = json_str.replace("\n", " ")
        json_dict = json.loads(json_str)
    except json.JSONDecodeError:
        if "```json" in json_str:
            json_str = json_str[json_str.find("```json") + len("```json") :]
            json_str = json_str[: json_str.find("```")]
        elif "```" in json_str:
            json_str = json_str[json_str.find("```") + len("```") :]
            # get the last ``` not one from an intermediate string
            json_str = json_str[: json_str.find("}```")]
        try:
            json_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            json_dict = _extract_sub_json(json_str)
            if json_dict is not None:
                return json_dict  # type: ignore
            error_msg = f"Could not extract JSON from the given str: {json_str}"
            _LOGGER.exception(error_msg)
            raise ValueError(error_msg) from e

    return json_dict  # type: ignore


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
