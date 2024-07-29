import json
import logging
import sys
from typing import Any, Dict

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)


def extract_json(json_str: str) -> Dict[str, Any]:
    try:
        json_dict = json.loads(json_str)
    except json.JSONDecodeError:
        input_json_str = json_str
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
            error_msg = f"Could not extract JSON from the given str: {json_str}.\nFunction input:\n{input_json_str}"
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
