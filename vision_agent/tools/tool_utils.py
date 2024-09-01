import inspect
import logging
import os
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple

import pandas as pd
from IPython.display import display
from pydantic import BaseModel
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vision_agent.utils.exceptions import RemoteToolCallFailed
from vision_agent.utils.execute import Error, MimeType
from vision_agent.utils.type_defs import LandingaiAPIKey

_LOGGER = logging.getLogger(__name__)
_LND_API_KEY = os.environ.get("LANDINGAI_API_KEY", LandingaiAPIKey().api_key)
_LND_BASE_URL = os.environ.get("LANDINGAI_URL", "https://api.landing.ai")
_LND_API_URL = f"{_LND_BASE_URL}/v1/agent/model"
_LND_API_URL_v2 = f"{_LND_BASE_URL}/v1/tools"


class ToolCallTrace(BaseModel):
    endpoint_url: str
    request: MutableMapping[str, Any]
    response: MutableMapping[str, Any]
    error: Optional[Error]


def send_inference_request(
    payload: Dict[str, Any],
    endpoint_name: str,
    files: Optional[List[Tuple[Any, ...]]] = None,
    v2: bool = False,
    metadata_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # TODO: runtime_tag and function_name should be metadata_payload and now included
    # in the service payload
    try:
        if runtime_tag := os.environ.get("RUNTIME_TAG", ""):
            payload["runtime_tag"] = runtime_tag

        url = f"{_LND_API_URL_v2 if v2 else _LND_API_URL}/{endpoint_name}"
        if "TOOL_ENDPOINT_URL" in os.environ:
            url = os.environ["TOOL_ENDPOINT_URL"]

        tool_call_trace = ToolCallTrace(
            endpoint_url=url,
            request=payload,
            response={},
            error=None,
        )
        headers = {"apikey": _LND_API_KEY}
        if "TOOL_ENDPOINT_AUTH" in os.environ:
            headers["Authorization"] = os.environ["TOOL_ENDPOINT_AUTH"]
            headers.pop("apikey")

        session = _create_requests_session(
            url=url,
            num_retry=3,
            headers=headers,
        )

        if files is not None:
            res = session.post(url, data=payload, files=files)
        else:
            res = session.post(url, json=payload)
        if res.status_code != 200:
            tool_call_trace.error = Error(
                name="RemoteToolCallFailed",
                value=f"{res.status_code} - {res.text}",
                traceback_raw=[],
            )
            _LOGGER.error(f"Request failed: {res.status_code} {res.text}")
            # TODO: function_name should be in metadata_payload
            function_name = "unknown"
            if "function_name" in payload:
                function_name = payload["function_name"]
            elif metadata_payload is not None and "function_name" in metadata_payload:
                function_name = metadata_payload["function_name"]
            raise RemoteToolCallFailed(function_name, res.status_code, res.text)

        resp = res.json()
        tool_call_trace.response = resp
        # TODO: consider making the response schema the same between below two sources
        return resp if "TOOL_ENDPOINT_AUTH" in os.environ else resp["data"]  # type: ignore
    finally:
        trace = tool_call_trace.model_dump()
        trace["type"] = "tool_call"
        display({MimeType.APPLICATION_JSON: trace}, raw=True)


def _create_requests_session(
    url: str, num_retry: int, headers: Dict[str, str]
) -> Session:
    """Create a requests session with retry"""
    session = Session()
    retries = Retry(
        total=num_retry,
        backoff_factor=2,
        raise_on_redirect=True,
        raise_on_status=False,
        allowed_methods=["GET", "POST", "PUT"],
        status_forcelist=[
            408,  # Request Timeout
            429,  # Too Many Requests (ie. rate limiter).
            502,  # Bad Gateway
            503,  # Service Unavailable (include cloud circuit breaker)
            504,  # Gateway Timeout
        ],
    )
    session.mount(url, HTTPAdapter(max_retries=retries if num_retry > 0 else 0))
    session.headers.update(headers)
    return session


def get_tool_documentation(funcs: List[Callable[..., Any]]) -> str:
    docstrings = ""
    for func in funcs:
        docstrings += f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}\n\n"

    return docstrings


def get_tool_descriptions(funcs: List[Callable[..., Any]]) -> str:
    descriptions = ""
    for func in funcs:
        description = func.__doc__
        if description is None:
            description = ""

        if "Parameters:" in description:
            description = (
                description[: description.find("Parameters:")]
                .replace("\n", " ")
                .strip()
            )

        description = " ".join(description.split())
        descriptions += f"- {func.__name__}{inspect.signature(func)}: {description}\n"
    return descriptions


def get_tool_descriptions_by_names(
    tool_name: Optional[List[str]],
    funcs: List[Callable[..., Any]],
    util_funcs: List[
        Callable[..., Any]
    ],  # util_funcs will always be added to the list of functions
) -> str:
    if tool_name is None:
        return get_tool_descriptions(funcs + util_funcs)

    invalid_names = [
        name for name in tool_name if name not in {func.__name__ for func in funcs}
    ]

    if invalid_names:
        raise ValueError(f"Invalid customized tool names: {', '.join(invalid_names)}")

    filtered_funcs = (
        funcs
        if not tool_name
        else [func for func in funcs if func.__name__ in tool_name]
    )
    return get_tool_descriptions(filtered_funcs + util_funcs)


def get_tools_df(funcs: List[Callable[..., Any]]) -> pd.DataFrame:
    data: Dict[str, List[str]] = {"desc": [], "doc": []}

    for func in funcs:
        desc = func.__doc__
        if desc is None:
            desc = ""
        desc = desc[: desc.find("Parameters:")].replace("\n", " ").strip()
        desc = " ".join(desc.split())

        doc = f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}"
        data["desc"].append(desc)
        data["doc"].append(doc)

    return pd.DataFrame(data)  # type: ignore


def get_tools_info(funcs: List[Callable[..., Any]]) -> Dict[str, str]:
    data: Dict[str, str] = {}

    for func in funcs:
        desc = func.__doc__
        if desc is None:
            desc = ""

        data[func.__name__] = f"{func.__name__}{inspect.signature(func)}:\n{desc}"

    return data
