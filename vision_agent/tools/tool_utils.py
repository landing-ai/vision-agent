import logging
import os
from typing import Any, Dict, MutableMapping, Optional

from IPython.display import display
from pydantic import BaseModel
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vision_agent.utils.exceptions import RemoteToolCallFailed
from vision_agent.utils.execute import Error, MimeType
from vision_agent.utils.type_defs import LandingaiAPIKey

_LOGGER = logging.getLogger(__name__)
_LND_API_KEY = LandingaiAPIKey().api_key
_LND_API_URL = "https://api.landing.ai/v1/agent"


class ToolCallTrace(BaseModel):
    endpoint_url: str
    request: MutableMapping[str, Any]
    response: MutableMapping[str, Any]
    error: Optional[Error]


def send_inference_request(
    payload: Dict[str, Any], endpoint_name: str
) -> Dict[str, Any]:
    try:
        if runtime_tag := os.environ.get("RUNTIME_TAG", ""):
            payload["runtime_tag"] = runtime_tag

        url = f"{_LND_API_URL}/model/{endpoint_name}"
        if "TOOL_ENDPOINT_URL" in os.environ:
            url = os.environ["TOOL_ENDPOINT_URL"]

        tool_call_trace = ToolCallTrace(
            endpoint_url=url,
            request=payload,
            response={},
            error=None,
        )
        headers = {"Content-Type": "application/json", "apikey": _LND_API_KEY}
        if "TOOL_ENDPOINT_AUTH" in os.environ:
            headers["Authorization"] = os.environ["TOOL_ENDPOINT_AUTH"]
            headers.pop("apikey")

        session = _create_requests_session(
            url=url,
            num_retry=3,
            headers=headers,
        )
        res = session.post(url, json=payload)
        if res.status_code != 200:
            tool_call_trace.error = Error(
                name="RemoteToolCallFailed",
                value=f"{res.status_code} - {res.text}",
                traceback_raw=[],
            )
            _LOGGER.error(f"Request failed: {res.status_code} {res.text}")
            raise RemoteToolCallFailed(payload["tool"], res.status_code, res.text)

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
