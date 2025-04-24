import logging
import os
from base64 import b64encode
from functools import cache
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import numpy as np
from IPython.display import display
from pydantic import BaseModel
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vision_agent.utils.exceptions import RemoteToolCallFailed
from vision_agent.utils.execute import Error, MimeType
from vision_agent.utils.image_utils import normalize_bbox

_LOGGER = logging.getLogger(__name__)
_LND_BASE_URL = os.environ.get("LANDINGAI_URL", "https://api.va.landing.ai")
_LND_API_URL = f"{_LND_BASE_URL}/v1/agent/model"
_LND_API_URL_v2 = f"{_LND_BASE_URL}/v1/tools"


@cache
def get_vision_agent_api_key() -> str:
    vision_agent_api_key = os.environ.get("VISION_AGENT_API_KEY")
    if vision_agent_api_key:
        return vision_agent_api_key
    else:
        raise ValueError(
            "VISION_AGENT_API_KEY not found in environment variables, required for tool usage. You can get a free key from https://va.landing.ai/settings/api-key"
        )


def should_report_tool_traces() -> bool:
    return bool(os.environ.get("REPORT_TOOL_TRACES", False))


class ToolCallTrace(BaseModel):
    endpoint_url: str
    type: str
    request: MutableMapping[str, Any]
    response: MutableMapping[str, Any]
    error: Optional[Error]
    files: Optional[List[tuple[str, str]]]


def send_inference_request(
    payload: Dict[str, Any],
    endpoint_name: str,
    files: Optional[List[Tuple[Any, ...]]] = None,
    v2: bool = False,
    metadata_payload: Optional[Dict[str, Any]] = None,
    is_form: bool = False,
) -> Any:
    url = f"{_LND_API_URL_v2 if v2 else _LND_API_URL}/{endpoint_name}"
    if "TOOL_ENDPOINT_URL" in os.environ:
        url = os.environ["TOOL_ENDPOINT_URL"]

    vision_agent_api_key = get_vision_agent_api_key()
    headers = {
        "Authorization": f"Basic {vision_agent_api_key}",
        "X-Source": "vision_agent",
    }

    if runtime_tag := os.environ.get("RUNTIME_TAG", "vision-agent"):
        headers["runtime_tag"] = runtime_tag

    session = _create_requests_session(
        url=url,
        num_retry=3,
        headers=headers,
    )

    function_name = "unknown"
    if "function_name" in payload:
        function_name = payload["function_name"]
    elif metadata_payload is not None and "function_name" in metadata_payload:
        function_name = metadata_payload["function_name"]

    response = _call_post(url, payload, session, files, function_name, is_form)

    return response["data"]


def send_task_inference_request(
    payload: Dict[str, Any],
    task_name: str,
    files: Optional[List[Tuple[Any, ...]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    is_form: bool = False,
) -> Any:
    url = f"{_LND_API_URL_v2}/{task_name}"
    vision_agent_api_key = get_vision_agent_api_key()
    headers = {
        "Authorization": f"Basic {vision_agent_api_key}",
        "X-Source": "vision_agent",
    }
    session = _create_requests_session(
        url=url,
        num_retry=3,
        headers=headers,
    )

    function_name = "unknown"
    if metadata is not None and "function_name" in metadata:
        function_name = metadata["function_name"]
    response = _call_post(url, payload, session, files, function_name, is_form)
    return response["data"]


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


def _call_post(
    url: str,
    payload: dict[str, Any],
    session: Session,
    files: Optional[List[Tuple[Any, ...]]] = None,
    function_name: str = "unknown",
    is_form: bool = False,
) -> Any:
    files_in_b64 = None
    if files:
        files_in_b64 = [(file[0], b64encode(file[1]).decode("utf-8")) for file in files]

    tool_call_trace = None
    try:
        if files is not None:
            response = session.post(url, data=payload, files=files)
        elif is_form:
            response = session.post(url, data=payload)
        else:
            response = session.post(url, json=payload)

        tool_call_trace_payload = (
            payload
            if "function_name" in payload
            else {**payload, **{"function_name": function_name}}
        )
        tool_call_trace = ToolCallTrace(
            endpoint_url=url,
            type="tool_call",
            request=tool_call_trace_payload,
            response={},
            error=None,
            files=files_in_b64,
        )

        if response.status_code != 200:
            tool_call_trace.error = Error(
                name="RemoteToolCallFailed",
                value=f"{response.status_code} - {response.text}",
                traceback_raw=[],
            )
            _LOGGER.error(f"Request failed: {response.status_code} {response.text}")
            raise RemoteToolCallFailed(
                function_name, response.status_code, response.text
            )

        result = response.json()
        tool_call_trace.response = result
        return result
    finally:
        if tool_call_trace is not None and should_report_tool_traces():
            trace = tool_call_trace.model_dump()
            display({MimeType.APPLICATION_JSON: trace}, raw=True)


def add_bboxes_from_masks(
    all_preds: List[List[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    for frame_preds in all_preds:
        for preds in frame_preds:
            mask = preds["mask"]
            if mask.sum() == 0:
                preds["bbox"] = []
            else:
                # Get indices where mask is True using axis operations
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)

                # Find boundaries using argmax/argmin
                y_min = np.argmax(rows)
                y_max = len(rows) - np.argmax(rows[::-1])
                x_min = np.argmax(cols)
                x_max = len(cols) - np.argmax(cols[::-1])

                bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
                bbox = normalize_bbox(bbox, mask.shape)
                preds["bbox"] = bbox

    return all_preds


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    intersection = x_overlap * y_overlap

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def single_nms(
    preds: List[Dict[str, Any]], iou_threshold: float
) -> List[Dict[str, Any]]:
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            if calculate_iou(preds[i]["bbox"], preds[j]["bbox"]) > iou_threshold:
                if preds[i]["score"] > preds[j]["score"]:
                    preds[j]["score"] = 0
                else:
                    preds[i]["score"] = 0

    return [pred for pred in preds if pred["score"] > 0]


def nms(
    all_preds: List[List[Dict[str, Any]]], iou_threshold: float
) -> List[List[Dict[str, Any]]]:
    if not isinstance(all_preds[0], List):
        all_preds = [all_preds]

    return_preds = []
    for frame_preds in all_preds:
        frame_preds = single_nms(frame_preds, iou_threshold)
        return_preds.append(frame_preds)

    return return_preds
