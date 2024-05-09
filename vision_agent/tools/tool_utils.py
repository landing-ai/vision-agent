import logging
import os
from typing import Any, Dict

import requests

from vision_agent.utils.type_defs import LandingaiAPIKey

_LOGGER = logging.getLogger(__name__)
_LND_API_KEY = LandingaiAPIKey().api_key
_LND_API_URL = "https://api.dev.landing.ai/v1/agent"


def _send_inference_request(
    payload: Dict[str, Any], endpoint_name: str
) -> Dict[str, Any]:
    # runtime_tag is used to differentiate different internal callers
    runtime_tag = os.environ.get("RUNTIME_TAG", "")
    res = requests.post(
        f"{_LND_API_URL}/model/{endpoint_name}",
        headers={
            "Content-Type": "application/json",
            "apikey": _LND_API_KEY,
            "runtime-tag": runtime_tag,
        },
        json=payload,
    )
    if res.status_code != 200:
        _LOGGER.error(f"Request failed: {res.text}")
        raise ValueError(f"Request failed: {res.text}")
    return res.json()["data"]  # type: ignore
