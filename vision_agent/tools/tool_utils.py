import logging
from typing import Any, Dict

import requests

from vision_agent.type_defs import LandingaiAPIKey

_LOGGER = logging.getLogger(__name__)
_LND_API_KEY = LandingaiAPIKey().api_key
_LND_API_URL = "https://api.dev.landing.ai/v1/agent"


def _send_inference_request(
    payload: Dict[str, Any], endpoint_name: str
) -> Dict[str, Any]:
    res = requests.post(
        f"{_LND_API_URL}/model/{endpoint_name}",
        headers={
            "Content-Type": "application/json",
            "apikey": _LND_API_KEY,
        },
        json=payload,
    )
    if res.status_code != 200:
        _LOGGER.error(f"Request failed: {res.text}")
        raise ValueError(f"Request failed: {res.text}")
    return res.json()["data"]  # type: ignore
