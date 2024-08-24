import json
import logging
from typing import Any, Dict, Optional

from requests import Session
from requests.adapters import HTTPAdapter

_LOGGER = logging.getLogger(__name__)


class BaseHTTP:
    _TIMEOUT = 30  # seconds
    _MAX_RETRIES = 3

    def __init__(
        self, base_endpoint: str, *, headers: Optional[Dict[str, Any]] = None
    ) -> None:
        self._headers = headers
        if headers is None:
            self._headers = {
                "Content-Type": "application/json",
            }
        self._base_endpoint = base_endpoint
        self._session = Session()
        self._session.headers.update(self._headers)  # type: ignore
        self._session.mount(
            self._base_endpoint, HTTPAdapter(max_retries=self._MAX_RETRIES)
        )

    def post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        formatted_url = f"{self._base_endpoint}/{url}"
        _LOGGER.info(f"Sending data to {formatted_url}")
        try:
            response = self._session.post(
                url=formatted_url, json=payload, timeout=self._TIMEOUT
            )
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            _LOGGER.info(json.dumps(result))
        except json.JSONDecodeError:
            resp_text = response.text
            _LOGGER.warning(f"Response seems incorrect: '{resp_text}'.")
            raise
        return result

    def get(self, url: str) -> Dict[str, Any]:
        formatted_url = f"{self._base_endpoint}/{url}"
        _LOGGER.info(f"Sending data to {formatted_url}")
        try:
            response = self._session.get(url=formatted_url, timeout=self._TIMEOUT)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            _LOGGER.info(json.dumps(result))
        except json.JSONDecodeError:
            resp_text = response.text
            _LOGGER.warning(f"Response seems incorrect: '{resp_text}'.")
            raise
        return result
