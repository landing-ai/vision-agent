import json
import logging
from typing import Any, Dict, Optional

from requests import Session
from requests.adapters import HTTPAdapter

_LOGGER = logging.getLogger(__name__)


class BaseHTTP:
    """A simple HTTP client for making GET and POST requests.

    This class provides methods to send HTTP requests to a specified base endpoint
    with configurable headers and automatic retries.

    Attributes:
        _TIMEOUT (int): Timeout for HTTP requests in seconds.
        _MAX_RETRIES (int): Maximum number of retry attempts for failed requests.

    Methods:
        __init__(base_endpoint, headers=None):
            Initializes a BaseHTTP instance with a base endpoint and optional headers.
        
        post(url, payload):
            Sends a POST request to the specified URL with the given payload.
        
        get(url):
            Sends a GET request to the specified URL.

    Args:
        base_endpoint (str): The base URL for the HTTP client.
        headers (Optional[Dict[str, Any]]): Optional headers to include in each request.

    Returns:
        Dict[str, Any]: The JSON-decoded response from the server for GET and POST requests.

    Raises:
        json.JSONDecodeError: If the response cannot be parsed as JSON."""
    _TIMEOUT = 30  # seconds
    _MAX_RETRIES = 3

    def __init__(
        self, base_endpoint: str, *, headers: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initializes a client for interacting with the specified API endpoint.

    This constructor sets up a session with default JSON headers and retries for
    HTTP requests. It allows for custom headers if provided.

    Args:
        base_endpoint (str): The base URL for the API endpoint.
        headers (Optional[Dict[str, Any]]): Optional headers to include in the requests.
            If not provided, defaults to JSON content type.

    Returns:
        None"""
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
        """Sends a POST request with a JSON payload to a specified endpoint.

    Args:
        url (str): The endpoint path to append to the base URL.
        payload (Dict[str, Any]): The JSON-compatible data to send in the request body.

    Returns:
        Dict[str, Any]: The JSON response from the server as a dictionary.

    Raises:
        json.JSONDecodeError: If the response cannot be parsed as JSON."""
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
        """Fetches JSON data from a specified endpoint.

    Constructs a full URL using the provided endpoint path and the base endpoint,
    sends a GET request, and returns the JSON response as a dictionary.

    Args:
        url (str): The endpoint path to append to the base URL.

    Returns:
        Dict[str, Any]: The JSON response from the server, parsed into a dictionary.

    Raises:
        JSONDecodeError: If the response cannot be decoded as JSON."""
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
