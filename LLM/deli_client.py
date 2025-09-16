import logging
import os
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = ""  # Place your API base URL here
DEFAULT_SERVICE_KEY = ""  # Place your service key here

BASE_URL_ENV_VAR = "DELI_BASE_URL"
SERVICE_KEY_ENV_VAR = "DELI_SERVICE_KEY"


def _resolve_base_url(base_url: Optional[str]) -> str:
    """Return the configured base URL, falling back to environment variables."""

    return (
        (base_url or os.getenv(BASE_URL_ENV_VAR) or DEFAULT_BASE_URL)
        .strip()
    )


def _resolve_service_key(service_key: Optional[str]) -> str:
    """Return the configured service key, falling back to environment variables."""

    return (
        (service_key or os.getenv(SERVICE_KEY_ENV_VAR) or DEFAULT_SERVICE_KEY)
        .strip()
    )


def search_law(
    query: str,
    *,
    base_url: Optional[str] = None,
    service_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query the external law search service.

    When required configuration such as the base URL or the service key is
    missing, the function logs a warning and returns an empty list instead of
    attempting the HTTP request. This prevents requests from being issued with
    incomplete parameters, which previously resulted in `MissingSchema` errors.
    """

    resolved_base_url = _resolve_base_url(base_url)
    resolved_service_key = _resolve_service_key(service_key)

    if not resolved_base_url or not resolved_service_key:
        missing_bits = []
        if not resolved_base_url:
            missing_bits.append("base URL")
        if not resolved_service_key:
            missing_bits.append("service key")
        logger.warning(
            "Skipping law search request because %s %s missing.",
            " and ".join(missing_bits),
            "are" if len(missing_bits) > 1 else "is",
        )
        return []

    params = {"question": query, "serviceKey": resolved_service_key}

    try:
        response = requests.get(resolved_base_url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Law search request failed: %s", exc)
        return []

    try:
        data = response.json()
    except ValueError:
        logger.warning("Law search response was not valid JSON.")
        return []

    if isinstance(data, dict):
        # Some services wrap the actual results in a dedicated key.
        for key in ("data", "results", "items"):
            nested = data.get(key)
            if isinstance(nested, list):
                data = nested
                break

    if not isinstance(data, list):
        logger.warning(
            "Law search response had unexpected type %s; expected a list of laws.",
            type(data).__name__,
        )
        return []

    return data


if __name__ == "__main__":
    query = "중화인민공화국 노동법 제43조는 무엇을 규정하고 있나요?"
    print(search_law(query))
