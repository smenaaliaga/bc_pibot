"""HTTP helpers for simple JSON POST requests."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


def post_json(url: str, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """POST JSON payload and parse JSON response.

    Raises RuntimeError on HTTP/parse errors.
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8") if resp else ""
    except HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8")
        except Exception:
            detail = str(err)
        raise RuntimeError(f"HTTP {err.code} calling {url}: {detail}") from err
    except URLError as err:
        raise RuntimeError(f"Failed to call {url}: {err}") from err

    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {body[:200]}") from exc
    if isinstance(parsed, dict):
        return parsed
    raise RuntimeError(f"Unexpected response from {url}: {type(parsed)}")
