"""
Lightweight healthcheck script used by Docker HEALTHCHECK.
Exit 0 → healthy, exit 1 → unhealthy.
"""

import sys

import httpx

URL = "http://localhost:8000/health"

try:
    r = httpx.get(URL, timeout=4)
    data = r.json()
    if r.status_code == 200 and data.get("model_loaded"):
        sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
