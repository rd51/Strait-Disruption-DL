"""
Copernicus Data Space OAuth — client-credentials flow with token caching.

CDSE access tokens are short-lived — measured 2026-07-27: `expires_in` came back
as **1800s (30 min)**, not the ~10 min assumed when this was written. Do not
hardcode either number; the cache reads `expires_in` from the response. A
long-running collector that fetches a fresh token per request will hit rate
limits; one that caches without checking expiry will start 401-ing mid-job.
This caches with a safety margin.
"""

from __future__ import annotations

import logging
import threading
import time

import requests

from .constants import TOKEN_URL, load_credentials

log = logging.getLogger("satellite.auth")

# Refresh this many seconds before actual expiry, so a request never sets off
# with a token that dies in flight.
REFRESH_MARGIN_S = 60


class TokenCache:
    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

    def get(self, force: bool = False) -> str:
        with self._lock:
            if not force and self._token and time.time() < self._expires_at:
                return self._token

            client_id, client_secret = load_credentials()
            resp = requests.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
            if resp.status_code != 200:
                # The body explains *why* (bad secret vs unknown client vs
                # disabled account) and contains no credential material.
                raise RuntimeError(
                    f"CDSE token request failed: HTTP {resp.status_code} — {resp.text[:300]}"
                )
            payload = resp.json()
            self._token = payload["access_token"]
            self._expires_at = time.time() + payload.get("expires_in", 600) - REFRESH_MARGIN_S
            log.info("obtained CDSE token, valid ~%ss", payload.get("expires_in", "?"))
            return self._token


_CACHE = TokenCache()


def get_token(force: bool = False) -> str:
    return _CACHE.get(force=force)


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {get_token()}"}
