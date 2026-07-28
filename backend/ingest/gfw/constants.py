"""
Global Fishing Watch API — access constants for the Gulf coverage question.

WHY THIS ARM EXISTS: aisstream.io was measured on 2026-07-27 to have ZERO
receiver coverage in the Gulf (world box 6758 msgs/45s; Hormuz box 0/30s;
Gulf+Arabian Sea 0/60s). GFW's AIS is SATELLITE-sourced rather than aggregated
from volunteer terrestrial receivers, so it is the most credible free candidate
for Gulf vessel data.

⚠️ EXPECT A RESEARCH PRODUCT, NOT A LIVE FEED. GFW publishes with a lag of days
and its 4wings layers are spatially aggregated. If coverage exists, this is a
strong source for BACKTEST and congestion ground truth — and a weak one for a
real-time vessel map. Do not wire it to anything labelled "live" without
checking the actual timestamp of the newest data it returns.
"""

from __future__ import annotations

from ...common.secrets import load_secret

API_BASE = "https://gateway.api.globalfishingwatch.org"

# Same box the AIS collector uses — covers all 13 UAE ports/terminals, the
# Hormuz lanes and the Gulf of Oman approaches.
HORMUZ_BBOX = {"min_lat": 23.5, "min_lon": 51.0, "max_lat": 27.0, "max_lon": 57.5}

# Wider box, used to distinguish "no coverage at Hormuz" from "no coverage in
# the whole region" — the same control that diagnosed the aisstream gap.
GULF_WIDE_BBOX = {"min_lat": 10.0, "min_lon": 40.0, "max_lat": 32.0, "max_lon": 70.0}

# Control region with known-good coverage. If the Gulf is silent but this is
# not, the cause is coverage; if both are silent, the cause is the request.
CONTROL_BBOX = {"min_lat": 50.0, "min_lon": -5.0, "max_lat": 54.0, "max_lon": 5.0}


def bbox_to_geojson(bbox: dict) -> dict:
    """Bounding box -> the GeoJSON Polygon the 4wings endpoints expect."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [bbox["min_lon"], bbox["min_lat"]],
            [bbox["max_lon"], bbox["min_lat"]],
            [bbox["max_lon"], bbox["max_lat"]],
            [bbox["min_lon"], bbox["max_lat"]],
            [bbox["min_lon"], bbox["min_lat"]],
        ]],
    }


def load_token() -> str:
    """
    GFW API token: GFW_TOKEN env var, else secrets/gfw_token.txt.

    GFW issues long-lived JWTs, so min_len is generous — a real token is
    hundreds of characters and anything short is a paste accident.
    """
    return load_secret(
        label="Global Fishing Watch API token",
        env_var="GFW_TOKEN",
        filenames=["gfw_token.txt", "gfw_token.key", "gfw.txt"],
        signup_url="https://globalfishingwatch.org/our-apis/tokens",
        min_len=40,
    )


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}
