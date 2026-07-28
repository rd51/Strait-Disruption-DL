"""
Copernicus Data Space — credentials, endpoints and port AOIs for the CNN arm.

WHY THIS ARM IS NOW LOAD-BEARING. Measured 2026-07-27: aisstream has zero Gulf
coverage, and Global Fishing Watch covers the Gulf but lags 4 days. Sentinel-1
returned a Fujairah pass ~2 hours old in the same session. SAR is therefore the
*timeliest* source of physical Gulf observation available to this project, and
it detects vessels regardless of transponder state or receiver coverage — the
original justification for the arm, now confirmed by the failure of the others.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ...common.secrets import SecretNotFound, repo_root

# ─────────────────────────────────────────────────────────────── endpoints

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
# Catalogue search — open, needs no credentials (verified 2026-07-27).
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
# Sentinel Hub Process API — server-side crop, returns only the AOI.
# Preferred over downloading whole scenes: an IW_GRDH product is 0.9-1.6 GB and
# an IW_SLC is 7.1 GB (measured), which is exactly how containers run out of
# memory. Quota-limited, so ODATA_URL stays the fallback for bulk pulls.
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

# ─────────────────────────────────────────────────────────── credentials

# Accept the many spellings these get saved under.
_ID_KEYS = ("client_id", "clientid", "id", "cdse_client_id", "copernicus_client_id")
_SECRET_KEYS = ("client_secret", "clientsecret", "secret", "cdse_client_secret",
                "copernicus_client_secret")


def _parse_credentials(text: str) -> dict:
    """
    Parse credentials from JSON *or* `KEY = VALUE` text.

    The file is named .json but may well contain env-style lines — that is what
    actually happened here. Demanding one format would just bounce the user
    back to re-save a file that already contains everything needed.
    """
    text = text.strip()
    if not text:
        return {}

    # JSON first, including nested shapes some providers emit.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            flat = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    flat.update({f"{k}".lower(): v for k, v in value.items()})
                else:
                    flat[str(key).lower()] = value
            return flat
    except json.JSONDecodeError:
        pass

    # Fall back to KEY=VALUE / KEY: VALUE lines.
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*[:=]\s*(.+)$", line)
        if m:
            key = m.group(1).strip().lower()
            value = m.group(2).strip().strip('"').strip("'").rstrip(",")
            out[key] = value
    return out


def _pick(data: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        if data.get(name):
            return str(data[name]).strip()
    return None


def load_credentials() -> tuple[str, str]:
    """
    Resolve (client_id, client_secret) for Copernicus Data Space.

    Order: env vars, then any credential file in secrets/. Raises with the
    locations tried and — critically — *what was found in them*, so a
    half-filled file is distinguishable from a missing one.
    """
    import os

    env_id = os.environ.get("CDSE_CLIENT_ID", "").strip()
    env_secret = os.environ.get("CDSE_CLIENT_SECRET", "").strip()
    if env_id and env_secret:
        return env_id, env_secret

    root = repo_root()
    candidates = [
        root / "secrets" / "copernicus.json",
        root / "secrets" / "copernicus.txt",
        root / "secrets" / "cdse.json",
        root / "secrets" / "copernicus_credentials.json",
    ]
    report: list[str] = []
    for path in candidates:
        if not path.exists():
            report.append(f"{path.name}: not found")
            continue
        # utf-8-sig — this file arrived with a UTF-8 BOM.
        data = _parse_credentials(path.read_text(encoding="utf-8-sig"))
        if not data:
            report.append(f"{path.name}: unparseable (not JSON, not KEY=VALUE)")
            continue
        client_id = _pick(data, _ID_KEYS)
        client_secret = _pick(data, _SECRET_KEYS)
        if client_id and client_secret:
            return client_id, client_secret
        found = ", ".join(sorted(data)) or "nothing"
        report.append(
            f"{path.name}: parsed but missing "
            f"{'client_id' if not client_id else 'client_secret'} (keys found: {found})"
        )

    detail = "\n".join(f"      {r}" for r in report)
    raise SecretNotFound(
        "No usable Copernicus credentials.\n"
        f"    CDSE_CLIENT_ID / CDSE_CLIENT_SECRET env vars: "
        f"{'partially set' if (env_id or env_secret) else 'unset'}\n"
        f"    files searched in secrets/:\n{detail}\n"
        "  Create OAuth client credentials at dataspace.copernicus.eu\n"
        "  (User Settings -> OAuth clients), then save either:\n"
        '    {"client_id": "...", "client_secret": "..."}\n'
        "  or:\n"
        "    CLIENT_ID = ...\n"
        "    CLIENT_SECRET = ...\n"
        "  into secrets/copernicus.json (both formats are accepted)."
    )


# ──────────────────────────────────────────────────────────── port AOIs

# ⚠️ PROVISIONAL. uae_ports.geojson carries POINTS; the CNN arm needs POLYGONS
# covering anchorage + berths. These boxes are a first approximation built by
# buffering each point, and four ports (Hamriyah, Jebel Dhanna, Das, Zirku) are
# flagged `approximate` in the source data — their centres may not sit on the
# harbour at all. Verify every box in ports/uae_ports_viewer.html (it loads any
# geojson from disk) BEFORE using these to cut training chips.
PRIORITY_PORTS = {
    # name: (lat, lon, half-width km) — Fujairah first: outside the strait, the
    # reroute destination, and the earliest physical tell of disruption.
    "fujairah": (25.1727756, 56.3635322, 8.0),
    "khor_fakkan": (25.3526481, 56.367736, 5.0),
    "jebel_ali": (25.0236111, 55.0402778, 8.0),
    "khalifa": (24.8099016, 54.6494965, 6.0),
}


def bbox_around(lat: float, lon: float, half_km: float) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) box of ~half_km around a point."""
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.2, abs(__import__("math").cos(__import__("math").radians(lat)))))
    return (round(lon - dlon, 6), round(lat - dlat, 6),
            round(lon + dlon, 6), round(lat + dlat, 6))


def port_bbox(name: str) -> tuple[float, float, float, float]:
    lat, lon, half = PRIORITY_PORTS[name]
    return bbox_around(lat, lon, half)
