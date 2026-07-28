"""
Port reference geography and SAR chip rendering for the dashboard.

WHY CHIPS ARE RENDERED SERVER-SIDE. A Sentinel-1 chip is a 1024x1024 float32
GeoTIFF of raw backscatter — roughly 8 MB, and not an image a browser can
display. The values are also heavy-tailed (measured VV median 0.030 against a
max of 740.7, a 24,000x span), so a naive linear stretch renders as an almost
black frame with a few white specks. Converting to decibels and clipping to
fixed bounds is what makes a vessel visible at all.

FIXED dB BOUNDS, NOT PER-CHIP PERCENTILES. Auto-stretching each chip to its own
contrast would make a busy anchorage and an empty stretch of sea look
identical, which is precisely the difference the dashboard exists to show. The
same bounds are used here as in the CNN preprocessing, so what a viewer sees is
what the model sees.
"""

from __future__ import annotations

import io
import json
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from ..common.paths import repo_root

log = logging.getLogger(__name__)

VV_DB_LO, VV_DB_HI = -35.0, 5.0


# ────────────────────────────────────────────────────────── port geography
@lru_cache(maxsize=1)
def ports() -> list[dict]:
    """
    The 13 UAE ports/terminals with the coastline split that IS the thesis.

    `hormuz_dependent` is the load-bearing field: 11 ports sit inside the
    strait, 2 (Fujairah, Khor Fakkan) are on the Gulf of Oman coast and bypass
    it entirely. Reroute pressure should show up at the bypass pair.
    """
    for candidate in (repo_root() / "data" / "reference" / "uae_ports.csv",
                      repo_root() / "ports" / "uae_ports.csv"):
        if candidate.exists():
            df = pd.read_csv(candidate)
            break
    else:
        return []

    df = df.where(pd.notna(df), None)
    out = []
    for r in df.to_dict("records"):
        out.append({
            "name": r["name"], "emirate": r["emirate"],
            "coast_side": r["coast_side"],
            "hormuz_dependent": bool(r["hormuz_dependent"]),
            "type": r["type"], "role": r["role"],
            "primary_routes": r.get("primary_routes"),
            "lat": float(r["latitude"]), "lon": float(r["longitude"]),
            # Four coordinates are approximate (Hamriyah plus the three offshore
            # terminals). Surfaced so the map can mark them rather than imply a
            # precision the reference data does not have.
            "coord_precision": r.get("coord_precision", "unknown"),
        })
    return out


# ──────────────────────────────────────────────────────────── SAR imagery
@lru_cache(maxsize=1)
def _detections_index() -> dict:
    """CFAR results keyed by (port, date), for overlaying on a rendered chip."""
    p = repo_root() / "data" / "derived" / "sar_cfar" / "cfar_detections.json"
    if not p.exists():
        return {}
    out = {}
    for rec in json.loads(p.read_text()):
        out[(rec["port"], rec["date"])] = rec
    return out


def chip_catalogue() -> list[dict]:
    """Every chip on disk, with its coverage class and orbit direction."""
    base = repo_root() / "data" / "raw" / "satellite" / "chips"
    dets = _detections_index()
    rows = []
    for sidecar in sorted(base.glob("*/dt=*/*.json")):
        try:
            m = json.loads(sidecar.read_text())
        except Exception:                               # noqa: BLE001
            continue
        key = (m.get("port"), m.get("date"))
        d = dets.get(key, {})
        rows.append({
            "port": m.get("port"), "date": m.get("date"),
            "orbit_state": m.get("orbit_state"), "platform": m.get("platform"),
            "window": m.get("window"),
            "coverage": m.get("coverage"),
            "coverage_status": m.get("coverage_status"),
            "vessels_on_water": d.get("n_on_water"),
            "detections_on_land": d.get("n_on_land"),
        })
    rows.sort(key=lambda r: (r["port"] or "", r["date"] or ""), reverse=True)
    return rows


def _chip_path(port: str, date: str):
    p = (repo_root() / "data" / "raw" / "satellite" / "chips" / port
         / f"dt={date}" / f"{port}_{date}_s1_vvvh.tiff")
    return p if p.exists() else None


def render_chip(port: str, date: str, overlay: bool = True,
                max_px: int = 700) -> bytes:
    """
    Render one chip to PNG, optionally circling CFAR water detections.

    Returns PNG bytes. Raises FileNotFoundError if the chip is not on disk.
    """
    import rasterio
    from PIL import Image, ImageDraw

    path = _chip_path(port, date)
    if path is None:
        raise FileNotFoundError(f"no chip for {port} {date}")

    with rasterio.open(path) as src:
        vv = src.read(1).astype("float32")

    # dB, then a FIXED stretch — see the module docstring.
    db = 10.0 * np.log10(np.clip(vv, 1e-6, None))
    norm = np.clip((db - VV_DB_LO) / (VV_DB_HI - VV_DB_LO), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    img = Image.fromarray((norm * 255).astype("uint8"), mode="L").convert("RGB")

    scale = 1.0
    if overlay:
        rec = _detections_index().get((port, date))
        if rec:
            draw = ImageDraw.Draw(img)
            for d in rec.get("detections", []):          # water only
                r, c = d["row"], d["col"]
                rad = max(4, min(14, d.get("size_px", 4)))
                draw.ellipse([c - rad, r - rad, c + rad, r + rad],
                             outline=(88, 166, 255), width=2)

    if img.width > max_px:
        scale = max_px / img.width
        img = img.resize((max_px, int(img.height * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
