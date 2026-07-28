"""
Build a per-port WATER MASK from OpenStreetMap coastline geometry.

WHY THIS IS PART A's REAL FIX. Land is bright in SAR: cranes, quays, tanks and
buildings return far more strongly than any hull. Run CFAR over a raw port chip
and most "vessels" are infrastructure, and worse, they are the SAME
infrastructure every pass — so a congestion index built on unmasked counts is
dominated by a large constant plus noise, and the actual ship signal is the
small residual. Masking to water is what makes the count mean something.

WHY OSM RATHER THAN HAND-DRAWN BOXES. `uae_ports.geojson` carries POINTS, and
four of its thirteen are flagged `approximate` — their centres may not even sit
on the harbour. Drawing rectangles by eye would inherit that error and add
more. OSM coastline is real surveyed geometry, and `verify before trust` means
checking what actually comes back per port rather than assuming it exists.

USAGE
    python -m ingest.satellite.watermask --port fujairah
    python -m ingest.satellite.watermask --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import requests

from ...common.secrets import safe_stdout, repo_root
from .constants import PRIORITY_PORTS, port_bbox

log = logging.getLogger("satellite.watermask")

OVERPASS = "https://overpass-api.de/api/interpreter"
# Overpass returns 406 without a User-Agent — it rejects anonymous clients.
HEADERS = {"User-Agent": "hormuz-disruption-engine/0.1 (research)"}


def overpass_query(bbox: tuple[float, float, float, float]) -> str:
    """bbox is (min_lon, min_lat, max_lon, max_lat); Overpass wants S,W,N,E."""
    min_lon, min_lat, max_lon, max_lat = bbox
    s, w, n, e = min_lat, min_lon, max_lat, max_lon
    return (
        "[out:json][timeout:90];("
        f'way["natural"="coastline"]({s},{w},{n},{e});'
        f'way["man_made"="breakwater"]({s},{w},{n},{e});'
        f'way["man_made"="pier"]({s},{w},{n},{e});'
        ");out geom;"
    )


def fetch_coastline(port: str) -> dict:
    bbox = port_bbox(port)
    r = requests.post(OVERPASS, data={"data": overpass_query(bbox)},
                      headers=HEADERS, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Overpass HTTP {r.status_code}: {r.text[:200]}")
    elements = r.json().get("elements", [])

    features = []
    for el in elements:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[p["lon"], p["lat"]] for p in geom]},
            "properties": {
                "osm_id": el.get("id"),
                "kind": tags.get("natural") or tags.get("man_made") or "unknown",
                "name": tags.get("name"),
            },
        })
    return {"type": "FeatureCollection", "bbox": list(bbox),
            "port": port, "features": features}


def rasterize_water(coastline: dict, bbox, width: int, height: int) -> np.ndarray:
    """
    Turn coastline LINESTRINGS into a boolean water mask on the chip grid.

    OSM coastline is an oriented line, not a polygon: by convention land is on
    the LEFT of the direction of travel. Reconstructing filled land from that
    reliably needs proper polygonisation. Rather than fake it, this draws the
    coastline as a barrier and flood-fills water inward from the seaward edge —
    which for these AOIs is the open-sea side of the box.

    Returns True where water.
    """
    from scipy import ndimage

    min_lon, min_lat, max_lon, max_lat = bbox
    barrier = np.zeros((height, width), dtype=bool)

    def to_px(lon, lat):
        x = int((lon - min_lon) / (max_lon - min_lon) * (width - 1))
        y = int((max_lat - lat) / (max_lat - min_lat) * (height - 1))
        return np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)

    for feat in coastline["features"]:
        pts = feat["geometry"]["coordinates"]
        for (lon0, lat0), (lon1, lat1) in zip(pts, pts[1:]):
            x0, y0 = to_px(lon0, lat0)
            x1, y1 = to_px(lon1, lat1)
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for t in range(steps + 1):
                x = int(round(x0 + (x1 - x0) * t / steps))
                y = int(round(y0 + (y1 - y0) * t / steps))
                barrier[y, x] = True

    # Thicken slightly so single-pixel gaps do not leak the flood fill.
    barrier = ndimage.binary_dilation(barrier, iterations=1)

    # Flood-fill from every edge pixel that is not barrier. For a coastal AOI
    # the open sea reaches the box edge, so this captures water; enclosed land
    # behind the coastline is left unfilled.
    free = ~barrier
    labels, _ = ndimage.label(free)
    edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    edge_labels = set(int(v) for v in edge if v > 0)

    water = np.isin(labels, list(edge_labels)) if edge_labels else free
    return water


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build per-port water masks from OSM")
    p.add_argument("--port", choices=sorted(PRIORITY_PORTS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--size", type=int, default=512)
    args = p.parse_args(argv)

    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stdout)

    ports = sorted(PRIORITY_PORTS) if args.all else [args.port or "fujairah"]
    out_dir = repo_root() / "data" / "reference" / "watermask"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'port':<14} {'osm ways':>9} {'nodes':>7} {'water %':>9}  status")
    print("  " + "-" * 60)

    for port in ports:
        try:
            coast = fetch_coastline(port)
        except Exception as exc:
            print(f"  {port:<14} {'--':>9} {'--':>7} {'--':>9}  FETCH FAILED: {exc}")
            continue

        n_ways = len(coast["features"])
        n_nodes = sum(len(f["geometry"]["coordinates"]) for f in coast["features"])

        (out_dir / f"{port}_coastline.geojson").write_text(
            json.dumps(coast, indent=2), encoding="utf-8")

        if n_ways == 0:
            # Honest failure: no coastline in the AOI means either the box is
            # entirely offshore (fine — all water) or OSM lacks coverage here.
            print(f"  {port:<14} {0:>9} {0:>7} {'n/a':>9}  NO COASTLINE — treat AOI as all-water?")
            continue

        water = rasterize_water(coast, port_bbox(port), args.size, args.size)
        frac = float(water.mean())
        np.save(out_dir / f"{port}_water_{args.size}.npy", water)

        status = "OK"
        if frac > 0.98:
            status = "SUSPECT — almost no land found"
        elif frac < 0.15:
            status = "SUSPECT — almost no water; check AOI centre"
        print(f"  {port:<14} {n_ways:>9} {n_nodes:>7} {frac:>8.1%}  {status}")

    print(f"\n  written to {out_dir}")
    print("  ⚠️  VERIFY VISUALLY before trusting: load the *_coastline.geojson in")
    print("      ports/uae_ports_viewer.html (it accepts any geojson from disk).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
