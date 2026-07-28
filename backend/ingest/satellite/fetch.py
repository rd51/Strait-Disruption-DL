"""
Fetch small SAR/optical chips over port AOIs via the Sentinel Hub Process API.

THE WHOLE POINT: the server crops to the AOI and returns a few hundred KB.
Downloading the equivalent product would be 0.9-1.6 GB for IW_GRDH (7.1 GB for
SLC) — measured, and exactly how a container runs out of memory. Nothing here
ever holds a full scene.

Sentinel-1 GRD is the detection product: SAR sees through Gulf haze, dust and
darkness, and detects hulls regardless of whether a transponder is on. That
matters more than originally assumed — aisstream has no Gulf AIS coverage at
all, and GFW lags 4 days.

USAGE
    python -m ingest.satellite.fetch --port fujairah --date 2026-07-27
    python -m ingest.satellite.fetch --port fujairah --days 30 --all-passes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from .auth import get_token
from .constants import PRIORITY_PORTS, PROCESS_URL, port_bbox

log = logging.getLogger("satellite.fetch")

# VV + VH backscatter as float32. Both polarisations matter: VV responds to
# hulls and calm-water contrast, VH suppresses sea clutter and makes small
# metal targets pop. Keeping them separate lets the CNN learn the combination
# instead of us hardcoding a ratio.
EVALSCRIPT_S1 = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH"]}],
    output: {bands: 2, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) {
  return [s.VV, s.VH];
}
"""


def data_root() -> Path:
    import os
    env = os.environ.get("SATELLITE_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    from ...common.paths import raw_dir
    return raw_dir() / "satellite"


def fetch_chip(port: str, day: date, size: int = 1024,
               timeout: int = 180) -> tuple[bytes, dict]:
    """
    Request one AOI chip for a port on a given day.

    Returns (tiff_bytes, metadata). Raises on a non-200 so a failed pass is
    never silently written as an empty file.
    """
    bbox = port_bbox(port)
    start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    payload = {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {
                        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    "acquisitionMode": "IW",
                    "polarization": "DV",          # dual VV+VH
                },
                "processing": {
                    # Terrain-flattened gamma0 + orthorectification: without
                    # this, backscatter varies with incidence angle and the
                    # CNN learns viewing geometry instead of vessels.
                    "backCoeff": "GAMMA0_TERRAIN",
                    "orthorectify": True,
                },
            }],
        },
        "output": {
            "width": size,
            "height": size,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT_S1,
    }

    resp = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {get_token()}",
                 "Accept": "image/tiff", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Process API HTTP {resp.status_code} for {port} {day}: {resp.text[:400]}"
        )

    meta = {
        "port": port, "date": str(day), "bbox": bbox, "size_px": size,
        "bytes": len(resp.content),
        "processing_units": resp.headers.get("x-processingunits-spent"),
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta.update(assess_coverage(resp.content))
    return resp.content, meta


def assess_coverage(content: bytes) -> dict:
    """
    What fraction of the chip actually carries data?

    ⚠️ THE PARTIAL-CHIP TRAP. A chip clipped by the edge of the SAR swath is a
    perfectly valid, real image covering only part of the AOI — measured
    2026-07-26 over Fujairah at 59%. If a congestion index counts bright
    targets, that chip silently reports ~59% of the vessels, which reads as
    "congestion fell at Fujairah" when nothing changed but the satellite's
    footprint. Coverage must travel with every chip so downstream code can
    reject or normalise partial ones instead of trusting the count.

    Also distinguishes EMPTY (no acquisition at all) from partial. An empty
    request still costs the full processing-unit charge, which is why
    `catalog.find_passes` should gate requests.
    """
    try:
        import io

        import numpy as np
        import tifffile

        arr = tifffile.imread(io.BytesIO(content))
    except Exception as exc:            # inspection must never fail a fetch
        return {"coverage": None, "coverage_note": f"uninspectable: {exc!r}"}

    band = arr[..., 0] if arr.ndim == 3 else arr
    finite = np.isfinite(band)
    covered = float((finite & (band != 0)).sum()) / band.size

    if covered == 0.0:
        status = "EMPTY"
    elif covered < 0.90:
        status = "PARTIAL"
    else:
        status = "FULL"
    return {"coverage": round(covered, 4), "coverage_status": status}


def save_chip(content: bytes, port: str, day: date) -> Path:
    root = data_root() / "chips" / port / f"dt={day}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{port}_{day}_s1_vvvh.tiff"
    tmp = path.with_suffix(".tiff.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
    return path


def describe_tiff(content: bytes) -> str:
    """Identify the payload without needing rasterio installed."""
    if content[:2] in (b"II", b"MM"):
        endian = "little" if content[:2] == b"II" else "big"
        return f"valid TIFF ({endian}-endian, {len(content):,} bytes)"
    if content[:8].startswith(b"\x89PNG"):
        return f"PNG ({len(content):,} bytes)"
    return f"UNRECOGNISED format, first bytes {content[:8]!r}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch Sentinel-1 chips over port AOIs")
    p.add_argument("--port", default="fujairah", choices=sorted(PRIORITY_PORTS))
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--size", type=int, default=1024, help="output pixels per side")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stdout)

    day = date.fromisoformat(args.date) if args.date else date.today()

    print(f"\n  port {args.port} | date {day} | {args.size}x{args.size} px")
    print(f"  bbox {port_bbox(args.port)}")
    try:
        content, meta = fetch_chip(args.port, day, size=args.size)
    except Exception as exc:
        print(f"\n  ✗ {exc}\n")
        return 1

    print(f"  {describe_tiff(content)}")
    print(f"  processing units spent: {meta['processing_units']}")

    status = meta.get("coverage_status")
    cov = meta.get("coverage")
    if status == "EMPTY":
        print(f"  ✗ EMPTY CHIP — no acquisition on {day}. The PU was spent anyway;")
        print("    run `python -m ingest.satellite.catalog` first to find real pass dates.")
        return 2
    if status == "PARTIAL":
        print(f"  ⚠️  PARTIAL COVERAGE {cov:.1%} — the AOI is clipped by the swath edge.")
        print("    A vessel count from this chip UNDERCOUNTS and must not be compared")
        print("    against full chips without normalising.")
    else:
        print(f"  coverage: {cov:.1%}")

    path = save_chip(content, args.port, day)
    # Coverage must survive alongside the pixels, or downstream code cannot
    # tell a quiet port from a half-imaged one.
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"  ✓ saved -> {path}")
    print(f"    meta   -> {sidecar.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
