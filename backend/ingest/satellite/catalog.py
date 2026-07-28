"""
Find which dates actually have a Sentinel-1 pass over a port, BEFORE spending
processing units on a chip request.

WHY THIS EXISTS — measured 2026-07-27. A Process API request that matches no
acquisition still returns a well-formed TIFF (4,603 bytes of zeros) and still
spends the full 3.333 processing units. Requesting a date range blind therefore
burns quota on every day without a pass. Over a backtest window that is most
days: Sentinel-1 revisits the Gulf every ~1-3 days, so a naive 90-day pull
would waste PUs on roughly two thirds of its requests.

Catalogue search itself is free.

USAGE
    python -m ingest.satellite.catalog --port fujairah --days 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests

from .auth import get_token
from .constants import CATALOG_URL, PRIORITY_PORTS, port_bbox

log = logging.getLogger("satellite.catalog")


MAX_LIMIT = 100          # server-enforced: >100 returns HTTP 400


def find_passes(port: str, start: date, end: date, limit: int = MAX_LIMIT) -> list[dict]:
    """
    Sentinel-1 acquisitions intersecting a port AOI in [start, end].

    Returns one dict per acquisition, newest first.

    ⚠️ The API caps `limit` at 100 and rejects anything larger with HTTP 400.
    A long window can therefore be SILENTLY TRUNCATED at 100 results — which
    would understate pass counts and quietly shrink a collection budget. Query
    in chunks (see `find_passes_chunked`) for anything longer than ~2 months.
    """
    limit = min(limit, MAX_LIMIT)
    bbox = list(port_bbox(port))
    body = {
        "collections": ["sentinel-1-grd"],
        "datetime": (
            f"{datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
            "/"
            f"{datetime.combine(end, datetime.max.time().replace(microsecond=0), tzinfo=timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
        ),
        "bbox": bbox,
        "limit": limit,
    }
    resp = requests.post(
        CATALOG_URL,
        headers={"Authorization": f"Bearer {get_token()}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Catalog HTTP {resp.status_code}: {resp.text[:400]}")

    features = resp.json().get("features", [])
    out = []
    for f in features:
        props = f.get("properties", {})
        out.append({
            "id": f.get("id", "?"),
            "datetime": props.get("datetime", "?"),
            "date": (props.get("datetime") or "?")[:10],
            "platform": props.get("platform", "?"),
            "orbit_state": props.get("sat:orbit_state", props.get("orbitDirection", "?")),
            "polarizations": props.get("polarization", props.get("sar:polarizations", "?")),
        })
    out.sort(key=lambda r: r["datetime"], reverse=True)
    return out


def find_passes_chunked(port: str, start: date, end: date,
                        chunk_days: int = 45, pause_s: float = 0.3) -> list[dict]:
    """
    Same as find_passes but split into chunks, so the 100-result cap cannot
    silently truncate a long window.

    45 days is comfortably under 100 results even at the current ~1.4-day
    revisit (~32 passes), leaving headroom if the constellation grows again.
    """
    out: list[dict] = []
    seen: set[str] = set()
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        batch = find_passes(port, cursor, chunk_end)
        if len(batch) >= MAX_LIMIT:
            log.warning("chunk %s..%s hit the %d cap for %s — may be truncated",
                        cursor, chunk_end, MAX_LIMIT, port)
        for p in batch:
            if p["id"] not in seen:
                seen.add(p["id"])
                out.append(p)
        cursor = chunk_end + timedelta(days=1)
        time.sleep(pause_s)
    out.sort(key=lambda r: r["datetime"], reverse=True)
    return out


def pass_dates(port: str, start: date, end: date) -> list[str]:
    """Just the distinct YYYY-MM-DD strings that have an acquisition."""
    return sorted({p["date"] for p in find_passes_chunked(port, start, end)}, reverse=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="List Sentinel-1 passes over a port AOI")
    p.add_argument("--port", default="fujairah", choices=sorted(PRIORITY_PORTS))
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stdout)

    end = date.today()
    start = end - timedelta(days=args.days)
    passes = find_passes(args.port, start, end)

    print(f"\n  {args.port}  {start} -> {end}   bbox {port_bbox(args.port)}")
    print(f"  {len(passes)} acquisitions on {len(set(p['date'] for p in passes))} distinct days\n")
    for p_ in passes:
        print(f"    {p_['datetime'][:19]}  {p_['platform']:<12} {str(p_['orbit_state']):<12} {p_['polarizations']}")

    days = sorted({p_["date"] for p_ in passes})
    if len(days) > 1:
        gaps = [
            (date.fromisoformat(b) - date.fromisoformat(a)).days
            for a, b in zip(days, days[1:])
        ]
        print(f"\n  revisit gaps (days): {gaps}")
        print(f"  mean {sum(gaps)/len(gaps):.1f}  min {min(gaps)}  max {max(gaps)}")

    covered = len(set(p_["date"] for p_ in passes))
    print(f"\n  Requesting all {args.days} days blind would spend ~{args.days * 3.333:.0f} PU,")
    print(f"  of which ~{(args.days - covered) * 3.333:.0f} PU on days with NO acquisition.")
    print(f"  Requesting only the {covered} real pass days costs ~{covered * 3.333:.0f} PU.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
