"""
Does Global Fishing Watch actually cover the Gulf?

Runs the same controlled diagnosis that settled the aisstream question, so the
answer is measured rather than assumed:

  1. authenticate and list the datasets this token can actually reach
  2. query the Hormuz box
  3. query a much wider Gulf + Arabian Sea box
  4. query a known-covered control region

  Gulf silent + control returns data  -> genuine coverage gap
  both silent                         -> the request or token is wrong, not coverage

USAGE
    $env:GFW_TOKEN = "<token>"      # or put it in secrets/gfw_token.txt
    python -m ingest.gfw.coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta

import requests

from ...common.secrets import SecretNotFound, redact
from .constants import (
    API_BASE,
    CONTROL_BBOX,
    GULF_WIDE_BBOX,
    HORMUZ_BBOX,
    auth_headers,
    bbox_to_geojson,
    load_token,
)

TIMEOUT = 90


def _get(path: str, token: str, params: dict | None = None) -> requests.Response | None:
    try:
        return requests.get(f"{API_BASE}{path}", params=params or {},
                            headers=auth_headers(token), timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"    CONNECTION ERROR {exc!r}")
        return None


def _post(path: str, token: str, params: dict, body: dict) -> requests.Response | None:
    try:
        return requests.post(f"{API_BASE}{path}", params=params,
                             headers=auth_headers(token), json=body, timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"      CONNECTION ERROR {exc!r}")
        return None


def list_datasets(token: str) -> list[str]:
    """What can this token actually see? Avoids guessing dataset ids."""
    print("\n[1] datasets reachable with this token")
    # v3 rejects the request unless limit AND offset are BOTH supplied.
    resp = _get("/v3/datasets", token, {"limit": 250, "offset": 0})
    if resp is None:
        return []
    print(f"    HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"    {resp.text[:400]}")
        return []
    try:
        payload = resp.json()
    except ValueError:
        print(f"    non-JSON: {resp.text[:200]}")
        return []

    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    ids = [e.get("id", "?") for e in entries if isinstance(e, dict)]
    print(f"    {len(ids)} datasets")
    for i in ids[:25]:
        print(f"      {i}")
    if len(ids) > 25:
        print(f"      ... and {len(ids) - 25} more")
    return ids


def report(label: str, bbox: dict, token: str, dataset: str,
           start: str, end: str) -> int | None:
    """4wings report over one bounding box. Returns a crude activity count."""
    print(f"\n    {label}")
    # The geometry goes in the POST BODY — v3 rejects `geojson` as a query
    # param ("Query params geojson not supported").
    resp = _post(
        "/v3/4wings/report", token,
        {
            "spatial-resolution": "LOW",
            "temporal-resolution": "MONTHLY",
            "datasets[0]": dataset,
            "date-range": f"{start},{end}",
            "format": "JSON",
        },
        {"geojson": bbox_to_geojson(bbox)},
    )
    if resp is None:
        return None
    print(f"      HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"      {resp.text[:300]}")
        return None
    try:
        payload = resp.json()
    except ValueError:
        print(f"      non-JSON: {resp.text[:200]}")
        return None

    entries = payload.get("entries", [])
    total = 0
    for entry in entries:
        if isinstance(entry, dict):
            for value in entry.values():
                if isinstance(value, list):
                    total += len(value)
                elif isinstance(value, (int, float)):
                    total += 1
    print(f"      entries={len(entries)}  datapoints≈{total}")
    if total == 0:
        print(f"      raw (first 300 chars): {json.dumps(payload)[:300]}")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GFW Gulf coverage test")
    parser.add_argument("--days", type=int, default=90,
                        help="how far back to query (GFW publishes with a lag)")
    parser.add_argument("--dataset", default=None,
                        help="dataset id; default picks a presence/effort dataset")
    args = parser.parse_args(argv)

    try:
        token = load_token()
    except SecretNotFound as exc:
        print(f"\n{exc}\n")
        return 3

    print("=" * 76)
    print(f"GLOBAL FISHING WATCH — GULF COVERAGE TEST   token {redact(token)}")
    print("=" * 76)

    ids = list_datasets(token)

    dataset = args.dataset
    if not dataset:
        preferred = [i for i in ids if "presence" in i] or \
                    [i for i in ids if "fishing-effort" in i]
        dataset = preferred[0] if preferred else "public-global-presence:latest"
    print(f"\n[2] using dataset: {dataset}")

    # GFW publishes with a lag, so end the window well before today.
    end = date.today() - timedelta(days=10)
    start = end - timedelta(days=args.days)
    print(f"    window: {start} -> {end}  (ends 10d back; GFW lags real time)")

    print("\n[3] coverage probes")
    hormuz = report("HORMUZ box", HORMUZ_BBOX, token, dataset, str(start), str(end))
    wide = report("GULF + ARABIAN SEA box", GULF_WIDE_BBOX, token, dataset, str(start), str(end))
    control = report("CONTROL — English Channel/North Sea", CONTROL_BBOX, token,
                     dataset, str(start), str(end))

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    if control is None:
        print("  Inconclusive — the control query failed, so nothing can be concluded")
        print("  about the Gulf. Fix the request/dataset first.")
    elif hormuz or wide:
        print(f"  ✅ GFW HAS GULF COVERAGE  (hormuz={hormuz}, wide={wide}, control={control})")
        print("  Next: check the newest timestamp in the data before calling it 'live'.")
        print("  Treat as backtest / ground-truth source until that is verified.")
    elif control:
        print(f"  ❌ NO GULF COVERAGE  (hormuz={hormuz}, wide={wide}) while the control")
        print(f"     region returned {control} — same failure shape as aisstream.")
    else:
        print("  Inconclusive — every region returned nothing, which points at the")
        print("  request or dataset rather than at coverage.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
