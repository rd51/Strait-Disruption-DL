"""
Bulk Sentinel-1 chip collection for Arm A.

Cuts training chips across 4 ports x 4 anchor windows. This is the job that
turns Arm A from "access proven on 3 chips" into a trainable dataset.

FOUR THINGS THIS MUST GET RIGHT, each learned the expensive way:

  1. CATALOGUE FIRST, ALWAYS. A date with no acquisition still returns a
     well-formed 4,603-byte TIFF of zeros and still spends the full PU. Over a
     3-month window that is most of the budget burned on non-pass days.
     Catalogue search is free; blind fetching is not.

  2. HARD PU BUDGET. Overage is configured to 0, which makes the quota a wall
     rather than a bill: a job that overruns does not cost money, it DIES
     mid-run leaving a half-collected dataset. The budget is checked against
     the live accounting endpoint before starting and decremented per chip.

  3. RECORD ORBIT DIRECTION. Ascending (~14:20 UTC) and descending (~02:10 UTC)
     view the same water from opposite sides, so backscatter differs
     systematically. A CNN trained on a mix without this recorded will partly
     learn viewing geometry rather than vessels. GAMMA0_TERRAIN normalises
     terrain, NOT look direction.

  4. RESUME SAFELY. 674 chips over a slow API is a job that will be interrupted.
     Existing chips are skipped by path, so a rerun costs nothing.

Coverage class (FULL/PARTIAL/EMPTY) is written per chip by `assess_coverage`.
PARTIAL chips are KEPT here but flagged — the feature layer drops them. A chip
clipped by the swath edge is a real image of part of the AOI, and counting
vessels in it reports a congestion FALL that is pure satellite footprint.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from ...common.paths import repo_root
from ...common.secrets import safe_stdout
from .auth import get_token
from .catalog import find_passes_chunked
from .constants import PRIORITY_PORTS
from .fetch import fetch_chip, save_chip, data_root

log = logging.getLogger(__name__)

USAGE_URL = "https://sh.dataspace.copernicus.eu/api/v1/accounting/usage"

# The anchor windows, padded before each event so lead time is measurable.
ANCHOR_WINDOWS = {
    "2019_gulf_of_oman": ("2019-05-01", "2019-07-15"),
    "2019_abqaiq":       ("2019-08-15", "2019-10-15"),
    "2024_red_sea":      ("2023-11-01", "2024-02-15"),
    "2026_hormuz":       ("2026-01-15", "2026-04-30"),
}

# PU cost scales exactly with pixel count: 256->0.833, 512->3.333, 1024->13.333
# (measured, clean 4x per doubling).
PU_PER_CHIP = {256: 0.833, 512: 3.333, 1024: 13.333, 1600: 32.6}

# Keep a reserve so an overrun cannot strand the month with zero PU for
# incremental live collection.
PU_RESERVE = 3000.0


def quota() -> dict:
    r = requests.get(USAGE_URL, headers={"Authorization": f"Bearer {get_token()}"},
                     timeout=30)
    r.raise_for_status()
    d = r.json()
    return {
        "pu_remaining": float(d["processingUnitsMonthly"]["remaining"]),
        "requests_remaining": float(d["requestsMonthly"]["remaining"]),
    }


def chip_path(port: str, day: str):
    return data_root() / "chips" / port / f"dt={day}" / f"{port}_{day}_s1_vvvh.tiff"


def plan(ports: list[str], windows: list[str], size: int) -> pd.DataFrame:
    """
    Build the full work list from the FREE catalogue, before spending anything.

    One row per (port, date) with an acquisition. Orbit state and platform come
    from the catalogue, so they are recorded even for chips fetched later.
    """
    rows = []
    for port in ports:
        for wname in windows:
            lo, hi = ANCHOR_WINDOWS[wname]
            passes = find_passes_chunked(
                port, date.fromisoformat(lo), date.fromisoformat(hi))
            # Collapse to one chip per DAY. Two acquisitions on the same day
            # over one AOI is rare, and a second chip would double-count that
            # day in any congestion series.
            by_day: dict[str, dict] = {}
            for p in passes:
                by_day.setdefault(p["date"], p)
            for day, p in by_day.items():
                rows.append({
                    "port": port, "window": wname, "date": day,
                    "platform": p["platform"], "orbit_state": p["orbit_state"],
                    "polarizations": str(p["polarizations"]),
                    "exists": chip_path(port, day).exists(),
                })
            log.info("%-12s %-18s %3d passes on %3d distinct days",
                     port, wname, len(passes), len(by_day))
            time.sleep(0.3)
    df = pd.DataFrame(rows).sort_values(["port", "date"]).reset_index(drop=True)
    df["pu_estimate"] = PU_PER_CHIP.get(size, 13.333)
    return df


def collect(df: pd.DataFrame, size: int, budget_pu: float,
            pause_s: float = 1.0) -> pd.DataFrame:
    """Fetch every planned chip not already on disk, within the PU budget."""
    todo = df[~df.exists].copy()
    spent = 0.0
    results = []

    for i, row in enumerate(todo.itertuples(), 1):
        if spent + PU_PER_CHIP.get(size, 13.333) > budget_pu:
            log.warning("PU budget reached (%.1f spent) — stopping at %d/%d",
                        spent, i - 1, len(todo))
            break
        try:
            content, meta = fetch_chip(row.port, date.fromisoformat(row.date), size=size)
        except Exception as exc:                       # noqa: BLE001
            log.error("fetch failed %s %s: %s", row.port, row.date, exc)
            results.append({"port": row.port, "date": row.date, "status": "error",
                            "detail": str(exc)[:200]})
            time.sleep(pause_s * 3)
            continue

        # PU is spent whether or not the chip has content, so count it either way.
        spent += float(meta.get("processing_units") or PU_PER_CHIP.get(size, 13.333))

        # 🔴 KEY NAMES MUST MATCH assess_coverage's OUTPUT: it returns
        # `coverage_status` / `coverage`, NOT `coverage_class` / `coverage_pct`.
        # The first version of this file read the wrong names, so this guard
        # was INERT for a whole 446-chip run and the results table logged NaN.
        # No empty chip actually reached disk — but only because the
        # catalogue-first design meant none was ever requested. A silent guard
        # that happens not to be needed is still a broken guard.
        if meta.get("coverage_status") == "EMPTY":
            # The catalogue said there was a pass, so an empty chip means the
            # AOI fell outside the swath footprint. Not written — an all-zero
            # file in the store is indistinguishable from calm water.
            log.warning("EMPTY chip %s %s — not persisted", row.port, row.date)
            results.append({"port": row.port, "date": row.date, "status": "empty"})
            time.sleep(pause_s)
            continue

        path = save_chip(content, row.port, date.fromisoformat(row.date))
        # Orbit direction is metadata the pixels do not carry — persist it
        # beside them or the CNN can never be told which geometry it is seeing.
        meta.update(window=row.window, platform=row.platform,
                    orbit_state=row.orbit_state)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

        results.append({"port": row.port, "date": row.date, "status": "ok",
                        "coverage_status": meta.get("coverage_status"),
                        "coverage": meta.get("coverage"),
                        "orbit_state": row.orbit_state, "platform": row.platform,
                        "window": row.window, "path": str(path)})
        if i % 25 == 0:
            log.info("%d/%d chips · %.0f PU spent", i, len(todo), spent)
        time.sleep(pause_s)

    log.info("collection done: %d attempted, %.1f PU spent", len(results), spent)
    return pd.DataFrame(results)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bulk-collect Sentinel-1 chips for Arm A")
    p.add_argument("--ports", default="all")
    p.add_argument("--windows", default="all")
    p.add_argument("--size", type=int, default=1024, choices=[256, 512, 1024, 1600])
    p.add_argument("--budget-pu", type=float, default=None,
                   help="max PU to spend (default: live remaining minus reserve)")
    p.add_argument("--plan-only", action="store_true",
                   help="cost the job from the FREE catalogue and stop")
    a = p.parse_args(argv)

    ports = sorted(PRIORITY_PORTS) if a.ports == "all" else a.ports.split(",")
    windows = list(ANCHOR_WINDOWS) if a.windows == "all" else a.windows.split(",")

    q = quota()
    log.info("quota: %.0f PU remaining, %.0f requests remaining",
             q["pu_remaining"], q["requests_remaining"])

    df = plan(ports, windows, a.size)
    need = int((~df.exists).sum())
    est = need * PU_PER_CHIP.get(a.size, 13.333)
    log.info("PLAN: %d chips total, %d already on disk, %d to fetch",
             len(df), int(df.exists.sum()), need)
    log.info("estimated cost: %.0f PU (%.1f%% of the 30,000 monthly allowance)",
             est, est / 300.0)

    outdir = repo_root() / "data" / "raw" / "satellite"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "collection_plan.parquet", index=False)

    if a.plan_only:
        print(df.groupby(["port", "window"]).size().to_string())
        return 0

    budget = a.budget_pu if a.budget_pu is not None else max(
        0.0, q["pu_remaining"] - PU_RESERVE)
    if est > budget:
        log.warning("estimate %.0f PU exceeds budget %.0f — will stop early",
                    est, budget)

    res = collect(df, a.size, budget)
    if not res.empty:
        res.to_parquet(outdir / "collection_results.parquet", index=False)
        print(res.groupby("status").size().to_string())
    return 0


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
