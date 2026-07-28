"""
Collect Arm B market series from FRED (official) and Yahoo (supplementary).

USAGE
    python -m ingest.market.collect                 # everything, full history
    python -m ingest.market.collect --fred-only     # defensible sources only
    python -m ingest.market.collect --start 2019-01-01

OUTPUT
    data/raw/market/fred/<series_id>.parquet
    data/raw/market/yahoo/<ticker>.parquet
    data/raw/market/market_daily.parquet     wide, one row per calendar day
    data/raw/market/sources.json             provenance per series

⚠️ WEEKENDS AND HOLIDAYS ARE NOT ZEROS. Commodity series simply do not observe
on non-trading days. They are left as NaN and never forward-filled here —
filling is a modelling decision that belongs in the feature layer, where it can
be made explicitly and consistently with how the backtest splits time. Silently
filling at ingest would bake a look-ahead-shaped assumption into the raw store.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

import pandas as pd
import requests

from ...common.secrets import SecretNotFound, safe_stdout
from .constants import (
    FRED_BASE,
    FRED_SERIES,
    HISTORY_START,
    YAHOO_SERIES,
    data_root,
    load_fred_key,
)

log = logging.getLogger("market.collect")


def fetch_fred(series_id: str, key: str, start: str) -> pd.DataFrame:
    """One FRED series as a tidy daily frame."""
    r = requests.get(
        f"{FRED_BASE}/series/observations",
        params={"series_id": series_id, "api_key": key, "file_type": "json",
                "observation_start": start},
        timeout=90,
    )
    if r.status_code != 200:
        raise RuntimeError(f"FRED {series_id}: HTTP {r.status_code} {r.text[:200]}")

    obs = r.json().get("observations", [])
    rows = []
    for o in obs:
        v = o.get("value")
        if v in (".", "", None):        # FRED marks a non-observation as "."
            continue
        rows.append({"date": o["date"], "value": float(v)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["series_id"] = series_id
    df["label"] = FRED_SERIES[series_id]["label"]
    df["source"] = "FRED"
    df["official"] = True
    return df


def fetch_yahoo(ticker: str, start: str) -> pd.DataFrame:
    """One Yahoo series. Tagged unofficial — see constants.py for why."""
    import yfinance as yf

    hist = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=False)
    if hist.empty:
        return pd.DataFrame()

    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    df = pd.DataFrame({
        "date": idx,
        "value": hist["Close"].to_numpy(),
        "volume": hist["Volume"].to_numpy() if "Volume" in hist else None,
    })
    df["series_id"] = ticker
    df["label"] = YAHOO_SERIES[ticker]["label"]
    df["source"] = "yfinance"
    df["official"] = False
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect Arm B market series")
    p.add_argument("--start", default=HISTORY_START)
    p.add_argument("--fred-only", action="store_true",
                   help="skip yfinance — official sources only")
    args = p.parse_args(argv)

    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stdout)

    root = data_root()
    (root / "fred").mkdir(parents=True, exist_ok=True)
    (root / "yahoo").mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    provenance: list[dict] = []

    # ── FRED ──────────────────────────────────────────────────────────────
    try:
        key = load_fred_key()
    except SecretNotFound as exc:
        print(f"\n{exc}\n")
        return 3

    print(f"\n  FRED (official) — from {args.start}")
    for sid, meta in FRED_SERIES.items():
        try:
            df = fetch_fred(sid, key, args.start)
        except Exception as exc:
            log.error("FRED %s failed: %s", sid, exc)
            continue
        if df.empty:
            log.warning("FRED %s returned no observations", sid)
            continue
        df.to_parquet(root / "fred" / f"{sid}.parquet", index=False)
        frames.append(df)
        provenance.append({
            "series_id": sid, "label": meta["label"], "source": "FRED",
            "official": True, "url": f"{FRED_BASE}/series/observations?series_id={sid}",
            "desc": meta["desc"], "role": meta["role"],
            "obs": len(df), "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        })
        print(f"    {sid:<14} {len(df):>6,} obs   "
              f"{df['date'].min().date()} -> {df['date'].max().date()}")

    # ── Yahoo ─────────────────────────────────────────────────────────────
    if not args.fred_only:
        print(f"\n  Yahoo (UNOFFICIAL — supplement only, never a sole source)")
        for ticker, meta in YAHOO_SERIES.items():
            try:
                df = fetch_yahoo(ticker, args.start)
            except Exception as exc:
                log.error("yahoo %s failed: %r", ticker, exc)
                continue
            if df.empty:
                log.warning("yahoo %s returned nothing", ticker)
                continue
            safe = ticker.replace("=", "_")
            df.to_parquet(root / "yahoo" / f"{safe}.parquet", index=False)
            frames.append(df)
            provenance.append({
                "series_id": ticker, "label": meta["label"], "source": "yfinance",
                "official": False, "url": "https://finance.yahoo.com (unofficial scraper)",
                "desc": meta["desc"], "obs": len(df),
                "start": str(df["date"].min().date()),
                "end": str(df["date"].max().date()),
            })
            print(f"    {ticker:<14} {len(df):>6,} obs   "
                  f"{df['date'].min().date()} -> {df['date'].max().date()}")

    if not frames:
        print("\n  nothing collected\n")
        return 1

    # ── wide daily panel ──────────────────────────────────────────────────
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index="date", columns="label", values="value", aggfunc="last")
    wide = wide.sort_index()
    wide.to_parquet(root / "market_daily.parquet")

    (root / "sources.json").write_text(
        json.dumps({"generated_utc": str(pd.Timestamp.utcnow()),
                    "history_start": args.start,
                    "series": provenance}, indent=2),
        encoding="utf-8")

    official = [p["label"] for p in provenance if p["official"]]
    print(f"\n  wide panel: {wide.shape[0]:,} days x {wide.shape[1]} series "
          f"-> {root / 'market_daily.parquet'}")
    print(f"  official (FRED) series: {official}")
    print(f"  provenance -> {root / 'sources.json'}")

    # Coverage against the anchors — the point of collecting this at all.
    print("\n  coverage per anchor window (official series only):")
    windows = {
        "2019 Gulf of Oman": ("2019-04-01", "2019-07-31"),
        "2019 Abqaiq": ("2019-08-01", "2019-10-31"),
        "2024 Red Sea": ("2023-11-01", "2024-03-31"),
        "2026 Hormuz closure": ("2026-05-01", str(date.today())),
    }
    for wname, (a, b) in windows.items():
        sl = wide.loc[a:b, [c for c in official if c in wide.columns]]
        n = int(sl.notna().any(axis=1).sum())
        print(f"    {wname:<22} {n:>4} days with data")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
