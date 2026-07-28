"""
Build the common daily feature index from all three arms.

THE ONE RULE THIS FILE EXISTS TO ENFORCE: every feature must be indexed by
WHEN IT WAS KNOWABLE, not when the underlying event happened.

  · GDELT      -> aggregate on DATEADDED (publication), never on Day.
                  Measured: Day trails DATEADDED by up to 365 days, so
                  indexing on Day places information before anyone could have
                  had it. That is look-ahead bias, and it is the exact thing
                  the CI leakage gate exists to fail the build over.
  · Market     -> the close of day D is knowable at the end of D. Any feature
                  using it to predict D is using same-day information; the
                  shift lives in the model layer, and `lag_days` records it.
  · SAR        -> a chip is knowable at its acquisition timestamp.

Weekends/holidays stay NaN. Forward-filling here would silently manufacture
observations the market never made, and would do it BEFORE the train/test
split, which is how leakage gets baked in below the level anyone inspects.

USAGE
    python -m features.build
    python -m features.build --check-leakage
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.paths import derived_dir, raw_dir

log = logging.getLogger("features.build")

# Anchored to the project root, not the current working directory. A relative
# Path("data/raw") silently resolves against wherever the process happens to
# start — which meant this only worked when run from the repo root.
RAW = raw_dir()
OUT = derived_dir()
CONFLICT_CODES = [str(i) for i in range(13, 21)]


# ─────────────────────────────────────────────────────── Arm C : GDELT
def gdelt_daily() -> pd.DataFrame:
    """
    Daily geopolitical features from the historical windows, on DATEADDED.

    Volume is counted over DISTINCT ARTICLES, not raw event rows: GDELT emits
    one row per actor pair extracted from an article, so raw rows measure
    re-reporting (measured 60-75% overcount) rather than escalation.
    """
    files = sorted(glob.glob(str(RAW / "gdelt" / "historical" / "*.parquet")))
    if not files:
        return pd.DataFrame()

    cols = ["DATEADDED", "EventRootCode", "GoldsteinScale", "AvgTone",
            "NumArticles", "SOURCEURL"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)

    ts = pd.to_datetime(df["DATEADDED"], format="%Y%m%d%H%M%S", errors="coerce")
    df = df.assign(date=ts.dt.normalize()).dropna(subset=["date"])
    df["is_conflict"] = df["EventRootCode"].isin(CONFLICT_CODES)

    grouped = df.groupby("date")
    out = pd.DataFrame({
        "gdelt_rows": grouped.size(),
        "gdelt_articles": grouped["SOURCEURL"].nunique(),
        "gdelt_tone_mean": grouped["AvgTone"].mean(),
        "gdelt_goldstein_mean": grouped["GoldsteinScale"].mean(),
    })
    conf = df[df["is_conflict"]].groupby("date")
    out["gdelt_conflict_rows"] = conf.size()
    out["gdelt_conflict_articles"] = conf["SOURCEURL"].nunique()
    out["gdelt_conflict_tone"] = conf["AvgTone"].mean()
    out["gdelt_conflict_goldstein"] = conf["GoldsteinScale"].mean()

    # Share of the day's reporting that is conflict-coded — scale-free, so it
    # is comparable across windows with very different absolute volumes.
    out["gdelt_conflict_share"] = (
        out["gdelt_conflict_articles"] / out["gdelt_articles"].replace(0, np.nan)
    )
    return out.sort_index()


# ─────────────────────────────────────────────────────── Arm B : market
def market_daily() -> pd.DataFrame:
    """
    Market features. The Brent-WTI spread is the Gulf-specific discriminator:
    WTI is landlocked US crude, so a widening spread prices CHOKEPOINT risk,
    whereas a parallel move in both is a global oil story. Measured 2026: the
    spread went from ~4 to 25.94 at the crisis peak.
    """
    path = RAW / "market" / "market_daily.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    out = pd.DataFrame(index=df.index)

    if {"brent_spot_usd", "wti_spot_usd"}.issubset(df.columns):
        brent, wti = df["brent_spot_usd"], df["wti_spot_usd"]
        out["brent"] = brent
        out["wti"] = wti
        out["brent_wti_spread"] = brent - wti
        # fill_method=None is REQUIRED, not stylistic. pandas' pct_change pads
        # NaNs by default, which fabricates 0% returns on every non-trading day
        # and extends the series past the last real observation. Caught here
        # because brent_ret_1d had MORE non-nulls (2,203) than brent itself
        # (2,165) and ran a week beyond Brent's last close.
        out["brent_ret_1d"] = brent.pct_change(fill_method=None) * 100

        # Rolling stats are backward-looking (never centred), but they must
        # also be MASKED to where the source actually observed. On a reindexed
        # daily calendar a trailing 7-day window still finds enough points to
        # emit a value days after the last real close, inventing observations —
        # the same class of bug as pct_change's default padding. `_only_where`
        # confines every derived series to its source's own dates.
        def _only_where(derived: pd.Series, source: pd.Series) -> pd.Series:
            return derived.where(source.notna())

        out["brent_vol_7d"] = _only_where(
            out["brent_ret_1d"].rolling(7, min_periods=3).std(), out["brent_ret_1d"])
        out["brent_ret_7d"] = brent.pct_change(7, fill_method=None) * 100
        out["spread_chg_7d"] = _only_where(
            out["brent_wti_spread"].diff(7), out["brent_wti_spread"])
        # z-score against a trailing year, so "high" means high for this regime
        # rather than high against a fixed constant.
        roll_mu = out["brent_wti_spread"].rolling(252, min_periods=60).mean()
        roll_sd = out["brent_wti_spread"].rolling(252, min_periods=60).std()
        out["spread_z"] = _only_where(
            (out["brent_wti_spread"] - roll_mu) / roll_sd, out["brent_wti_spread"])

        # ── PERSISTENCE, not peak, is what makes the spread discriminate ──
        # Measured across every shock in the sample: the raw daily spread's
        # all-time maximum (54.34) belongs to 2020-04-20 — the day WTI settled
        # at -$36.98 because the May contract expired with Cushing storage
        # full. That is a US futures mechanic with zero Gulf content, and it
        # exceeds the 2026 Hormuz peak (25.94) by more than 2x. Ranking shocks
        # on the raw daily spread therefore puts a storage squeeze above a
        # strait closure.
        #
        # Smoothing over 10 days fixes it, because a chokepoint sustains the
        # premium and a contract expiry does not:
        #
        #   shock              smoothed max   days raw>15
        #   2026 Hormuz            22.98           19
        #   2019 Gulf of Oman      11.87            0
        #   COVID                  11.22            1
        #   2022 Ukraine           10.21            0
        #   Red Sea / Abqaiq    7.04 / 6.38         0
        out["spread_smooth_10d"] = _only_where(
            out["brent_wti_spread"].rolling(10, min_periods=5).mean(),
            out["brent_wti_spread"])
        # Count of stressed days in the trailing month. Deliberately a COUNT
        # rather than a mean: one 54-dollar day and twenty 16-dollar days have
        # similar means but mean entirely different things.
        stressed = (out["brent_wti_spread"] > 15).astype(float)
        out["spread_days_gt15_30d"] = _only_where(
            stressed.rolling(30, min_periods=10).sum(), out["brent_wti_spread"])

    if "henry_hub_gas_usd" in df.columns:
        out["gas"] = df["henry_hub_gas_usd"]

    return out.sort_index()


# ─────────────────────────────────────────────────────── Arm A : SAR
def sar_daily() -> pd.DataFrame:
    """
    Port congestion from the CFAR baseline, water-masked.

    Uses n_on_water only. Land detections are the same cranes and quays every
    pass (measured near-constant at 39/37/38 on Fujairah) and would add a large
    fixed offset that swamps the vessel signal.

    PARTIAL chips are DROPPED, not rescaled. A chip clipped by the swath edge
    undercounts, and a naive rescale assumes vessels are uniformly distributed
    across the AOI — they are not, they cluster in the anchorage.
    """
    path = OUT / "sar_cfar" / "cfar_detections.json"
    if not path.exists():
        return pd.DataFrame()

    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        if r.get("coverage", 0) < 0.90:
            log.info("dropping %s %s — coverage %.1f%% (partial chip)",
                     r["port"], r["date"], 100 * r.get("coverage", 0))
            continue
        rows.append({
            "date": pd.Timestamp(r["date"]),
            "port": r["port"],
            "vessels": r.get("n_on_water", r.get("n_detections")),
            "coverage": r.get("coverage"),
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="date", columns="port", values="vessels", aggfunc="mean")
    wide.columns = [f"sar_vessels_{c}" for c in wide.columns]
    return wide.sort_index()


# ─────────────────────────────────────────────────────── assembly
def build() -> pd.DataFrame:
    parts = {"gdelt": gdelt_daily(), "market": market_daily(), "sar": sar_daily()}
    for name, part in parts.items():
        log.info("%-7s %s", name, part.shape if len(part) else "EMPTY")

    frames = [p for p in parts.values() if len(p)]
    if not frames:
        raise SystemExit("no arm produced features")

    idx = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames])))
    full = pd.date_range(idx.min(), idx.max(), freq="D")
    combined = pd.concat([f.reindex(full) for f in frames], axis=1)
    combined.index.name = "date"
    return combined


def check_leakage(df: pd.DataFrame) -> list[str]:
    """
    Cheap structural leakage checks. Not a substitute for the CI gate, but
    these catch the mistakes that actually happen.
    """
    problems = []

    # 1. No feature may correlate ~perfectly with its own future.
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 60:
            continue
        fwd = s.shift(-1).dropna()
        common = s.index.intersection(fwd.index)
        if len(common) > 30:
            c = np.corrcoef(s.loc[common], fwd.loc[common])[0, 1]
            if abs(c) > 0.9999:
                problems.append(f"{col}: identical to its own 1-day-ahead value")

    # 2. Rolling features must not be defined before their window can be full.
    for col in [c for c in df.columns if "_7d" in c or "_z" in c]:
        s = df[col]
        first = s.first_valid_index()
        base = df["brent"].first_valid_index() if "brent" in df else None
        if first is not None and base is not None and first < base:
            problems.append(f"{col}: defined at {first.date()} before source data at {base.date()}")

    # 3. Index must be sorted and unique — an out-of-order index silently
    #    breaks every temporal split downstream.
    if not df.index.is_monotonic_increasing:
        problems.append("index is not monotonically increasing")
    if df.index.duplicated().any():
        problems.append(f"index has {int(df.index.duplicated().sum())} duplicate dates")

    # 4. A DERIVED feature must never outlive its SOURCE. pandas pct_change
    #    pads NaNs by default, which fabricates returns on non-trading days and
    #    extends the series past the last real close — caught exactly this way.
    derived = {
        "brent_ret_1d": "brent", "brent_ret_7d": "brent",
        "brent_vol_7d": "brent", "spread_chg_7d": "brent_wti_spread",
        "spread_z": "brent_wti_spread",
        "spread_smooth_10d": "brent_wti_spread",
        "spread_days_gt15_30d": "brent_wti_spread",
    }
    for child, parent in derived.items():
        if child not in df or parent not in df:
            continue
        c_last, p_last = df[child].last_valid_index(), df[parent].last_valid_index()
        if c_last is not None and p_last is not None and c_last > p_last:
            problems.append(
                f"{child} extends to {c_last.date()} but its source {parent} "
                f"ends {p_last.date()} — values were fabricated by a fill"
            )
        if int(df[child].notna().sum()) > int(df[parent].notna().sum()):
            problems.append(
                f"{child} has more observations ({int(df[child].notna().sum())}) "
                f"than its source {parent} ({int(df[parent].notna().sum())})"
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the common daily feature index")
    p.add_argument("--check-leakage", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stdout)

    df = build()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "features_daily.parquet"
    df.to_parquet(dest)

    print(f"\n  feature panel: {df.shape[0]:,} days x {df.shape[1]} features")
    print(f"  range        : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  written      : {dest}\n")

    print(f"  {'feature':<26} {'non-null':>9} {'coverage':>9}  {'first':<12} {'last':<12}")
    print("  " + "-" * 72)
    for c in df.columns:
        s = df[c]
        n = int(s.notna().sum())
        fv, lv = s.first_valid_index(), s.last_valid_index()
        print(f"  {c:<26} {n:>9,} {n/len(df):>8.1%}  "
              f"{str(fv.date()) if fv is not None else '-':<12} "
              f"{str(lv.date()) if lv is not None else '-':<12}")

    problems = check_leakage(df)
    print("\n  leakage checks:", "PASS — no structural issues" if not problems else "FAIL")
    for p_ in problems:
        print(f"    ✗ {p_}")
    print()
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
