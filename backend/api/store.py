"""
Query layer — DuckDB directly over the parquet store.

WHY NO SEPARATE DATABASE. The raw store IS the source of truth. Copying it into
a second database creates two things that can disagree, and an ETL step that
can silently fall behind. DuckDB reads hive-partitioned parquet natively with
real SQL, so the API queries exactly the files the collectors wrote — nothing
to sync, nothing to drift.

FRESHNESS IS A FIRST-CLASS RESULT HERE, not a footnote. Only ONE arm is
genuinely live:

    GDELT     15 minutes          live
    Market    daily close         ~1 day behind
    SAR       1.4-6 day revisit   "latest pass", never real-time
    Vessel    (aisstream has no Gulf coverage — nothing to serve)

A dashboard that renders all four identically implies a real-time physical read
of the Gulf that does not exist. Every series therefore ships with its own
as-of timestamp and age so the UI can show staleness rather than imply currency.
"""

from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from ..common.paths import repo_root
ROOT = repo_root()
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"

# Cadence each source can actually achieve. Anything older than `stale_after_h`
# is reported stale — thresholds reflect the SOURCE's real publication rhythm,
# not a uniform guess.
ARM_CADENCE = {
    "gdelt": {"label": "Geopolitical event text", "cadence": "15 minutes",
              "live": True, "stale_after_h": 1},
    "market": {"label": "Oil & equity series", "cadence": "daily close",
               "live": False, "stale_after_h": 96},
    "sar": {"label": "SAR port congestion", "cadence": "1.4-6 day revisit",
            "live": False, "stale_after_h": 336},
}


def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def jsonable(df: pd.DataFrame) -> list[dict]:
    """
    DataFrame -> JSON-safe records.

    Two conversions that both matter: numpy scalars (float64, int64) are not
    JSON-serializable and make FastAPI return 500, and NaN/NaT must become
    null rather than the string "NaN". `.astype(object)` demotes numpy types to
    Python ones; `.where(notna, None)` handles the missing values. Missing
    EITHER step produces a 500 that looks like a query bug — which is exactly
    how /features failed while /arms/market worked.
    """
    if df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out = out.astype(object).where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ────────────────────────────────────────────────────────────── freshness
def gdelt_latest_slot() -> datetime | None:
    files = glob.glob(str(RAW / "gdelt" / "live" / "events" / "**" / "*.parquet"),
                      recursive=True)
    if not files:
        return None
    stamp = max(Path(f).stem for f in files)      # YYYYMMDDHHMMSS
    try:
        return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def market_latest() -> datetime | None:
    path = RAW / "market" / "market_daily.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return pd.Timestamp(df.index.max()).to_pydatetime().replace(tzinfo=timezone.utc)


def sar_latest() -> datetime | None:
    path = DERIVED / "sar_cfar" / "cfar_detections.json"
    if not path.exists():
        return None
    recs = json.loads(path.read_text(encoding="utf-8"))
    dates = [r.get("date") for r in recs if r.get("date")]
    if not dates:
        return None
    return datetime.strptime(max(dates), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def freshness() -> dict:
    """Per-arm as-of timestamps and staleness. The UI's honesty contract."""
    now = _utcnow()
    getters = {"gdelt": gdelt_latest_slot, "market": market_latest, "sar": sar_latest}

    arms = {}
    for arm, meta in ARM_CADENCE.items():
        ts = getters[arm]()
        if ts is None:
            arms[arm] = {**meta, "as_of": None, "age_hours": None,
                         "status": "NO DATA"}
            continue
        age_h = (now - ts).total_seconds() / 3600.0
        arms[arm] = {
            **meta,
            "as_of": ts.isoformat(),
            "age_hours": round(age_h, 2),
            "status": "STALE" if age_h > meta["stale_after_h"] else "CURRENT",
        }

    arms["vessel"] = {
        "label": "Per-vessel AIS safety layer",
        "cadence": "n/a", "live": False, "as_of": None, "age_hours": None,
        "status": "UNAVAILABLE",
        "note": ("aisstream.io has zero AIS coverage in the Gulf (measured "
                 "2026-07-27: Hormuz 0 msgs/30s vs Europe 101 msgs/s). "
                 "Global Fishing Watch covers the Gulf but publishes 4 days "
                 "behind. No free source provides live Gulf vessel positions."),
    }

    return {
        "generated_utc": now.isoformat(),
        "live_arms": [a for a, m in ARM_CADENCE.items() if m["live"]],
        "arms": arms,
    }


# ───────────────────────────────────────────────────────────── features
def features(start: str | None = None, end: str | None = None,
             columns: list[str] | None = None) -> pd.DataFrame:
    path = DERIVED / "features_daily.parquet"
    if not path.exists():
        return pd.DataFrame()

    cols = "*" if not columns else ", ".join(['"date"'] + [f'"{c}"' for c in columns])
    where = []
    if start:
        where.append(f"date >= DATE '{start}'")
    if end:
        where.append(f"date <= DATE '{end}'")
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _con() as con:
        return con.execute(
            f"SELECT {cols} FROM read_parquet('{path.as_posix()}') {clause} ORDER BY date"
        ).df()


def gdelt_series(start: str | None = None, end: str | None = None,
                 default_days: int = 30) -> pd.DataFrame:
    """
    Daily GDELT aggregates keyed on DATEADDED.

    DATEADDED is publication time. `Day` is when the event allegedly happened
    and trails publication by up to a year, so indexing on it would place
    information before it was knowable.

    ⚠️ PARTITION PRUNING IS NOT OPTIONAL HERE. `live/events/` holds 35,000+
    slot files because the historical backfill writes into the same tree (so
    "live" is a misnomer — it is the whole slot store). An unbounded query
    scans every file and is far too slow to serve. When no range is given this
    defaults to the last `default_days`, and the WHERE clause filters on the
    `dt=` partition directory so DuckDB skips whole folders rather than opening
    every file.
    """
    base = RAW / "gdelt" / "live" / "events"
    if not glob.glob(str(base / "**" / "*.parquet"), recursive=True):
        return pd.DataFrame()

    if not start and not end:
        latest = gdelt_latest_slot()
        anchor = latest.date() if latest else datetime.now(timezone.utc).date()
        start = str(anchor - pd.Timedelta(days=default_days).to_pytimedelta())
        end = str(anchor)

    # Glob only the partitions in range — this is what makes it fast.
    dirs = sorted(p for p in base.glob("dt=*") if p.is_dir())
    selected = []
    for d in dirs:
        day = d.name.replace("dt=", "")
        if start and day < start:
            continue
        if end and day > end:
            continue
        selected.append(d)
    if not selected:
        return pd.DataFrame()

    pattern = "', '".join((d / "*.parquet").as_posix() for d in selected)
    pattern = f"['{pattern}']"
    clause = ""

    sql = f"""
        SELECT
            strptime(CAST(DATEADDED AS VARCHAR)[1:8], '%Y%m%d')::DATE AS date,
            COUNT(*)                                   AS rows,
            COUNT(DISTINCT SOURCEURL)                  AS articles,
            AVG(TRY_CAST(AvgTone AS DOUBLE))           AS tone_mean,
            AVG(TRY_CAST(GoldsteinScale AS DOUBLE))    AS goldstein_mean,
            COUNT(*) FILTER (WHERE EventRootCode IN
                ('13','14','15','16','17','18','19','20'))            AS conflict_rows,
            COUNT(DISTINCT SOURCEURL) FILTER (WHERE EventRootCode IN
                ('13','14','15','16','17','18','19','20'))            AS conflict_articles
        FROM read_parquet({pattern}, union_by_name=true)
        {clause}
        GROUP BY 1 ORDER BY 1
    """
    with _con() as con:
        return con.execute(sql).df()


def sar_detections() -> list[dict]:
    path = DERIVED / "sar_cfar" / "cfar_detections.json"
    if not path.exists():
        return []
    recs = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in recs:
        out.append({
            "port": r.get("port"), "date": r.get("date"),
            "vessels_on_water": r.get("n_on_water"),
            "detections_on_land": r.get("n_on_land"),
            "coverage": r.get("coverage"),
            "water_fraction": (r.get("water_mask") or {}).get("water_frac"),
            "usable": (r.get("coverage") or 0) >= 0.90,
            "note": None if (r.get("coverage") or 0) >= 0.90 else
                    "partial chip — count undercounts and is not comparable",
        })
    return sorted(out, key=lambda x: (x["port"] or "", x["date"] or ""))


def market_series(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    path = RAW / "market" / "market_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df.reset_index().rename(columns={"index": "date"})


def sources() -> dict:
    """Provenance for citeability — which series is official, which is not."""
    path = RAW / "market" / "sources.json"
    market = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        "market": market,
        "gdelt": {"source": "GDELT 2.0 Translingual", "official": True,
                  "url": "http://data.gdeltproject.org/gdeltv2/",
                  "note": "Free, no key. Aggregated on DATEADDED (publication time)."},
        "sar": {"source": "Copernicus Sentinel-1 GRD via Sentinel Hub Process API",
                "official": True, "url": "https://dataspace.copernicus.eu",
                "note": "CFAR baseline detections, water-masked. Not a trained model."},
    }


def gdelt_recent_titles(limit: int = 25) -> list[str]:
    """
    Headline-ish text from the most recent GDELT slots, for live inference.

    GDELT Events carries no article body, so text is recovered from the URL
    path — the same slug extraction Arm C is built on. Roughly half of URLs
    yield usable words; the rest are opaque numeric IDs and are skipped.
    """
    from ..arms.text.slugs import slug_text, MIN_WORDS

    base = RAW / "gdelt" / "live" / "events"
    parts = sorted(base.glob("dt=*"), reverse=True)[:3]
    files: list[str] = []
    for p in parts:
        files.extend(sorted(p.glob("*.parquet"), reverse=True)[:8])
    if not files:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["SOURCEURL"])
        except Exception:                               # noqa: BLE001
            continue
        for url in df["SOURCEURL"].dropna().unique():
            t = slug_text(url)
            if t and t.count(" ") + 1 >= MIN_WORDS and t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= limit:
                    return out
    return out
