"""
GDELT historical backfill via BigQuery — for the backtest anchor windows.

⚠️ BILLING. BigQuery on-demand pricing bills by BYTES SCANNED, and the public
GDELT tables are large. This module therefore refuses to run a billable query
until a dry run has reported the estimate and the caller has passed --execute.
The default action is always a dry run.

⚠️ COST MODEL — READ BEFORE TUNING QUERIES. Bytes scanned is a function of the
COLUMNS you reference, not the rows your WHERE clause returns, unless the table
is partitioned or clustered on the filter column. If `gdelt-bq.gdeltv2.events`
is not date-partitioned, narrowing the date range will NOT reduce the bill —
only naming fewer columns will. The dry run reports the truth for whatever the
table actually is; trust it over any assumption, including this comment.

Because row filters may not reduce cost, the SQL pulls the full Gulf-filtered
row set and leaves conflict filtering to the same Python code the live poller
uses (`transform.filter_conflict`). Identical filter logic on both paths avoids
train/serve skew in the backtest.

NOT VERIFIED FROM THIS ENVIRONMENT: no Google Cloud credentials, gcloud, or
google-cloud-bigquery were available when this was written, so the table name
and column spellings below are unconfirmed. Run `--probe-schema` first — it
queries INFORMATION_SCHEMA (metadata only, no table scan) and tells you whether
these names are right before you spend anything.

USAGE
    pip install -r ingest/requirements-backfill.txt
    gcloud auth application-default login       # or set GOOGLE_APPLICATION_CREDENTIALS

    python -m ingest.gdelt.backfill --probe-schema
    python -m ingest.gdelt.backfill --window 2019_gulf_of_oman              # dry run
    python -m ingest.gdelt.backfill --window 2019_gulf_of_oman --execute
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .schema import GDELT2_EPOCH, GULF_FIPS, GULF_LAT, GULF_LON, HORMUZ_TERMS
from .storage import RawStore

log = logging.getLogger("gdelt.backfill")

# Public GDELT 2.0 Events table. Override with GDELT_BQ_TABLE if this is wrong.
DEFAULT_TABLE = "gdelt-bq.gdeltv2.events"

# On-demand analysis pricing, US multi-region, USD per TiB scanned.
# ⚠️ VERIFY CURRENT PRICING — this changes, and a stale constant here produces a
# confidently wrong cost estimate. https://cloud.google.com/bigquery/pricing
USD_PER_TIB = float(os.environ.get("BQ_USD_PER_TIB", "6.25"))

# Refuse to execute above this scan size unless explicitly raised.
DEFAULT_MAX_GB = float(os.environ.get("BQ_MAX_SCAN_GB", "50"))

# ──────────────────────────────────────────────────────── backtest anchors
#
# Windows are padded well before each event so the backtest can measure LEAD
# TIME — how far ahead of the event the index moved — not just the reaction.
#
# Deliberately absent:
#   · 2011-12 Hormuz closure threat — predates GDELT 2.0 (starts 2015-02-18).
#     There is no GDELT 2.0 signal for it. Source separately or drop the label.
#   · 2026 window — dates were never pinned. Pin them from UKMTO advisories
#     before adding, rather than inventing a range.
ANCHOR_WINDOWS: dict[str, tuple[str, str]] = {
    # Fujairah anchorage attacks 12 May 2019; Gulf of Oman tankers 13 Jun 2019.
    "2019_gulf_of_oman": ("2019-04-01", "2019-07-31"),
    # Abqaiq/Khurais strike 14 Sep 2019.
    "2019_abqaiq": ("2019-08-01", "2019-10-31"),
    # Houthi Red Sea campaign — adjacent chokepoint, transfer signal.
    "2024_red_sea": ("2023-11-01", "2024-03-31"),
    # 🔴 ADDED 2026-07-28. This window was previously excluded on the grounds
    # that "the dates were never pinned" — that reason is now stale. The onset
    # is pinned at 2026-03-02 from FRED Brent (first +15% cross; peak $138.21
    # on 04-07), corroborated by DOC 2.0 headlines from 03-01/03-02 reporting
    # a tanker struck in the strait and Iran halting Hormuz traffic.
    # It is also the ONLY window where Arm B currently fires, so without it the
    # two arms cannot be compared on the same event.
    # Padded from mid-January so a ~6-week lead time is measurable.
    "2026_hormuz": ("2026-01-15", "2026-04-30"),
}

# BigQuery column spellings. The BQ table does NOT use the raw-CSV names
# throughout — notably SQLDATE (BQ) vs Day (CSV), and GLOBALEVENTID uppercase.
# `--probe-schema` verifies these against the live table.
BQ_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Actor1Name", "Actor1CountryCode",
    "Actor2Name", "Actor2CountryCode", "EventCode", "EventBaseCode",
    "EventRootCode", "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_Lat", "ActionGeo_Long", "DATEADDED", "SOURCEURL",
]


def _keyword_regex() -> str:
    return "|".join(t.replace(" ", r"\s+") for t in HORMUZ_TERMS)


def build_query(table: str, start: str, end: str) -> str:
    """
    Gulf-filtered Events for a date window.

    Mirrors `transform.filter_gulf`: bounding box OR FIPS country code OR
    chokepoint keyword. Never `SELECT *` — the column list is the cost lever.
    """
    cols = ",\n        ".join(BQ_COLUMNS)
    fips = ", ".join(f"'{c}'" for c in sorted(GULF_FIPS))
    return f"""
    SELECT
        {cols}
    FROM `{table}`
    WHERE
        SQLDATE BETWEEN @start_date AND @end_date
        AND (
            (   ActionGeo_Lat  BETWEEN {GULF_LAT[0]} AND {GULF_LAT[1]}
            AND ActionGeo_Long BETWEEN {GULF_LON[0]} AND {GULF_LON[1]} )
            OR ActionGeo_CountryCode IN ({fips})
            OR REGEXP_CONTAINS(LOWER(IFNULL(SOURCEURL, '')),           @kw)
            OR REGEXP_CONTAINS(LOWER(IFNULL(ActionGeo_FullName, '')),  @kw)
        )
    """.strip()


def _client():
    try:
        from google.cloud import bigquery
    except ImportError:
        raise SystemExit(
            "google-cloud-bigquery is not installed.\n"
            "  pip install -r ingest/requirements-backfill.txt"
        )
    return bigquery


def probe_schema(table: str) -> None:
    """
    Verify the table and column spellings before spending anything.

    INFORMATION_SCHEMA is metadata — it does not scan the table, so this is the
    cheap way to find out that a column is actually called SQLDATE and not Day.
    """
    bigquery = _client()
    client = bigquery.Client()

    project, dataset, tbl = table.split(".")
    sql = f"""
        SELECT column_name, data_type
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{tbl}'
        ORDER BY ordinal_position
    """
    rows = list(client.query(sql).result())
    if not rows:
        raise SystemExit(f"no columns returned — does `{table}` exist and is it readable?")

    actual = {r["column_name"] for r in rows}
    print(f"\n`{table}` — {len(rows)} columns\n")

    missing = [c for c in BQ_COLUMNS if c not in actual]
    if missing:
        print("  ✗ MISSING (BQ_COLUMNS is wrong — fix before querying):")
        for c in missing:
            near = [a for a in actual if a.lower() == c.lower()]
            hint = f"   (case differs: {near[0]})" if near else ""
            print(f"      {c}{hint}")
    else:
        print(f"  ✓ all {len(BQ_COLUMNS)} requested columns exist")

    print("\n  first 15 columns in the table:")
    for r in rows[:15]:
        print(f"      {r['column_name']:26} {r['data_type']}")
    print()


def run(window: str, execute: bool, max_gb: float, table: str) -> int:
    if window not in ANCHOR_WINDOWS:
        raise SystemExit(
            f"unknown window '{window}'. Available: {', '.join(ANCHOR_WINDOWS)}"
        )
    start, end = ANCHOR_WINDOWS[window]
    if start < GDELT2_EPOCH:
        raise SystemExit(
            f"window '{window}' starts {start}, before the GDELT 2.0 epoch "
            f"({GDELT2_EPOCH}). There is no GDELT 2.0 data for it."
        )

    bigquery = _client()
    client = bigquery.Client()

    sql = build_query(table, start, end)
    params = [
        bigquery.ScalarQueryParameter("start_date", "INT64", int(start.replace("-", ""))),
        bigquery.ScalarQueryParameter("end_date", "INT64", int(end.replace("-", ""))),
        bigquery.ScalarQueryParameter("kw", "STRING", _keyword_regex()),
    ]

    # ── mandatory dry run ────────────────────────────────────────────────
    dry_cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False,
                                      query_parameters=params)
    dry = client.query(sql, job_config=dry_cfg)
    scanned = dry.total_bytes_processed or 0
    gb = scanned / 1024 ** 3
    cost = (scanned / 1024 ** 4) * USD_PER_TIB

    print(f"\n  window   {window}   {start} → {end}")
    print(f"  table    {table}")
    print(f"  columns  {len(BQ_COLUMNS)} named (never SELECT *)")
    print(f"  SCAN     {gb:,.2f} GB")
    print(f"  COST     ~${cost:,.2f} USD  (at ${USD_PER_TIB}/TiB — verify current pricing)")
    print(f"  BUDGET   {max_gb:,.2f} GB\n")

    if gb > max_gb:
        print(f"  ✗ REFUSING: {gb:,.2f} GB exceeds the {max_gb:,.2f} GB budget.")
        print("    Narrow the window, drop columns, or raise --max-gb deliberately.\n")
        return 2

    if not execute:
        print("  Dry run only. Re-run with --execute to pull.\n")
        return 0

    # ── real query ───────────────────────────────────────────────────────
    log.info("executing query for %s ...", window)
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    df = job.to_dataframe()
    billed = job.total_bytes_billed or 0
    log.info("returned %d rows; billed %.2f GB", len(df), billed / 1024 ** 3)

    if df.empty:
        print("  ⚠️  zero rows returned — check the filter before trusting this.\n")
        return 1

    store = RawStore()
    path = store.write_historical(df, window)
    print(f"\n  ✓ wrote {len(df):,} rows -> {path}")
    print(f"    billed {billed / 1024 ** 3:,.2f} GB "
          f"(~${(billed / 1024 ** 4) * USD_PER_TIB:,.2f})\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GDELT historical backfill via BigQuery")
    p.add_argument("--window", choices=sorted(ANCHOR_WINDOWS),
                   help="backtest anchor window to pull")
    p.add_argument("--execute", action="store_true",
                   help="actually run the billable query (default is dry run only)")
    p.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB,
                   help=f"refuse to execute above this scan size (default {DEFAULT_MAX_GB})")
    p.add_argument("--table", default=os.environ.get("GDELT_BQ_TABLE", DEFAULT_TABLE))
    p.add_argument("--probe-schema", action="store_true",
                   help="verify table + column names via INFORMATION_SCHEMA (no table scan)")
    p.add_argument("--list-windows", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stdout)

    if args.list_windows:
        for name, (s, e) in ANCHOR_WINDOWS.items():
            print(f"  {name:22} {s} → {e}")
        return 0

    if args.probe_schema:
        probe_schema(args.table)
        return 0

    if not args.window:
        p.error("--window is required (or use --list-windows / --probe-schema)")

    return run(args.window, args.execute, args.max_gb, args.table)


if __name__ == "__main__":
    raise SystemExit(main())
