"""
GDELT historical backfill from RAW 15-minute files — Route A.

No Google Cloud account, no billing, no auth. Costs bandwidth and wall-clock
time, not money, and there is no way for it to produce a surprise bill.

It reuses `download_events` from the live poller, so the historical path runs
the EXACT parser and filters the live path runs. That is the point: if the
backtest and inference paths diverge, the backtest stops meaning anything.

Scale (verified by HEAD on the anchors): the three windows are ~366 days x 96
slots ≈ 35,000 files ≈ 3.5 GB. That is a background job, not an interactive
one. Therefore:

  · bounded concurrency (default 8) — enough to saturate a home line, polite
    enough not to hammer a free public archive
  · resumable — a slot already on disk is skipped, so Ctrl-C and restart is
    always safe and never re-downloads
  · missing slots are recorded, not fatal. GDELT genuinely has gaps in its
    history; one 404 must not kill a six-hour job
  · progress and ETA on one line, because a silent multi-hour job is
    indistinguishable from a hung one

USAGE
    python -m ingest.gdelt.backfill_raw --list
    python -m ingest.gdelt.backfill_raw --window 2019_gulf_of_oman --dry-run
    python -m ingest.gdelt.backfill_raw --window 2019_gulf_of_oman
    python -m ingest.gdelt.backfill_raw --all --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ...common.secrets import safe_stdout
from .backfill import ANCHOR_WINDOWS
from .poller import Config, SlotUnavailable, download_events
from .schema import GDELT2_EPOCH, RETAIN_COLS, SchemaError
from .storage import RawStore
from .transform import coerce_numeric, filter_gulf

log = logging.getLogger("gdelt.backfill_raw")

SLOT_MINUTES = 15


def slots_in_window(start: str, end: str) -> list[str]:
    """Every 15-minute slot stamp in [start, end], inclusive of both dates."""
    begin = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    finish = (datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
              + timedelta(days=1))
    epoch = datetime.strptime(GDELT2_EPOCH, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if begin < epoch:
        log.warning("window starts %s, before the GDELT 2.0 epoch %s — clamping",
                    start, GDELT2_EPOCH)
        begin = epoch

    out, cursor = [], begin
    while cursor < finish:
        out.append(cursor.strftime("%Y%m%d%H%M%S"))
        cursor += timedelta(minutes=SLOT_MINUTES)
    return out


class Progress:
    """One-line progress with ETA. A silent multi-hour job looks like a hung one."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.ok = 0
        self.missing = 0
        self.failed = 0
        self.skipped = 0
        self.rows = 0
        self.started = time.monotonic()

    def tick(self, status: str, rows: int = 0) -> None:
        self.done += 1
        setattr(self, status, getattr(self, status) + 1)
        self.rows += rows
        if self.done % 25 and self.done != self.total:
            return
        elapsed = time.monotonic() - self.started
        rate = self.done / elapsed if elapsed else 0
        remaining = (self.total - self.done) / rate if rate else 0
        pct = 100 * self.done / self.total if self.total else 100
        sys.stdout.write(
            f"\r  {self.done:>6}/{self.total} ({pct:5.1f}%)  "
            f"ok={self.ok} skip={self.skipped} miss={self.missing} fail={self.failed}  "
            f"gulf_rows={self.rows:,}  {rate:4.1f} slot/s  ETA {timedelta(seconds=int(remaining))}   "
        )
        sys.stdout.flush()


def fetch_slot(stamp: str, cfg: Config, store: RawStore) -> tuple[str, int]:
    """Download + filter + persist one slot. Returns (status, gulf_row_count)."""
    if store.has_slot(stamp):
        return "skipped", 0
    try:
        df_raw = download_events(stamp, cfg)
    except SlotUnavailable:
        return "missing", 0            # genuine gaps exist in GDELT's history
    except SchemaError as exc:
        log.error("schema error on %s: %s", stamp, exc)
        return "failed", 0
    except Exception as exc:
        log.debug("slot %s failed: %r", stamp, exc)
        return "failed", 0

    df = coerce_numeric(df_raw)
    df_gulf = filter_gulf(df)
    keep = [c for c in RETAIN_COLS if c in df_gulf.columns] + ["gulf_match"]
    store.write_events(df_gulf[keep], stamp)
    return "ok", len(df_gulf)


def run_window(name: str, cfg: Config, store: RawStore, workers: int,
               dry_run: bool) -> dict:
    start, end = ANCHOR_WINDOWS[name]
    stamps = slots_in_window(start, end)
    already = sum(1 for s in stamps if store.has_slot(s))
    todo = len(stamps) - already

    print(f"\n  {name}   {start} -> {end}")
    print(f"  {len(stamps):,} slots | {already:,} already on disk | {todo:,} to fetch")
    print(f"  estimated download ≈ {todo * 0.1:,.0f} MB at ~100 KB/slot")

    if dry_run:
        print("  DRY RUN — nothing downloaded.")
        return {"window": name, "slots": len(stamps), "todo": todo, "dry_run": True}
    if todo == 0:
        print("  Nothing to do — window already complete.")
        return {"window": name, "slots": len(stamps), "ok": 0, "skipped": already}

    progress = Progress(len(stamps))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_slot, s, cfg, store): s for s in stamps}
        for fut in as_completed(futures):
            try:
                status, rows = fut.result()
            except Exception as exc:
                log.debug("worker error: %r", exc)
                status, rows = "failed", 0
            progress.tick(status, rows)
    print()

    result = {
        "window": name, "slots": len(stamps), "ok": progress.ok,
        "skipped": progress.skipped, "missing": progress.missing,
        "failed": progress.failed, "gulf_rows": progress.rows,
    }
    log.info("%s complete: %s", name, result)
    return result


def consolidate(name: str, store: RawStore) -> Path | None:
    """
    Merge a window's per-slot parquet files into one file for the backtest.

    35,000 tiny parquet files is a miserable thing to feed a model; one file
    per anchor is what the arms and the backtest actually want to read.
    """
    start, end = ANCHOR_WINDOWS[name]
    stamps = [s for s in slots_in_window(start, end) if store.has_slot(s)]
    if not stamps:
        return None

    paths = [str(store.events_path(s)) for s in stamps]
    print(f"  consolidating {len(paths):,} slot files ...")

    # pyarrow.dataset reads the whole file set in one pass. Calling
    # pd.read_parquet once per file spends most of its time re-opening
    # footers — measurably slower at ~9k files, and the next windows are
    # 11.7k and 14.6k.
    try:
        import pyarrow.dataset as pads

        merged = pads.dataset(paths, format="parquet").to_table().to_pandas()
    except Exception as exc:
        log.warning("fast path failed (%r); falling back to per-file reads", exc)
        frames = []
        for path in paths:
            try:
                frames.append(pd.read_parquet(path))
            except Exception as inner:
                log.warning("unreadable slot %s: %r", path, inner)
        if not frames:
            return None
        merged = pd.concat(frames, ignore_index=True)

    # Sort by publication time, not event time. DATEADDED is when GDELT
    # actually published the record; Day is when the event allegedly
    # happened and can precede it by up to a year (measured: max 365d on
    # the 2019_abqaiq window). A backtest indexed on Day would place
    # information before it was knowable — look-ahead bias.
    if "DATEADDED" in merged.columns:
        merged = merged.sort_values("DATEADDED", kind="stable").reset_index(drop=True)

    path = store.write_historical(merged, name)
    print(f"  consolidated {len(paths):,} slots -> {len(merged):,} rows -> {path.name}")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GDELT historical backfill from raw files")
    p.add_argument("--window", choices=sorted(ANCHOR_WINDOWS))
    p.add_argument("--all", action="store_true", help="every anchor window")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent downloads (default 8; be polite to a free archive)")
    p.add_argument("--dry-run", action="store_true",
                   help="report scale and what is already on disk, download nothing")
    p.add_argument("--list", action="store_true")
    p.add_argument("--consolidate-only", action="store_true",
                   help="skip downloading; just merge what is already on disk")
    args = p.parse_args(argv)

    # Must run BEFORE any print: on Windows stdout defaults to cp1252 when
    # redirected, and one unencodable character aborts the whole job.
    safe_stdout()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stdout)

    if args.list:
        for name, (s, e) in ANCHOR_WINDOWS.items():
            n = len(slots_in_window(s, e))
            print(f"  {name:22} {s} -> {e}   {n:,} slots  (~{n * 0.1:,.0f} MB)")
        return 0

    if not args.window and not args.all:
        p.error("pass --window <name>, --all, or --list")

    names = sorted(ANCHOR_WINDOWS) if args.all else [args.window]
    cfg = Config()
    store = RawStore(cfg.data_root)

    print(f"\n  stream: {'translingual' if cfg.translingual else 'english'} "
          f"| workers: {args.workers} | store: {store.root}")

    results = []
    for name in names:
        if not args.consolidate_only:
            results.append(run_window(name, cfg, store, args.workers, args.dry_run))
        if not args.dry_run:
            consolidate(name, store)

    if results and not args.dry_run:
        print("\n  SUMMARY")
        for r in results:
            print(f"    {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
