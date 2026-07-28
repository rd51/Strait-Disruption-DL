"""
GDELT 2.0 live ingestion service (Arm C) — containerised, slot-aligned, persistent.

Runs as a long-lived service. Every 15 minutes GDELT publishes a new Events
export at :00/:15/:30/:45 UTC; this service waits for each slot, downloads it,
applies the validated Gulf + conflict filters, and persists the result.

Differences from a demo script, all of which matter in a container:

  · Slot-aligned scheduling. Sleeping a fixed 900s after processing drifts by
    the processing time every cycle and eventually straddles slot boundaries.
    This computes the next wall-clock quarter-hour instead.
  · Idempotent. A slot already on disk is skipped, so restarts never duplicate.
  · Catch-up after downtime. Missed slots are reconstructed directly from their
    deterministic URLs (bounded by GDELT_MAX_CATCHUP) so a restarted container
    does not leave a hole in the series.
  · SIGTERM-aware. `docker compose down` sends SIGTERM; the loop exits between
    slots rather than being SIGKILLed mid-write ten seconds later.
  · Logs to stdout as structured lines — the container runtime is the log sink.

USAGE
    python -m ingest.gdelt.poller --once     # single slot, then exit
    python -m ingest.gdelt.poller            # service loop
    docker compose up gdelt-poller           # the intended deployment

No API key required — GDELT raw files are open.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import signal
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .schema import (
    EVENT_COLS,
    LASTUPDATE_EN,
    LASTUPDATE_TRANSLINGUAL,
    RETAIN_COLS,
    SchemaError,
    validate_raw_frame,
)
from .storage import RawStore
from .transform import (
    coerce_numeric,
    filter_conflict,
    filter_gulf,
    summarise,
    tension_score,
)

log = logging.getLogger("gdelt.poller")

SLOT_MINUTES = 15
BASE_URL = "http://data.gdeltproject.org/gdeltv2"


# ──────────────────────────────────────────────────────────────────── config

class Config:
    """12-factor config — everything overridable by environment variable."""

    def __init__(self) -> None:
        # Translingual covers 65 languages. Arabic and Farsi reporting on the
        # Gulf is not optional for this project, so it defaults on.
        self.translingual = _env_bool("GDELT_TRANSLINGUAL", True)
        # GDELT publishes a few minutes after the nominal slot; poll after a lag.
        self.poll_lag_s = int(os.environ.get("GDELT_POLL_LAG_SECONDS", "240"))
        self.max_catchup = int(os.environ.get("GDELT_MAX_CATCHUP", "16"))
        self.http_timeout = int(os.environ.get("GDELT_HTTP_TIMEOUT", "120"))
        self.max_retries = int(os.environ.get("GDELT_MAX_RETRIES", "4"))
        self.data_root = os.environ.get("HORMUZ_DATA_ROOT", "").strip() or None
        self.log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    def __repr__(self) -> str:
        return (
            f"Config(translingual={self.translingual}, poll_lag_s={self.poll_lag_s}, "
            f"max_catchup={self.max_catchup}, data_root={self.data_root or 'default'})"
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ─────────────────────────────────────────────────────────── slot arithmetic

def slot_stamp(dt: datetime) -> str:
    """Floor a UTC datetime to its 15-minute GDELT slot -> `YYYYMMDDHHMMSS`."""
    dt = dt.astimezone(timezone.utc)
    floored = dt.replace(minute=(dt.minute // SLOT_MINUTES) * SLOT_MINUTES,
                         second=0, microsecond=0)
    return floored.strftime("%Y%m%d%H%M%S")


def stamp_to_dt(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def next_slot_dt(now: datetime) -> datetime:
    """Wall-clock datetime of the next 15-minute boundary strictly after `now`."""
    now = now.astimezone(timezone.utc)
    floored = now.replace(minute=(now.minute // SLOT_MINUTES) * SLOT_MINUTES,
                          second=0, microsecond=0)
    return floored + timedelta(minutes=SLOT_MINUTES)


def missed_stamps(last_stamp: str, current_stamp: str, cap: int) -> list[str]:
    """Slots strictly between the last processed one and the current one."""
    try:
        cur = stamp_to_dt(current_stamp)
        cursor = stamp_to_dt(last_stamp) + timedelta(minutes=SLOT_MINUTES)
    except ValueError:
        return []
    out: list[str] = []
    while cursor < cur and len(out) < cap:
        out.append(cursor.strftime("%Y%m%d%H%M%S"))
        cursor += timedelta(minutes=SLOT_MINUTES)
    return out


def export_url(stamp: str, translingual: bool) -> str:
    kind = "translation.export" if translingual else "export"
    return f"{BASE_URL}/{stamp}.{kind}.CSV.zip"


# ─────────────────────────────────────────────────────────────────── fetch

class SlotUnavailable(RuntimeError):
    """The slot's file is not published (404) — expected for the newest slot."""


def _get_with_retry(url: str, cfg: Config) -> requests.Response:
    """GET with exponential backoff. 404 short-circuits — retrying won't help."""
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = requests.get(url, timeout=cfg.http_timeout)
            if resp.status_code == 404:
                raise SlotUnavailable(f"404 {url}")
            resp.raise_for_status()
            return resp
        except SlotUnavailable:
            raise
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            if attempt == cfg.max_retries:
                break
            log.warning("fetch attempt %d/%d failed (%s); retrying in %.0fs",
                        attempt, cfg.max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up on {url} after {cfg.max_retries} attempts: {last_exc!r}")


def fetch_latest_stamp(cfg: Config) -> str | None:
    """Read lastupdate.txt and return the stamp of the newest published export."""
    url = LASTUPDATE_TRANSLINGUAL if cfg.translingual else LASTUPDATE_EN
    resp = _get_with_retry(url, cfg)
    for line in resp.text.strip().splitlines():
        parts = line.split()
        if parts and ".export." in parts[-1]:
            return parts[-1].split("/")[-1].split(".")[0]
    return None


def download_events(stamp: str, cfg: Config) -> pd.DataFrame:
    """Download + unzip one slot into a DataFrame under the validated schema."""
    resp = _get_with_retry(export_url(stamp, cfg.translingual), cfg)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(
                fh,
                sep="\t",
                names=EVENT_COLS,
                header=None,
                dtype=str,
                quoting=3,            # QUOTE_NONE — GDELT does not quote fields
                on_bad_lines="skip",
            )
    validate_raw_frame(df)
    return df


# ──────────────────────────────────────────────────────────────── pipeline

def process_slot(stamp: str, cfg: Config, store: RawStore) -> dict | None:
    """Fetch → validate → filter → score → persist one 15-minute slot."""
    if store.has_slot(stamp):
        log.info("slot %s already persisted — skipping", stamp)
        return None

    df_raw = download_events(stamp, cfg)
    df = coerce_numeric(df_raw)
    df_gulf = filter_gulf(df)
    df_conflict = filter_conflict(df_gulf)

    score, parts = tension_score(df_conflict)
    counts = summarise(df, df_gulf, df_conflict)

    keep = [c for c in RETAIN_COLS if c in df_gulf.columns] + ["gulf_match"]
    events_path = store.write_events(df_gulf[keep], stamp)

    record = {
        "stamp": stamp,
        "slot_utc": stamp_to_dt(stamp).isoformat(),
        "ingested_utc": datetime.now(timezone.utc).isoformat(),
        "stream": "translingual" if cfg.translingual else "english",
        "tension_score_placeholder": score,
        **counts,
        **parts,
    }
    store.append_score(record, stamp)

    log.info(
        "slot=%s total=%d gulf=%d conflict=%d distinct_articles=%s "
        "TENSION(placeholder)=%.1f -> %s",
        stamp, counts["rows_total"], counts["rows_gulf"], counts["rows_conflict"],
        parts.get("articles_distinct"), score, events_path.name,
    )
    log.debug("gulf filter legs: bbox=%d country=%d keyword=%d keyword_only=%d",
              counts["leg_bbox"], counts["leg_country"],
              counts["leg_keyword"], counts["leg_keyword_only"])
    return record


# ─────────────────────────────────────────────────────────────── the service

class Service:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = RawStore(cfg.data_root)
        self._stop = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, _frame) -> None:
        log.info("received signal %s — shutting down after current slot", signum)
        self._stop = True

    def _sleep_until(self, target: datetime) -> None:
        """Interruptible sleep so SIGTERM is honoured within a second."""
        while not self._stop:
            remaining = (target - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))

    def run_once(self) -> dict | None:
        stamp = fetch_latest_stamp(self.cfg)
        if not stamp:
            log.warning("no export entry in lastupdate.txt — nothing to do")
            return None
        record = self._process_with_catchup(stamp)
        self.store.heartbeat()
        return record

    def _process_with_catchup(self, stamp: str) -> dict | None:
        state = self.store.read_state()
        last = state.get("last_stamp")

        if last and last != stamp:
            gaps = missed_stamps(last, stamp, self.cfg.max_catchup)
            if gaps:
                log.info("catching up %d missed slot(s) since %s", len(gaps), last)
                for gap in gaps:
                    if self._stop:
                        break
                    try:
                        process_slot(gap, self.cfg, self.store)
                    except SlotUnavailable:
                        log.warning("catch-up slot %s not published — skipping", gap)
                    except (SchemaError, RuntimeError) as exc:
                        log.error("catch-up slot %s failed: %s", gap, exc)

        record = process_slot(stamp, self.cfg, self.store)
        self.store.write_state({
            "last_stamp": stamp,
            "last_run_utc": datetime.now(timezone.utc).isoformat(),
            "stream": "translingual" if self.cfg.translingual else "english",
        })
        return record

    def run_forever(self) -> None:
        log.info("gdelt poller starting | %s", self.cfg)
        log.info("data root: %s", self.store.root)

        # Process whatever is current immediately, then align to slot boundaries.
        try:
            self.run_once()
        except Exception as exc:
            log.error("initial poll failed: %r", exc)

        while not self._stop:
            target = next_slot_dt(datetime.now(timezone.utc)) + timedelta(seconds=self.cfg.poll_lag_s)
            log.debug("sleeping until %s", target.isoformat())
            self._sleep_until(target)
            if self._stop:
                break
            try:
                self.run_once()
            except SlotUnavailable:
                log.warning("current slot not yet published — will retry next cycle")
            except SchemaError as exc:
                # Schema drift is not transient; surface it loudly but keep the
                # service alive so the outage is visible rather than a crashloop.
                log.error("SCHEMA ERROR — GDELT format may have changed: %s", exc)
            except Exception as exc:
                log.error("poll cycle failed: %r", exc)

        log.info("gdelt poller stopped cleanly")


# ─────────────────────────────────────────────────────────────────── entry

def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
    )
    logging.Formatter.converter = time.gmtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GDELT 2.0 live ingestion service")
    parser.add_argument("--once", action="store_true",
                        help="process the current slot once, then exit")
    parser.add_argument("--data-root", default=None,
                        help="override the raw-store root directory")
    args = parser.parse_args(argv)

    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root
    configure_logging(cfg.log_level)

    service = Service(cfg)
    if args.once:
        try:
            service.run_once()
        except SlotUnavailable:
            log.warning("current slot not published yet")
            return 0
        except Exception as exc:
            log.error("run failed: %r", exc)
            return 1
        return 0

    service.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
