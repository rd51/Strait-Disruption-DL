"""
Raw-store layout and atomic writes for the GDELT arm.

Layout under the data root (default `<repo>/data/raw/gdelt`, `/data/raw/gdelt`
inside the container):

    live/events/dt=YYYY-MM-DD/<stamp>.parquet   Gulf-filtered rows, one file per slot
    live/scores/dt=YYYY-MM-DD/scores.jsonl      one placeholder-score line per slot
    historical/<window>.parquet                 BigQuery backfill, one file per anchor
    _state/poller.json                          last processed slot (restart safety)
    _state/heartbeat                            mtime probed by the Docker healthcheck

Everything is written temp-then-rename. A container killed mid-write must never
leave a half-written parquet that pandas will happily read as truncated data.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("gdelt.storage")


def default_data_root() -> Path:
    """Env override first, then the repo-relative default."""
    env = os.environ.get("HORMUZ_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    # ingest/gdelt/storage.py -> ingest/gdelt -> ingest -> repo root
    from ...common.paths import raw_dir
    return raw_dir() / "gdelt"


class RawStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else default_data_root()
        self.events_dir = self.root / "live" / "events"
        self.scores_dir = self.root / "live" / "scores"
        self.historical_dir = self.root / "historical"
        self.state_dir = self.root / "_state"
        for d in (self.events_dir, self.scores_dir, self.historical_dir, self.state_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────────── atomic helpers

    @staticmethod
    def _atomic_write_bytes(path: Path, write_fn) -> None:
        """Write via a temp file in the same directory, then atomically rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            write_fn(tmp_path)
            os.replace(tmp_path, path)   # atomic on POSIX and on NTFS
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # ─────────────────────────────────────────────────────────────── writers

    @staticmethod
    def _partition(stamp: str) -> str:
        """GDELT stamp `YYYYMMDDHHMMSS` -> Hive-style partition `dt=YYYY-MM-DD`."""
        return f"dt={stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"

    def events_path(self, stamp: str) -> Path:
        return self.events_dir / self._partition(stamp) / f"{stamp}.parquet"

    def write_events(self, df: pd.DataFrame, stamp: str) -> Path:
        path = self.events_path(stamp)
        self._atomic_write_bytes(
            path, lambda p: df.to_parquet(p, engine="pyarrow", compression="snappy", index=False)
        )
        return path

    def append_score(self, record: dict, stamp: str) -> Path:
        """
        Append one score line to the day's JSONL.

        Single-writer append: the poller is the only process writing this file,
        so an O_APPEND write of one line is atomic enough. Not safe to run two
        pollers against one data root.
        """
        path = self.scores_dir / self._partition(stamp) / "scores.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return path

    def write_historical(self, df: pd.DataFrame, name: str) -> Path:
        path = self.historical_dir / f"{name}.parquet"
        self._atomic_write_bytes(
            path, lambda p: df.to_parquet(p, engine="pyarrow", compression="snappy", index=False)
        )
        return path

    def has_slot(self, stamp: str) -> bool:
        """Idempotency check — has this 15-minute slot already been persisted?"""
        return self.events_path(stamp).exists()

    # ───────────────────────────────────────────────────────────────── state

    @property
    def state_path(self) -> Path:
        return self.state_dir / "poller.json"

    def read_state(self) -> dict:
        """
        Load poller state, tolerating a UTF-8 BOM.

        An unreadable state file must be LOUD. Returning {} silently is worse
        than crashing: the poller carries on looking healthy while catch-up is
        disabled, so downtime gaps never get backfilled and the series quietly
        develops holes that only surface during the backtest. Editors on Windows
        write BOMs routinely, so `utf-8-sig` is the tolerant read.
        """
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError) as exc:
            log.error(
                "state file %s is unreadable (%s) — catch-up is DISABLED for this "
                "cycle and downtime gaps will not be backfilled. Delete the file "
                "to reset cleanly.", self.state_path, exc,
            )
            return {}

    def write_state(self, state: dict) -> None:
        self._atomic_write_bytes(
            self.state_path,
            lambda p: p.write_text(json.dumps(state, indent=2), encoding="utf-8"),
        )

    def heartbeat(self) -> None:
        """Touch the file the container healthcheck watches."""
        hb = self.state_dir / "heartbeat"
        hb.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
