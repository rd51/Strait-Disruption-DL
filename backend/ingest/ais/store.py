"""
Vessel state and persistence for the AIS collector.

Two outputs, deliberately different in kind:

  positions/dt=YYYY-MM-DD/hour=HH/<ts>.parquet
      Append-only position history. This is the time series that downstream
      work reads — including last-seen gap analysis for transponder-dark
      inference. NOTE: darkness is detected by ABSENCE, downstream. A dark
      vessel transmits nothing, so it cannot appear here by definition; do not
      try to solve it in the collector.

  vessels_snapshot.json
      Current state of every vessel seen, rewritten atomically. This is a
      convenience view for the dashboard, not a historical record.

The previous client kept state in an in-memory dict and printed it. Nothing was
persisted, despite the docs claiming an "atomic JSON snapshot" — so every
restart lost the entire session.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("ais.store")

POSITION_COLS = [
    "mmsi", "ts_utc", "lat", "lon", "sog", "cog", "heading",
    "nav_status", "name", "ship_type", "destination", "draught",
]


def _atomic_write(path: Path, write_fn) -> None:
    """Temp-file-then-rename so a killed container never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def default_data_root() -> Path:
    env = os.environ.get("AIS_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    from ...common.paths import raw_dir
    return raw_dir() / "ais"


def _num(value):
    """aisstream sends nulls for fields a vessel isn't reporting — don't crash on them."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class VesselStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else default_data_root()
        self.positions_dir = self.root / "positions"
        self.state_dir = self.root / "_state"
        for d in (self.positions_dir, self.state_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.vessels: dict[str, dict] = {}
        self._buffer: list[dict] = []
        self.messages_seen = 0
        self.positions_written = 0

    # ───────────────────────────────────────────────────────────── parsing

    def handle_message(self, msg: dict) -> dict | None:
        """
        Merge one aisstream message into the vessel store.

        Position and static messages arrive separately and are joined by MMSI:
        PositionReport carries movement, ShipStaticData carries identity. A
        vessel is only fully described once both have been seen.
        """
        mtype = msg.get("MessageType")
        meta = msg.get("MetaData") or {}
        mmsi = meta.get("MMSI") or meta.get("MMSI_String")
        if mmsi is None:
            return None

        mmsi = str(mmsi)
        now = datetime.now(timezone.utc)
        rec = self.vessels.setdefault(mmsi, {"mmsi": mmsi, "first_seen": now.isoformat()})
        rec["last_seen"] = now.isoformat()

        if meta.get("ShipName"):
            rec["name"] = str(meta["ShipName"]).strip()
        lat, lon = _num(meta.get("latitude")), _num(meta.get("longitude"))
        if lat is not None and lon is not None:
            rec["lat"], rec["lon"] = round(lat, 5), round(lon, 5)

        body = msg.get("Message") or {}

        if mtype == "PositionReport":
            pr = body.get("PositionReport") or {}
            lat, lon = _num(pr.get("Latitude")), _num(pr.get("Longitude"))
            if lat is not None:
                rec["lat"] = round(lat, 5)
            if lon is not None:
                rec["lon"] = round(lon, 5)
            rec["sog"] = _num(pr.get("Sog"))
            rec["cog"] = _num(pr.get("Cog"))
            rec["heading"] = _num(pr.get("TrueHeading"))
            rec["nav_status"] = pr.get("NavigationalStatus")

            # Only position messages become time-series rows.
            self._buffer.append({
                "mmsi": mmsi,
                "ts_utc": now,
                "lat": rec.get("lat"),
                "lon": rec.get("lon"),
                "sog": rec.get("sog"),
                "cog": rec.get("cog"),
                "heading": rec.get("heading"),
                "nav_status": rec.get("nav_status"),
                "name": rec.get("name"),
                "ship_type": rec.get("ship_type"),
                "destination": rec.get("destination"),
                "draught": rec.get("draught"),
            })

        elif mtype == "ShipStaticData":
            sd = body.get("ShipStaticData") or {}
            rec["ship_type"] = sd.get("Type")
            rec["destination"] = (sd.get("Destination") or "").strip()
            rec["callsign"] = (sd.get("CallSign") or "").strip()
            rec["draught"] = _num(sd.get("MaximumStaticDraught"))
            imo = sd.get("ImoNumber")
            if imo:
                rec["imo"] = imo

        self.messages_seen += 1
        return rec

    # ──────────────────────────────────────────────────────────── flushing

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def flush_positions(self) -> Path | None:
        """Write buffered positions to an hour-partitioned parquet file."""
        if not self._buffer:
            return None

        df = pd.DataFrame(self._buffer, columns=POSITION_COLS)
        now = datetime.now(timezone.utc)
        path = (
            self.positions_dir
            / f"dt={now:%Y-%m-%d}"
            / f"hour={now:%H}"
            / f"{now:%Y%m%dT%H%M%S}.parquet"
        )
        _atomic_write(
            path,
            lambda p: df.to_parquet(p, engine="pyarrow", compression="snappy", index=False),
        )
        self.positions_written += len(df)
        log.info("flushed %d positions -> %s", len(df), path.name)
        self._buffer.clear()
        return path

    def write_snapshot(self) -> Path:
        path = self.root / "vessels_snapshot.json"
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "vessel_count": len(self.vessels),
            "messages_seen": self.messages_seen,
            "vessels": list(self.vessels.values()),
        }
        _atomic_write(
            path,
            lambda p: p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8"),
        )
        return path

    def heartbeat(self) -> None:
        (self.state_dir / "heartbeat").write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )
