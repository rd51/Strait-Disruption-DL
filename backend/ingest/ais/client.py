"""
aisstream.io live vessel collector — containerised service (vessel-safety layer).

⚠️ NOT A PREDICTOR. This feeds the live, rule-based vessel-safety layer. It is
never backtested and makes no forecast claim. Keep it walled off from the
disruption engine's ML validity.

Service behaviours that the previous script lacked, each of which matters:

  · Persistence. The old client kept an in-memory dict and printed — every
    restart lost the session. This writes an append-only position history plus
    an atomic current-state snapshot.
  · Stall detection. A malformed bounding box is accepted by the server, which
    then returns SILENCE with no error. "No messages" is therefore a real
    failure mode that looks identical to "quiet water", so a stall is logged
    loudly with the causes named rather than waited out forever.
  · Fatal vs transient errors. A bad API key is fatal — reconnect-looping on it
    forever just hides the problem. Network drops are transient and retried
    with backoff.
  · SIGTERM-aware, flushing buffered positions before exit.

USAGE
    python -m ingest.ais.client --duration 60     # bounded first-run check
    python -m ingest.ais.client                   # run as a service
    docker compose up -d ais-collector
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import websockets

from .constants import (
    MESSAGE_TYPES,
    WS_URL,
    KeyError_,
    load_api_key,
    load_bbox,
    redact,
)
from .store import VesselStore

log = logging.getLogger("ais.client")


class Config:
    def __init__(self) -> None:
        self.flush_seconds = int(os.environ.get("AIS_FLUSH_SECONDS", "300"))
        self.flush_rows = int(os.environ.get("AIS_FLUSH_ROWS", "2000"))
        self.stall_seconds = int(os.environ.get("AIS_STALL_SECONDS", "120"))
        self.ping_interval = int(os.environ.get("AIS_PING_INTERVAL", "20"))
        self.data_root = os.environ.get("AIS_DATA_ROOT", "").strip() or None
        self.log_level = os.environ.get("LOG_LEVEL", "INFO").upper()


class FatalStreamError(RuntimeError):
    """Server rejected the subscription — retrying will not help."""


class Collector:
    def __init__(self, cfg: Config, duration: int | None = None):
        self.cfg = cfg
        self.duration = duration
        self.store = VesselStore(cfg.data_root)
        self.api_key = load_api_key()
        self.bbox = load_bbox()
        self._stop = False
        self._started = time.monotonic()
        self._last_flush = time.monotonic()

    def request_stop(self, signum=None, _frame=None) -> None:
        if signum:
            log.info("received signal %s — flushing and shutting down", signum)
        self._stop = True

    @property
    def _expired(self) -> bool:
        return self.duration is not None and (time.monotonic() - self._started) >= self.duration

    def _maybe_flush(self, force: bool = False) -> None:
        due = (
            force
            or self.store.buffered >= self.cfg.flush_rows
            or (time.monotonic() - self._last_flush) >= self.cfg.flush_seconds
        )
        if due and self.store.buffered:
            self.store.flush_positions()
            self.store.write_snapshot()
            self._last_flush = time.monotonic()
        self.store.heartbeat()

    async def _consume(self) -> None:
        subscribe = {
            "APIKey": self.api_key,
            "BoundingBoxes": self.bbox,
            "FilterMessageTypes": MESSAGE_TYPES,
        }
        log.info("connecting to aisstream (key %s), bbox=%s",
                 redact(self.api_key), self.bbox)

        async with websockets.connect(
            WS_URL, ping_interval=self.cfg.ping_interval, ping_timeout=20
        ) as ws:
            await ws.send(json.dumps(subscribe))
            log.info("subscribed; waiting for messages")
            last_msg = time.monotonic()
            stall_warned = False

            while not self._stop and not self._expired:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    idle = time.monotonic() - last_msg
                    if idle > self.cfg.stall_seconds and not stall_warned:
                        log.warning(
                            "NO MESSAGES for %.0fs on bbox=%s. The server accepts a bad "
                            "subscription and returns silence, so check in order: "
                            "(1) RECEIVER COVERAGE — measured 2026-07-27, aisstream has "
                            "NO coverage in the Gulf; a correct Hormuz box yields zero "
                            "while a European box yields ~100 msg/s; "
                            "(2) bounding box nesting must be three levels; "
                            "(3) API key valid and not rate-limited.",
                            idle, self.bbox,
                        )
                        stall_warned = True
                    self._maybe_flush()
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if "error" in msg:
                    raise FatalStreamError(str(msg["error"]))

                last_msg = time.monotonic()
                stall_warned = False
                self.store.handle_message(msg)

                if self.store.messages_seen % 100 == 0:
                    log.info("%d messages | %d unique vessels | %d buffered",
                             self.store.messages_seen, len(self.store.vessels),
                             self.store.buffered)
                self._maybe_flush()

    async def run(self) -> int:
        backoff = 2.0
        while not self._stop and not self._expired:
            try:
                await self._consume()
                backoff = 2.0
            except FatalStreamError as exc:
                log.error("SERVER REJECTED THE SUBSCRIPTION: %s", exc)
                log.error("This is fatal (usually an invalid or expired API key). "
                          "Not retrying — fix the key and restart.")
                self._maybe_flush(force=True)
                return 2
            except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                if self._stop or self._expired:
                    break
                log.warning("connection dropped (%r); reconnecting in %.0fs", exc, backoff)
                self._maybe_flush()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

        self._maybe_flush(force=True)
        snap = self.store.write_snapshot()

        log.info("stopped | %d messages, %d unique vessels, %d positions written",
                 self.store.messages_seen, len(self.store.vessels),
                 self.store.positions_written)
        log.info("snapshot: %s", snap)

        if self.store.messages_seen == 0:
            log.error("ZERO messages received — the feed was never confirmed. "
                      "Do not treat this run as a successful validation.")
            return 1
        return 0


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
    )
    logging.Formatter.converter = time.gmtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="aisstream.io live vessel collector")
    parser.add_argument("--duration", type=int, default=None,
                        help="stop after N seconds (use for the first-run feed check)")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)

    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root
    configure_logging(cfg.log_level)

    try:
        collector = Collector(cfg, duration=args.duration)
    except KeyError_ as exc:
        log.error("%s", exc)
        return 3

    signal.signal(signal.SIGTERM, collector.request_stop)
    signal.signal(signal.SIGINT, collector.request_stop)

    started = datetime.now(timezone.utc)
    log.info("ais collector starting at %s", started.isoformat())
    try:
        return asyncio.run(collector.run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
