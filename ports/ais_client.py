"""
Step 1 — aisstream.io live vessel feed for the Hormuz Disruption Engine.

Connects to aisstream.io over WebSocket, subscribes to a bounding box covering
the full UAE coastline (Arabian Gulf + Gulf of Oman, including the Strait of
Hormuz), and prints/stores live vessel position + static messages.

USAGE:
    1. Register at https://aisstream.io and copy your free API key.
    2. export AISSTREAM_KEY="your-key-here"      (or paste into API_KEY below)
    3. pip install websockets
    4. python ais_client.py

Notes:
- aisstream bounding boxes are [[lat, lon], [lat, lon]] = [[SW corner],[NE corner]].
- The free tier streams live positions; there is no historical replay here.
- 'dark' vessels (transponder off) can't appear in AIS by definition — you detect
  them by their ABSENCE / last-seen gaps, handled later in the pipeline, not here.
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import websockets

def _load_api_key() -> str:
    env_key = os.environ.get("AISSTREAM_KEY", "").strip()
    if env_key:
        return env_key

    key_file = Path(__file__).with_name("aisstream.key")
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()

    return "PASTE_YOUR_KEY_HERE"


API_KEY = _load_api_key()

# Full UAE coastline: Arabian Gulf side + Gulf of Oman side, incl. Strait of Hormuz.
# SW corner ~ (23.5 N, 51.0 E) below Abu Dhabi/offshore terminals;
# NE corner ~ (27.0 N, 57.5 E) past Hormuz into the Gulf of Oman approaches.
# NOTE: aisstream wants a LIST OF BOXES, each box = two [lat, lon] corners.
# Passing [[lat,lon],[lat,lon]] (one level too shallow) is read as two
# zero-area boxes and silently matches nothing — that was the 0-vessel bug.
UAE_HORMUZ_BBOX = [[[23.5, 51.0], [27.0, 57.5]]]

WS_URL = "wss://stream.aisstream.io/v0/stream"

# Message types we care about:
#   PositionReport         -> live lat/lon/speed/course/heading
#   ShipStaticData         -> name, type, destination, dimensions
SUBSCRIBE = {
    "APIKey": API_KEY,
    "BoundingBoxes": UAE_HORMUZ_BBOX,
    "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
}

# In-memory vessel store: mmsi -> latest merged record.
vessels = {}


def _now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def handle_message(msg: dict):
    """Parse one aisstream message and update the vessel store."""
    mtype = msg.get("MessageType")
    meta = msg.get("MetaData", {}) or {}
    mmsi = meta.get("MMSI") or meta.get("MMSI_String")
    if mmsi is None:
        return

    rec = vessels.setdefault(str(mmsi), {"mmsi": str(mmsi)})
    rec["last_seen"] = _now()
    # MetaData usually carries a name + a lat/lon snapshot on every message.
    if meta.get("ShipName"):
        rec["name"] = meta["ShipName"].strip()
    if meta.get("latitude") is not None:
        rec["lat"] = round(meta["latitude"], 5)
        rec["lon"] = round(meta["longitude"], 5)

    body = (msg.get("Message") or {})

    if mtype == "PositionReport":
        pr = body.get("PositionReport", {})
        rec["lat"] = round(pr.get("Latitude", rec.get("lat", 0)), 5)
        rec["lon"] = round(pr.get("Longitude", rec.get("lon", 0)), 5)
        rec["sog"] = pr.get("Sog")          # speed over ground (knots)
        rec["cog"] = pr.get("Cog")          # course over ground (deg)
        rec["heading"] = pr.get("TrueHeading")
        rec["nav_status"] = pr.get("NavigationalStatus")

    elif mtype == "ShipStaticData":
        sd = body.get("ShipStaticData", {})
        rec["type"] = sd.get("Type")
        rec["destination"] = (sd.get("Destination") or "").strip()
        rec["callsign"] = (sd.get("CallSign") or "").strip()
        rec["draught"] = sd.get("MaximumStaticDraught")

    return rec


async def stream():
    if API_KEY in ("", "PASTE_YOUR_KEY_HERE"):
        raise SystemExit("Set AISSTREAM_KEY or add your key to ports/aisstream.key.")

    print(f"[{_now()}] connecting to aisstream, bbox={UAE_HORMUZ_BBOX} ...")
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(SUBSCRIBE))
        print(f"[{_now()}] subscribed. waiting for messages (Ctrl-C to stop)...")

        count = 0
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # aisstream sends an {"error": "..."} object on bad subscriptions.
            if "error" in msg:
                print(f"[{_now()}] SERVER ERROR: {msg['error']}")
                break

            rec = handle_message(msg)
            count += 1

            # Print a compact line every message; summarize the store periodically.
            if rec:
                name = rec.get("name", "?")
                sog = rec.get("sog", "-")
                dest = rec.get("destination", "")
                print(f"[{_now()}] {rec['mmsi']:>10}  {name[:22]:22}  "
                      f"({rec.get('lat')},{rec.get('lon')})  sog={sog}  dest={dest}")

            if count % 50 == 0:
                print(f"----- {_now()} store now tracking {len(vessels)} unique vessels -----")


async def main():
    # Auto-reconnect loop — aisstream drops idle/long connections periodically.
    while True:
        try:
            await stream()
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[{_now()}] connection dropped ({e!r}); reconnecting in 5s ...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print(f"\n[{_now()}] stopped. {len(vessels)} vessels seen this session.")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n[{_now()}] stopped. {len(vessels)} vessels seen this session.")
