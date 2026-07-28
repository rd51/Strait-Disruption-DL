"""
ONE-SHOT AIS TEST — settles whether the problem is the KEY or the BOX.

Tries, in order, each for 15 seconds:
  1. WORLDWIDE   — if this is silent, the key/account/network is the problem.
  2. SW->NE      — [[23.5,51.0],[27.0,57.5]]   (what we've been using)
  3. NW->SE      — [[27.0,51.0],[23.5,57.5]]   (some APIs want this)
  4. lon/lat     — [[51.0,23.5],[57.5,27.0]]   (in case order is lon-first)
  5. BIG GULF    — a deliberately huge box around the whole region

Prints a verdict at the end.

Put your key in  aisstream.key  (this folder), or set $env:AISSTREAM_KEY.
    python ais_test.py
"""

import asyncio, json, os
from datetime import datetime, timezone
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
WS_URL = "wss://stream.aisstream.io/v0/stream"


def load_key():
    k = os.environ.get("AISSTREAM_KEY", "").strip()
    if k:
        return k, "env var"
    p = os.path.join(HERE, "aisstream.key")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip(), "aisstream.key"
    return "", "NOT FOUND"


KEY, SRC = load_key()

TESTS = [
    ("1. WORLDWIDE",  [[[-90.0, -180.0], [90.0, 180.0]]]),
    ("2. SW->NE",     [[[23.5, 51.0], [27.0, 57.5]]]),
    ("3. NW->SE",     [[[27.0, 51.0], [23.5, 57.5]]]),
    ("4. lon-first",  [[[51.0, 23.5], [57.5, 27.0]]]),
    ("5. BIG GULF",   [[[20.0, 45.0], [32.0, 62.0]]]),
]


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


async def one(label, bbox, secs=15):
    print(f"\n{'='*58}\n{label}   bbox={bbox}\n{'='*58}")
    n = 0
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            await ws.send(json.dumps({"APIKey": KEY, "BoundingBoxes": bbox}))
            print(f"  [{now()}] subscribed, listening {secs}s ...")

            async def rx():
                nonlocal n
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"  [{now()}] non-JSON: {raw[:150]}")
                        continue
                    if isinstance(m, dict) and (m.get("error") or m.get("Error")):
                        print(f"  [{now()}] ✗ SERVER SAYS: {m.get('error') or m.get('Error')}")
                        return
                    n += 1
                    if n <= 2:
                        md = m.get("MetaData", {}) or {}
                        print(f"  [{now()}] #{n} {m.get('MessageType')} "
                              f"{(md.get('ShipName') or '?').strip()[:18]} "
                              f"({md.get('latitude')},{md.get('longitude')})")
                    if n >= 25:
                        return

            await asyncio.wait_for(rx(), timeout=secs)
    except asyncio.TimeoutError:
        pass
    except websockets.ConnectionClosed as e:
        print(f"  [{now()}] closed: code={e.code} reason={e.reason!r}")
    except Exception as e:
        print(f"  [{now()}] {type(e).__name__}: {e}")
    print(f"  RESULT: {n} messages")
    return n


async def main():
    print("=" * 58)
    if not KEY:
        print("NO KEY FOUND.")
        print(f"  Create {os.path.join(HERE,'aisstream.key')} with your key inside,")
        print("  or run:  $env:AISSTREAM_KEY=\"your-key\"")
        return
    print(f"key source: {SRC}   key: {KEY[:4]}...{KEY[-4:]}  len={len(KEY)}")
    print("=" * 58)

    res = {}
    for label, bbox in TESTS:
        res[label] = await one(label, bbox)
        if label.startswith("1.") and res[label] == 0:
            print("\n  ⚠ Worldwide returned NOTHING — no point testing boxes.")
            break

    print("\n" + "=" * 58 + "\nVERDICT\n" + "=" * 58)
    for k, v in res.items():
        print(f"  {k:<16} {v:>4} messages")
    w = res.get("1. WORLDWIDE", 0)
    if w == 0:
        print("\n  ✗ Worldwide silent => NOT a geography problem.")
        print("    The key, the account, or the network is blocking it.")
        print("    - Confirm the key is active/enabled on aisstream.io")
        print("    - Check for any email-verification or quota notice")
        print("    - Try a phone hotspot (corporate/uni nets block long WS)")
    else:
        good = [k for k, v in res.items() if not k.startswith("1.") and v > 0]
        if good:
            print(f"\n  ✓ Key is FINE. Working box format(s): {good}")
            print("    Use that ordering in ais_client.py.")
        else:
            print("\n  ⚠ Worldwide works but NO regional box does.")
            print("    Paste this output — the box semantics differ from assumed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped.")
