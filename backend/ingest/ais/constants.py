"""
aisstream.io connection constants and key loading for the vessel-safety layer.

⚠️ SCOPE. This collector feeds the LIVE, RULE-BASED vessel-safety layer. It is
NOT part of the backtested disruption engine and nothing here may be presented
as predictive. The only permitted coupling is the risk index modulating the
alert threshold — which happens downstream, not in this module.
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path

WS_URL = "wss://stream.aisstream.io/v0/stream"

# ─────────────────────────────────────────────────────────── THE BBOX GOTCHA
#
# aisstream wants a LIST OF BOXES, each box being two [lat, lon] corners.
# That is THREE bracket levels. Two levels is read as two zero-area boxes and
# silently matches nothing — the server accepts it, returns no error, and the
# feed is simply dead. This cost real debugging time; do not "tidy" it.
#
#   CORRECT  [[[23.5, 51.0], [27.0, 57.5]]]
#   WRONG     [[23.5, 51.0], [27.0, 57.5]]
#
# SW corner (23.5N, 51.0E) sits below Abu Dhabi and the offshore terminals;
# NE corner (27.0N, 57.5E) reaches past Hormuz into the Gulf of Oman approaches.
# Verified to cover all 13 ports/terminals in uae_ports.csv.
UAE_HORMUZ_BBOX = [[[23.5, 51.0], [27.0, 57.5]]]

# ⚠️ COVERAGE, NOT GEOMETRY. Measured 2026-07-27: this box is geographically
# correct and correctly formatted, and still returns ZERO messages, because
# aisstream has no receiver coverage in the Gulf. See the CLAUDE.md gotcha.
# Use AIS_BBOX to point the collector at a region that does have coverage when
# testing the pipeline itself.
MESSAGE_TYPES = ["PositionReport", "ShipStaticData"]


def load_bbox() -> list:
    """
    Bounding boxes from AIS_BBOX (JSON), else the UAE/Hormuz default.

    Enforces the three-bracket-level rule in code. A two-level box is read by
    the server as two zero-area boxes and returns silence with no error, so
    validating here converts a silent dead feed into a startup crash.
    """
    raw = os.environ.get("AIS_BBOX", "").strip()
    if not raw:
        return UAE_HORMUZ_BBOX

    import json
    try:
        boxes = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AIS_BBOX is not valid JSON: {exc}") from exc

    def depth(x):
        return 1 + depth(x[0]) if isinstance(x, list) and x else 0

    if depth(boxes) != 3:
        raise ValueError(
            f"AIS_BBOX must be a LIST OF BOXES — three bracket levels, e.g. "
            f'[[[23.5,51.0],[27.0,57.5]]]. Got nesting depth {depth(boxes)}. '
            "Two levels is silently read as two zero-area boxes and matches nothing."
        )
    for box in boxes:
        if len(box) != 2 or any(len(corner) != 2 for corner in box):
            raise ValueError(f"each box needs exactly two [lat, lon] corners; got {box}")
    return boxes


class KeyError_(RuntimeError):
    """Raised when no usable aisstream key is available."""


def _looks_like_placeholder(value: str) -> bool:
    """
    Detect placeholder text of ANY shape, not one hardcoded string.

    The previous guard compared against the literal "PASTE_YOUR_KEY_HERE", so
    the placeholder actually sitting in ports/aisstream.key sailed straight
    past it and got sent to the server as though it were a real credential.
    Anything all-caps-and-underscores, or containing an instruction word, is
    a placeholder — no real key looks like that.
    """
    if not value:
        return True
    upper_underscore = set(value) <= set(string.ascii_uppercase + string.digits + "_")
    if upper_underscore:
        return True
    lowered = value.lower()
    return any(word in lowered for word in ("paste", "your", "here", "xxxx", "changeme", "todo"))


def load_api_key(key_file: Path | str | None = None) -> str:
    """
    Resolve the aisstream key: environment variable first, then a local file.

    Fails loudly with an actionable message rather than returning a placeholder
    that produces a confusing server-side error later.
    """
    env_key = os.environ.get("AISSTREAM_KEY", "").strip()
    if env_key and not _looks_like_placeholder(env_key):
        return env_key

    # Try several filenames, not one. Windows File Explorer silently appends
    # .txt to extensionless files, and this key has in practice lived under
    # both `aisstream.key` and `aisstream.txt` — hardcoding one spelling makes
    # a present, valid key look missing.
    if key_file:
        candidates = [Path(key_file)]
    else:
        from ...common.paths import ports_dir
        ports = ports_dir()
        candidates = [
            ports / "aisstream.key",
            ports / "aisstream.txt",
            ports / "aisstream.key.txt",
        ]

    checked: list[str] = []
    for path in candidates:
        if not path.exists():
            checked.append(f"{path.name}: not found")
            continue
        # utf-8-sig: Windows editors add a BOM to "plain text" files routinely,
        # and a BOM silently corrupts the key when it is sent to the server.
        value = path.read_text(encoding="utf-8-sig").strip()
        if not value:
            checked.append(f"{path.name}: empty")
        elif _looks_like_placeholder(value):
            checked.append(f"{path.name}: PLACEHOLDER TEXT ({len(value)} chars)")
        else:
            return value

    detail = "\n".join(f"    {c}" for c in checked)
    raise KeyError_(
        "No usable aisstream key.\n"
        f"  AISSTREAM_KEY env var: {'placeholder text' if env_key else 'unset'}\n"
        f"  Key files searched in {candidates[0].parent}:\n{detail}\n"
        "Register at https://aisstream.io, then either:\n"
        "  $env:AISSTREAM_KEY = '<key>'\n"
        f"  Set-Content '{candidates[0]}' '<key>' -NoNewline -Encoding ascii\n"
        "Do not paste the key into a chat or commit it."
    )


def redact(key: str) -> str:
    """Safe-to-log form of a key — never log the value itself."""
    if len(key) <= 8:
        return "****"
    return f"{key[:3]}{'*' * (len(key) - 6)}{key[-3:]}"
