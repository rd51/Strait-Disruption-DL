"""
Arm B market series — definitions, provenance, and credential loading.

PROVENANCE IS A FIRST-CLASS FIELD HERE, not documentation. Every series carries
`source` and `official`, because the two feeds have very different standing:

  · FRED  — official and citable. `DCOILBRENTEU` is EIA's Europe Brent Spot FOB
    redistributed by the St. Louis Fed (confirmed 2026-07-27: the series notes
    point at eia.doe.gov). This is what a backtest claim may rest on, and it
    removes the need for a separate EIA registration.
  · yfinance — an UNOFFICIAL scraper of Yahoo's private endpoints. It breaks
    without warning and its ToS position is ambiguous. It fills what FRED does
    not carry (intraday futures, equity ETFs) and must NEVER be the sole source
    behind a backtest claim.

Downstream code can therefore filter on `official=True` to build a defensible
series, rather than trusting that whoever assembled the frame remembered the
distinction.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ...common.secrets import SecretNotFound, repo_root

FRED_BASE = "https://api.stlouisfed.org/fred"

# ── FRED: official, citable, daily ────────────────────────────────────────
# Verified 2026-07-27 — all three cover every anchor window INCLUDING the 2026
# Hormuz crisis (55 obs between 2026-05-01 and 2026-07-20).
FRED_SERIES = {
    "DCOILBRENTEU": {
        "label": "brent_spot_usd",
        "desc": "Crude Oil Prices: Brent - Europe (EIA via FRED), $/bbl",
        "role": "The sharpest single macro proxy for Hormuz: ~20M bpd transits "
                "the strait, so Brent prices the risk before ships pile up.",
    },
    "DCOILWTICO": {
        "label": "wti_spot_usd",
        "desc": "Crude Oil Prices: WTI - Cushing (EIA via FRED), $/bbl",
        "role": "Non-Gulf benchmark. The Brent-WTI SPREAD isolates Gulf-specific "
                "risk from a global oil move — a spread widening is a chokepoint "
                "signal, a parallel move is not.",
    },
    "DHHNGSP": {
        "label": "henry_hub_gas_usd",
        "desc": "Henry Hub Natural Gas Spot Price, $/MMBtu",
        "role": "Qatar routes ~20% of global LNG through Hormuz; US gas is the "
                "non-Gulf comparator.",
    },
}

# ── yfinance: unofficial, supplementary ───────────────────────────────────
YAHOO_SERIES = {
    "BZ=F": {"label": "brent_futures", "desc": "Brent crude futures (front month)"},
    "CL=F": {"label": "wti_futures", "desc": "WTI crude futures (front month)"},
    # Exporters — supply side, ship through Hormuz.
    "KSA": {"label": "etf_saudi", "desc": "iShares MSCI Saudi Arabia ETF"},
    # Importers — demand side, Hormuz-dependent crude.
    "FXI": {"label": "etf_china", "desc": "iShares China Large-Cap ETF"},
    "INDA": {"label": "etf_india", "desc": "iShares MSCI India ETF"},
    "EWJ": {"label": "etf_japan", "desc": "iShares MSCI Japan ETF"},
    "EWY": {"label": "etf_skorea", "desc": "iShares MSCI South Korea ETF"},
    # Regional proximity.
    "EIS": {"label": "etf_israel", "desc": "iShares MSCI Israel ETF"},
}

# Full history: earliest anchor lead-in through today. The 2019 windows start
# 2019-04-01, so 2018 gives a year of pre-event baseline for anomaly training.
HISTORY_START = "2018-01-01"


def load_fred_key() -> str:
    """FRED_API_KEY env var, then .env, then secrets/. Never logged."""
    env = os.environ.get("FRED_API_KEY", "").strip()
    if env:
        return env

    root = repo_root()
    dotenv = root / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^\s*FRED_API_KEY\s*=\s*(.+)$", line, re.I)
            if m:
                v = m.group(1).strip().strip('"').strip("'").split("#")[0].strip()
                if v:
                    return v

    secrets = root / "secrets"
    if secrets.exists():
        for f in sorted(secrets.iterdir()):
            if "fred" not in f.name.lower():
                continue
            text = f.read_text(encoding="utf-8-sig").strip()
            if not text:
                continue
            try:
                data = json.loads(text)
                for k in ("api_key", "key", "fred_api_key"):
                    if data.get(k):
                        return str(data[k]).strip()
            except json.JSONDecodeError:
                m = re.search(r"([0-9a-zA-Z]{32,})", text)
                if m:
                    return m.group(1)

    raise SecretNotFound(
        "No FRED API key.\n"
        "  Looked at: FRED_API_KEY env var, .env, secrets/*fred*\n"
        "  Get a free key at https://fredaccount.stlouisfed.org/apikeys\n"
        "  then add   FRED_API_KEY=<key>   to .env (gitignored)."
    )


def data_root() -> Path:
    env = os.environ.get("MARKET_DATA_ROOT", "").strip()
    return Path(env) if env else repo_root() / "data" / "raw" / "market"
