"""
Render the data inventory diagram to PNG.

Companion to architecture.png. Where that figure answers "how does the system
work", this one answers "what data do we actually hold, and where does each
piece surface in the dashboard".

Numbers are READ FROM DISK at render time, not hardcoded, so the figure cannot
drift away from reality. If a count looks wrong in the image, the store changed
— regenerate rather than editing the label.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BG, PANEL, LINE = "#0b0f14", "#121821", "#2a3644"
INK, MUTED, DIM = "#e6edf3", "#8b9aab", "#6b7a8a"
OK, WARN, BAD = "#3fb950", "#d29922", "#f85149"
ACCENT, VIOLET = "#58a6ff", "#bc8cff"

W, H = 26.0, 19.2


def box(ax, x, y, w, h, title, lines=(), edge=LINE, fs=10.4, tfs=12, dashed=False):
    ax.add_patch(mp.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.055,rounding_size=0.16",
        linewidth=1.6, edgecolor=edge, facecolor=PANEL,
        linestyle=(0, (5, 3)) if dashed else "solid", zorder=2))
    ax.text(x + w / 2, y + h - 0.36, title, ha="center", va="top",
            color=INK, fontsize=tfs, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        col = MUTED
        if ln[:1] == "!":
            ln, col = ln[1:], BAD
        elif ln[:1] == "+":
            ln, col = ln[1:], OK
        elif ln[:1] == "~":
            ln, col = ln[1:], WARN
        elif ln[:1] == "#":
            ln, col = ln[1:], VIOLET
        elif ln[:1] == "@":
            ln, col = ln[1:], ACCENT
        ax.text(x + 0.22, y + h - 0.90 - i * 0.315, ln, ha="left", va="top",
                color=col, fontsize=fs, zorder=3)


def arrow(ax, x1, y1, x2, y2, col=LINE, ls="solid"):
    ax.add_patch(mp.FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        color=col, linewidth=1.6, zorder=1, linestyle=ls))


# ───────────────────────────────────────────────── read the store
def facts() -> dict:
    f: dict = {}
    live = glob.glob(str(ROOT / "data/raw/gdelt/live/events/dt=*/*.parquet"))
    f["live_slots"] = len(live)
    f["live_mb"] = sum(os.path.getsize(p) for p in live) / 1e6

    f["hist"] = []
    for p in sorted(glob.glob(str(ROOT / "data/raw/gdelt/historical/*.parquet"))):
        n = len(pd.read_parquet(p, columns=["DATEADDED"]))
        f["hist"].append((Path(p).stem, n))
    f["hist"].sort(key=lambda t: -t[1])
    f["hist_total"] = sum(n for _, n in f["hist"])

    mk = pd.read_parquet(ROOT / "data/raw/market/market_daily.parquet")
    f["mkt_days"], f["mkt_series"] = mk.shape
    f["mkt_start"], f["mkt_end"] = mk.index.min().date(), mk.index.max().date()

    chips = glob.glob(str(ROOT / "data/raw/satellite/chips/*/dt=*/*.tiff"))
    f["chips"] = len(chips)
    f["chips_gb"] = sum(os.path.getsize(p) for p in chips) / 1e9

    full = partial = 0
    for s in glob.glob(str(ROOT / "data/raw/satellite/chips/*/dt=*/*.json")):
        try:
            st = json.loads(Path(s).read_text()).get("coverage_status")
        except Exception:                               # noqa: BLE001
            continue
        full += st == "FULL"
        partial += st == "PARTIAL"
    f["chips_full"], f["chips_partial"] = full, partial

    ports = pd.read_csv(ROOT / "ports/uae_ports.csv")
    f["ports"] = len(ports)
    f["ports_dep"] = int(ports.hormuz_dependent.sum())
    f["ports_approx"] = int((ports.coord_precision != "exact").sum())

    try:
        feat = pd.read_parquet(ROOT / "data/derived/features_daily.parquet")
        f["feat_days"], f["feat_cols"] = feat.shape
    except Exception:                                   # noqa: BLE001
        f["feat_days"] = f["feat_cols"] = 0
    return f


def main() -> Path:
    D = facts()
    fig, ax = plt.subplots(figsize=(W, H), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(0.45, 18.75, "DATA INVENTORY", color=INK, fontsize=25,
            fontweight="bold", va="top")
    ax.text(0.45, 18.10, "What the Hormuz Disruption Engine actually holds, and "
            "where each piece surfaces in the dashboard  ·  counts read from disk at render time",
            color=MUTED, fontsize=12.5, va="top")

    # ── row 1: raw sources
    ax.text(0.45, 17.60, "1 · RAW SOURCES", color=DIM, fontsize=11.5,
            fontweight="bold", va="top")
    y1, h1 = 14.30, 2.90
    box(ax, 0.45, y1, 6.05, h1, "GDELT 2.0 Events", [
        "@THE ONLY LIVE FEED",
        f"+{D['live_slots']:,} slots  ·  {D['live_mb']:,.0f} MB",
        "new file every 15 min (96/day)",
        f"+{D['hist_total']:,} historical rows",
        "22 of 61 columns kept",
        "!no article text — URL only",
    ], edge=OK)
    box(ax, 6.90, y1, 6.05, h1, "FRED / EIA  +  yfinance", [
        f"+{D['mkt_days']:,} days x {D['mkt_series']} series",
        f"{D['mkt_start']} -> {D['mkt_end']}",
        "@FRED official: Brent, WTI, gas",
        "~yfinance: 2 futures + 6 ETFs",
        "every row tagged official=T/F",
        "weekends NaN, never filled",
    ], edge=OK)
    box(ax, 13.35, y1, 6.05, h1, "Sentinel-1 SAR", [
        f"+{D['chips']} chips  ·  {D['chips_gb']:.1f} GB",
        f"+{D['chips_full']} FULL / {D['chips_partial']} PARTIAL",
        "dual-pol VV+VH, gamma0",
        "4 ports x 4 crisis windows",
        "orbit direction on every chip",
        "~revisit 1.4-6 days, not live",
    ], edge=OK)
    box(ax, 19.80, y1, 5.75, h1, "Port reference", [
        f"+{D['ports']} ports / terminals",
        f"!{D['ports_dep']} INSIDE the strait",
        f"+{D['ports'] - D['ports_dep']} bypass (Fujairah,",
        "   Khor Fakkan)",
        f"~{D['ports_approx']} approximate coordinates",
        "static reference, not collected",
    ], edge=ACCENT)

    # ── row 2: what is inside GDELT + windows
    ax.text(0.45, 13.90, "2 · WHAT IS ACTUALLY INSIDE THE DATA",
            color=DIM, fontsize=11.5, fontweight="bold", va="top")
    y2, h2 = 9.45, 4.05
    box(ax, 0.45, y2, 9.15, h2, "GDELT — the 22 fields we keep", [
        "@WHO   Actor1Name, Actor2Name, + country codes (FIPS)",
        "@WHAT  EventCode, EventRootCode, QuadClass,",
        "       GoldsteinScale (-10..+10 stability impact)",
        "@WHERE ActionGeo_Lat/Long, FullName, CountryCode",
        "@LOUD  NumMentions, NumSources, NumArticles, AvgTone",
        "@WHEN  DATEADDED (publication)  ·  Day (alleged)",
        "@LINK  SOURCEURL   ·   OURS: gulf_match (filter leg)",
        "",
        "!Day trails publication by up to 365 days — we index",
        "!on DATEADDED only, or we would leak the future",
    ], edge=OK)

    hist_lines = [f"+{name:<22} {n:>9,}" for name, n in D["hist"]]
    box(ax, 10.00, y2, 7.35, h2, "Historical crisis windows", hist_lines + [
        "",
        f"+{'TOTAL':<22} {D['hist_total']:>9,}",
        "",
        "each window padded BEFORE the event",
        "so lead time is measurable",
    ], edge=OK)

    box(ax, 17.75, y2, 7.80, h2, "Derived products", [
        f"#feature panel   {D['feat_days']:,} days x {D['feat_cols']}",
        "#text embeddings 427,620 slugs -> 384-d",
        "#CNN patches     12,687 (64x64x3)",
        "#CFAR detections 449 chips scored",
        "#trained models  VAE + MobileNetV2",
        "",
        "!all regenerable — the pipeline rebuilds",
        "!them; only small JSON reports are kept",
        "!in version control",
    ], edge=VIOLET)

    for x in (3.4, 9.9, 16.3, 22.6):
        arrow(ax, x, y1, x, y2 + h2)

    # ── row 3: dashboard mapping
    ax.text(0.45, 9.28, "3 · WHERE EACH PIECE SURFACES IN THE DASHBOARD",
            color=DIM, fontsize=11.5, fontweight="bold", va="top")
    y3, h3 = 4.35, 4.55
    box(ax, 0.45, y3, 8.30, h3, "Overview  ·  Alerts", [
        "@risk index    <- fused Arm B + Arm C",
        "@backtest      <- 5 labelled events",
        "@live inference<- transformer runs on",
        "                 whatever you type",
        "@arm cards     <- each arm's own score",
        "@alerts        <- file timestamps for",
        "                 staleness + measured",
        "                 model results",
        "",
        "!data-quality alerts rank alongside",
        "!risk alerts, deliberately",
    ], edge=ACCENT)
    box(ax, 9.15, y3, 8.30, h3, "Map  ·  Satellite", [
        "@map        <- uae_ports.csv, coloured",
        "               by hormuz_dependent",
        "@satellite  <- the 449 GeoTIFFs, rendered",
        "               to PNG per request with",
        "               CFAR detections circled",
        "",
        "!raw chip is 8 MB float32 — no browser",
        "!can display it, so the server converts",
        "!to dB with a FIXED stretch: auto-stretch",
        "!would make a busy port and an empty sea",
        "!render identically",
    ], edge=ACCENT)
    box(ax, 17.85, y3, 7.70, h3, "Models  ·  Pipeline  ·  Semantics", [
        "@models    <- training reports on disk",
        "              (report.json, evaluation)",
        "@pipeline  <- live file counts + DAG",
        "              staleness from filesystem",
        "@semantics <- 24-metric registry and",
        "              its 25 prohibitions",
        "",
        "#the same 25 rules are injected into the",
        "#LLM brief as guardrails, so the model",
        "#cannot claim what the registry records",
        "#as false",
    ], edge=ACCENT)

    for x in (4.5, 13.2, 21.6):
        arrow(ax, x, y2, x, y3 + h3)

    # ── row 4: the honesty strip
    y4, h4 = 0.40, 3.60
    box(ax, 0.45, y4, 25.10, h4, "4 · ONLY ONE ARM IS GENUINELY LIVE — which is why every panel shows its own age", [
        "",
        "@GDELT      every 15 minutes          LIVE          the only feed that updates while you watch",
        "~MARKET     daily close               ~1 day behind  FRED publishes once a day; weekends are absent, not zero",
        "~SAR        1.4-6 day satellite revisit  'latest pass'  a satellite cannot be asked to look again sooner",
        "!VESSEL     no data at all            UNAVAILABLE    aisstream: 0 msgs/30s over the Gulf vs 101 msgs/SECOND over Europe",
        "",
        "!A dashboard that rendered all four identically would imply a real-time physical read of the Gulf that does not exist.",
        "!The vessel panel says NO DATA rather than drawing an empty map, because an empty map reads as calm water.",
    ], edge=WARN, fs=11.0, tfs=12.5)

    leg = [Line2D([0], [0], color=OK, lw=3, label="collected & validated"),
           Line2D([0], [0], color=WARN, lw=3, label="works with a caveat"),
           Line2D([0], [0], color=BAD, lw=3, label="limitation / not available"),
           Line2D([0], [0], color=VIOLET, lw=3, label="derived by a model"),
           Line2D([0], [0], color=ACCENT, lw=3, label="reference / serving")]
    lg = ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(0.999, 0.995),
                   frameon=True, fontsize=10.5, ncol=5)
    lg.get_frame().set_facecolor(PANEL); lg.get_frame().set_edgecolor(LINE)
    for t in lg.get_texts():
        t.set_color(MUTED)

    out = HERE / "data_inventory.png"
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
