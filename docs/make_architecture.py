"""
Render the project architecture diagram to PNG.

Drawn with matplotlib primitives rather than a diagramming tool so the figure
is reproducible from source and can be regenerated when the architecture
changes. Every box carries the DEEP LEARNING CONCEPT it embodies, because the
diagram doubles as the concept map for the writeup.

Honest labelling is enforced here too: Arm A is drawn in the "does not earn its
place" colour, and unbuilt components are drawn dashed. A diagram that renders
a failed component identically to a working one is a diagram that lies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BG      = "#0b0f14"
PANEL   = "#121821"
LINE    = "#2a3644"
INK     = "#e6edf3"
MUTED   = "#8b9aab"
DIM     = "#6b7a8a"
OK      = "#3fb950"
WARN    = "#d29922"
BAD     = "#f85149"
ACCENT  = "#58a6ff"
VIOLET  = "#bc8cff"

W, H = 26.0, 17.2


def box(ax, x, y, w, h, title, lines=(), edge=LINE, face=PANEL,
        tcol=INK, fs=10.5, tfs=12, dashed=False, lw=1.6):
    ax.add_patch(mp.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.055,rounding_size=0.16",
        linewidth=lw, edgecolor=edge, facecolor=face,
        linestyle=(0, (5, 3)) if dashed else "solid", zorder=2))
    ax.text(x + w / 2, y + h - 0.36, title, ha="center", va="top",
            color=tcol, fontsize=tfs, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        col = MUTED
        if ln.startswith("!"):
            ln, col = ln[1:], BAD
        elif ln.startswith("+"):
            ln, col = ln[1:], OK
        elif ln.startswith("~"):
            ln, col = ln[1:], WARN
        elif ln.startswith("#"):
            ln, col = ln[1:], VIOLET
        ax.text(x + 0.22, y + h - 0.92 - i * 0.325, ln, ha="left", va="top",
                color=col, fontsize=fs, zorder=3, family="DejaVu Sans")


def arrow(ax, x1, y1, x2, y2, col=ACCENT, style="-|>", lw=1.7, rad=0.0, ls="solid"):
    ax.add_patch(mp.FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
        color=col, linewidth=lw, zorder=1, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}"))


def main() -> Path:
    fig, ax = plt.subplots(figsize=(W, H), dpi=170)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(0.45, 16.60, "HORMUZ DISRUPTION ENGINE",
            color=INK, fontsize=25, fontweight="bold", va="top")
    ax.text(0.45, 15.95,
            "Chokepoint early-warning  ·  three deep-learning arms  ·  "
            "every component labelled with the DL concept it embodies",
            color=MUTED, fontsize=12.5, va="top")

    # ─────────────────────────────── row 1: sources
    ax.text(0.45, 15.45, "1 · DATA SOURCES", color=DIM, fontsize=11.5,
            fontweight="bold", va="top")
    y1, h1 = 12.70, 2.35
    box(ax, 0.45, y1, 5.7, h1, "GDELT 2.0 Events", [
        "15-min slots · 96/day · free",
        "61 columns · FIPS codes",
        "+45,356 slots · 737 MB live",
        "+935k historical rows",
    ], edge=OK)
    box(ax, 6.55, y1, 5.7, h1, "FRED / EIA market", [
        "Brent · WTI · Henry Hub",
        "official, citable primary",
        "+2,204 days × 11 series",
        "weekends NaN, never filled",
    ], edge=OK)
    box(ax, 12.65, y1, 5.7, h1, "Sentinel-1 SAR", [
        "IW GRDH dual-pol VV+VH",
        "1.4–6 day revisit",
        "+449 chips · 5,947 PU",
        "orbit direction recorded",
    ], edge=OK)
    box(ax, 18.75, y1, 6.8, h1, "AIS (vessel layer)", [
        "!aisstream: 0 msgs/30s in Gulf",
        "!vs 101 msgs/s in Europe",
        "!NO free live Gulf source",
        "~GFW covers Gulf, 4-day lag",
    ], edge=BAD, dashed=True)

    # ─────────────────────────────── row 2: arms
    ax.text(0.45, 12.30, "2 · THE THREE ARMS  —  each a different modality, "
            "each earning its place or declared not to",
            color=DIM, fontsize=11.5, fontweight="bold", va="top")
    y2, h2 = 7.35, 4.55

    box(ax, 0.45, y2, 8.05, h2, "ARM A  ·  SAR vision", [
        "#CNN · TRANSFER LEARNING",
        "#depthwise-separable conv",
        "#weak supervision · group split",
        "",
        "CA-CFAR + Otsu water mask",
        "MobileNetV2 (ImageNet) → 3-class",
        "12,687 patches / 371 acquisitions",
        "",
        "!CNN 0.837  vs  baseline 0.946",
        "!NO reroute signal: +0.0%",
        "!zero fusion weight",
    ], edge=BAD)

    box(ax, 8.9, y2, 8.05, h2, "ARM B  ·  market VAE", [
        "#VAE · LSTM · ELBO + KL",
        "#reparameterisation trick",
        "#β-annealing · posterior collapse",
        "",
        "LSTM(48) → μ,logσ² ∈ ℝ⁸ → LSTM(48)",
        "trained ONLY on non-event windows",
        "score = reconstruction error",
        "",
        "+AUC 0.680 · 38-day lead",
        "~detects 1 of 5 anchors",
        "~fires on COVID & Ukraine too",
    ], edge=WARN)

    box(ax, 17.35, y2, 8.2, h2, "ARM C  ·  semantic text", [
        "#TRANSFORMER · zero-shot",
        "#cross-lingual embeddings",
        "#contrastive scoring · distillation",
        "",
        "MiniLM-L12 → 384-d unit vectors",
        "cos(v, disruption) − cos(v, calm)",
        "427,620 slugs · no labels consumed",
        "",
        "+AUC 0.848 — best arm",
        "+EN↔AR cosine 0.866 vs 0.073",
        "~2-day lead on 2026 only",
    ], edge=OK)

    for x in (4.5, 12.9, 21.4):
        arrow(ax, x, y1, x, y2 + h2, col=LINE)
    arrow(ax, 22.1, y1, 22.1, y2 + h2, col=BAD, ls=(0, (4, 3)))

    # ─────────────────────────────── row 3: features + fusion
    ax.text(0.45, 6.95, "3 · FEATURE LAYER, FUSION AND VALIDATION",
            color=DIM, fontsize=11.5, fontweight="bold", va="top")
    y3, h3 = 3.25, 3.30

    box(ax, 0.45, y3, 7.3, h3, "Feature panel + semantic layer", [
        "3,129 days × 24 features",
        "indexed on DATEADDED, never Day",
        "",
        "#leakage gate: no derived feature",
        "#may outlive or out-count its source",
        "+24 metrics · 25 prohibitions",
        "+100% of panel defined",
    ], edge=ACCENT)

    box(ax, 8.15, y3, 8.05, h3, "FUSION  ·  weighted rule", [
        "#causal expanding percentile",
        "#(a full-sample rank leaks the future)",
        "",
        "w_B 0.50 · w_C 0.50 · w_A 0.00",
        "NOT fitted — 5 labels cannot",
        "validate a trained combiner",
        "~GBM runs alongside: degenerate",
    ], edge=VIOLET)

    box(ax, 16.6, y3, 8.95, h3, "BACKTEST  ·  causal LOEO", [
        "#leave-one-event-out · embargo 10d",
        "#one fold per event by construction",
        "",
        "+2 of 3 scoreable · median 36d lead",
        "+Abqaiq 30d · Hormuz 43d",
        "~Red Sea missed · 2 unscoreable",
        "+fused 43d > B alone 38d > C 2d",
    ], edge=OK)

    for x in (4.1, 12.2, 21.4):
        arrow(ax, x, y2, x, y3 + h3, col=LINE)
    arrow(ax, 7.75, y3 + h3 / 2, 8.15, y3 + h3 / 2, col=ACCENT)
    arrow(ax, 16.2, y3 + h3 / 2, 16.6, y3 + h3 / 2, col=ACCENT)

    # ─────────────────────────────── row 4: serving
    ax.text(0.45, 2.85, "4 · SERVING  —  the backend RUNS the models, "
            "it does not serve stored numbers",
            color=DIM, fontsize=11.5, fontweight="bold", va="top")
    y4, h4 = 0.30, 2.15

    box(ax, 0.45, y4, 6.0, h4, "FastAPI + DuckDB", [
        "24 endpoints · OpenAPI",
        "/freshness per-arm staleness",
        "!/vessels → 501 deliberately",
        "no placeholder is ever served",
    ], edge=ACCENT)
    box(ax, 6.85, y4, 6.0, h4, "LIVE INFERENCE", [
        "#models loaded & cached in RAM",
        "/predict/text  any language",
        "/predict/market  causal 20d window",
        "+2026-02-20 → HIGH, 10d early",
    ], edge=OK)
    box(ax, 13.25, y4, 5.6, h4, "Orchestration", [
        "12-stage DAG · toposorted",
        "staleness from the filesystem",
        "quota stages never auto-run",
        "validation GATE ≠ crash",
    ], edge=ACCENT)
    box(ax, 19.25, y4, 6.3, h4, "Dashboard + agent", [
        "7 tabs · Leaflet · SAR PNG render",
        "#LLM brief w/ 25 prohibitions",
        "#injected as guardrails",
        "~Kafka: 0.0001% utilisation",
    ], edge=ACCENT)

    for x in (3.4, 12.2, 21.4):
        arrow(ax, x, y3, x, y4 + h4, col=LINE)

    # legend
    leg = [
        Line2D([0], [0], color=OK, lw=3, label="works / validated"),
        Line2D([0], [0], color=WARN, lw=3, label="works with a material caveat"),
        Line2D([0], [0], color=BAD, lw=3, label="measured NULL or not achievable"),
        Line2D([0], [0], color=VIOLET, lw=3, label="deep-learning concept"),
    ]
    lg = ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(0.999, 0.995),
                   frameon=True, fontsize=10.5, ncol=4)
    lg.get_frame().set_facecolor(PANEL)
    lg.get_frame().set_edgecolor(LINE)
    for t in lg.get_texts():
        t.set_color(MUTED)

    out = Path(__file__).resolve().parent / "architecture.png"
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = main()
    print("wrote", p, f"({p.stat().st_size/1024:.0f} KB)")
