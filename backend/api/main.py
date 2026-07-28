"""
FastAPI backend for the Hormuz Disruption Engine.

DESIGN COMMITMENTS, all of which are refusals as much as features:

1. **/risk returns 501, not a number.** The GBM fusion does not exist yet.
   Serving a placeholder index — even one labelled "demo" — is how a prototype
   ends up with a headline figure nobody can trace. The endpoint exists so the
   contract is visible, and it fails loudly until there is something real
   behind it.

2. **/freshness is a primary endpoint.** Only GDELT is genuinely live. Market
   is daily, SAR is "latest pass" at 1.4-6 day revisit, and the vessel layer
   has no Gulf data at all. A UI that renders these identically implies a
   real-time read of the Gulf that does not exist.

3. **Polling, not WebSockets.** The fastest arm updates every 15 minutes. A
   30-second poll is already 30x faster than the data changes; a socket would
   be complexity serving nothing.

4. **Provenance travels with the data.** /sources reports which series are
   official (FRED, GDELT, Copernicus) and which are not (yfinance), so a claim
   can be traced to a citeable source rather than assumed.

RUN
    uvicorn api.main:app --reload --port 8000
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from . import store
from ..semantic import registry as semantic

app = FastAPI(
    title="Hormuz Disruption Engine API",
    version="0.1.0",
    description=(
        "Chokepoint early-warning backend. Serves per-arm data with explicit "
        "freshness. The fused risk index is NOT implemented — /risk returns 501 "
        "rather than a placeholder."
    ),
)

# Local dashboard development only. Tighten before anything is exposed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def dashboard():
    """
    The dashboard, served by the API that feeds it.

    Single self-contained HTML file: no CDN, no build step, no npm. Eight
    panels consuming a REST API do not need a toolchain — the same reasoning
    applied to the Kafka layer applies here.
    """
    from ..common.paths import repo_root
    f = repo_root() / "backend" / "api" / "static" / "dashboard.html"
    if not f.exists():
        raise HTTPException(404, "dashboard.html missing")
    return FileResponse(f, media_type="text/html")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "utc": datetime.now(timezone.utc).isoformat()}


@app.get("/freshness", tags=["meta"])
def freshness():
    """
    Per-arm as-of timestamps and staleness.

    The dashboard must render this next to every panel. "Congestion: 4 days
    old" is an honest statement; a bare number implies currency the data does
    not have.
    """
    return store.freshness()


@app.get("/sources", tags=["meta"])
def sources():
    """Provenance per series — official vs unofficial, with URLs."""
    return store.sources()


@app.get("/semantic/metrics", tags=["semantic"])
def semantic_metrics(arm: str | None = Query(None, description="filter: A_sar|B_market|C_gdelt|label")):
    """
    The semantic layer — what every number in this system actually means.

    A metric without a definition is a number nobody can defend. This endpoint
    is what lets the frontend render a caveat beside a value instead of a bare
    figure, and what lets a model-training script refuse an undefined column.
    """
    metrics = [m.to_dict() for m in semantic.REGISTRY.values()
               if arm is None or m.arm == arm]
    return {"count": len(metrics), "metrics": metrics}


@app.get("/semantic/metrics/{name}", tags=["semantic"])
def semantic_metric(name: str):
    m = semantic.get(name)
    if m is None:
        raise HTTPException(404, f"'{name}' is not a defined metric")
    return m.to_dict()


@app.get("/semantic/forbidden", tags=["semantic"])
def semantic_forbidden():
    """
    Every prohibited use, in one place.

    These are not generic warnings — each was learned by getting it wrong and
    measuring the consequence. Encoded here so the lesson survives the person.
    """
    return {"rules": semantic.forbidden_uses()}


@app.get("/semantic/audit", tags=["semantic"])
def semantic_audit():
    """Which columns of the live feature panel are semantically defined?"""
    df = store.features(None, None, None)
    if df.empty:
        raise HTTPException(404, "no feature panel to audit")
    cols = [c for c in df.columns if c != "date"]
    return semantic.audit_frame(cols)


@app.get("/features", tags=["data"])
def features(
    start: str | None = Query(None, description="YYYY-MM-DD inclusive"),
    end: str | None = Query(None, description="YYYY-MM-DD inclusive"),
    columns: str | None = Query(None, description="comma-separated feature names"),
):
    """The common daily feature panel across all arms."""
    cols = [c.strip() for c in columns.split(",")] if columns else None
    df = store.features(start, end, cols)
    if df.empty:
        raise HTTPException(404, "no feature panel — run `python -m features.build`")
    return {
        "rows": len(df),
        "columns": [c for c in df.columns if c != "date"],
        "data": store.jsonable(df),
    }


@app.get("/arms/gdelt", tags=["arms"])
def arm_gdelt(start: str | None = None, end: str | None = None):
    """
    Arm C — daily GDELT aggregates from the LIVE store, keyed on DATEADDED.

    Article counts are de-duplicated on SOURCEURL: GDELT emits one row per
    actor pair per article, so raw rows overcount by 60-75% and measure
    re-reporting rather than escalation.
    """
    df = store.gdelt_series(start, end)
    if df.empty:
        raise HTTPException(404, "no GDELT data in that range")
    return {"rows": len(df), "keyed_on": "DATEADDED (publication time)",
            "note": "defaults to the last 30 days; the slot store holds 35k+ files",
            "data": store.jsonable(df)}


@app.get("/arms/market", tags=["arms"])
def arm_market(start: str | None = None, end: str | None = None):
    """
    Arm B — oil and equity series.

    The Brent-WTI spread is the Gulf-specific discriminator: WTI is landlocked
    US crude, so a widening spread prices chokepoint risk while a parallel move
    in both is a global oil story.
    """
    df = store.market_series(start, end)
    if df.empty:
        raise HTTPException(404, "no market data — run `python -m ingest.market.collect`")
    return {"rows": len(df), "data": store.jsonable(df)}


@app.get("/arms/sar", tags=["arms"])
def arm_sar():
    """
    Arm A — SAR port congestion from the CFAR baseline.

    `vessels_on_water` is the signal. Land detections are excluded because they
    are the same cranes and quays every pass (measured near-constant at
    39/37/38 on Fujairah) and would add a large fixed offset that swamps the
    vessel variation. Partial chips are returned but flagged `usable: false`.
    """
    dets = store.sar_detections()
    if not dets:
        raise HTTPException(404, "no SAR detections — run `python -m arms.sar.run_cfar`")
    return {"count": len(dets), "usable": sum(d["usable"] for d in dets),
            "method": "CA-CFAR baseline (not a trained model)", "data": dets}


@app.get("/arms/market/anomaly", tags=["arms"])
def arm_market_anomaly(start: str | None = None, end: str | None = None):
    """
    Arm B VAE reconstruction error — the market-regime anomaly score.

    Returned WITH its evaluation, deliberately. The score is only interpretable
    beside the fact that it detects 1 of 5 labelled anchors; serving the series
    alone would invite a lead-time claim the evidence does not support.
    """
    import json as _json
    from ..common.paths import repo_root

    d = repo_root() / "data" / "models" / "arm_b_vae"
    if not (d / "scores.parquet").exists():
        raise HTTPException(
            503, "Arm B VAE not trained — run `python -m backend.arms.market.vae`")

    df = pd.read_parquet(d / "scores.parquet")
    df["date"] = pd.to_datetime(df["date"])
    if start:
        df = df[df.date >= pd.Timestamp(start)]
    if end:
        df = df[df.date <= pd.Timestamp(end)]

    ev_path = d / "evaluation.json"
    return {
        "rows": len(df),
        "model": _json.loads((d / "report.json").read_text()),
        "evaluation": _json.loads(ev_path.read_text()) if ev_path.exists() else None,
        "caveat": (
            "Detects 1 of 5 labelled anchors (2026 Hormuz, 38-day lead). The "
            "other four did not move the market enough to be anomalous. Fires "
            "on COVID and the 2022 Ukraine invasion too — this arm detects "
            "seaborne/market dislocation, not Gulf geography."
        ),
        "data": store.jsonable(df),
    }


class TextRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64,
                             description="headlines or sentences, any language")


@app.post("/predict/text", tags=["inference"])
def predict_text(req: TextRequest):
    """
    LIVE INFERENCE — run the multilingual encoder on arbitrary text.

    Zero-shot: any headline in any supported language, including ones this
    project has never collected. The score is contrastive (similarity to
    disruption references MINUS similarity to ordinary maritime language), so
    it measures escalation rather than "this text is about shipping".
    """
    from . import inference
    try:
        return {"results": inference.score_text(req.texts)}
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(503, f"text model unavailable: {exc}")


@app.get("/predict/market", tags=["inference"])
def predict_market(as_of: str | None = Query(None, description="YYYY-MM-DD")):
    """
    LIVE INFERENCE — run the Arm B VAE on the trailing 20-trading-day window.

    Causal by construction: the window ends at `as_of` and uses nothing the
    market had not yet printed.
    """
    from . import inference
    try:
        return inference.score_market(as_of)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/predict/live", tags=["inference"])
def predict_live():
    """Both model arms on the most recent available data."""
    from . import inference
    out: dict = {"generated_utc": datetime.now(timezone.utc).isoformat()}
    try:
        out["market"] = inference.score_market(None)
    except Exception as exc:                            # noqa: BLE001
        out["market"] = {"error": str(exc)}

    # Score the most recent real GDELT headlines rather than a canned string,
    # so this endpoint reflects live input end to end.
    try:
        d = store.gdelt_recent_titles(limit=25)
        out["text"] = {"n_scored": len(d),
                       "top": sorted(inference.score_text(d),
                                     key=lambda r: -r["chokepoint_score"])[:5]} \
            if d else {"error": "no recent text available"}
    except Exception as exc:                            # noqa: BLE001
        out["text"] = {"error": str(exc)}
    out["reminder"] = (
        "Two independent arms, deliberately NOT combined into a single risk "
        "index — fusion is not built and /risk returns 501 until it is."
    )
    return out


@app.post("/predict/warm", tags=["inference"])
def predict_warm():
    """Force models into memory so the first user request is not slow."""
    from . import inference
    return inference.warm()


@app.get("/models", tags=["engine"])
def models():
    """
    Every trained model with its evaluation AND its honest caveat.

    Deliberately returns the caveat inline rather than in separate docs. A
    frontend that renders an accuracy without the condition attached will be
    read as a claim the evidence does not support — the Arm B lead time is one
    event, and the Arm A CNN loses to a brightness baseline.
    """
    import json as _json
    from ..common.paths import repo_root

    root = repo_root() / "data" / "models"
    specs = [
        ("arm_b_vae", "Arm B — market-regime sequence VAE",
         "Detects 1 of 5 labelled anchors. The 38-day lead is ONE event. "
         "Also fires on COVID and the 2022 Ukraine invasion, so it detects "
         "seaborne/market dislocation, not Gulf geography."),
        ("arm_a_cnn", "Arm A — SAR patch CNN (MobileNetV2 transfer)",
         "LOSES to a random forest on brightness statistics (0.837 vs 0.946). "
         "At ~15.6 m/px a vessel is 1-3 px, so there is little spatial "
         "structure for convolution to exploit. Not a validated detector. "
         "Vessel labels come from CFAR, so agreement there is circular; only "
         "the infrastructure and sea_clutter classes carry reliable labels."),
    ]
    out = []
    for name, title, caveat in specs:
        d = root / name
        if not (d / "report.json").exists():
            out.append({"name": name, "title": title, "status": "not trained"})
            continue
        entry = {"name": name, "title": title, "status": "trained",
                 "caveat": caveat, "report": _json.loads((d / "report.json").read_text())}
        ev = d / "evaluation.json"
        if ev.exists():
            entry["evaluation"] = _json.loads(ev.read_text())
        out.append(entry)
    return {"models": out}


@app.get("/ports", tags=["geo"])
def ports():
    """
    The 13 UAE ports/terminals, with the coastline split that IS the thesis.

    11 sit INSIDE the strait and are Hormuz-dependent; Fujairah and Khor Fakkan
    are on the Gulf of Oman coast and bypass it. If traffic reroutes under
    stress it should appear at the bypass pair — a claim Arm A measured and
    did NOT support.
    """
    from . import imagery
    rows = imagery.ports()
    if not rows:
        raise HTTPException(404, "port reference data not found")
    return {
        "count": len(rows),
        "hormuz_dependent": sum(r["hormuz_dependent"] for r in rows),
        "bypass": [r["name"] for r in rows if not r["hormuz_dependent"]],
        "caveat": ("4 coordinates are APPROXIMATE (Hamriyah and the three "
                   "offshore terminals) — flagged per row in coord_precision."),
        "ports": rows,
    }


@app.get("/chips", tags=["geo"])
def chips(port: str | None = None, limit: int = Query(60, le=500)):
    """Sentinel-1 chips on disk, with coverage class and orbit direction."""
    from . import imagery
    rows = imagery.chip_catalogue()
    if port:
        rows = [r for r in rows if r["port"] == port]
    return {
        "count": len(rows),
        "note": ("PARTIAL chips are listed but must not be compared against "
                 "FULL ones without normalising — a swath-clipped chip reports "
                 "fewer vessels purely because of the satellite footprint."),
        "chips": rows[:limit],
    }


@app.get("/chips/{port}/{date}.png", tags=["geo"])
def chip_png(port: str, date: str, overlay: bool = True):
    """
    Render one SAR chip as a viewable PNG, CFAR detections circled.

    The raw chip is a 1024x1024 float32 GeoTIFF (~8 MB) that no browser can
    display; this converts to dB with a FIXED stretch so a busy anchorage and
    an empty sea do not render identically.
    """
    from . import imagery
    try:
        png = imagery.render_chip(port, date, overlay=overlay)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/pipeline", tags=["meta"])
def pipeline():
    """
    Orchestration DAG state, derived from the filesystem.

    Staleness is a fact about files (input newer than output), not bookkeeping
    in a scheduler database that can drift from reality. Quota-spending stages
    are flagged `manual` and never run automatically.
    """
    from ..orchestration.dag import graph
    return graph()


@app.get("/pipeline/volumes", tags=["meta"])
def pipeline_volumes():
    """Measured data volumes — what the system actually ingests and holds."""
    from ..common.paths import repo_root
    from ..streaming.pipeline import VOLUME_FACTS
    import glob as _glob

    root = repo_root() / "data"
    def _n(pat):
        return len(_glob.glob(str(root / pat), recursive=True))
    def _mb(pat):
        return round(sum(__import__("os").path.getsize(f)
                         for f in _glob.glob(str(root / pat), recursive=True)) / 1e6, 1)
    return {
        "ingest_rate": VOLUME_FACTS,
        "gdelt_live_slots": _n("raw/gdelt/live/events/dt=*/*.parquet"),
        "gdelt_live_mb": _mb("raw/gdelt/live/events/dt=*/*.parquet"),
        "gdelt_historical_windows": _n("raw/gdelt/historical/*.parquet"),
        "sar_chips": _n("raw/satellite/chips/*/dt=*/*.tiff"),
        "sar_chips_mb": _mb("raw/satellite/chips/*/dt=*/*.tiff"),
        "cadence": {
            "gdelt": "15 minutes (96 slots/day)",
            "market": "daily close",
            "sar": "1.4-6 day revisit, per port",
        },
    }


@app.get("/events", tags=["data"])
def events():
    """
    Backtest label calendar.

    The 2026 anchor is dated from PRICE onset (2026-03-02), not the headline
    closure announcement (2026-06-12). The June date sits in the
    de-escalation — Brent fell 24% across it — so labelling there would score
    an early-warning system on an event that had already ended.
    """
    from ..features.splits import EVENTS
    return {
        "count": len(EVENTS),
        "caveat": ("~5 labels total. This cannot support conventional supervised "
                   "training; the engine is framed as anomaly detection / early "
                   "warning, not a classifier with robust event odds."),
        "events": [{"name": k, "start": v[0], "end": v[1]} for k, v in EVENTS.items()],
    }


@app.get("/risk", tags=["engine"])
def risk(start: str | None = None, end: str | None = None):
    """
    The fused chokepoint risk index (0-100).

    A WEIGHTED RULE, NOT A TRAINED MODEL. With five labelled events a fitted
    combiner cannot be validated, so the weights come from each arm's
    separately measured performance and are not tuned here. The GBM is
    reported alongside as an explicitly illustrative comparison and is never
    the shipped number.

    Backtested leave-one-event-out: 3 of 5 anchors detected, median lead 30
    days. Read the `honest_reading` field before quoting any of it.
    """
    import json as _json
    from ..common.paths import repo_root

    d = repo_root() / "data" / "derived" / "fusion"
    if not (d / "risk_index.parquet").exists():
        raise HTTPException(503, "fusion not built — run `python -m backend.fusion.combine`")

    df = pd.read_parquet(d / "risk_index.parquet").reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    if start:
        df = df[df.date >= pd.Timestamp(start)]
    if end:
        df = df[df.date <= pd.Timestamp(end)]

    report = _json.loads((d / "fusion_report.json").read_text())
    idx = df["risk_index"].dropna()
    return {
        "rows": len(df),
        "latest": round(float(idx.iloc[-1]), 1) if len(idx) else None,
        "weights": report["weights"],
        "weight_justification": report["weight_justification"],
        "backtest_loeo": report["backtest_loeo"],
        "gbm_comparison": report["gbm_comparison"],
        "honest_reading": (
            "CAUSAL leave-one-event-out. 2 of 3 SCOREABLE events detected with "
            "usable lead (Abqaiq 30d, Hormuz 43d, median 36d); Red Sea missed. "
            "The two earliest 2019 events are UNSCOREABLE — under causal rules "
            "they lack enough prior history to have been predictable at the "
            "time, so they are excluded rather than counted as hits or misses. "
            "An earlier version scored all five and claimed '3 of 5', which "
            "credited a 2019 Gulf of Oman detection the system could not have "
            "made in 2019. Arm A carries ZERO weight — no measured signal."
        ),
        "data": store.jsonable(df),
    }


@app.get("/brief", tags=["engine"])
def brief(dry_run: bool = Query(True, description="inspect the grounded prompt without calling the API")):
    """
    LLM analyst brief, grounded in the semantic registry.

    Every number comes from the store; the registry's 25 prohibitions are
    injected as guardrails, so the model cannot claim things this project has
    measured to be false. `dry_run=true` returns the grounded state and prompt
    size without spending an API call.
    """
    from ..agent import brief as agent
    try:
        return agent.generate(dry_run=dry_run)
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(503, f"brief unavailable: {exc}")


@app.get("/streaming/facts", tags=["engine"])
def streaming_facts():
    """
    Why this project does NOT need Kafka, with the measured numbers.

    Served as an endpoint deliberately: the honest framing should travel with
    the component rather than living only in a docstring.
    """
    from ..streaming.pipeline import VOLUME_FACTS
    return VOLUME_FACTS


@app.get("/vessels", tags=["engine"], status_code=501)
def vessels():
    """NOT AVAILABLE — no free source provides live Gulf vessel positions."""
    raise HTTPException(
        status_code=501,
        detail={
            "error": "data_unavailable",
            "message": (
                "aisstream.io has zero AIS coverage in the Gulf. Measured "
                "2026-07-27: Hormuz box 0 messages in 30s, while a "
                "comparably-sized European box returned 101 messages/second on "
                "the same key. Global Fishing Watch covers the Gulf but "
                "publishes 4 days behind."
            ),
            "why_this_matters": (
                "Transponder-dark detection works by ABSENCE — a vessel that had "
                "a track and stopped transmitting. With no tracks at all, every "
                "vessel would read as dark. An empty map here would be "
                "confidently wrong, not merely blank."
            ),
            "alternative": (
                "SAR detects hulls regardless of transponder state. See "
                "/arms/sar. Cross-matching SAR against GFW gives retrospective "
                "dark-vessel detection at a 4-day lag."
            ),
        },
    )
