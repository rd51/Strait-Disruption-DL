"""
Live model inference — the backend actually RUNS the trained models.

Until this module existed the API served precomputed parquet and JSON reports,
with the trained `.keras` files sitting on disk unopened. That made it a data
service with model artifacts beside it, not a model service. Here the models
are loaded into memory and executed per request.

LAZY, CACHED LOADING. Nothing loads at import. A TensorFlow model plus a
384-dimension sentence transformer take tens of seconds to initialise and
several hundred MB of RAM; paying that at startup would make the server slow to
boot and would crash it outright when a model is missing. Each loader is called
on first use and memoised, so the first request to an endpoint is slow and every
one after it is fast.

WHAT EACH ENDPOINT ACTUALLY COMPUTES

  /predict/text     encodes arbitrary text with the multilingual encoder and
                    scores it contrastively against the disruption and calm
                    reference sentences. Genuinely zero-shot — any headline in
                    any supported language, including ones the project has
                    never seen.

  /predict/market   builds the trailing 20-trading-day window ending on a given
                    date, standardises it with the TRAINING scaler, runs the
                    VAE, and returns reconstruction error against the trained
                    thresholds.

  /predict/live     both arms on the most recent available data.

THE SCALER MUST COME FROM TRAINING, NOT FROM THE REQUEST. Standardising an
inference window by its own mean and standard deviation would rescale a crisis
to look normal — the very deviation the model is meant to find surprising would
be normalised away. `scaler.npz` holds the training statistics and is the only
correct source.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from ..common.paths import repo_root

log = logging.getLogger(__name__)

WINDOW = 20
VAE_FEATURES = [
    "brent_ret_1d", "brent_ret_7d", "brent_vol_7d", "brent_wti_spread",
    "spread_chg_7d", "spread_z", "spread_smooth_10d",
    "spread_days_gt15_30d", "gas",
]


# ────────────────────────────────────────────────────── lazy model loaders
@lru_cache(maxsize=1)
def load_text_model():
    """Multilingual sentence encoder + precomputed anchor vectors."""
    from sentence_transformers import SentenceTransformer
    from ..arms.text.embed import MODEL_NAME, DISRUPTION_ANCHORS, CALM_ANCHORS

    log.info("loading sentence encoder %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    # Anchors never change, so encode them once at load rather than per request.
    dis = model.encode(DISRUPTION_ANCHORS, normalize_embeddings=True,
                       convert_to_numpy=True)
    calm = model.encode(CALM_ANCHORS, normalize_embeddings=True,
                        convert_to_numpy=True)
    return model, dis, calm


@lru_cache(maxsize=1)
def load_vae():
    """Encoder + decoder + the TRAINING scaler and thresholds."""
    import json
    import keras

    d = repo_root() / "data" / "models" / "arm_b_vae"
    if not (d / "encoder.keras").exists():
        raise FileNotFoundError(
            "Arm B VAE not trained — run `python -m backend.arms.market.vae`")
    log.info("loading Arm B VAE from %s", d)
    enc = keras.models.load_model(d / "encoder.keras")
    dec = keras.models.load_model(d / "decoder.keras")
    sc = np.load(d / "scaler.npz")
    report = json.loads((d / "report.json").read_text())
    return enc, dec, sc["mu"], sc["sd"], report


@lru_cache(maxsize=1)
def feature_panel() -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "derived" / "features_daily.parquet")


# ─────────────────────────────────────────────────────────────── scoring
def score_text(texts: list[str]) -> list[dict]:
    """
    Contrastive chokepoint score for arbitrary text.

    All vectors are unit-norm, so a dot product IS the cosine similarity.
    The score subtracts similarity to ordinary maritime/energy language so it
    measures ESCALATION rather than "this sentence is about shipping".
    """
    model, dis, calm = load_text_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    sim_d = (vecs @ dis.T).max(axis=1)
    sim_c = (vecs @ calm.T).max(axis=1)
    score = sim_d - sim_c

    # Calibration from the historical slug corpus, so a raw cosine difference
    # can be reported as "how unusual is this". Falls back gracefully.
    try:
        ref = pd.read_parquet(
            repo_root() / "data" / "derived" / "text" / "slug_features_daily.parquet"
        )["slug_chokepoint_top10"]
        p90, p99 = float(ref.quantile(0.90)), float(ref.quantile(0.99))
    except Exception:                                   # noqa: BLE001
        p90 = p99 = float("nan")

    out = []
    for t, s, sd_, sc_ in zip(texts, score, sim_d, sim_c):
        if np.isnan(p90):
            level = "uncalibrated"
        elif s >= p99:
            level = "HIGH"
        elif s >= p90:
            level = "ELEVATED"
        else:
            level = "normal"
        out.append({
            "text": t,
            "chokepoint_score": round(float(s), 4),
            "sim_disruption": round(float(sd_), 4),
            "sim_calm": round(float(sc_), 4),
            "level": level,
            "reference_p90": round(p90, 4) if not np.isnan(p90) else None,
            "reference_p99": round(p99, 4) if not np.isnan(p99) else None,
        })
    return out


def score_market(as_of: str | None = None) -> dict:
    """
    Run the VAE on the trailing window ending at `as_of`.

    The window is built from days at or BEFORE `as_of`, so the score is causal:
    it uses nothing the market had not yet printed.
    """
    enc, dec, mu, sd, report = load_vae()
    df = feature_panel()[VAE_FEATURES].dropna()
    if as_of:
        df = df[df.index <= pd.Timestamp(as_of)]
    if len(df) < WINDOW:
        raise ValueError(
            f"need {WINDOW} complete trading days, have {len(df)}")

    win = df.iloc[-WINDOW:]
    x = ((win.to_numpy(dtype="float32") - mu) / sd)[None, ...]

    # Posterior MEAN, never a sample: a sampled latent makes the same window
    # score differently on each call, which is indefensible in a system that
    # reports a risk number.
    z_mean, z_log_var = enc.predict(x, verbose=0)
    recon = dec.predict(z_mean, verbose=0)
    err = float(np.mean((x - recon) ** 2))

    p95, p99 = report["threshold_p95"], report["threshold_p99"]
    level = "HIGH" if err > p99 else "ELEVATED" if err > p95 else "normal"
    return {
        "as_of": str(win.index[-1].date()),
        "window_start": str(win.index[0].date()),
        "window_days": WINDOW,
        "reconstruction_error": round(err, 4),
        "threshold_p95": round(p95, 4),
        "threshold_p99": round(p99, 4),
        "level": level,
        "latent_dim": int(z_mean.shape[-1]),
        "latent_mean_norm": round(float(np.linalg.norm(z_mean)), 4),
        "caveat": (
            "Trained only on non-event windows. Detects 1 of 5 labelled "
            "anchors; fires on COVID and the 2022 Ukraine invasion too, so it "
            "measures seaborne/market dislocation, not Gulf geography."
        ),
    }


def warm() -> dict:
    """Force both models into memory so the first user request is not slow."""
    status = {}
    for name, fn in (("text", load_text_model), ("vae", load_vae)):
        try:
            fn()
            status[name] = "loaded"
        except Exception as exc:                        # noqa: BLE001
            status[name] = f"unavailable: {exc}"
    return status
