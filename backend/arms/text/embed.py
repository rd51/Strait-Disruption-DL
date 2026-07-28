"""
Arm C — multilingual sentence embeddings over GDELT headlines.

WHAT THIS BUYS THAT ARM C'S EVENT CODES DO NOT. GDELT's CAMEO coding already
gives a conflict/no-conflict signal, and the feature panel already uses it. But
CAMEO is a fixed taxonomy applied by an automated coder: "MAKE STATEMENT",
"THREATEN", "USE CONVENTIONAL MILITARY FORCE". It cannot distinguish a threat
to close the Strait of Hormuz from a threat about fishing rights — both code as
13x. Embeddings put the headline's ACTUAL MEANING into the model, which is the
only way the text arm contributes geography rather than a second volume count.

WHY MULTILINGUAL, NOT ENGLISH-ONLY. Measured on one day of collected headlines:
English 218, Indonesian 12, Arabic 9, Chinese 8. On the 2023-11-19 sample the
split was English 123, Arabic 34, Russian 23, Chinese 16 — a QUARTER of the
signal is Arabic. An English-only encoder would silently drop exactly the
regional reporting this project treats as load-bearing.

The chosen encoder maps translations into the same region of space. Verified:

    "Iran closes the Strait of Hormuz"  vs  "ايران تغلق مضيق هرمز"   cos 0.866
    "Iran closes the Strait of Hormuz"  vs  "Oil prices steady..."   cos 0.073

ZERO-SHOT BY DESIGN — the same reason Arm B is a VAE. With ~5 labelled events
a supervised text classifier cannot be validated. Instead the disruption
concept is expressed as REFERENCE SENTENCES, and each headline is scored by
cosine similarity to them. No labels are consumed, so none are burned.

CONTRASTIVE SCORING. Similarity to disruption anchors alone is not enough:
any headline mentioning oil or shipping scores moderately, so the series would
track "how much maritime news happened" — which is the volume signal already
available. Subtracting similarity to CALM anchors ("oil prices steady in quiet
trading") removes that common component and leaves the escalation-specific
part.

TOP-K AGGREGATION, NOT MEAN. A day carries up to 250 headlines, most unrelated.
The daily mean is dominated by the irrelevant majority and barely moves during
a crisis; the max is one headline away from noise. The mean of the top 10 is
the compromise, and is what the daily features use.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import repo_root
from ...common.secrets import safe_stdout

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
TOP_K = 10

# Reference sentences describing the TARGET phenomenon. Written as complete
# statements rather than keywords because the encoder is a SENTENCE model —
# it embeds propositions, and a bag of keywords lands somewhere unhelpful.
DISRUPTION_ANCHORS = [
    "The Strait of Hormuz has been closed to shipping.",
    "Iran threatens to block oil exports through the Strait of Hormuz.",
    "Oil tankers have been attacked in the Persian Gulf.",
    "Vessels are rerouting to avoid attacks on shipping lanes.",
    "Naval warships are escorting commercial tankers through the strait.",
    "Mines have been laid in a maritime chokepoint.",
    "Shipping traffic through the strait has halted.",
    "A tanker was struck by a missile near the Gulf of Oman.",
    "Maritime insurance rates have surged for Gulf shipping.",
    "GPS jamming is disrupting navigation in the Gulf.",
]

# Contrast set — ordinary energy/shipping coverage with no disruption content.
# These absorb the "this is a maritime or oil story" component that would
# otherwise make the score a volume proxy.
CALM_ANCHORS = [
    "Oil prices were steady in quiet trading.",
    "The port completed routine maintenance on schedule.",
    "A shipping company reported quarterly earnings.",
    "Container throughput rose modestly at the port.",
    "Energy ministers met to discuss long-term supply contracts.",
    "The tanker completed its scheduled voyage without incident.",
]


def headlines_dir() -> Path:
    return repo_root() / "data" / "raw" / "gdelt" / "headlines"


def cache_dir() -> Path:
    d = repo_root() / "data" / "derived" / "text"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_headlines() -> pd.DataFrame:
    files = sorted(headlines_dir().glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            "no headlines — run `python -m backend.ingest.gdelt.headlines`")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["title", "seendate"])
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["seendate"], utc=True).dt.tz_convert(None).dt.normalize()
    log.info("%d headlines over %s -> %s across %d windows",
             len(df), df.date.min().date(), df.date.max().date(),
             df["window"].nunique())
    return df


def get_model():
    from sentence_transformers import SentenceTransformer
    log.info("loading %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def embed_headlines(df: pd.DataFrame, batch_size: int = 128,
                    force: bool = False) -> np.ndarray:
    """
    Encode every headline, caching to disk.

    Embeddings are normalised at encode time so a cosine similarity is a plain
    dot product downstream — cheaper, and it removes any chance of an
    unnormalised vector silently inflating a similarity.
    """
    cache = cache_dir() / "headline_embeddings.npy"
    keys = cache_dir() / "headline_keys.parquet"
    if cache.exists() and keys.exists() and not force:
        prev = pd.read_parquet(keys)
        if len(prev) == len(df) and (prev["url"].to_numpy() == df["url"].to_numpy()).all():
            log.info("reusing cached embeddings (%d)", len(prev))
            return np.load(cache)
        log.info("cache stale (%d cached vs %d current) — re-encoding",
                 len(prev), len(df))

    model = get_model()
    vecs = model.encode(df["title"].tolist(), batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)
    np.save(cache, vecs)
    df[["url"]].to_parquet(keys, index=False)
    log.info("encoded %d headlines -> %s", len(vecs), vecs.shape)
    return vecs


def anchor_vectors(model) -> tuple[np.ndarray, np.ndarray]:
    d = model.encode(DISRUPTION_ANCHORS, normalize_embeddings=True,
                     convert_to_numpy=True)
    c = model.encode(CALM_ANCHORS, normalize_embeddings=True,
                     convert_to_numpy=True)
    return d, c


def score_headlines(vecs: np.ndarray, disruption: np.ndarray,
                    calm: np.ndarray) -> pd.DataFrame:
    """
    Per-headline semantic scores.

    All vectors are unit-norm, so `vecs @ anchors.T` IS the cosine matrix.
    """
    sim_d = vecs @ disruption.T          # (n, n_disruption)
    sim_c = vecs @ calm.T                # (n, n_calm)
    return pd.DataFrame({
        "sim_disruption": sim_d.max(axis=1),
        "sim_calm": sim_c.max(axis=1),
        # The contrastive score: how much MORE this headline looks like
        # disruption than like ordinary maritime coverage.
        "chokepoint_score": sim_d.max(axis=1) - sim_c.max(axis=1),
    })


def daily_features(df: pd.DataFrame, scores: pd.DataFrame,
                   top_k: int = TOP_K) -> pd.DataFrame:
    """Aggregate per-headline scores to the daily index used by the panel."""
    work = df[["date", "language"]].copy()
    work["chokepoint_score"] = scores["chokepoint_score"].to_numpy()
    work["sim_disruption"] = scores["sim_disruption"].to_numpy()

    def _topk_mean(s: pd.Series) -> float:
        return float(s.nlargest(min(top_k, len(s))).mean())

    g = work.groupby("date")
    out = pd.DataFrame({
        # THE headline Arm C feature: how strongly the day's most relevant
        # coverage matches a chokepoint disruption.
        "text_chokepoint_top10": g["chokepoint_score"].apply(_topk_mean),
        "text_chokepoint_mean": g["chokepoint_score"].mean(),
        "text_disruption_top10": g["sim_disruption"].apply(_topk_mean),
        "text_n_headlines": g.size(),
        # Share of coverage that is strongly disruption-like. A COUNT-based
        # measure, so it rises when many outlets report escalation rather than
        # when one outlet writes one vivid headline.
        "text_share_above_0": g["chokepoint_score"].apply(lambda s: float((s > 0).mean())),
        "text_n_nonenglish": g["language"].apply(lambda s: int((s != "English").sum())),
    })
    out.index.name = "date"
    return out.sort_index()


def build(force: bool = False) -> pd.DataFrame:
    df = load_headlines()
    vecs = embed_headlines(df, force=force)
    model = get_model()
    disruption, calm = anchor_vectors(model)
    scores = score_headlines(vecs, disruption, calm)
    daily = daily_features(df, scores)

    out = cache_dir() / "text_features_daily.parquet"
    daily.to_parquet(out)
    log.info("wrote %s — %d days x %d features", out, len(daily), daily.shape[1])

    # Persist per-headline scores too: the brief layer needs to quote the
    # actual headlines behind a spike, not just the number.
    per = df[["date", "title", "domain", "language", "url", "window"]].copy()
    per = pd.concat([per.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    per.to_parquet(cache_dir() / "headline_scores.parquet", index=False)
    return daily


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Arm C headline embeddings")
    p.add_argument("--force", action="store_true", help="re-encode, ignore cache")
    a = p.parse_args(argv)
    daily = build(force=a.force)
    print(daily.describe().to_string())
    return 0


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
