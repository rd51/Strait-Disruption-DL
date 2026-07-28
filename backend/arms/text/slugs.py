"""
Arm C text extraction from GDELT SOURCEURL slugs.

WHY THIS EXISTS. The DOC 2.0 API is the only free source of real headlines, and
it is not reliably reachable — measured 1/5 success at 30s spacing, 2/4 at 60s,
and repeated blocks even at 180s. Waiting on it makes Arm C hostage to a shared
public endpoint.

But 424,473 distinct article URLs are ALREADY on disk from the historical
backfill, and many news CMSs put the headline in the URL path. Measured:
**205,696 URLs (48.5%) yield four or more alphabetic words** — a real
multilingual corpus, free, with no API in the loop.

    tbmm baskani mustafa sentop bagdata gidecek
    arabia saudi iran amenazar seguridad regional mundial

WHAT IS LOST RELATIVE TO REAL HEADLINES. Slugs are lowercased, stripped of
function words and punctuation by the CMS, often truncated, and sometimes
transliterated. They are a DEGRADED proxy — but they keep the content words,
which is most of what a sentence encoder actually attends to. Treat any score
built on them as noisier than one built on DOC headlines, and say so.

The other 51.5% are opaque numeric IDs (`lratvakan.com/news/619886.html`).
That is a MISSING-NOT-AT-RANDOM bias: outlets using numeric-ID CMSs are
systematically excluded, and they skew toward certain regions and languages.
Check the domain mix before treating coverage as representative.

⚠️ ALWAYS `safe_stdout()` BEFORE PRINTING THESE. They contain Arabic, Farsi,
Cyrillic and accented Latin; printing them under Windows cp1252 raises
UnicodeEncodeError and kills the job AFTER the work is done — which has already
happened twice in this project.
"""

from __future__ import annotations

import argparse
import logging
import re
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from ...common.paths import repo_root
from ...common.secrets import safe_stdout

log = logging.getLogger(__name__)

MIN_WORDS = 4
# Accept Latin, Arabic and Cyrillic word characters. Restricting to ASCII would
# silently drop exactly the Arabic/Farsi coverage this project treats as
# load-bearing.
WORD = re.compile(r"[A-Za-z؀-ۿЀ-ӿ]{3,}")
EXT = re.compile(r"\.(html?|php|aspx?|shtml|jsp)$", re.I)
# Path noise that is never headline content.
STOP_SEG = {"news", "article", "articles", "story", "stories", "index",
            "world", "politics", "business", "en", "ar", "amp", "www"}


def slug_text(url: str) -> str:
    """Extract headline-ish words from a URL path, or '' if there are none."""
    try:
        path = unquote(urlparse(url).path)
    except Exception:                                   # noqa: BLE001
        return ""
    segs = [s for s in path.split("/") if s]
    if not segs:
        return ""
    # The headline is usually the LAST meaningful segment; fall back to the
    # longest one when the last is a bare id or a known section name.
    cand = EXT.sub("", segs[-1])
    words = [w for w in WORD.findall(cand) if w.lower() not in STOP_SEG]
    if len(words) < MIN_WORDS and len(segs) > 1:
        alt = max((EXT.sub("", s) for s in segs), key=len)
        alt_words = [w for w in WORD.findall(alt) if w.lower() not in STOP_SEG]
        if len(alt_words) > len(words):
            words = alt_words
    return " ".join(words).lower()


def load_corpus() -> pd.DataFrame:
    """Every historical article with a usable slug, dated on DATEADDED."""
    files = sorted((repo_root() / "data" / "raw" / "gdelt" / "historical").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no historical GDELT parquet files")
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["SOURCEURL", "DATEADDED", "EventRootCode"])
        df["window"] = f.stem
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Dedup on the article, not the event row — GDELT emits one row per actor
    # pair, a 60-75% overcount that would weight re-reported stories heavily.
    df = df.drop_duplicates(subset=["SOURCEURL"])
    # DATEADDED is publication time (YYYYMMDDHHMMSS). Day is the alleged event
    # date and trails publication by up to a year — indexing on it is
    # look-ahead bias.
    df["date"] = pd.to_datetime(df["DATEADDED"].astype("int64").astype(str),
                                format="%Y%m%d%H%M%S", errors="coerce").dt.normalize()
    df["text"] = df["SOURCEURL"].map(slug_text)
    df["n_words"] = df["text"].str.count(" ") + 1
    keep = df[(df["text"] != "") & (df["n_words"] >= MIN_WORDS)].dropna(subset=["date"])
    log.info("corpus: %d articles -> %d with usable slugs (%.1f%%), %s -> %s",
             len(df), len(keep), 100 * len(keep) / max(len(df), 1),
             keep["date"].min().date(), keep["date"].max().date())
    return keep.reset_index(drop=True)


def build(sample_per_day: int = 120, batch_size: int = 256) -> pd.DataFrame:
    """
    Embed the slug corpus and produce daily contrastive chokepoint features.

    Sampling per day rather than embedding all 205k: coverage volume varies by
    an order of magnitude across the sample, and an unsampled daily mean would
    partly track how much news existed rather than what it said. A fixed cap
    makes days comparable and keeps CPU encoding tractable.
    """
    from .embed import (DISRUPTION_ANCHORS, CALM_ANCHORS, get_model, TOP_K)

    df = load_corpus()
    df = (df.groupby("date", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), sample_per_day), random_state=42))
            .reset_index(drop=True))
    log.info("sampled to %d rows across %d days", len(df), df["date"].nunique())

    model = get_model()
    vecs = model.encode(df["text"].tolist(), batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)
    dis = model.encode(DISRUPTION_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    calm = model.encode(CALM_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)

    # Unit-norm vectors, so a dot product IS the cosine.
    df["sim_disruption"] = (vecs @ dis.T).max(axis=1)
    df["chokepoint_score"] = df["sim_disruption"] - (vecs @ calm.T).max(axis=1)

    def topk_mean(s: pd.Series) -> float:
        return float(s.nlargest(min(TOP_K, len(s))).mean())

    g = df.groupby("date")
    daily = pd.DataFrame({
        "slug_chokepoint_top10": g["chokepoint_score"].apply(topk_mean),
        "slug_chokepoint_mean": g["chokepoint_score"].mean(),
        "slug_share_above_0": g["chokepoint_score"].apply(lambda s: float((s > 0).mean())),
        "slug_n_sampled": g.size(),
    })
    daily.index.name = "date"

    out = repo_root() / "data" / "derived" / "text"
    out.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out / "slug_features_daily.parquet")
    df[["date", "text", "window", "chokepoint_score", "sim_disruption"]].to_parquet(
        out / "slug_scores.parquet", index=False)
    log.info("wrote %d days of slug features", len(daily))
    return daily


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Arm C features from GDELT URL slugs")
    p.add_argument("--per-day", type=int, default=120)
    a = p.parse_args()
    d = build(a.per_day)
    print(d.describe().to_string())
