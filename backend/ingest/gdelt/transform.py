"""
Filtering, de-duplication and (placeholder) scoring for the GDELT arm.

The Gulf filter is deliberately three-way — bounding box OR FIPS country code OR
chokepoint keyword — and every retained row is tagged with which leg caught it,
so the filter's behaviour is auditable after the fact instead of being a black
box that returns "some rows".
"""

from __future__ import annotations

import pandas as pd

from .schema import (
    CONFLICT_ROOT_CODES,
    GULF_FIPS,
    GULF_LAT,
    GULF_LON,
    HORMUZ_TERMS,
    NUMERIC_COLS,
)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the numeric columns once, up front. Everything downstream assumes this ran."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def gulf_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    The three legs of the Gulf filter, returned separately so callers can see
    what each contributed rather than only their union.
    """
    in_box = (
        df["ActionGeo_Lat"].between(*GULF_LAT)
        & df["ActionGeo_Long"].between(*GULF_LON)
    )
    in_country = df["ActionGeo_CountryCode"].isin(GULF_FIPS)

    name = df["ActionGeo_FullName"].fillna("").astype(str).str.lower()
    url = df["SOURCEURL"].fillna("").astype(str).str.lower()
    keyword = pd.Series(False, index=df.index)
    for term in HORMUZ_TERMS:
        keyword |= name.str.contains(term, regex=False) | url.str.contains(term, regex=False)

    return {"bbox": in_box.fillna(False), "country": in_country, "keyword": keyword}


def filter_gulf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep events geolocated to the Gulf **or** textually about the chokepoint.

    The keyword leg is not redundant with the geographic legs: a London- or
    Yemen-datelined report about Hormuz tanker traffic is highly relevant but
    geolocates far outside the box. Dropping that leg loses exactly the
    reporting that matters most.
    """
    masks = gulf_masks(df)
    union = masks["bbox"] | masks["country"] | masks["keyword"]
    out = df[union].copy()

    # Provenance: which leg(s) caught each row, e.g. "country" or "bbox+country".
    #
    # ⚠️ Built with a list comprehension, NOT DataFrame.apply(axis=1). On an
    # EMPTY frame `apply(axis=1)` returns a DataFrame rather than a Series, and
    # assigning that to one column raises
    #   ValueError: Cannot set a DataFrame with multiple columns to the single
    #   column gulf_match
    # A 15-minute slot with zero Gulf events is uncommon but perfectly normal,
    # so that crashed deterministically on exactly those slots — silently
    # dropping them from both the backfill and the live poller, and looking
    # like a transient network fault because it only hit a small fraction.
    legs = ("bbox", "country", "keyword")
    columns = [masks[leg][union].to_numpy() for leg in legs]
    out["gulf_match"] = [
        "+".join(leg for leg, hit in zip(legs, row) if hit)
        for row in zip(*columns)
    ] if len(out) else pd.Series(dtype="object")
    return out


def filter_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the CAMEO escalation ladder (root codes 13-20)."""
    return df[df["EventRootCode"].isin(CONFLICT_ROOT_CODES)].copy()


def dedup_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per source article, keeping the highest-reach record.

    GDELT emits a separate event row per actor pair extracted from a single
    article, so one report becomes several rows with distinct GlobalEventIDs.
    Measured on the 2026-07-27 live window: 16 conflict rows came from only 10
    distinct SOURCEURLs — a 60% overcount if rows are treated as independent
    events. Any volume-based signal must dedup first or it inflates on
    re-reporting rather than on escalation.
    """
    if df.empty or "SOURCEURL" not in df.columns:
        return df.copy()
    return (
        df.sort_values("NumArticles", ascending=False, na_position="last")
          .drop_duplicates(subset="SOURCEURL", keep="first")
    )


def tension_score(df_conflict: pd.DataFrame) -> tuple[float, dict]:
    """
    ⚠️ PLACEHOLDER HEURISTIC — NOT A MODEL. ⚠️

    A crude 0-100 window score so the pipeline has an end-to-end output before
    the transformer arm exists. It combines conflict volume, Goldstein
    negativity and tone negativity. It is **not** trained, **not** validated,
    and must never appear in a backtest result or be described as a model
    output. Arm C replaces it.

    Volume is counted over DISTINCT ARTICLES, not raw event rows — see
    `dedup_articles` for why raw rows overcount by ~60%.
    """
    if df_conflict.empty:
        return 0.0, {
            "events": 0, "articles_distinct": 0, "articles_total": 0,
            "avg_goldstein": None, "avg_tone": None, "placeholder": True,
        }

    deduped = dedup_articles(df_conflict)
    n_distinct = len(deduped)

    goldstein = deduped["GoldsteinScale"].mean()   # -10 (worst) .. +10
    tone = deduped["AvgTone"].mean()               # typically -20 .. +20
    total_articles = int(df_conflict["NumArticles"].fillna(0).sum())

    volume_c = min(n_distinct / 50.0, 1.0) * 40.0        # 0..40
    goldstein_c = max(0.0, (-goldstein) / 10.0) * 35.0   # 0..35
    tone_c = max(0.0, (-tone) / 10.0) * 25.0             # 0..25

    score = round(min(volume_c + goldstein_c + tone_c, 100.0), 1)
    parts = {
        "events": int(len(df_conflict)),
        "articles_distinct": int(n_distinct),
        "articles_total": total_articles,
        "avg_goldstein": round(float(goldstein), 2),
        "avg_tone": round(float(tone), 2),
        "component_volume": round(volume_c, 1),
        "component_goldstein": round(goldstein_c, 1),
        "component_tone": round(tone_c, 1),
        "placeholder": True,
    }
    return score, parts


def summarise(df_all: pd.DataFrame, df_gulf: pd.DataFrame, df_conflict: pd.DataFrame) -> dict:
    """Per-window counts, including the contribution of each Gulf filter leg."""
    masks = gulf_masks(df_all)
    return {
        "rows_total": int(len(df_all)),
        "rows_gulf": int(len(df_gulf)),
        "rows_conflict": int(len(df_conflict)),
        "leg_bbox": int(masks["bbox"].sum()),
        "leg_country": int(masks["country"].sum()),
        "leg_keyword": int(masks["keyword"].sum()),
        "leg_keyword_only": int(
            (masks["keyword"] & ~masks["bbox"] & ~masks["country"]).sum()
        ),
    }
