"""
The semantic layer — canonical definitions for every quantity in the system.

WHY THIS IS THE MOST IMPORTANT LAYER. A number in a dashboard is meaningless
without three things travelling with it: what it actually measures, where it
came from, and what it must never be used for. This project has already been
bitten by all three gaps — a "vessel count" that was 70% cranes, a Brent return
series that pandas silently fabricated, a congestion feature whose sampling
density encodes the label. Each was a *semantic* failure, not a code failure:
the number was computed correctly and meant something other than it appeared to.

This registry is the single source of truth for what each metric means. It is
machine-readable so downstream code can validate against it rather than trusting
that whoever built the frame remembered the caveats.

Four things every metric carries:
  · definition   — what it measures, precisely enough to argue with
  · provenance   — source, whether that source is officially citable
  · caveats      — the measured gotchas, not generic warnings
  · forbidden    — uses that are WRONG, with the reason

`forbidden` is the unusual field and the most valuable one. Every entry in it
was learned by being wrong first.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Arm = Literal["A_sar", "B_market", "C_gdelt", "fusion", "label"]
Cadence = Literal["15min", "daily", "per_pass", "event", "static"]


@dataclass(frozen=True)
class Metric:
    name: str
    arm: Arm
    definition: str
    unit: str
    source: str
    official: bool          # citable primary source vs convenience/unofficial
    cadence: Cadence
    computation: str
    caveats: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────── ARM C — geopolitical text
GDELT_METRICS = [
    Metric(
        name="gdelt_articles",
        arm="C_gdelt",
        definition=(
            "Count of DISTINCT source articles published in the Gulf-filtered "
            "GDELT stream on a given publication day."
        ),
        unit="articles/day",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="COUNT(DISTINCT SOURCEURL) grouped by DATEADDED date",
        caveats=[
            "GDELT emits one event row per actor pair extracted from an article, "
            "so raw row counts overcount by 60-75% (measured: 21 rows from 12 "
            "URLs in one slot, 16 from 10 in another).",
            "Aggregated on DATEADDED (publication), never Day (alleged event "
            "date) — Day trails publication by up to 365 days.",
        ],
        forbidden=[
            "Do NOT use gdelt_rows as a volume proxy — it measures re-reporting "
            "volume, not event volume, and spikes when wire services repeat a "
            "story rather than when something new happens.",
            "Do NOT index this series on Day. That places information before it "
            "was knowable (look-ahead bias).",
        ],
    ),
    Metric(
        name="gdelt_conflict_articles",
        arm="C_gdelt",
        definition=(
            "Distinct articles whose CAMEO EventRootCode falls on the escalation "
            "ladder (13 Threaten, 14 Protest, 15 Exhibit force, 16 Reduce "
            "relations, 17 Coerce, 18 Assault, 19 Fight, 20 Mass violence)."
        ),
        unit="articles/day",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="COUNT(DISTINCT SOURCEURL) WHERE EventRootCode IN 13..20",
        caveats=[
            "Conflict coding is imperfect; a raw spike is a signal to corroborate, "
            "not a fact.",
        ],
        forbidden=[],
    ),
    Metric(
        name="gdelt_conflict_share",
        arm="C_gdelt",
        definition=(
            "Fraction of the day's Gulf reporting that is conflict-coded. "
            "Scale-free, so comparable across windows with very different "
            "absolute article volumes."
        ),
        unit="ratio 0-1",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="gdelt_conflict_articles / gdelt_articles",
        caveats=[
            "Preferred over raw counts for cross-window comparison: total GDELT "
            "coverage volume grows over time independently of Gulf events.",
        ],
        forbidden=[],
    ),
    Metric(
        name="gdelt_rows",
        arm="C_gdelt",
        definition=(
            "RAW count of Gulf-filtered GDELT event rows on a publication day. "
            "Retained for diagnostics and for computing the re-reporting ratio "
            "(gdelt_rows / gdelt_articles) — NOT as a volume signal."
        ),
        unit="rows/day",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="COUNT(*) grouped by DATEADDED date",
        caveats=[
            "Overcounts articles by 60-75% because GDELT emits one row per actor "
            "pair extracted from each article.",
        ],
        forbidden=[
            "Do NOT feed to a model as a volume feature. It measures how many "
            "actor pairs a story mentioned, which rises when wire services "
            "repeat a story — use gdelt_articles instead.",
        ],
    ),
    Metric(
        name="gdelt_conflict_rows",
        arm="C_gdelt",
        definition="Raw conflict-coded row count. Diagnostic counterpart to gdelt_rows.",
        unit="rows/day",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="COUNT(*) WHERE EventRootCode IN 13..20, grouped by DATEADDED date",
        caveats=["Same re-reporting inflation as gdelt_rows."],
        forbidden=[
            "Do NOT use as a volume feature — use gdelt_conflict_articles.",
        ],
    ),
    Metric(
        name="gdelt_conflict_tone",
        arm="C_gdelt",
        definition=(
            "Mean AvgTone restricted to conflict-coded rows. Measures how "
            "negatively the escalatory reporting itself is written, separate "
            "from how much of it there is."
        ),
        unit="tone (-100..+100)",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="AVG(AvgTone) WHERE EventRootCode IN 13..20",
        caveats=[
            "NaN on days with zero conflict rows — that is correct and must not "
            "be filled with 0, which would read as neutral tone rather than "
            "no reporting.",
        ],
        forbidden=[
            "Do NOT impute missing values with 0. Absence of conflict reporting "
            "is not neutral-toned conflict reporting.",
        ],
    ),
    Metric(
        name="gdelt_conflict_goldstein",
        arm="C_gdelt",
        definition="Mean Goldstein scale over conflict-coded rows only.",
        unit="goldstein (-10..+10)",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="AVG(GoldsteinScale) WHERE EventRootCode IN 13..20",
        caveats=[
            "Bounded below by the ladder's own floor: restricting to codes 13-20 "
            "already selects negative Goldstein values, so the interesting "
            "variation is WITHIN the escalation band, not around zero.",
        ],
        forbidden=[
            "Do NOT impute missing values with 0 — 0 sits ABOVE this metric's "
            "entire realistic range and would read as de-escalation.",
        ],
    ),
    Metric(
        name="gdelt_tone_mean",
        arm="C_gdelt",
        definition="Mean GDELT AvgTone across the day's Gulf-filtered rows.",
        unit="tone (-100..+100, typically -20..+20)",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="AVG(AvgTone) grouped by DATEADDED date",
        caveats=["Tone coding is imperfect and language-dependent."],
        forbidden=[],
    ),
    Metric(
        name="gdelt_goldstein_mean",
        arm="C_gdelt",
        definition=(
            "Mean Goldstein scale — a -10..+10 measure of an event type's "
            "theoretical impact on political stability."
        ),
        unit="goldstein (-10..+10)",
        source="GDELT 2.0 Translingual Events",
        official=True,
        cadence="daily",
        computation="AVG(GoldsteinScale) grouped by DATEADDED date",
        caveats=[
            "Goldstein is a property of the EVENT TYPE (a fixed lookup per CAMEO "
            "code), not a measurement of this specific event's severity.",
        ],
        forbidden=[
            "Do NOT interpret a Goldstein change as a severity change — it is a "
            "change in the MIX of event types being reported.",
        ],
    ),
]

# ─────────────────────────────────────────────────── ARM B — market series
MARKET_METRICS = [
    Metric(
        name="brent",
        arm="B_market",
        definition="Europe Brent Spot Price FOB — the Gulf-exposed crude benchmark.",
        unit="USD/barrel",
        source="EIA, redistributed via FRED (DCOILBRENTEU)",
        official=True,
        cadence="daily",
        computation="FRED series DCOILBRENTEU, no transformation",
        caveats=[
            "Weekends and holidays are absent (NaN), never zero. Never "
            "forward-filled at ingest — filling is a modelling decision that "
            "belongs in the feature layer.",
            "Series notes point at eia.doe.gov, so this IS the official EIA "
            "series; a separate EIA key is unnecessary.",
        ],
        forbidden=[
            "Do NOT forward-fill before a train/test split — that leaks the last "
            "known price across the boundary.",
        ],
    ),
    Metric(
        name="wti",
        arm="B_market",
        definition=(
            "West Texas Intermediate spot, Cushing OK. The LANDLOCKED control "
            "series — it exists in this project to be subtracted from Brent."
        ),
        unit="USD/barrel",
        source="EIA, redistributed via FRED (DCOILWTICO)",
        official=True,
        cadence="daily",
        computation="FRED series DCOILWTICO, no transformation",
        caveats=[
            "WTI prices at an inland hub with no seaborne chokepoint exposure. "
            "That insensitivity is the point: it isolates the Gulf premium.",
        ],
        forbidden=[
            "Do NOT treat a WTI move as Gulf evidence. WTI rising WITH Brent is "
            "a global oil story, not a chokepoint story.",
        ],
    ),
    Metric(
        name="brent_ret_1d",
        arm="B_market",
        definition="One-trading-day percentage change in Brent.",
        unit="percent",
        source="derived from FRED",
        official=True,
        cadence="daily",
        computation="brent.pct_change(fill_method=None) * 100",
        caveats=[
            "Defined only on consecutive TRADING days. A Monday's return spans "
            "the weekend and is mechanically larger — a known property, not a bug.",
            "Largest measured 2026 escalation days: 03-12 (+12.53%), 03-18 "
            "(+8.95%), 03-05 (+8.62%), 04-07 (+8.31%), 03-02 (+8.30%).",
        ],
        forbidden=[
            "Do NOT call pct_change() without fill_method=None. The pandas "
            "default pads NaNs and fabricates 0% returns on non-trading days.",
        ],
    ),
    Metric(
        name="brent_ret_7d",
        arm="B_market",
        definition="Seven-calendar-day percentage change in Brent — trend, not noise.",
        unit="percent",
        source="derived from FRED",
        official=True,
        cadence="daily",
        computation="brent.pct_change(7, fill_method=None) * 100, masked to source dates",
        caveats=[
            "Computed on a reindexed daily calendar, so the lookback is 7 "
            "CALENDAR days (~5 trading days), not 7 trading days.",
        ],
        forbidden=[
            "Do NOT compare against brent_ret_1d * 7 — the horizons differ.",
        ],
    ),
    Metric(
        name="spread_chg_7d",
        arm="B_market",
        definition=(
            "Seven-day ABSOLUTE change in the Brent-WTI spread. The rate at "
            "which chokepoint risk is being repriced."
        ),
        unit="USD/barrel",
        source="derived from FRED",
        official=True,
        cadence="daily",
        computation="brent_wti_spread.diff(7), masked to source dates",
        caveats=[
            "Absolute difference, not percentage — the spread crosses and "
            "approaches zero, so a percentage change is unstable there.",
        ],
        forbidden=[
            "Do NOT convert to a percentage change. Near-zero denominators "
            "produce meaningless spikes.",
        ],
    ),
    Metric(
        name="gas",
        arm="B_market",
        definition="Henry Hub natural gas spot — a second, partly independent energy read.",
        unit="USD per million BTU",
        source="EIA, redistributed via FRED (DHHNGSP)",
        official=True,
        cadence="daily",
        computation="FRED series DHHNGSP, no transformation",
        caveats=[
            "Henry Hub is a US domestic benchmark. Qatari LNG transits Hormuz, "
            "but that flow prices in Asian/European LNG markets (JKM, TTF), not "
            "Henry Hub. Its link to Gulf disruption is INDIRECT and weak.",
        ],
        forbidden=[
            "Do NOT present a Henry Hub move as evidence of Hormuz LNG "
            "disruption — the relevant benchmarks are JKM and TTF, which this "
            "project does not have.",
        ],
    ),
    Metric(
        name="brent_wti_spread",
        arm="B_market",
        definition=(
            "Brent minus WTI. THE GULF-SPECIFIC DISCRIMINATOR: WTI is landlocked "
            "US crude, so a widening spread prices CHOKEPOINT risk, whereas a "
            "parallel move in both is a global oil story."
        ),
        unit="USD/barrel",
        source="EIA via FRED (DCOILBRENTEU - DCOILWTICO)",
        official=True,
        cadence="daily",
        computation="brent - wti",
        caveats=[
            "Normal range is roughly 3-5. Measured peak during the 2026 Hormuz "
            "crisis: 25.94 on 2026-04-08 — a ~6x normal level.",
            "This is the single most diagnostic market feature in the system.",
        ],
        forbidden=[],
    ),
    Metric(
        name="spread_z",
        arm="B_market",
        definition=(
            "Brent-WTI spread expressed as a z-score against its own trailing "
            "252-day (one trading year) distribution."
        ),
        unit="standard deviations",
        source="derived from FRED",
        official=True,
        cadence="daily",
        computation="(spread - rolling_mean_252) / rolling_std_252, backward-looking only",
        caveats=[
            "Regime-relative: 'high' means high for the current regime, not high "
            "against a fixed constant.",
            "Undefined for the first 60 observations (min_periods=60).",
        ],
        forbidden=[
            "Do NOT use a centred rolling window — that reads the future.",
        ],
    ),
    Metric(
        name="spread_smooth_10d",
        arm="B_market",
        definition=(
            "10-day trailing mean of the Brent-WTI spread. The PERSISTENT "
            "chokepoint premium, with single-day mechanics averaged out."
        ),
        unit="USD/barrel",
        source="derived from FRED",
        official=True,
        cadence="daily",
        computation="brent_wti_spread.rolling(10, min_periods=5).mean(), masked to source",
        caveats=[
            "This is the spread measure that actually discriminates. Measured "
            "smoothed maxima: 2026 Hormuz 22.98, 2019 Gulf of Oman 11.87, "
            "COVID 11.22, 2022 Ukraine 10.21, Red Sea 7.04, Abqaiq 6.38.",
            "Baseline: p50 4.37, p90 8.37, p99 15.24.",
        ],
        forbidden=[
            "Do NOT rank shocks on the RAW daily spread instead. Its all-time "
            "maximum (54.34) is 2020-04-20, when WTI settled at -$36.98 on the "
            "May contract expiry with Cushing storage full — a US futures "
            "mechanic with zero Gulf content that outranks the 2026 Hormuz "
            "peak (25.94) by more than 2x.",
        ],
    ),
    Metric(
        name="spread_days_gt15_30d",
        arm="B_market",
        definition=(
            "Number of days in the trailing 30 on which the Brent-WTI spread "
            "exceeded $15 — a count of STRESSED days, not an average."
        ),
        unit="days (0-30)",
        source="derived from FRED",
        official=True,
        cadence="daily",
        computation="(brent_wti_spread > 15).rolling(30, min_periods=10).sum(), masked",
        caveats=[
            "The cleanest single separator measured so far. Days above 15 per "
            "shock window: 2026 Hormuz 19, COVID 1, and ZERO for 2019 Gulf of "
            "Oman, 2019 Abqaiq, 2022 Ukraine and the 2023-24 Red Sea.",
            "Deliberately a COUNT, not a mean — one $54 day and twenty $16 days "
            "give similar means and mean entirely different things.",
            "The $15 threshold sits near the raw spread's p99 (14.46), so it is "
            "calibrated to the sample rather than chosen to fit the label.",
        ],
        forbidden=[
            "Do NOT read a non-zero value as Hormuz-specific on its own. It "
            "separates SUSTAINED seaborne dislocation from transient mechanics; "
            "geography comes from Arm C, not from the price.",
        ],
    ),
    Metric(
        name="brent_vol_7d",
        arm="B_market",
        definition="7-day trailing standard deviation of daily Brent returns.",
        unit="percent",
        source="derived from FRED",
        official=True,
        cadence="daily",
        computation="rolling(7, min_periods=3).std() of brent_ret_1d, masked to source dates",
        caveats=[
            "MUST be masked to where brent_ret_1d is defined. On a reindexed "
            "daily calendar, rolling() keeps emitting values after the source "
            "ends because a trailing window still finds enough points — that "
            "fabricated data past the last real close until it was caught.",
        ],
        forbidden=[
            "Do NOT compute with pandas defaults. pct_change() pads NaNs unless "
            "given fill_method=None, inventing 0% returns on non-trading days.",
        ],
    ),
]

# ────────────────────────────────────────────────────── ARM A — SAR / vision
SAR_METRICS = [
    Metric(
        name="sar_vessels_<port>",
        arm="A_sar",
        definition=(
            "Count of CFAR detections falling on WATER within a port's AOI on a "
            "given Sentinel-1 pass. The congestion signal."
        ),
        unit="detections/pass",
        source="Copernicus Sentinel-1 GRD via Sentinel Hub Process API",
        official=True,
        cadence="per_pass",
        computation=(
            "CA-CFAR (guard 4, background 12, k=5) on VV AND VH, size-filtered "
            "2-400px, intersected with a SAR-derived water mask"
        ),
        caveats=[
            "Land detections are EXCLUDED. They are the same cranes and quays "
            "every pass (measured near-constant at 39/37/38 on Fujairah) and "
            "would add a large fixed offset: unmasked, a real 14->19 vessel "
            "change (+36%) reads as 53->57 (+7.5%).",
            "PARTIAL chips (coverage < 90%) are dropped, not rescaled — vessels "
            "cluster in the anchorage, so rescaling by covered area assumes a "
            "uniformity that does not hold.",
            "CFAR is a statistical detector, NOT a trained model. It is the "
            "baseline the CNN must beat.",
        ],
        forbidden=[
            "Do NOT compare counts across ports. Fujairah/Khor Fakkan get ~2x "
            "the passes of Jebel Ali/Khalifa in every window (swath geometry).",
            "Do NOT expose raw observation COUNTS as a model feature. Sampling "
            "density is correlated with the label — 2026 has ~2x the revisit of "
            "historical windows AND is an event window, so a model could learn "
            "'dense sampling => crisis'. TimeSeriesSplit cannot catch this.",
            "Do NOT compare a partial chip against a full one without normalising.",
        ],
    ),
]

# ─────────────────────────────────── ARM C — semantic / embedding features
TEXT_METRICS = [
    Metric(
        name="text_chokepoint_top10",
        arm="C_gdelt",
        definition=(
            "Mean CONTRASTIVE semantic score of the day's 10 most "
            "disruption-like headlines. Per headline: cosine similarity to a "
            "set of chokepoint-disruption reference sentences MINUS similarity "
            "to ordinary maritime/energy reference sentences."
        ),
        unit="cosine difference (roughly -1..1)",
        source="GDELT DOC 2.0 headlines + multilingual sentence embeddings",
        official=True,
        cadence="daily",
        computation=(
            "paraphrase-multilingual-MiniLM-L12-v2 (384-d, unit-normalised); "
            "max-sim to 10 disruption anchors minus max-sim to 6 calm anchors; "
            "mean of the day's top 10"
        ),
        caveats=[
            "The CONTRAST is what stops this being a volume proxy. Similarity "
            "to disruption anchors alone scores any oil/shipping headline "
            "moderately, so the raw series would track how much maritime news "
            "happened — which the article count already measures.",
            "Top-10 rather than mean or max: a day carries up to 250 headlines "
            "and the mean is dominated by the irrelevant majority, while the "
            "max is one odd headline away from noise.",
            "Multilingual by necessity. Measured language mix on one day: "
            "English 123, Arabic 34, Russian 23, Chinese 16 — an English-only "
            "encoder would drop a quarter of the signal. Cross-lingual "
            "alignment verified: EN/AR translations of the same sentence score "
            "cosine 0.866 against 0.073 for unrelated text.",
            "ZERO-SHOT — consumes no labels, so none are burned. Same reasoning "
            "as the Arm B VAE.",
        ],
        forbidden=[
            "Do NOT compare values across days with very different headline "
            "counts without checking text_n_headlines. A day with 12 headlines "
            "has a top-10 that is nearly its whole sample.",
            "Do NOT treat the anchor sentences as tuned parameters. They were "
            "written before scoring and must not be edited to improve a result "
            "— that would fit the label through the back door.",
        ],
    ),
    Metric(
        name="text_n_headlines",
        arm="C_gdelt",
        definition="Headlines returned by GDELT DOC 2.0 for the day's query.",
        unit="headlines/day",
        source="GDELT DOC 2.0",
        official=True,
        cadence="daily",
        computation="row count per seendate day, after URL dedup",
        caveats=[
            "CAPPED AT 250 by the API's maxrecords limit. On heavy news days "
            "the true count is higher and this saturates.",
        ],
        forbidden=[
            "Do NOT use as a volume feature — it is censored at 250, so it "
            "flattens exactly when coverage intensifies, which is backwards.",
        ],
    ),
]

# ───────────────────────────────────────────────────────────────── labels
LABEL_METRICS = [
    Metric(
        name="event_onset",
        arm="label",
        definition=(
            "The date a disruption began to be PRICED, not the date it was "
            "announced. What an early-warning system must anticipate."
        ),
        unit="date",
        source="corroborated across GDELT headlines and FRED Brent",
        official=True,
        cadence="event",
        computation="first sustained deviation from the pre-event price baseline",
        caveats=[
            "The 2026 anchor is 2026-03-02 (price onset), NOT 2026-06-12 (the "
            "closure announcement). Brent FELL 24% across the June window — the "
            "announcement sits in the de-escalation, after the premium unwound.",
            "Corroborated across TWO arms: GDELT said June, the market said "
            "March. Neither alone would have caught it.",
        ],
        forbidden=[
            "Do NOT label on the headline peak. An early-warning system scored "
            "against it would be credited for detecting an event already over, "
            "and its measured 'lead time' would be meaningless.",
            "Do NOT admit a label corroborated by only one arm into the "
            "backtest gate.",
        ],
    ),
]

REGISTRY: dict[str, Metric] = {
    m.name: m for m in (GDELT_METRICS + MARKET_METRICS + SAR_METRICS
                        + TEXT_METRICS + LABEL_METRICS)
}


def get(name: str) -> Metric | None:
    """Exact lookup, falling back to the templated per-port SAR entry."""
    if name in REGISTRY:
        return REGISTRY[name]
    if name.startswith("sar_vessels_"):
        return REGISTRY["sar_vessels_<port>"]
    return None


def describe(name: str) -> str:
    m = get(name)
    if m is None:
        return f"{name}: NOT IN REGISTRY — undefined metric"
    lines = [f"{m.name}  [{m.arm}]  ({m.unit}, {m.cadence})", f"  {m.definition}",
             f"  source: {m.source} ({'official' if m.official else 'UNOFFICIAL'})"]
    for c in m.caveats:
        lines.append(f"  ⚠ {c}")
    for f_ in m.forbidden:
        lines.append(f"  ✗ {f_}")
    return "\n".join(lines)


def audit_frame(columns: list[str]) -> dict:
    """
    Which columns of a feature frame are semantically defined?

    An undefined column is a number nobody can explain — it should not reach a
    model, and certainly not a dashboard.
    """
    defined, undefined = [], []
    for c in columns:
        (defined if get(c) else undefined).append(c)
    return {
        "n_columns": len(columns),
        "defined": defined,
        "undefined": undefined,
        "coverage": round(len(defined) / max(len(columns), 1), 3),
    }


def forbidden_uses() -> list[dict]:
    """Every prohibition in one place — the project's accumulated hard lessons."""
    out = []
    for m in REGISTRY.values():
        for f_ in m.forbidden:
            out.append({"metric": m.name, "arm": m.arm, "rule": f_})
    return out
