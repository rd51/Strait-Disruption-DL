"""
Fusion — combining the arms into a chokepoint risk index (0-100).

🔴 WHY THIS IS A WEIGHTED RULE AND NOT A TRAINED MODEL.
The project has FIVE labelled events. Fitting a gradient-boosted model to
combine three arm scores means fitting a decision surface to five positive
examples. Whatever AUC that produced would be an artefact of which event landed
in which fold, not a measurement of skill. A model that cannot be validated
should not be the thing making the claim.

So the PRIMARY fusion is a transparent weighted rule whose weights come from
each arm's SEPARATELY MEASURED performance — numbers that were established
before fusion existed and are not tuned here. The GBM is trained alongside as
an explicitly ILLUSTRATIVE comparison, reported with its own caveat, never as
the shipped index.

    weight   arm  justification (measured, documented in CLAUDE.md)
    ------   ---  ---------------------------------------------------------
      0.50   C    AUC 0.848 on point events; detected 3/3 sharp incidents at
                  z = 3.4-4.7; top-scoring day in a 366-day corpus was the day
                  after the Gulf of Oman attacks
      0.50   B    AUC 0.680; detected the 2026 sustained dislocation with a
                  38-day held-out lead. Misses sharp incidents entirely —
                  exactly complementary to C, which is why both carry weight
      0.00   A    NO measured signal. Fujairah vessel counts moved +0.0% across
                  the 2026 onset; Mann-Whitney p = 0.26-0.89 with bypass ports
                  NEGATIVE in both orbits. Included in the schema and given
                  zero weight rather than quietly dropped — the null is a
                  finding, and a reader should see it was considered

B and C are equal not because that is convenient but because neither dominates:
each detects precisely what the other misses, so there is no measured basis for
preferring one.

🔴 NORMALISATION MUST BE CAUSAL. Arm scores live on incomparable scales (VAE
reconstruction error is unbounded and positive; the semantic score is a cosine
difference roughly in [-1, 1]). Converting each to a percentile rank against a
FULL-SAMPLE distribution would leak the future into every historical point — a
day in 2019 would be ranked against 2026 data that did not exist yet. Ranks are
therefore computed against a TRAILING EXPANDING window only.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from ..common.paths import repo_root
from ..common.secrets import safe_stdout
from ..features.splits import EVENTS

log = logging.getLogger(__name__)

# Weights are DECLARED from prior measurement, never fitted here.
WEIGHTS = {"arm_b": 0.50, "arm_c": 0.50, "arm_a": 0.00}
MIN_HISTORY = 120        # days before a causal percentile is meaningful


def load_arms() -> pd.DataFrame:
    """Every arm's daily score on one index. Missing arms are simply absent."""
    root = repo_root() / "data"
    frames = {}

    p = root / "models" / "arm_b_vae" / "scores.parquet"
    if p.exists():
        s = pd.read_parquet(p)
        s["date"] = pd.to_datetime(s["date"])
        frames["arm_b"] = s.set_index("date")["recon_error"]

    p = root / "derived" / "text" / "slug_features_daily.parquet"
    if p.exists():
        frames["arm_c"] = pd.read_parquet(p)["slug_chokepoint_top10"]

    p = root / "derived" / "features_daily.parquet"
    if p.exists():
        f = pd.read_parquet(p)
        sar = [c for c in f.columns if c.startswith("sar_vessels_")]
        if sar:
            # Congestion enters as a deviation, not a level: ports differ by an
            # order of magnitude in absolute vessel count.
            z = f[sar].apply(lambda c: (c - c.expanding(20).mean())
                             / c.expanding(20).std())
            frames["arm_a"] = z.mean(axis=1)

    if not frames:
        raise FileNotFoundError("no arm scores found — run the arms first")
    df = pd.DataFrame(frames).sort_index()
    log.info("arms loaded: %s | %s -> %s", list(df.columns),
             df.index.min().date(), df.index.max().date())
    return df


def causal_rank(s: pd.Series, min_history: int = MIN_HISTORY) -> pd.Series:
    """
    Percentile rank of each value against ONLY its own past.

    Deliberately O(n^2)-ish rather than a vectorised full-sample rank: the
    vectorised version is what leaks. At a few thousand rows the cost is
    irrelevant and the guarantee is worth more than the speed.
    """
    v = s.to_numpy(dtype="float64")
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        past = v[:i + 1]
        past = past[~np.isnan(past)]
        if len(past) < min_history:
            continue
        out[i] = (past < v[i]).sum() / len(past)
    return pd.Series(out, index=s.index)


def risk_index(arms: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted causal-percentile fusion -> a 0-100 chokepoint risk index.

    Weights are renormalised over the arms actually PRESENT on each day, so a
    day where Arm C has no coverage is not silently scored as if C were zero —
    which would read as "no geopolitical tension" rather than "no data".
    """
    ranks = pd.DataFrame({c: causal_rank(arms[c]) for c in arms.columns},
                         index=arms.index)
    w = pd.Series({c: WEIGHTS.get(c, 0.0) for c in ranks.columns})

    contrib = ranks.mul(w, axis=1)
    present = ranks.notna().mul(w, axis=1).sum(axis=1)      # weight actually available
    idx = (contrib.sum(axis=1) / present.replace(0, np.nan)) * 100.0

    out = ranks.add_prefix("rank_")
    out["arms_available"] = ranks.notna().sum(axis=1)
    out["weight_available"] = present
    out["risk_index"] = idx
    # Require at least half the declared weight before publishing a number.
    out.loc[present < 0.5 * w.sum(), "risk_index"] = np.nan
    return out


# ─────────────────────────────────────────────── leave-one-event-out backtest
def loeo_backtest(fused: pd.DataFrame, threshold_pct: float = 90.0) -> dict:
    """
    Leave-one-event-out evaluation.

    WHY LOEO RATHER THAN UNIFORM TimeSeriesSplit. With five events, an evenly
    spaced 5-fold split leaves two test windows containing NO labelled event —
    scoring them says nothing about early-warning skill, and averaging across
    them quotes an accuracy for periods where nothing happened. LOEO gives one
    fold per event by construction, so every fold is informative.

    The threshold is set from the OTHER events' quiet periods only, never from
    the held-out event's own window.
    """
    from ..features.splits import leave_one_event_out_splits

    idx = fused["risk_index"].dropna()
    results = []

    # 🔴 CAUSALITY GATE. The split module decides which events can honestly be
    # scored: under causal LOEO an event needs enough PRIOR history to have
    # been predictable at the time. Measured on this panel, the two earliest
    # events (2019 Fujairah, 2019 Gulf of Oman) have only 298 and 321 training
    # days against a 365-day minimum, so they are UNSCOREABLE — not "missed"
    # and not "detected". An earlier version of this backtest scored all five
    # and reported "3 of 5 detected", which credited the system for a Gulf of
    # Oman detection it could not have made in 2019.
    #
    # WHICH MINIMUM APPLIES DEPENDS ON WHETHER ANYTHING IS FITTED.
    # The weighted rule trains NOTHING — weights are declared from prior
    # measurement. So "training days" is not a meaningful requirement for it;
    # the binding constraint is the causal percentile's warm-up (MIN_HISTORY
    # days), which `causal_rank` already enforces by emitting NaN before it.
    # Applying the 365-day fitted-model standard here discarded three events
    # for a reason that does not apply to this estimator.
    # The GBM below IS fitted, and the stricter standard does apply to it.
    scoreability = {f["event"]: f for f in
                    leave_one_event_out_splits(idx.index, causal=True,
                                               min_train_days=MIN_HISTORY)}

    for name, (start, end) in EVENTS.items():
        fold = scoreability.get(name, {})
        if not fold.get("scoreable", False):
            results.append({"event": name, "onset": start,
                            "status": "UNSCOREABLE (causal LOEO)",
                            "reason": fold.get("reason", "no fold"),
                            "detected": None})
            continue
        onset = pd.Timestamp(start)
        # Everything outside ANY event window is "quiet"; the threshold comes
        # from quiet days that are also OUTSIDE the held-out event.
        quiet = idx.copy()
        for other, (s2, e2) in EVENTS.items():
            m = (quiet.index >= pd.Timestamp(s2) - pd.Timedelta("30D")) & \
                (quiet.index <= pd.Timestamp(e2) + pd.Timedelta("30D"))
            quiet = quiet[~m]
        if len(quiet) < 50:
            results.append({"event": name, "status": "insufficient quiet history"})
            continue
        thr = float(np.percentile(quiet.to_numpy(), threshold_pct))

        pre = idx[(idx.index >= onset - pd.Timedelta("60D")) & (idx.index <= onset)]
        if pre.empty:
            results.append({"event": name, "onset": start, "status": "no index coverage"})
            continue
        crossed = pre[pre > thr]
        entry = {
            "event": name, "onset": start,
            "threshold": round(thr, 2),
            "peak_pre_onset": round(float(pre.max()), 2),
            "detected": bool(len(crossed)),
            "lead_days": int((onset - crossed.index[0]).days) if len(crossed) else None,
            "first_cross": str(crossed.index[0].date()) if len(crossed) else None,
            "n_days_covered": int(len(pre)),
        }
        results.append(entry)

    scoreable = [r for r in results if r.get("detected") is not None]
    detected = [r for r in scoreable if r["detected"]]
    unscoreable = [r for r in results if r.get("detected") is None]
    # Lead of 0 means the index crossed ON the onset. That is a detection but
    # NOT early warning, and averaging it into a "median lead" flatters the
    # result — so usable lead is counted separately.
    usable = [r for r in detected if (r.get("lead_days") or 0) > 0]
    leads = [r["lead_days"] for r in usable]
    return {
        "method": ("causal leave-one-event-out; threshold from quiet days "
                   "excluding all events; events without sufficient prior "
                   "history are UNSCOREABLE rather than counted as misses"),
        "threshold_pct": threshold_pct,
        "n_events_total": len(EVENTS),
        "n_scoreable": len(scoreable),
        "n_unscoreable": len(unscoreable),
        "n_detected": len(detected),
        "n_with_usable_lead": len(usable),
        "median_usable_lead_days": int(np.median(leads)) if leads else None,
        "headline": (
            f"{len(usable)} of {len(scoreable)} scoreable events detected with "
            f"usable lead"
            + (f" (median {int(np.median(leads))} days)" if leads else "")
            + f"; {len(unscoreable)} event(s) unscoreable for lack of prior history."
        ),
        "folds": results,
    }


def gbm_comparison(fused: pd.DataFrame, arms: pd.DataFrame) -> dict:
    """
    ILLUSTRATIVE ONLY — a GBM fitted on 5 events is fitting noise.

    Reported so the comparison exists and so the weighted rule is not the only
    thing anyone has seen, but it must never be presented as the shipped index.
    Trained leave-one-event-out; per-fold scores are given individually because
    averaging five numbers from five single-event folds hides their spread.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    X = arms.reindex(fused.index).ffill(limit=5)
    y = pd.Series(0, index=fused.index)
    for name, (s, e) in EVENTS.items():
        y[(y.index >= pd.Timestamp(s) - pd.Timedelta("30D")) & (y.index <= pd.Timestamp(e))] = 1

    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    if y.sum() < 2 or len(X) < 200:
        return {"status": "insufficient data", "n_rows": int(len(X)), "n_pos": int(y.sum())}

    folds = []
    for name, (s, e) in EVENTS.items():
        held = (X.index >= pd.Timestamp(s) - pd.Timedelta("30D")) & \
               (X.index <= pd.Timestamp(e) + pd.Timedelta("30D"))
        if held.sum() < 10 or y[~held].sum() < 2:
            continue
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=2,
                                         learning_rate=0.05, random_state=42)
        clf.fit(X[~held], y[~held])
        p = clf.predict_proba(X[held])[:, 1]
        yt = y[held].to_numpy()
        if yt.min() == yt.max():
            folds.append({"event": name, "auc": None, "note": "single-class fold"})
            continue
        pooled = pd.Series(p).rank().to_numpy()
        pos = pooled[yt == 1]
        u = pos.sum() - len(pos) * (len(pos) + 1) / 2
        auc = float(u / (len(pos) * (len(yt) - len(pos))))
        folds.append({"event": name, "auc": round(auc, 3), "n_test": int(held.sum())})

    aucs = [f["auc"] for f in folds if f.get("auc") is not None]
    return {
        "status": "ILLUSTRATIVE ONLY — fitted on 5 events, cannot be validated",
        "n_rows": int(len(X)), "n_positive_days": int(y.sum()),
        "folds": folds,
        "auc_spread": [round(min(aucs), 3), round(max(aucs), 3)] if aucs else None,
        "warning": (
            "Per-fold AUCs are reported individually and deliberately NOT "
            "averaged: five folds each containing one event have a spread that "
            "a mean would hide."
        ),
    }


def build() -> dict:
    arms = load_arms()
    fused = risk_index(arms)

    out = repo_root() / "data" / "derived" / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(out / "risk_index.parquet")

    report = {
        "method": "weighted causal-percentile fusion",
        "weights": WEIGHTS,
        "weight_justification": {
            "arm_c": "AUC 0.848 point events; 3/3 sharp incidents at z=3.4-4.7",
            "arm_b": "AUC 0.680; 38-day held-out lead on the 2026 sustained event",
            "arm_a": "ZERO weight — no measured signal (+0.0% across the onset)",
        },
        "coverage": {
            "start": str(fused.index.min().date()),
            "end": str(fused.index.max().date()),
            "days_with_index": int(fused["risk_index"].notna().sum()),
            "arms_present": list(arms.columns),
        },
        "backtest_loeo": loeo_backtest(fused),
        "gbm_comparison": gbm_comparison(fused, arms),
    }
    (out / "fusion_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description="Fusion + LOEO backtest").parse_args()
    print(json.dumps(build(), indent=2))
