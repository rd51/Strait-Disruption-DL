"""
Arm B evaluation — does the VAE anomaly score actually detect anything?

An unsupervised model trained without labels can still be evaluated WITH them,
and must be. Reconstruction error that does not separate crisis windows from
calm ones is a number, not a detector.

Three questions, in increasing order of how easy it is to fool yourself:

  1. SEPARATION — do Hormuz windows score above calm windows at all?
     Reported as AUC. Note this is a DIAGNOSTIC, not a backtest score: the
     threshold comes from the training distribution, and the windows being
     scored were held out of training but not out of feature construction.

  2. LEAD TIME — the question the project exists to answer. On what date does
     the score first cross threshold relative to the labelled onset? A detector
     that fires ON the onset has zero warning value.

  3. SPECIFICITY — does it fire on COVID too? Almost certainly yes, and saying
     so is the point. A market anomaly detector detects market anomalies; what
     makes a detection HORMUZ-specific is the Brent-WTI spread, so that is
     measured separately rather than assumed.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ...common.paths import repo_root
from ...features.splits import EVENTS
from .vae import NON_HORMUZ_EXCLUSIONS

log = logging.getLogger(__name__)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """
    Rank-based AUC via the Mann-Whitney U identity.

    AUC = P(score_pos > score_neg), estimated as
        (U) / (n_pos · n_neg),  U = R_pos − n_pos(n_pos+1)/2
    where R_pos is the sum of the positives' ranks in the pooled sample. Ties
    get average ranks, which is what makes this equal the trapezoidal ROC area
    rather than approximating it.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pooled = np.concatenate([pos, neg])
    ranks = pd.Series(pooled).rank().to_numpy()
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def evaluate() -> dict:
    d = repo_root() / "data" / "models" / "arm_b_vae"
    s = pd.read_parquet(d / "scores.parquet")
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)
    rep = json.loads((d / "report.json").read_text())
    thr95, thr99 = rep["threshold_p95"], rep["threshold_p99"]

    calm = s[s.is_train_window]
    hormuz = s[s.is_hormuz_window]
    other = s[s.is_declared_other]

    out: dict = {
        "thresholds": {"p95": thr95, "p99": thr99},
        "separation": {
            "calm_median": float(calm.recon_error.median()),
            "hormuz_median": float(hormuz.recon_error.median()),
            "covid_median": float(other.recon_error.median()) if len(other) else None,
            "auc_hormuz_vs_calm": _auc(hormuz.recon_error.to_numpy(),
                                       calm.recon_error.to_numpy()),
            "calm_pct_above_p95": float((calm.recon_error > thr95).mean() * 100),
            "hormuz_pct_above_p95": float((hormuz.recon_error > thr95).mean() * 100),
            "covid_pct_above_p95": float((other.recon_error > thr95).mean() * 100)
            if len(other) else None,
        },
        "lead_time": [],
        "specificity": {},
    }

    # ── 2. lead time, per event ────────────────────────────────────────────
    # First crossing inside a 60-day pre-onset window. Searching only BEFORE
    # the onset is deliberate: a crossing after the fact is not a warning.
    for name, (start, end) in EVENTS.items():
        onset = pd.Timestamp(start)
        pre = s[(s.date >= onset - pd.Timedelta("60D")) & (s.date <= onset)]
        if pre.empty:
            out["lead_time"].append({"event": name, "onset": start,
                                     "status": "no market data in window"})
            continue
        for label, thr in (("p95", thr95), ("p99", thr99)):
            crossed = pre[pre.recon_error > thr]
            entry = {
                "event": name, "onset": start, "threshold": label,
                "peak_score_in_window": float(pre.recon_error.max()),
            }
            if crossed.empty:
                entry.update(first_cross=None, lead_days=None, detected=False)
            else:
                first = crossed.date.iloc[0]
                entry.update(first_cross=str(first.date()),
                             lead_days=int((onset - first).days),
                             detected=True)
            out["lead_time"].append(entry)

    # ── 3. specificity ─────────────────────────────────────────────────────
    # Every window above p99 that is NEITHER a Hormuz event NOR a declared
    # shock is, by construction, a false positive.
    fp = s[(s.recon_error > thr99) & ~s.is_hormuz_window & ~s.is_declared_other]
    out["specificity"] = {
        "false_positive_windows": int(len(fp)),
        "false_positive_rate_pct": float(len(fp) / len(s) * 100),
        "false_positive_dates": [str(x.date()) for x in fp.date.head(15)],
        "covid_fires": bool(len(other) and (other.recon_error > thr99).any()),
    }
    return out


if __name__ == "__main__":
    from ...common.secrets import safe_stdout
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    r = evaluate()
    (repo_root() / "data" / "models" / "arm_b_vae" / "evaluation.json").write_text(
        json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
