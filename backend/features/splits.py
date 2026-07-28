"""
Temporal splits — TimeSeriesSplit with an embargo, plus the leakage gate.

WHY PLAIN TimeSeriesSplit IS NOT ENOUGH HERE. sklearn's TimeSeriesSplit gives
expanding train / forward test folds, which prevents the obvious sin of
training on the future. But two project-specific hazards survive it:

  1. FEATURE-WINDOW BLEED. `brent_vol_7d` on the first test day is computed
     from the preceding 7 days — which are in TRAIN. The feature therefore
     carries training information across the boundary. The standard fix is an
     EMBARGO: drop a gap of days between train and test at least as long as the
     longest feature lookback.

  2. EVENT SPLITTING. With ~5 labelled events, a fold boundary landing inside a
     crisis puts its beginning in train and its end in test. The model then
     "predicts" an event it has already partly seen — which looks like skill and
     is not. Folds must be checked against the event calendar, not just dates.

Both are enforced here rather than left as comments, because with this few
labels a single leak is not a small bias — it can be the whole result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Longest feature lookback in features/build.py is the 252-day rolling z-score.
# The embargo only needs to cover the windows that touch the boundary; 252 days
# would consume the dataset, so embargo on the practical horizon and treat the
# long z-score as a documented approximation.
DEFAULT_EMBARGO_DAYS = 10

# Known events. Onset dates matter, not headline peaks — see CLAUDE.md for why
# the 2026 anchor is 2026-03-02 (price onset) and not 2026-06-12 (the headline).
EVENTS = {
    "2019_fujairah_attacks": ("2019-05-12", "2019-05-12"),
    "2019_gulf_of_oman": ("2019-06-13", "2019-06-13"),
    "2019_abqaiq": ("2019-09-14", "2019-09-16"),
    "2024_red_sea": ("2023-11-19", "2024-03-31"),
    "2026_hormuz": ("2026-03-02", "2026-06-26"),
}


def time_series_splits(
    index: pd.DatetimeIndex,
    n_splits: int = 5,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    min_train_days: int = 180,
) -> list[dict]:
    """
    Expanding-window splits with an embargo gap between train and test.

    Returns dicts rather than raw index arrays so every fold carries its own
    provenance — the boundaries are auditable after the fact instead of being
    implicit in an integer array.
    """
    index = pd.DatetimeIndex(index).sort_values()
    n = len(index)
    if n < min_train_days + n_splits:
        raise ValueError(f"only {n} days — too few for {n_splits} splits")

    fold_size = (n - min_train_days) // n_splits
    folds = []
    for i in range(n_splits):
        train_end_pos = min_train_days + i * fold_size
        test_start_pos = train_end_pos
        test_end_pos = min(train_end_pos + fold_size, n)
        if test_start_pos >= n:
            break

        train_end_date = index[train_end_pos - 1]
        embargo_cutoff = train_end_date + pd.Timedelta(embargo_days, unit="D")

        train_idx = index[:train_end_pos]
        test_mask = (index > embargo_cutoff) & (index <= index[test_end_pos - 1])
        test_idx = index[test_mask]
        if len(test_idx) == 0:
            continue

        folds.append({
            "fold": i,
            "train_start": train_idx[0], "train_end": train_idx[-1],
            "train_days": len(train_idx),
            "embargo_days": embargo_days,
            "embargo_start": train_end_date,
            "embargo_end": embargo_cutoff,
            "test_start": test_idx[0], "test_end": test_idx[-1],
            "test_days": len(test_idx),
            "train_index": train_idx, "test_index": test_idx,
        })
    return folds


def leave_one_event_out_splits(
    index: pd.DatetimeIndex,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    lead_window_days: int = 60,
    causal: bool = True,
    min_train_days: int = 365,
) -> list[dict]:
    """
    One fold per labelled event — the split strategy this project actually uses.

    WHY THIS REPLACES UNIFORM FOLDS. Events cluster in 2019, 2024 and 2026, so
    an evenly spaced 5-fold split leaves folds 1 and 2 testing on windows
    (2020-02 -> 2021-10, 2021-10 -> 2023-06) containing NO labelled event.
    Scoring those says nothing about early-warning skill, and averaging them
    into a headline number quotes an accuracy for periods where nothing
    happened. LOEO gives one event per fold BY CONSTRUCTION, so every fold is
    informative and the empty-fold problem cannot arise.

    THE TEST WINDOW STARTS BEFORE THE ONSET. `lead_window_days` of pre-onset
    days are included, because that is precisely where early warning has to
    happen. A test window beginning at the onset could only ever measure
    detection, never lead time.

    🔴 CAUSAL vs POOLED — AN EXPLICIT CHOICE, NOT A DEFAULT TO IGNORE.
    LOEO is normally run "pooled": train on every other event, including ones
    that occur LATER in time. For a classifier that is standard. For an
    EARLY-WARNING system it is indefensible — a 2019 event scored by a model
    that has seen 2026 is not a forecast, and the lead time it reports could
    not have been achieved in 2019.

      causal=True  (default) train strictly BEFORE the test window. Honest, and
                             the price is that the earliest events have little
                             or no training history and are marked unscoreable.
      causal=False           train on both sides. Uses all five events, but any
                             lead time it reports is retrospective. Every fold
                             is tagged `anti_causal: True` so this cannot be
                             quoted by accident.

    Report which mode produced a number. They are not interchangeable.
    """
    index = pd.DatetimeIndex(index).sort_values()
    emb = pd.Timedelta(embargo_days, unit="D")
    lead = pd.Timedelta(lead_window_days, unit="D")
    folds = []

    for i, (name, (start, end)) in enumerate(sorted(
            EVENTS.items(), key=lambda kv: kv[1][0])):
        onset, closed = pd.Timestamp(start), pd.Timestamp(end)
        test_lo, test_hi = onset - lead, closed
        test_idx = index[(index >= test_lo) & (index <= test_hi)]
        if len(test_idx) == 0:
            folds.append({"fold": i, "event": name, "scoreable": False,
                          "reason": "no feature coverage in the test window",
                          "causal": causal})
            continue

        if causal:
            train_idx = index[index < test_lo - emb]
        else:
            train_idx = index[(index < test_lo - emb) | (index > test_hi + emb)]

        if len(train_idx) < min_train_days:
            folds.append({
                "fold": i, "event": name, "scoreable": False,
                "reason": (f"only {len(train_idx)} training days before this event "
                           f"(need {min_train_days}) — unavoidable under causal "
                           f"LOEO for the earliest events"),
                "causal": causal, "test_days": len(test_idx),
            })
            continue

        folds.append({
            "fold": i, "event": name, "scoreable": True, "causal": causal,
            "anti_causal": not causal,
            "train_start": train_idx[0], "train_end": train_idx[-1],
            "train_days": len(train_idx),
            "embargo_days": embargo_days,
            "test_start": test_idx[0], "test_end": test_idx[-1],
            "test_days": len(test_idx),
            "onset": onset, "lead_window_days": lead_window_days,
            "train_index": train_idx, "test_index": test_idx,
        })
    return folds


def audit_loeo(folds: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Validate LOEO folds. The failures here differ from the uniform-split ones.

    An empty test window is impossible by construction, so the checks that
    matter are: does the held-out event leak into training, is the embargo
    respected, and is the fold anti-causal.
    """
    problems: list[str] = []
    report = []

    for f in folds:
        if not f.get("scoreable"):
            report.append({"fold": f["fold"], "event": f["event"],
                           "status": "UNSCOREABLE", "reason": f["reason"]})
            continue

        tr = events_in(f["train_start"], f["train_end"])
        if f["event"] in tr:
            problems.append(
                f"fold {f['fold']} ({f['event']}): the held-out event appears in "
                "TRAINING — LOEO is violated"
            )
        gap = (f["test_start"] - f["train_end"]).days
        if gap < f["embargo_days"]:
            problems.append(
                f"fold {f['fold']} ({f['event']}): train/test gap {gap}d < "
                f"embargo {f['embargo_days']}d"
            )
        if f.get("anti_causal"):
            problems.append(
                f"fold {f['fold']} ({f['event']}): ANTI-CAUSAL — training data "
                "postdates the test window. Any lead time here is retrospective."
            )

        report.append({
            "fold": f["fold"], "event": f["event"], "status": "OK",
            "train": f"{f['train_start'].date()} -> {f['train_end'].date()} ({f['train_days']}d)",
            "test": f"{f['test_start'].date()} -> {f['test_end'].date()} ({f['test_days']}d)",
            "train_events": tr,
        })

    scoreable = sum(1 for f in folds if f.get("scoreable"))
    if scoreable < 2:
        problems.append(
            f"only {scoreable} scoreable fold(s) — too few to report a backtest"
        )
    return problems, report


def events_in(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    out = []
    for name, (a, b) in EVENTS.items():
        ea, eb = pd.Timestamp(a), pd.Timestamp(b)
        if eb >= start and ea <= end:
            out.append(name)
    return out


def audit_folds(folds: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Check folds for the failures that matter with ~5 labels.

    Returns (problems, per-fold event report).
    """
    problems: list[str] = []
    report = []

    for f in folds:
        tr = events_in(f["train_start"], f["train_end"])
        te = events_in(f["test_start"], f["test_end"])

        # An event straddling the boundary is the dangerous case: partly seen
        # in training, then "predicted" in test.
        straddling = []
        for name in set(tr) & set(te):
            straddling.append(name)
            problems.append(
                f"fold {f['fold']}: event '{name}' spans the train/test boundary "
                f"(train ends {f['train_end'].date()}, test starts {f['test_start'].date()}) "
                "— the model sees part of the event it is scored on"
            )

        if f["train_end"] >= f["test_start"]:
            problems.append(f"fold {f['fold']}: train overlaps test")
        if (f["test_start"] - f["train_end"]).days < f["embargo_days"]:
            problems.append(
                f"fold {f['fold']}: gap {(f['test_start'] - f['train_end']).days}d "
                f"< embargo {f['embargo_days']}d"
            )
        if not te:
            problems.append(
                f"fold {f['fold']}: test window contains NO labelled event — "
                "scoring it says nothing about early-warning skill"
            )

        report.append({
            "fold": f["fold"],
            "train": f"{f['train_start'].date()} -> {f['train_end'].date()} ({f['train_days']}d)",
            "test": f"{f['test_start'].date()} -> {f['test_end'].date()} ({f['test_days']}d)",
            "train_events": tr, "test_events": te, "straddling": straddling,
        })

    return problems, report


def assert_no_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Hard assertion for use inside a training loop or CI."""
    if X_train.index.max() >= X_test.index.min():
        raise AssertionError(
            f"LEAKAGE: train ends {X_train.index.max()} >= test starts {X_test.index.min()}"
        )
    if len(X_train.index.intersection(X_test.index)):
        raise AssertionError("LEAKAGE: train and test share dates")


def main() -> int:
    path = "data/derived/features_daily.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)

    # Split only over days that actually carry features; the full calendar is
    # mostly NaN and folds over empty stretches are meaningless.
    usable = df.dropna(how="all")
    print(f"\n  feature panel {df.shape} | days with any feature: {len(usable):,}")
    print(f"  range {usable.index.min().date()} -> {usable.index.max().date()}\n")

    # ── LOEO is the strategy this project uses. Reported FIRST because it is
    #    the one whose numbers are quotable.
    print("  === LEAVE-ONE-EVENT-OUT (causal) — the strategy in use ===\n")
    loeo = leave_one_event_out_splits(usable.index, causal=True)
    problems, report = audit_loeo(loeo)
    for r in report:
        if r["status"] == "UNSCOREABLE":
            print(f"  fold {r['fold']}  {r['event']:24s} UNSCOREABLE")
            print(f"    {r['reason']}\n")
            continue
        print(f"  fold {r['fold']}  {r['event']}")
        print(f"    train {r['train']}   prior events: {r['train_events'] or '-'}")
        print(f"    test  {r['test']}\n")

    scoreable = sum(1 for f in loeo if f.get("scoreable"))
    print(f"  scoreable folds: {scoreable}/{len(loeo)}")
    print("  LOEO audit:", "PASS" if not problems else f"{len(problems)} PROBLEM(S)")
    for p in problems:
        print(f"    x {p}")

    # ── The uniform split is retained ONLY as a documented counter-example.
    #    It is what the project used first, and its failure mode is the reason
    #    LOEO exists — keeping it visible stops the lesson being lost.
    print("\n  === UNIFORM 5-FOLD (retained as a counter-example, NOT used) ===\n")
    uni_problems, uni_report = audit_folds(time_series_splits(usable.index, n_splits=5))
    empty = [r for r in uni_report if not r["test_events"]]
    print(f"  {len(empty)} of {len(uni_report)} folds have an EVENT-FREE test window:")
    for r in empty:
        print(f"    fold {r['fold']}: test {r['test']} — nothing to detect")
    print(f"  uniform audit: {len(uni_problems)} problem(s) — this is why LOEO is used\n")

    # The gate passes or fails on LOEO. The uniform split's known failure must
    # not keep the pipeline red forever.
    return 1 if problems else 0


if __name__ == "__main__":
    # Windows consoles default to cp1252, which cannot encode the check/cross
    # marks this report prints — the job then dies AFTER doing all its work,
    # with a UnicodeEncodeError that looks nothing like the real cause. This is
    # the THIRD time this bug class has bitten the project (an arrow in the
    # backfill, Arabic text in an inference test, these glyphs here). Any module
    # that prints non-ASCII must call safe_stdout() first.
    from ..common.secrets import safe_stdout
    safe_stdout()
    raise SystemExit(main())
