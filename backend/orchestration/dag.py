"""
Pipeline orchestration — the layer that stitches every arm together.

THE PROBLEM THIS SOLVES. Until now every stage was a manual CLI invocation.
The pipeline existed in a shell history, not in the repository: nobody could
tell which stages were stale, what order they run in, or whether the feature
panel currently on disk reflects the GDELT data currently on disk. That is not
reproducible, and "run these nine commands in the right order" is not an
architecture.

DESIGN — FILESYSTEM-DERIVED STATE, NOT A SCHEDULER DATABASE.
Each stage declares its INPUT paths and OUTPUT paths. Staleness is then a fact
about the filesystem rather than bookkeeping that can drift from reality:

    missing output                      -> STALE  (never run)
    any input newer than oldest output  -> STALE  (upstream moved)
    otherwise                           -> FRESH

There is no state database to corrupt, and deleting an output is enough to
force a rerun. Airflow/Dagster/Prefect would each add a scheduler, a metadata
DB and a web server to express the same nine-node graph — more infrastructure
than this pipeline earns.

EXECUTION ORDER IS DERIVED, NOT DECLARED. Stages name their dependencies and
the runner topologically sorts them, so adding a stage cannot silently break
the order. A cycle raises instead of hanging.

⚠️ EXPENSIVE AND CREDENTIALLED STAGES ARE `manual=True` AND NEVER AUTO-RUN.
Satellite collection spends real quota (~5,947 PU for a full pass, against a
30,000/month HARD CAP where overage is refused rather than billed), and the
GDELT backfill is a multi-hour download. An orchestrator that "helpfully"
re-runs those because a timestamp moved would be actively dangerous. They are
reported in the graph, and skipped unless explicitly named.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..common.paths import repo_root
from ..common.secrets import safe_stdout

log = logging.getLogger(__name__)
ROOT = repo_root()


@dataclass
class Stage:
    name: str
    title: str
    module: str                       # python -m <module>
    args: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)     # globs, repo-relative
    outputs: list[str] = field(default_factory=list)
    manual: bool = False              # never run automatically
    # A GATE exits non-zero to REPORT a validation failure, not to signal a
    # crash. Conflating the two is how a working guard gets "fixed" into
    # silence — the split audit failing the build is design rule 3 doing its
    # job, and downstream stages should still be blocked, but the run must not
    # be reported as an error.
    gate: bool = False
    note: str = ""


# The pipeline. Order here is documentation only — the runner sorts by `needs`.
STAGES: list[Stage] = [
    # ── ingest ─────────────────────────────────────────────────────────────
    Stage("ingest_gdelt", "GDELT events (live poller slot)",
          "backend.ingest.gdelt.poller", ["--once"],
          outputs=["data/raw/gdelt/live/events/dt=*/*.parquet"],
          note="Runs continuously as a container; --once fetches a single slot."),
    Stage("ingest_gdelt_backfill", "GDELT historical backfill",
          "backend.ingest.gdelt.backfill_raw", ["--all"],
          outputs=["data/raw/gdelt/historical/*.parquet"],
          manual=True, note="Hours of download. Run deliberately."),
    Stage("ingest_market", "Market panel (FRED + yfinance)",
          "backend.ingest.market.collect", [],
          outputs=["data/raw/market/market_daily.parquet"],
          note="Needs a FRED key. Daily cadence — no value in running more often."),
    Stage("ingest_satellite", "Sentinel-1 chip collection",
          "backend.ingest.satellite.collect", ["--size", "1024"],
          outputs=["data/raw/satellite/chips/*/dt=*/*.tiff"],
          manual=True,
          note="SPENDS QUOTA (~5,947 PU). Hard cap 30,000/month, overage refused."),

    # ── arm A: SAR ─────────────────────────────────────────────────────────
    Stage("arm_a_cfar", "CFAR detection + water mask",
          "backend.arms.sar.run_cfar", [],
          needs=["ingest_satellite"],
          inputs=["data/raw/satellite/chips/*/dt=*/*.tiff"],
          outputs=["data/derived/sar_cfar/cfar_detections.json"]),
    Stage("arm_a_patches", "CNN patch dataset",
          "backend.arms.sar.patches", [],
          needs=["arm_a_cfar"],
          inputs=["data/derived/sar_cfar/cfar_detections.json"],
          outputs=["data/derived/sar_patches/X.npy"]),
    Stage("arm_a_cnn", "MobileNetV2 transfer learning",
          "backend.arms.sar.cnn", [],
          needs=["arm_a_patches"],
          inputs=["data/derived/sar_patches/X.npy"],
          outputs=["data/models/arm_a_cnn/report.json"],
          note="Currently LOSES to a brightness baseline (0.837 vs 0.946)."),

    # ── arm C: text ────────────────────────────────────────────────────────
    Stage("arm_c_slugs", "Semantic scoring of GDELT URL slugs",
          "backend.arms.text.slugs", [],
          needs=["ingest_gdelt_backfill"],
          inputs=["data/raw/gdelt/historical/*.parquet"],
          outputs=["data/derived/text/slug_features_daily.parquet"]),

    # ── features ───────────────────────────────────────────────────────────
    Stage("features", "Common daily feature panel + leakage checks",
          "backend.features.build", [],
          needs=["ingest_gdelt", "ingest_market", "arm_a_cfar"],
          inputs=["data/raw/market/market_daily.parquet",
                  "data/derived/sar_cfar/cfar_detections.json"],
          outputs=["data/derived/features_daily.parquet"]),

    # ── arm B: market ──────────────────────────────────────────────────────
    Stage("arm_b_vae", "Sequence VAE (LSTM) training",
          "backend.arms.market.vae", [],
          needs=["features"],
          inputs=["data/derived/features_daily.parquet"],
          outputs=["data/models/arm_b_vae/encoder.keras",
                   "data/models/arm_b_vae/report.json"]),
    Stage("arm_b_eval", "VAE evaluation — AUC, lead time, specificity",
          "backend.arms.market.evaluate", [],
          needs=["arm_b_vae"],
          inputs=["data/models/arm_b_vae/scores.parquet"],
          outputs=["data/models/arm_b_vae/evaluation.json"]),

    # ── validation ─────────────────────────────────────────────────────────
    Stage("splits", "Temporal split audit (embargo + event straddle)",
          "backend.features.splits", [],
          needs=["features"],
          inputs=["data/derived/features_daily.parquet"],
          outputs=[], gate=True,
          note="GATE: fails the build when a fold straddles an event or a test "
               "window contains none. Currently FAILING — 2 folds are event-free."),
]

BY_NAME = {s.name: s for s in STAGES}


# ─────────────────────────────────────────────────────────── graph utils
def toposort(stages: list[Stage]) -> list[Stage]:
    """Dependency order. Raises on a cycle rather than looping forever."""
    order, seen, visiting = [], set(), set()

    def visit(s: Stage):
        if s.name in seen:
            return
        if s.name in visiting:
            raise ValueError(f"dependency cycle at '{s.name}'")
        visiting.add(s.name)
        for dep in s.needs:
            if dep in BY_NAME:
                visit(BY_NAME[dep])
        visiting.discard(s.name)
        seen.add(s.name)
        order.append(s)

    for s in stages:
        visit(s)
    return order


def _newest(globs: list[str]) -> float | None:
    """Most recent mtime across a set of globs, or None if nothing matches."""
    best = None
    for g in globs:
        for p in ROOT.glob(g):
            if p.is_file():
                m = p.stat().st_mtime
                best = m if best is None else max(best, m)
    return best


def _oldest(globs: list[str]) -> float | None:
    best = None
    for g in globs:
        for p in ROOT.glob(g):
            if p.is_file():
                m = p.stat().st_mtime
                best = m if best is None else min(best, m)
    return best


def _count(globs: list[str]) -> int:
    return sum(1 for g in globs for p in ROOT.glob(g) if p.is_file())


def state(s: Stage) -> dict:
    """FRESH / STALE / MISSING_INPUT for one stage, derived from the filesystem."""
    n_out = _count(s.outputs)
    if not s.outputs:
        status, reason = "ALWAYS_RUN", "reports only, no persisted output"
    elif n_out == 0:
        status, reason = "STALE", "no output on disk"
    else:
        out_t = _oldest(s.outputs)
        in_t = _newest(s.inputs) if s.inputs else None
        if in_t is not None and in_t > out_t:
            status, reason = "STALE", "input newer than output"
        else:
            status, reason = "FRESH", "output up to date"

    if s.inputs and _count(s.inputs) == 0:
        status, reason = "MISSING_INPUT", "declared inputs not present"

    return {"name": s.name, "title": s.title, "status": status, "reason": reason,
            "manual": s.manual, "needs": s.needs,
            "n_outputs": n_out, "n_inputs": _count(s.inputs) if s.inputs else None,
            "note": s.note}


_GRAPH_CACHE: dict = {"t": 0.0, "v": None}
_GRAPH_TTL_S = 20.0


def graph(use_cache: bool = True) -> dict:
    """
    DAG state. Cached briefly because computing it globs the raw store, and
    the live GDELT partition alone holds 45,000+ slot files — a measured 1.9s
    per uncached call. Pipeline state does not change second-to-second, so a
    20s TTL costs nothing in accuracy and keeps the dashboard responsive.
    """
    if use_cache and _GRAPH_CACHE["v"] is not None             and time.time() - _GRAPH_CACHE["t"] < _GRAPH_TTL_S:
        return _GRAPH_CACHE["v"]
    ordered = toposort(STAGES)
    rows = [state(s) for s in ordered]
    out = {
        "n_stages": len(rows),
        "order": [r["name"] for r in rows],
        "runnable_now": [r["name"] for r in rows
                         if r["status"] in ("STALE", "ALWAYS_RUN") and not r["manual"]],
        "manual_only": [r["name"] for r in rows if r["manual"]],
        "stages": rows,
    }
    _GRAPH_CACHE.update(t=time.time(), v=out)
    return out


# ───────────────────────────────────────────────────────────── execution
def run_stage(s: Stage, timeout: int = 7200) -> dict:
    cmd = [sys.executable, "-m", s.module, *s.args]
    log.info("RUN  %-22s %s", s.name, " ".join(cmd[2:]))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    ok = proc.returncode == 0
    log.info("%s %-22s %.1fs", "OK  " if ok else "FAIL", s.name, dt)
    if not ok:
        # Surface the real error. A swallowed exception here is exactly how a
        # deterministic bug gets misdiagnosed as a transient one.
        log.error("stderr tail:\n%s", (proc.stderr or "")[-1200:])
    return {"stage": s.name, "ok": ok, "kind": "gate" if s.gate else "stage",
            "gate_failed": bool(s.gate and not ok),
            "returncode": proc.returncode,
            "seconds": round(dt, 1), "stderr_tail": (proc.stderr or "")[-600:] if not ok else ""}


def run(only: list[str] | None = None, force: bool = False,
        include_manual: bool = False, dry_run: bool = False) -> dict:
    targets = [BY_NAME[n] for n in only] if only else STAGES
    ordered = toposort(targets)
    results, skipped = [], []

    for s in ordered:
        st = state(s)
        explicitly_named = bool(only and s.name in only)
        if s.manual and not (include_manual or explicitly_named):
            skipped.append({"stage": s.name, "why": "manual — spends quota or hours"})
            continue
        if st["status"] == "MISSING_INPUT":
            skipped.append({"stage": s.name, "why": st["reason"]})
            continue
        if st["status"] == "FRESH" and not force:
            skipped.append({"stage": s.name, "why": "fresh"})
            continue
        if dry_run:
            results.append({"stage": s.name, "would_run": True, "reason": st["reason"]})
            continue
        r = run_stage(s)
        results.append(r)
        if not r["ok"]:
            if s.gate:
                log.warning("GATE '%s' reported validation problems — this is the "
                            "guard working, not a crash. Downstream still blocked.", s.name)
            # Stop on failure: downstream stages consume this stage's output,
            # and running them on stale inputs produces numbers that look valid.
            log.error("stopping — '%s' failed and downstream depends on it", s.name)
            break

    gates_failed = [r["stage"] for r in results if r.get("gate_failed")]
    return {"ran": results, "skipped": skipped,
            "gates_failed": gates_failed,
            "ok": all(r.get("ok", True) for r in results if not r.get("gate_failed")),
            "validation_clean": not gates_failed}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hormuz pipeline orchestration")
    p.add_argument("--status", action="store_true", help="show the DAG and exit")
    p.add_argument("--run", action="store_true", help="execute stale stages")
    p.add_argument("--only", default=None, help="comma-separated stage names")
    p.add_argument("--force", action="store_true", help="run even if fresh")
    p.add_argument("--include-manual", action="store_true",
                   help="also run quota-spending / multi-hour stages")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    if a.run:
        only = a.only.split(",") if a.only else None
        res = run(only, a.force, a.include_manual, a.dry_run)
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1

    g = graph()
    print(f"\n  PIPELINE — {g['n_stages']} stages\n")
    print(f"  {'stage':<24} {'status':<14} {'out':>5}  note")
    print("  " + "-" * 78)
    for r in g["stages"]:
        flag = " [manual]" if r["manual"] else ""
        print(f"  {r['name']:<24} {r['status']:<14} {r['n_outputs']:>5}  {r['reason']}{flag}")
    print(f"\n  runnable now : {', '.join(g['runnable_now']) or '(none — all fresh)'}")
    print(f"  manual only  : {', '.join(g['manual_only'])}\n")
    return 0


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
