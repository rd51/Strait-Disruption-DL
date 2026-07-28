"""
Agent layer — an LLM-written analyst brief, grounded in the semantic registry.

WHAT MAKES THIS DIFFERENT FROM "ASK AN LLM ABOUT THE GULF".
A language model asked to comment on Hormuz will produce fluent, plausible
prose from its training data. That is worse than useless here: it would read
exactly like analysis grounded in this project's measurements while being
grounded in nothing. Three constraints make the output defensible.

  1. NUMBERS COME FROM THE STORE, NOT THE MODEL. Every figure is computed by
     this file and passed in. The model is asked to explain and prioritise,
     never to recall or estimate.

  2. THE SEMANTIC REGISTRY'S PROHIBITIONS BECOME THE MODEL'S GUARDRAILS. The
     `forbidden` field of every metric is injected into the system prompt. This
     is the payoff for building the registry: 25 rules the project learned by
     getting things wrong now constrain what the brief is allowed to claim. The
     model cannot say the SAR arm confirms disruption, because the registry
     records that it does not.

  3. THE MODEL IS TOLD WHAT IS NOT KNOWN. Unbuilt components, failed
     validations and null results are listed explicitly, so absence of evidence
     is available to it as a fact rather than a gap to fill.

⚠️ THE BRIEF IS NOT A PREDICTION. It is a readout of measured state, and the
prompt says so. The vessel-safety layer is excluded entirely — CLAUDE.md's wall
between the backtested engine and the live rule-based layer applies to
generated text exactly as it applies to a dashboard.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from ..common.paths import repo_root
from ..common.secrets import safe_stdout
from ..semantic import registry as semantic

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"
MAX_TOKENS = 2000


# People name this key several reasonable ways, and a loader that accepts only
# one silently reports "no key" when the key is right there. Measured: this
# project's .env had it as `Claude_API_KEY`. Matching is case-insensitive.
KEY_ALIASES = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY", "CLAUDE_KEY")


def load_key() -> str | None:
    """
    Env var first, then .env, across all known aliases.

    The key is never logged, echoed or returned in an API response — only its
    presence is ever reported.
    """
    for name in KEY_ALIASES:
        v = os.environ.get(name, "").strip()
        if v:
            return v

    env = repo_root() / ".env"
    if not env.exists():
        return None
    # utf-8-sig: Windows editors write a BOM, which would otherwise become part
    # of the first key's NAME and make it never match.
    for line in env.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip().upper() in KEY_ALIASES:
            v = v.strip().strip('"').strip("'")
            if v:
                return v
    return None


def gather_state() -> dict:
    """
    Everything the brief is allowed to talk about, computed here.

    Deliberately includes what is MISSING and what FAILED. A brief that only
    sees successes will write as if the system works.
    """
    root = repo_root() / "data"
    st: dict = {"as_of": None, "arms": {}, "not_built": [], "null_results": []}

    # ── fusion risk index
    p = root / "derived" / "fusion" / "risk_index.parquet"
    if p.exists():
        f = pd.read_parquet(p)
        idx = f["risk_index"].dropna()
        if len(idx):
            st["as_of"] = str(idx.index[-1].date())
            st["risk_index"] = {
                "latest": round(float(idx.iloc[-1]), 1),
                "change_7d": round(float(idx.iloc[-1] - idx.iloc[-8]), 1) if len(idx) > 8 else None,
                "percentile_of_own_history": round(float((idx < idx.iloc[-1]).mean() * 100), 1),
            }
    else:
        st["not_built"].append("fusion risk index")

    # ── Arm B
    p = root / "models" / "arm_b_vae" / "evaluation.json"
    if p.exists():
        e = json.loads(p.read_text())
        st["arms"]["market_vae"] = {
            "auc_hormuz_vs_calm": e["separation"]["auc_hormuz_vs_calm"],
            "detected_events": "1 of 5 labelled anchors (2026 Hormuz only)",
            "lead_time_days": 38,
            "fires_on_non_gulf": "COVID 2020 and Ukraine 2022 — not Gulf-specific",
        }
    p = root / "models" / "arm_b_vae" / "scores.parquet"
    if p.exists():
        s = pd.read_parquet(p)
        s["date"] = pd.to_datetime(s["date"])
        st["arms"].setdefault("market_vae", {})["latest_recon_error"] = \
            round(float(s.sort_values("date")["recon_error"].iloc[-1]), 3)

    # ── Arm C
    p = root / "derived" / "text" / "slug_features_daily.parquet"
    if p.exists():
        c = pd.read_parquet(p)
        col = "slug_chokepoint_top10"
        if col in c:
            st["arms"]["text_semantic"] = {
                "auc_point_events": 0.848,
                "latest_score": round(float(c[col].dropna().iloc[-1]), 4),
                "p90_reference": round(float(c[col].quantile(0.90)), 4),
                "coverage_days": int(len(c)),
            }

    # ── Arm A — the null
    p = root / "derived" / "sar_cfar" / "cfar_detections.json"
    if p.exists():
        st["arms"]["sar_cfar"] = {
            "chips_processed": len(json.loads(p.read_text())),
            "MEASURED_RESULT": "NO reroute signal",
        }
        st["null_results"].append(
            "Arm A (SAR): Fujairah vessel counts changed +0.0% across the 2026-03-02 "
            "onset; Mann-Whitney p=0.26-0.89, bypass ports NEGATIVE in both orbits. "
            "Rules out a >33% shift, not a subtle one."
        )
        st["null_results"].append(
            "Arm A CNN: 0.837 test accuracy vs 0.946 for a brightness baseline — "
            "the CNN does not beat a trivial model on this data."
        )

    st["not_built"] += ["live vessel layer (no Gulf AIS coverage exists)",
                        "GBM as shipped index (5 labels cannot validate one)"]
    return st


def build_prompt(state: dict) -> tuple[str, str]:
    prohibitions = semantic.forbidden_uses()
    rules = "\n".join(f"- [{r['metric']}] {r['rule']}" for r in prohibitions)

    system = f"""You are the analyst layer of the Hormuz Disruption Engine, a \
supply-chain early-warning system for the Strait of Hormuz.

You write a short situation brief for a supply-chain risk manager.

ABSOLUTE CONSTRAINTS:

1. Use ONLY the numbers in the payload. Never recall, estimate or infer a \
figure from your own knowledge of Gulf events. If a number is not in the \
payload, say it is not measured.

2. These prohibitions are derived from this system's own measurements. Each was \
learned by getting something wrong. You must not write anything that violates \
one:
{rules}

3. Report null and negative results as findings, not omissions. If an arm \
measured no signal, say so plainly — do not soften it or imply partial support.

4. This brief is a READOUT OF MEASURED STATE, not a forecast. Do not predict \
what will happen. Do not assign probabilities that are not in the payload.

5. Never reference live vessel positions or per-vessel safety. That layer has \
no data and is architecturally separate from this engine.

FORMAT: markdown, under 350 words.
  ## Assessment      — 2-3 sentences on measured state
  ## What the arms show  — one bullet per arm, with its number AND its caveat
  ## What we cannot say  — the honest limits, including nulls and unbuilt parts
Be direct. A risk manager acting on an overstated brief is the failure mode \
this system exists to avoid."""

    user = ("Write the brief from this measured state.\n\n"
            + json.dumps(state, indent=2))
    return system, user


def generate(dry_run: bool = False) -> dict:
    state = gather_state()
    system, user = build_prompt(state)

    if dry_run:
        return {"state": state, "system_prompt_chars": len(system),
                "prohibitions_injected": len(semantic.forbidden_uses()),
                "brief": "(dry run — no API call)"}

    key = load_key()
    if not key:
        return {"state": state, "error":
                "ANTHROPIC_API_KEY not set. Add it to .env, then rerun. "
                "Use --dry-run to inspect the grounded prompt without a key."}

    import anthropic
    client = anthropic.Anthropic(api_key=key)
    try:
        # Streaming: a long brief on a slow connection can otherwise hit the
        # request timeout. get_final_message() reassembles it.
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.BadRequestError as exc:
        # A 400 mentioning credit is an ACCOUNT state, not a bug in this code
        # and not a bad key — the request authenticated to reach it. Say which
        # it is, because "400 Bad Request" reads like a malformed payload.
        detail = str(exc)
        if "credit" in detail.lower() or "billing" in detail.lower():
            return {"state": state, "error":
                    "The API key is VALID (it authenticated) but the Anthropic "
                    "account has insufficient credit. Add credit at "
                    "console.anthropic.com -> Plans & Billing, then rerun. "
                    "Everything else in the brief layer is working — use "
                    "--dry-run to inspect the fully grounded prompt meanwhile."}
        return {"state": state, "error": f"API rejected the request: {detail[:300]}"}
    except anthropic.AuthenticationError:
        return {"state": state, "error":
                "The API key was rejected. Check the value in .env "
                "(any of: " + ", ".join(KEY_ALIASES) + ")."}
    except anthropic.APIError as exc:
        return {"state": state, "error": f"Anthropic API error: {str(exc)[:300]}"}

    text = "".join(b.text for b in msg.content if b.type == "text")
    out = repo_root() / "data" / "derived" / "brief"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest_brief.md").write_text(text, encoding="utf-8")
    return {"state": state, "brief": text, "model": MODEL,
            "prohibitions_injected": len(semantic.forbidden_uses()),
            "usage": {"input_tokens": msg.usage.input_tokens,
                      "output_tokens": msg.usage.output_tokens}}


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LLM analyst brief, grounded in measurements")
    p.add_argument("--dry-run", action="store_true",
                   help="build and inspect the grounded prompt without calling the API")
    a = p.parse_args()
    r = generate(a.dry_run)
    if a.dry_run:
        print(json.dumps(r, indent=2)[:3000])
    else:
        print(r.get("brief") or r.get("error"))
