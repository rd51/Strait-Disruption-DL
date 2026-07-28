"""
Launch and status for the whole backend.

    python -m backend.launch            # bring everything up, then report
    python -m backend.launch --status   # report only, change nothing
    python -m backend.launch --down     # stop the containerised services

DEPLOYMENT SPLIT, AND WHY IT IS DELIBERATE.
The collectors and the API do NOT belong in the same container:

  · the GDELT poller is a 24/7 service whose image is kept lean on purpose
    (pandas, requests, pyarrow, websockets — four packages). It restarts
    unattended, so a small image is a real operational property.

  · the API now performs live model inference, which drags in TensorFlow,
    torch, sentence-transformers and rasterio — roughly 6 GB. Putting that in
    the poller image would multiply a 603 MB always-on container by ten to
    carry libraries it never imports.

So the poller runs in Docker and the API runs where the ML environment lives.
Both are the same codebase; only the runtime differs. `--build-api-image`
exists for when the API genuinely needs containerising, and is not the default
because the build is long and the artifact large.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .common.paths import repo_root
from .common.secrets import safe_stdout

API_HOST, API_PORT = "127.0.0.1", 8000
ROOT = repo_root()


def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _docker(*args: str, timeout: int = 600) -> tuple[int, str]:
    if not shutil.which("docker"):
        return 127, "docker not on PATH"
    p = subprocess.run(["docker", *args], cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def docker_up() -> dict:
    """Start the containerised collectors. A dead daemon is reported, not raised."""
    rc, out = _docker("ps", "--format", "{{.Names}}")
    if rc == 127:
        return {"ok": False, "detail": "docker not installed"}
    if "cannot find the file specified" in out or "daemon" in out.lower() and rc != 0:
        # Measured twice in this project: the DAEMON dies while containers are
        # listed as running. Distinguish it from a stopped container, because
        # the fix is different (start Docker Desktop, not `compose up`).
        return {"ok": False, "detail": "Docker daemon is DOWN — start Docker Desktop"}

    rc, out = _docker("compose", "up", "-d", "gdelt-poller")
    return {"ok": rc == 0, "detail": out.strip().splitlines()[-1] if out.strip() else ""}


def api_up() -> dict:
    """Start uvicorn on the host if nothing is already serving the port."""
    if _port_open(API_HOST, API_PORT):
        return {"ok": True, "detail": "already serving", "started": False}
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.main:app",
         "--host", API_HOST, "--port", str(API_PORT)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        if _port_open(API_HOST, API_PORT):
            return {"ok": True, "detail": "started", "started": True}
        time.sleep(1)
    return {"ok": False, "detail": "did not come up within 40s", "started": False}


def _get(path: str, timeout: int = 90):
    with urllib.request.urlopen(f"http://{API_HOST}:{API_PORT}{path}", timeout=timeout) as r:
        return json.load(r)


def status() -> dict:
    st: dict = {"api": {}, "containers": [], "arms": {}, "pipeline": {}, "models": []}

    rc, out = _docker("ps", "--format", "{{.Names}}|{{.Status}}")
    if rc == 0:
        st["containers"] = [dict(zip(("name", "status"), ln.split("|", 1)))
                            for ln in out.strip().splitlines() if "|" in ln]
    elif rc == 127:
        st["containers"] = [{"name": "docker", "status": "not installed"}]
    else:
        st["containers"] = [{"name": "docker", "status": "daemon DOWN"}]

    if not _port_open(API_HOST, API_PORT):
        st["api"] = {"up": False}
        return st

    st["api"] = {"up": True, "url": f"http://{API_HOST}:{API_PORT}"}
    try:
        f = _get("/freshness")
        st["arms"] = {k: {"status": v.get("status"), "age_h": v.get("age_hours")}
                      for k, v in f["arms"].items()}
    except Exception as exc:                            # noqa: BLE001
        st["arms"] = {"error": str(exc)[:120]}
    try:
        v = _get("/pipeline/volumes")
        st["pipeline"] = {
            "gdelt_slots": v["gdelt_live_slots"], "gdelt_mb": v["gdelt_live_mb"],
            "sar_chips": v["sar_chips"], "rows_per_sec": v["ingest_rate"]["rows_per_second"],
        }
    except Exception as exc:                            # noqa: BLE001
        st["pipeline"] = {"error": str(exc)[:120]}
    try:
        st["models"] = [{"name": m["name"], "status": m["status"]}
                        for m in _get("/models")["models"]]
    except Exception:                                   # noqa: BLE001
        pass
    try:
        r = _get("/risk?start=2026-06-01")
        st["risk"] = {"latest": r["latest"],
                      "detected": r["backtest_loeo"]["n_detected"],
                      "headline": r["backtest_loeo"]["headline"]}
    except Exception:                                   # noqa: BLE001
        pass
    return st


def report(st: dict) -> None:
    P = print
    P("\n  " + "=" * 66)
    P("  HORMUZ DISRUPTION ENGINE — BACKEND")
    P("  " + "=" * 66)

    P("\n  CONTAINERS")
    if st["containers"]:
        for c in st["containers"]:
            P(f"    {c['name']:<26} {c['status']}")
    else:
        P("    (none running)")

    a = st["api"]
    P("\n  API")
    if a.get("up"):
        P(f"    serving      {a['url']}")
        P(f"    dashboard    {a['url']}/")
        P(f"    openapi      {a['url']}/docs")
    else:
        P("    DOWN")
        return

    P("\n  ARM FRESHNESS")
    for k, v in st["arms"].items():
        if k == "error":
            P(f"    error: {v}")
            continue
        age = "no data" if v["age_h"] is None else f"{v['age_h']:.1f}h"
        P(f"    {k:<12} {str(v['status']):<12} {age}")

    p = st.get("pipeline") or {}
    if "error" not in p and p:
        P("\n  DATA ON DISK")
        P(f"    GDELT slots  {p['gdelt_slots']:,} ({p['gdelt_mb']:,.0f} MB)")
        P(f"    SAR chips    {p['sar_chips']}")
        P(f"    ingest rate  {p['rows_per_sec']} rows/sec")

    if st.get("models"):
        P("\n  MODELS (loaded on first inference request, then cached)")
        for m in st["models"]:
            P(f"    {m['name']:<16} {m['status']}")

    if st.get("risk"):
        r = st["risk"]
        P("\n  RISK INDEX")
        P(f"    latest       {r['latest']}")
        P(f"    backtest     {r['headline']}")

    P("\n  " + "-" * 66)
    P("  Inference endpoints are LIVE — the backend runs models, not stored numbers:")
    P("    POST /predict/text     multilingual encoder, zero-shot, any language")
    P("    GET  /predict/market   VAE on the trailing 20-trading-day window")
    P("    POST /predict/warm     preload both (first call is ~30s otherwise)")
    P("  " + "-" * 66 + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Launch the Hormuz backend")
    p.add_argument("--status", action="store_true", help="report only")
    p.add_argument("--down", action="store_true", help="stop containerised services")
    p.add_argument("--no-warm", action="store_true", help="skip model preloading")
    a = p.parse_args(argv)

    if a.down:
        rc, out = _docker("compose", "down")
        print(out.strip() or f"exit {rc}")
        return 0

    if not a.status:
        d = docker_up()
        print(f"  docker : {'ok' if d['ok'] else 'FAILED'} — {d['detail']}")
        r = api_up()
        print(f"  api    : {'ok' if r['ok'] else 'FAILED'} — {r['detail']}")
        if r["ok"] and not a.no_warm:
            # Preload so the first user request is not a 30s cold start.
            try:
                req = urllib.request.Request(
                    f"http://{API_HOST}:{API_PORT}/predict/warm", method="POST")
                with urllib.request.urlopen(req, timeout=300) as resp:
                    print(f"  models : {json.load(resp)}")
            except Exception as exc:                    # noqa: BLE001
                print(f"  models : warm-up skipped ({str(exc)[:80]})")

    report(status())
    return 0


if __name__ == "__main__":
    safe_stdout()
    raise SystemExit(main())
