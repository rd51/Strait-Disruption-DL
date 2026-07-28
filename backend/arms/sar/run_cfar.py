"""
Run the CFAR baseline over the downloaded Sentinel-1 chips.

USAGE
    python -m arms.sar.run_cfar
    python -m arms.sar.run_cfar --port fujairah --k 6
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from ...common.paths import derived_dir, raw_dir
from .cfar import detect_vessels
from .watermask import water_mask

CHIP_ROOT = raw_dir() / "satellite" / "chips"


def load_chip(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict]:
    with rasterio.open(path) as src:
        vv = src.read(1).astype("float64")
        vh = src.read(2).astype("float64") if src.count > 1 else None
        meta = {
            "crs": str(src.crs), "bounds": tuple(src.bounds),
            "width": src.width, "height": src.height,
            "res_deg": (src.bounds.right - src.bounds.left) / src.width,
        }
    return vv, vh, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CFAR vessel-detection baseline")
    p.add_argument("--port", default=None)
    p.add_argument("--k", type=float, default=5.0, help="threshold in background sigmas")
    p.add_argument("--guard", type=int, default=4)
    p.add_argument("--background", type=int, default=12)
    p.add_argument("--min-px", type=int, default=2)
    p.add_argument("--max-px", type=int, default=400)
    p.add_argument("--no-mask", action="store_true",
                   help="skip the water mask (shows how much land inflates the count)")
    args = p.parse_args(argv)

    pattern = str(CHIP_ROOT / (args.port or "*") / "dt=*" / "*.tiff")
    chips = sorted(glob.glob(pattern))
    if not chips:
        print(f"no chips matched {pattern}")
        return 1

    print(f"\n  CFAR baseline | k={args.k} guard={args.guard} bg={args.background} "
          f"size {args.min_px}-{args.max_px}px | water mask: {'OFF' if args.no_mask else 'ON'}\n")
    print(f"  {'port':<12} {'date':<12} {'cover':>6} {'water':>6} {'raw':>6} "
          f"{'on water':>9} {'on land':>8}  note")
    print("  " + "-" * 74)

    results = []
    for chip in chips:
        path = Path(chip)
        port = path.parts[-3]
        day = path.parts[-2].replace("dt=", "")
        vv, vh, meta = load_chip(path)

        res = detect_vessels(vv, vh, guard=args.guard, background=args.background,
                             k=args.k, min_px=args.min_px, max_px=args.max_px)

        # Split detections by water/land. Land detections are cranes, quays and
        # tanks — they recur in the SAME places every pass, so leaving them in
        # makes the congestion index a large constant plus a small ship signal.
        wm, wdiag = (np.ones_like(vv, dtype=bool), {"water_frac": 1.0, "status": "MASK OFF"}) \
            if args.no_mask else water_mask(vv)

        on_water, on_land = [], []
        for d in res["detections"]:
            r, c = int(round(d["row"])), int(round(d["col"]))
            r = min(max(r, 0), wm.shape[0] - 1)
            c = min(max(c, 0), wm.shape[1] - 1)
            (on_water if wm[r, c] else on_land).append(d)

        note = wdiag["status"] if wdiag["status"] != "OK" else ""
        if res["coverage"] < 0.90:
            note = (note + "  " if note else "") + "PARTIAL COVER - not comparable"

        print(f"  {port:<12} {day:<12} {res['coverage']:>6.1%} {wdiag['water_frac']:>6.1%} "
              f"{res['n_detections']:>6,} {len(on_water):>9,} {len(on_land):>8,}  {note}")

        res.update({
            "port": port, "date": day, "chip": str(path), "geo": meta,
            "water_mask": wdiag,
            "n_on_water": len(on_water), "n_on_land": len(on_land),
            "detections": on_water,          # vessels = water detections only
            "detections_land": on_land,
        })
        results.append(res)

    out = derived_dir() / "sar_cfar"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "cfar_detections.json"
    dest.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {dest}")

    print("\n  Use `n_on_water` as the congestion signal. Measured on Fujairah:")
    print("  land detections are near-constant across passes (39/37/38) because")
    print("  they ARE the same cranes and quays; leaving them in turns a 14->19")
    print("  vessel change (+36%) into 53->57 (+7.5%) by adding a fixed offset.")
    print("  Still compare same-port, equal-coverage only — a partial chip")
    print("  undercuts both the count and the water fraction.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
