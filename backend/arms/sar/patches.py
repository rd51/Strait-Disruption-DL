"""
Patch dataset for the Arm A CNN — cut from CFAR detections as WEAK labels.

THE LABEL PROBLEM, AND THE WAY ROUND IT. There are no vessel bounding boxes for
these chips. xView3 has them but is a multi-hundred-GB download of different
scenes. So the CNN cannot be trained on ground truth here.

What IS available is a weak teacher. CFAR marks bright compact targets, and the
water mask says whether each sits on water or on land. That yields three
classes with genuinely different meanings:

    vessel_candidate  CFAR detection ON WATER  — mostly ships, some clutter
    infrastructure    CFAR detection ON LAND   — cranes, quays, tanks
    sea_clutter       random water patch with NO detection — speckle, waves

THIS IS THE POINT: `infrastructure` is a class with RELIABLE labels. A
detection on land at a port is definitionally not a ship at sea, and land
detections were measured near-constant across passes (Fujairah 157/158/164/171)
precisely because they are fixed structures. So the model can be scored on a
task where the labels are trustworthy, even though the vessel class is noisy.

WHAT THIS CAN AND CANNOT CLAIM. It can claim: the CNN learns to separate
vessel-like targets from fixed infrastructure and from sea clutter, measured on
held-out chips. It CANNOT claim a validated vessel count — that needs ground
truth this project does not have. Distilling CFAR and then reporting agreement
with CFAR would be circular; say what was measured and no more.

SPLIT BY DATE, NOT BY PATCH. Patches from one chip are highly correlated — same
sea state, same speckle realisation, same vessels. Splitting randomly would put
near-duplicates in train and test and inflate the score. Chips are assigned to
folds by DATE so an entire acquisition lands on one side.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from ...common.paths import repo_root
from ...common.secrets import safe_stdout

log = logging.getLogger(__name__)

PATCH = 64                 # px per side; ~1 km at 15.6 m/px
CLASSES = ["sea_clutter", "vessel_candidate", "infrastructure"]
RNG = np.random.default_rng(42)


def db(x: np.ndarray) -> np.ndarray:
    """
    Backscatter -> decibels.

    SAR amplitude is heavy-tailed: measured VV median 0.030 against a max of
    740.7, a 24,000x span. Feeding that to a CNN linearly means the loss is
    dominated by a handful of bright pixels. dB compresses it to a range a
    network can actually use.
    """
    return 10.0 * np.log10(np.clip(x, 1e-6, None))


def to_rgb(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Two-channel SAR -> three-channel input for an ImageNet-pretrained backbone.

    The pretrained weights expect 3 channels. Rather than duplicating VV, the
    third channel carries the VV/VH RATIO, which is the physically meaningful
    combination: metal hulls depolarise differently from water, so the ratio
    separates ships from sea better than either polarisation alone. In dB the
    ratio is simply a difference.
    """
    vv_db, vh_db = db(vv), db(vh)
    ratio = vv_db - vh_db
    # Fixed clipping bounds, NOT per-chip percentiles. Per-chip normalisation
    # would rescale every image to its own contrast, destroying exactly the
    # absolute-brightness information that distinguishes a ship from the sea
    # — and making a busy chip look identical to an empty one.
    def norm(a, lo, hi):
        return np.clip((a - lo) / (hi - lo), 0, 1)
    out = np.stack([norm(vv_db, -35, 5), norm(vh_db, -40, 0), norm(ratio, 0, 25)], -1)
    # 🔴 SANITISE. np.clip PROPAGATES NaN — a NaN pixel survives clipping and
    # reaches the model. Exactly ONE patch in 12,687 carried a non-finite value
    # and it was enough to destroy training: a single NaN in a batch makes the
    # loss NaN, one NaN gradient step turns every weight NaN permanently, and
    # the network then emits a constant. Observed symptom was a 35% "accuracy"
    # with recall 1.0 on one class and 0.0 on the other two — which reads like
    # a hard task rather than a corrupted model. A hand-feature baseline on the
    # same patches scores 94%, which is what exposed it.
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def cut(img: np.ndarray, r: int, c: int, size: int = PATCH) -> np.ndarray | None:
    """Extract a patch centred on (r, c), or None if it would leave the chip."""
    h = size // 2
    r0, c0 = int(r) - h, int(c) - h
    if r0 < 0 or c0 < 0 or r0 + size > img.shape[0] or c0 + size > img.shape[1]:
        return None
    return img[r0:r0 + size, c0:c0 + size]


def build(max_per_class_per_chip: int = 12) -> dict:
    import rasterio
    from ..sar.watermask import water_mask

    det = json.loads((repo_root() / "data" / "derived" / "sar_cfar"
                      / "cfar_detections.json").read_text())
    X, y, meta = [], [], []

    for rec in det:
        if rec["coverage"] < 0.90:
            continue
        try:
            with rasterio.open(rec["chip"]) as src:
                vv = src.read(1).astype("float32")
                vh = src.read(2).astype("float32") if src.count > 1 else vv
        except Exception as exc:                       # noqa: BLE001
            log.warning("cannot read %s: %s", rec["chip"], exc)
            continue

        img = to_rgb(vv, vh)
        wm, _ = water_mask(vv)

        def take(dets, label):
            n = 0
            for d in dets:
                if n >= max_per_class_per_chip:
                    break
                p = cut(img, d["row"], d["col"])
                if p is None:
                    continue
                X.append(p.astype("float32"))
                y.append(CLASSES.index(label))
                meta.append({"port": rec["port"], "date": rec["date"], "cls": label})
                n += 1

        take(rec.get("detections", []), "vessel_candidate")
        take(rec.get("detections_land", []), "infrastructure")

        # Negatives: water pixels far from any detection. Sampled at random
        # rather than on a grid so the model cannot learn a position prior.
        occupied = {(int(d["row"]) // PATCH, int(d["col"]) // PATCH)
                    for d in rec.get("detections", []) + rec.get("detections_land", [])}
        n = 0
        for _ in range(300):
            if n >= max_per_class_per_chip:
                break
            r = int(RNG.integers(PATCH, img.shape[0] - PATCH))
            c = int(RNG.integers(PATCH, img.shape[1] - PATCH))
            if (r // PATCH, c // PATCH) in occupied or not wm[r, c]:
                continue
            p = cut(img, r, c)
            if p is None:
                continue
            X.append(p.astype("float32"))
            y.append(CLASSES.index("sea_clutter"))
            meta.append({"port": rec["port"], "date": rec["date"], "cls": "sea_clutter"})
            n += 1

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int64")
    dfm = pd.DataFrame(meta)

    out = repo_root() / "data" / "derived" / "sar_patches"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    dfm.to_parquet(out / "meta.parquet", index=False)

    counts = dfm["cls"].value_counts().to_dict()
    log.info("patches: %s -> %s", X.shape, counts)
    return {"shape": list(X.shape), "counts": counts,
            "chips": int(dfm[["port", "date"]].drop_duplicates().shape[0])}


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Cut CNN patches from CFAR detections")
    p.add_argument("--per-class-per-chip", type=int, default=12)
    a = p.parse_args()
    print(json.dumps(build(a.per_class_per_chip), indent=2))
