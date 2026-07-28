"""
Derive a water mask FROM THE SAR IMAGE, not from an external map.

WHY NOT OSM. The first attempt pulled OSM coastline and flood-filled water from
the AOI edge. It returned 98.6% water for Fujairah — obviously wrong for a
13 km box around a major port. OSM coastline is an oriented LINE (land on the
left of travel), not a polygon, so reconstructing filled land needs real
polygonisation; a flood fill leaks anywhere the line fails to span the box.
Overpass also rate-limits (HTTP 429 after two ports).

WHY THE IMAGE ITSELF IS BETTER HERE. Calm water is specular — it reflects radar
energy away from the sensor and comes back very dark. Land, quays, cranes and
tanks are rough and bright. That contrast is exactly what a threshold finds,
and it is measured per scene, so it adapts to tide, wind and incidence angle
instead of assuming a fixed shoreline.

THE ONE TRAP: ships are bright too. Threshold naively and every vessel is
classified as land, which would mask out precisely what we are counting. The
fix is scale — smooth the image first with a kernel much larger than a ship but
much smaller than a landmass. Ships vanish into the background; land does not.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _to_db(x: np.ndarray) -> np.ndarray:
    """Backscatter is log-normal; work in dB so a threshold is meaningful."""
    return 10.0 * np.log10(np.maximum(x, 1e-6))


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method — the between-class variance maximiser."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype("float64")
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    p = hist / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(centres[int(np.nanargmax(sigma_b))])


def water_mask(
    vv: np.ndarray,
    smooth_px: int = 15,
    min_land_px: int = 200,
    close_px: int = 5,
) -> tuple[np.ndarray, dict]:
    """
    Boolean water mask (True = water) plus diagnostics.

    smooth_px    : median-filter size. Must exceed a ship (a few px) and stay
                   well under a landmass. At ~35 m/px, 15 px ≈ 500 m.
    min_land_px  : land blobs smaller than this are re-labelled water — a
                   200-px island at 35 m/px is ~250 m across, which at a port is
                   far more likely to be a moored vessel than actual land.
    close_px     : morphological closing to fill quay gaps and small harbour
                   inlets so the landmass is contiguous.
    """
    valid = np.isfinite(vv) & (vv > 0)
    if not valid.any():
        return np.zeros_like(vv, dtype=bool), {"status": "EMPTY", "water_frac": 0.0}

    db = _to_db(np.where(valid, vv, np.nan))

    # Median filter, not Gaussian: median is far less swayed by the very bright
    # outliers (ships) we are specifically trying to make disappear.
    filled = np.where(np.isfinite(db), db, np.nanmedian(db))
    smoothed = ndimage.median_filter(filled, size=smooth_px, mode="nearest")

    thresh = otsu_threshold(smoothed[valid])
    land = (smoothed > thresh) & valid

    if close_px > 1:
        land = ndimage.binary_closing(land, structure=np.ones((close_px, close_px)))

    # Drop small bright blobs that are almost certainly vessels, not land.
    labels, n = ndimage.label(land)
    if n:
        sizes = ndimage.sum(land, labels, index=np.arange(1, n + 1))
        too_small = {i for i, s in enumerate(sizes, start=1) if s < min_land_px}
        if too_small:
            land = land & ~np.isin(labels, list(too_small))

    water = valid & ~land
    frac = float(water.sum() / max(valid.sum(), 1))

    status = "OK"
    if frac > 0.97:
        status = "SUSPECT — nearly all water; AOI may be offshore or threshold failed"
    elif frac < 0.10:
        status = "SUSPECT — nearly all land; check AOI centre"

    return water, {
        "status": status,
        "water_frac": round(frac, 4),
        "threshold_db": round(thresh, 3),
        "valid_frac": round(float(valid.mean()), 4),
        "land_blobs": int(n),
        "smooth_px": smooth_px,
        "min_land_px": min_land_px,
    }
