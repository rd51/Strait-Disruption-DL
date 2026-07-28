"""
CA-CFAR vessel detection on Sentinel-1 chips — the baseline for Arm A.

WHY A CLASSICAL BASELINE EXISTS AT ALL. The CNN has no labels of its own: our
Gulf chips are raw backscatter with no ground truth about which bright pixels
are ships. CFAR needs none — it is a statistical test, not a trained model — so
it produces a working congestion index immediately AND gives the CNN a number
to beat. Design rule 4 says every arm must earn its place; "our CNN beats CFAR
by X on the same chips" is how Arm A earns it. If the CNN cannot beat this,
that is worth discovering now rather than after the writeup.

THE METHOD. Cell-Averaging CFAR compares each pixel against the statistics of a
ring of background around it:

    detect  <=>  x_pixel  >  mu_background + k * sigma_background

A guard band between the test pixel and the background ring stops a large
vessel's own energy leaking into its background estimate and suppressing the
detection ("target masking"). The threshold adapts per-pixel, which is the
point: sea clutter varies hugely with wind and incidence angle, so any single
global threshold is wrong somewhere in the scene.

⚠️ LAND IS BRIGHT IN SAR. Buildings, cranes and quays return far more strongly
than any ship. Run naively over a port chip and the "vessels" are mostly
infrastructure. Two defences here, neither sufficient alone:
  · a size filter — vessels are small and compact; land is large and contiguous
  · an explicit land-mask hook, to be filled once per-port water polygons exist
Until real water polygons are drawn, treat counts as RELATIVE (day-over-day at
the same port) and never as an absolute vessel census.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def ca_cfar(
    image: np.ndarray,
    guard: int = 4,
    background: int = 12,
    k: float = 5.0,
    min_background_frac: float = 0.3,
) -> np.ndarray:
    """
    Cell-Averaging CFAR. Returns a boolean detection mask.

    guard      : half-width of the guard band, in pixels
    background : half-width of the background window (> guard)
    k          : threshold in background standard deviations

    Implemented with summed-area arithmetic via uniform filters, so the
    background ring statistics cost two convolutions rather than a Python loop
    over every pixel.
    """
    if background <= guard:
        raise ValueError("background half-width must exceed guard half-width")

    img = np.nan_to_num(image.astype("float64"), nan=0.0)
    valid = np.isfinite(image) & (image > 0)

    outer = 2 * background + 1
    inner = 2 * guard + 1

    # Sum and sum-of-squares over the outer box and the inner (guard) box; the
    # ring is their difference. Using counts of VALID pixels only, so nodata
    # borders do not drag the background mean toward zero.
    def boxsum(a):
        return ndimage.uniform_filter(a, size=outer, mode="constant") * outer ** 2

    def boxsum_inner(a):
        return ndimage.uniform_filter(a, size=inner, mode="constant") * inner ** 2

    v = valid.astype("float64")
    x = img * v

    ring_n = boxsum(v) - boxsum_inner(v)
    ring_sum = boxsum(x) - boxsum_inner(x)
    ring_sq = boxsum(x * x) - boxsum_inner(x * x)

    enough = ring_n > (outer ** 2 - inner ** 2) * min_background_frac
    ring_n_safe = np.where(ring_n > 0, ring_n, 1.0)

    mu = ring_sum / ring_n_safe
    var = np.maximum(ring_sq / ring_n_safe - mu ** 2, 0.0)
    sigma = np.sqrt(var)

    return valid & enough & (img > mu + k * sigma)


def detections(
    mask: np.ndarray,
    min_px: int = 2,
    max_px: int = 400,
) -> tuple[np.ndarray, list[dict]]:
    """
    Group detected pixels into objects and filter by size.

    min_px drops single-pixel speckle (thermal noise survives CFAR
    occasionally). max_px drops land and large infrastructure: at ~10 m
    resolution a 400-pixel blob is roughly 200 m across, which is a very large
    ship, so anything bigger is almost certainly not a vessel.
    """
    labels, n = ndimage.label(mask)
    if n == 0:
        return labels, []

    objs = ndimage.find_objects(labels)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))

    out = []
    for i, (sl, size) in enumerate(zip(objs, sizes), start=1):
        if size < min_px or size > max_px:
            continue
        ys, xs = sl
        cy = (ys.start + ys.stop - 1) / 2.0
        cx = (xs.start + xs.stop - 1) / 2.0
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        out.append({
            "label": i,
            "size_px": int(size),
            "row": float(cy), "col": float(cx),
            "height_px": int(h), "width_px": int(w),
            "elongation": float(max(h, w) / max(1, min(h, w))),
        })
    return labels, out


def detect_vessels(
    vv: np.ndarray,
    vh: np.ndarray | None = None,
    guard: int = 4,
    background: int = 12,
    k: float = 5.0,
    min_px: int = 2,
    max_px: int = 400,
) -> dict:
    """
    Full detection pass over one chip.

    When VH is present a detection must appear in BOTH polarisations. VH
    suppresses sea clutter far better than VV, so a target bright in both is
    much more likely to be a metal hull than a wave crest or a wind streak.
    """
    mask = ca_cfar(vv, guard=guard, background=background, k=k)
    if vh is not None:
        mask &= ca_cfar(vh, guard=guard, background=background, k=k)

    labels, objs = detections(mask, min_px=min_px, max_px=max_px)

    coverage = float((np.isfinite(vv) & (vv != 0)).mean())
    return {
        "n_detections": len(objs),
        "detections": objs,
        "mask_px": int(mask.sum()),
        "coverage": round(coverage, 4),
        # Counts are only comparable between chips of equal coverage — a chip
        # clipped by the swath edge undercounts in exact proportion.
        "density_per_covered_px": (
            len(objs) / max(coverage * vv.size, 1.0)
        ),
        "params": {"guard": guard, "background": background, "k": k,
                   "min_px": min_px, "max_px": max_px,
                   "dual_pol": vh is not None},
    }
