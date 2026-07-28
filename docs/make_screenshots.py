"""
Capture a screenshot of every dashboard tab.

Uses a real headless Chromium rather than a DOM dump so the figures in the
writeup show what a reader actually sees — Leaflet tiles, rendered SAR imagery,
the colour coding on caveats.

Each tab is given an explicit settle period because several panels are
network-bound: the map fetches map tiles, the satellite tab renders a 1024x1024
GeoTIFF to PNG server-side, and the pipeline tab globs the raw store. Screencap
before those land produces a picture of a loading state.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000/"
OUT = Path(__file__).resolve().parent / "screenshots"
VIEWPORT = {"width": 1600, "height": 1150}

# (tab id, filename, settle seconds) — settle tuned per tab's slowest fetch.
TABS = [
    ("overview", "01_overview", 6),
    ("alerts",   "02_alerts",   3),
    ("map",      "03_map",      8),   # Leaflet tiles
    ("sat",      "04_satellite", 9),  # server-side GeoTIFF -> PNG
    ("models",   "05_models",   4),
    ("pipe",     "06_pipeline", 7),   # globs the raw store
    ("sem",      "07_semantics", 3),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        time.sleep(4)

        # Warm the inference panel once so the Overview screenshot shows real
        # scores rather than an empty box. The first call loads the encoder.
        try:
            page.click("#go")
            page.wait_for_selector("#inf .score-row", timeout=120_000)
        except Exception as exc:                        # noqa: BLE001
            print(f"  ! inference panel did not populate: {str(exc)[:90]}")

        for tab, name, settle in TABS:
            page.click(f'nav button[data-tab="{tab}"]')
            time.sleep(settle)
            # full_page so tall tabs are not cropped mid-table.
            path = OUT / f"{name}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"  {name:15s} {path.stat().st_size/1024:6.0f} KB")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
