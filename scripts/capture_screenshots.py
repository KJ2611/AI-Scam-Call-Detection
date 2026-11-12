"""Capture real screenshots of the running app (headless Chromium via Playwright).

    streamlit run app/streamlit_app.py --server.port 8501 &
    python scripts/capture_screenshots.py [--url http://localhost:8501]
"""
import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def wait_ready(page, extra=2.5):
    page.wait_for_load_state("networkidle")
    for _ in range(120):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def main(url):
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        for slug, name in [("", "01_home"), ("models", "03_models"), ("dataset", "04_dataset"), ("advanced", "05_advanced"), ("about", "06_about")]:
            pg.goto(f"{url}/{slug}")
            wait_ready(pg, 4)
            pg.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
            print("saved", name)
        # detection flow: scam
        pg.goto(f"{url}/detect")
        wait_ready(pg)
        pg.get_by_role("button", name="Scam sample").click()
        wait_ready(pg, 1)
        pg.get_by_role("button", name="Analyze conversation").click()
        wait_ready(pg, 3)
        pg.locator(".verdict").scroll_into_view_if_needed()
        pg.mouse.wheel(0, -120)
        time.sleep(1)
        pg.screenshot(path=str(OUT / "02_detect_scam.png"))
        pg.get_by_role("tab", name="Why").click()
        time.sleep(1.5)
        pg.locator(".verdict").scroll_into_view_if_needed()
        pg.mouse.wheel(0, 420)
        time.sleep(1.5)
        pg.screenshot(path=str(OUT / "02a_detect_why.png"))
        pg.get_by_role("tab", name="Red flags & advice").click()
        time.sleep(1.5)
        pg.locator(".ind").first.scroll_into_view_if_needed()
        pg.mouse.wheel(0, 250)
        time.sleep(1)
        pg.screenshot(path=str(OUT / "02b_detect_redflags.png"))
        pg.get_by_role("tab", name="Highlighted transcript").click()
        time.sleep(2)
        pg.locator(".transcript").scroll_into_view_if_needed()
        pg.mouse.wheel(0, 200)
        time.sleep(1)
        pg.screenshot(path=str(OUT / "02c_detect_highlight.png"))
        # legit
        pg.get_by_role("button", name="Legit sample").click()
        wait_ready(pg, 1)
        pg.get_by_role("button", name="Analyze conversation").click()
        wait_ready(pg, 3)
        pg.locator(".verdict").scroll_into_view_if_needed()
        time.sleep(1)
        pg.screenshot(path=str(OUT / "02d_detect_legit.png"))
        print("saved detection screenshots")
        b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    main(ap.parse_args().url)
