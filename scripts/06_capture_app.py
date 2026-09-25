"""Capture screenshots of the running Streamlit app for the README.

Optional: needs `pip install playwright && playwright install chromium`.
Start the app first:  streamlit run app/streamlit_app.py --server.port 8511
"""

from __future__ import annotations

import sys

from _bootstrap import ensure_src_on_path  # noqa: E402  (must precede dq_anomaly)

ensure_src_on_path()

from dq_anomaly.config import PATHS

URL = "http://localhost:8511"
VIEWPORT = {"width": 1500, "height": 1000}


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed; skipping screenshots.")
        return

    PATHS.figures.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(25_000)

        page.screenshot(path=PATHS.figures / "app_overview.png")
        print("app_overview.png")

        # Switch to the credit card batch so the trained model is exercised.
        page.get_by_text("Credit card transactions", exact=True).click()
        page.wait_for_timeout(30_000)

        page.get_by_role("tab", name="Anomalies").click()
        page.wait_for_timeout(6_000)
        page.screenshot(path=PATHS.figures / "app_anomalies.png")
        print("app_anomalies.png")

        page.get_by_role("tab", name="Issues & report").click()
        page.wait_for_timeout(6_000)
        page.screenshot(path=PATHS.figures / "app_issues.png")
        print("app_issues.png")

        browser.close()


if __name__ == "__main__":
    sys.exit(main())
