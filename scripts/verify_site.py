"""Click every tab of the static site from a plain file server and report what breaks.

A static export is easy to ship broken: the build succeeds, the pages render, and three
figures 404 because the recorded listing points at paths the export did not write. That is
what the first version of this shipped, and a human clicking five tabs would have missed it,
because a missing SVG is a blank the eye slides over.

So this checks what a browser actually did: every request that failed, every console error,
that the gallery replay advances when clicked, and that no image on the page has a natural
width of zero. It needs playwright and a built site/.

    python scripts/verify_site.py
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
PORT = 8231

server = subprocess.Popen(
    [sys.executable, "-m", "http.server", str(PORT)],
    cwd=SITE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
try:
    base = f"http://127.0.0.1:{PORT}"
    for _ in range(40):
        try:
            urllib.request.urlopen(base, timeout=1)
            break
        except Exception:
            time.sleep(0.25)

    from playwright.sync_api import sync_playwright

    problems = []
    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 950})

        failed_requests = []
        console_errors = []
        page.on("requestfailed", lambda r: failed_requests.append(r.url))
        page.on(
            "response",
            lambda r: failed_requests.append(f"{r.status} {r.url}") if r.status >= 400 else None,
        )
        page.on(
            "console",
            lambda m: console_errors.append(m.text) if m.type == "error" else None,
        )

        for tab in ("problem", "architecture", "live", "gallery", "eval"):
            failed_requests.clear()
            console_errors.clear()
            page.goto(f"{base}/#{tab}", wait_until="networkidle")
            page.wait_for_timeout(2000)

            body = page.inner_text("body")
            marker = {
                "problem": "storygit",
                "architecture": "Architecture",
                "live": "static page",
                "gallery": "Gallery",
                "eval": "Eval",
            }[tab]
            ok = marker.lower() in body.lower()
            errors = [e for e in console_errors if "favicon" not in e.lower()]
            bad = [r for r in failed_requests if "favicon" not in r.lower()]

            print(
                f"[{tab:13}] text={'ok' if ok else 'MISSING'} chars={len(body):5} "
                f"failed_requests={len(bad)} console_errors={len(errors)}"
            )
            if not ok:
                problems.append(f"{tab}: expected text not found")
            for r in bad[:3]:
                problems.append(f"{tab}: request failed {r}")
            for e in errors[:3]:
                problems.append(f"{tab}: console error {e[:120]}")

            if tab == "gallery":
                # A replay has to actually play, not just list.
                first = page.get_by_role("button", name="replay", exact=False).first
                if first.count():
                    first.click()
                    page.wait_for_timeout(1800)
                    after = page.inner_text("body")
                    print(f"                 gallery replay opened, {len(after)} chars")
                    if len(after) <= len(body):
                        problems.append("gallery: clicking a session added nothing")
                else:
                    problems.append("gallery: no session control found")

            if tab == "eval":
                imgs = page.locator("img").count()
                print(f"                 {imgs} figures on the eval tab")
                if imgs == 0:
                    problems.append("eval: no figures rendered")

            if tab == "live":
                shots = page.locator(".shot img").count()
                broken = page.evaluate(
                    "Array.from(document.querySelectorAll('img'))"
                    ".filter(i => i.complete && i.naturalWidth === 0).length"
                )
                print(f"                 {shots} screenshots, {broken} broken images")
                if shots == 0:
                    problems.append("live: no screenshots")
                if broken:
                    problems.append(f"live: {broken} broken images")

        browser.close()

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("every tab renders, every request succeeded, no console errors")
finally:
    server.terminate()
    server.wait(timeout=10)
