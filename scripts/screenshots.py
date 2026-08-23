#!/usr/bin/env python
"""Screenshot every tab, so the visual contract can be audited against real pixels.

The prohibition list in the design contract — no gradients, no icons as decoration, no
emoji, no card shadows, one typeface — is only checkable against what actually renders, so
this drives a headless browser over a real server and writes a PNG per tab.

    .venv/bin/python scripts/screenshots.py [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

TABS = ("problem", "architecture", "live", "gallery", "eval")
PORT = int(os.environ.get("STORYGIT_SHOT_PORT", "8124"))


def wait_for(url: str, timeout: float = 45.0) -> bool:
    """Poll until the server answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    """Boot the server, drive a browser over every tab, and write the PNGs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="../arush/logs/chunk_6_screens")
    parser.add_argument("--db", default="")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import tempfile

    db = args.db or str(Path(tempfile.mkdtemp()) / "shots.db")
    env = {**os.environ, "STORYGIT_DB": db, "STORYGIT_PORT": str(PORT)}
    server = subprocess.Popen(
        [sys.executable, "scripts/e2e_server.py"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for(f"http://127.0.0.1:{PORT}/api/health"):
            print("server did not come up", file=sys.stderr)
            return 1

        # Build a little story first, so the Live tab has something in it. A screenshot of
        # an empty tool proves nothing about the tool.
        client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=60.0)
        for level in ("episode", "scene", "beat", "prose"):
            tree = client.get("/api/tree").json()
            parents = {
                "episode": "story",
                "scene": "episode",
                "beat": "scene",
                "prose": "beat",
            }
            candidates = [n for n in tree["nodes"] if n["node_type"] == parents[level]]
            if not candidates:
                break
            proposed = client.post(
                "/api/propose",
                json={"node_id": candidates[-1]["id"], "level": level, "intent": "go on"},
            ).json()
            if proposed["candidates"]:
                client.post(
                    "/api/action/accept",
                    json={"proposal_id": proposed["candidates"][0]["proposal_id"]},
                )
        # Put something in the writer ledger too. "None yet." proves nothing about the
        # panel that carries the agency argument, and the remove controls only exist to
        # be audited when there is something to remove.
        client.post("/api/ledger/style-note", json={"text": "shorter sentences at the turn"})
        client.post(
            "/api/ledger/criterion",
            json={
                "name": "menace",
                "description": "the sense that something is about to go wrong",
                "weight": 1.0,
            },
        )
        client.close()

        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            for tab in TABS:
                page.goto(f"http://127.0.0.1:{PORT}/#{tab}", wait_until="networkidle")
                page.wait_for_timeout(900)
                if tab == "live":
                    # Ask for candidates from the browser, so the shot shows the centre
                    # pane doing its actual job rather than an empty intent box.
                    page.fill("input[type=text]", "Kael's ability shows itself")
                    page.click("button.primary")
                    page.wait_for_timeout(2500)
                target = out / f"{tab}.png"
                page.screenshot(path=str(target), full_page=tab not in ("live",))
                print(f"wrote {target}")
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
