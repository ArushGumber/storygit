"""Export a fully static `site/` that needs no Python process.

Four of the five tabs are already static in everything but their transport: Problem,
Architecture, Gallery and Eval read JSON from the API and render it, and none of them
writes anything. So the export boots the real application over the committed demo story,
records the responses of exactly the routes those tabs call, and writes them to disk at the
paths the client already fetches. A dumb file server then serves them unchanged, and the
client cannot tell the difference.

The Live tab is the exception and is not faked: it needs a writable state and provider
keys, so the static build replaces it with screenshots of the running tool and the commands
to start it. That substitution happens in the frontend at build time, keyed on
``VITE_STATIC``, and the page it renders promises nothing this page can do.

    python scripts/build_site.py [--out site] [--no-shots]

Screenshots need Playwright with a Chromium; without it the build still produces a working
site and says which images are missing rather than shipping broken ones silently.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DEMO_DB = ROOT / "demo" / "story.db"

# Every GET the static tabs make, in the order a reader hits them. Recorded verbatim to the
# same path the client fetches, so nothing in the client changes for the static build.
ROUTES: tuple[str, ...] = (
    "/api/health",
    "/api/tree",
    "/api/ledger",
    "/api/authorship",
    "/api/branches",
    "/api/history",
    "/api/flags",
    "/api/gallery",
    "/api/eval/plots",
    "/api/eval/summary",
)

SHOTS: tuple[tuple[str, str], ...] = (
    ("live-tree", "#live"),
    ("live-candidates", "#live"),
    ("live-world", "#live"),
    ("live-audit", "#live"),
)


def build_frontend(out: Path) -> None:
    """Build the frontend in static mode and copy it into ``out``."""
    print("building the frontend in static mode")
    env_line = "VITE_STATIC=1"
    subprocess.run(
        ["npm", "run", "build"],
        cwd=FRONTEND,
        check=True,
        env={**_env(), "VITE_STATIC": "1"},
        stdout=subprocess.DEVNULL,
    )
    dist = FRONTEND / "dist"
    if not (dist / "index.html").exists():
        raise SystemExit(f"the frontend build produced no index.html ({env_line})")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dist, out)


def build_frontend_normal() -> None:
    """Build the frontend the way the local application serves it.

    The screenshots have to be of the real Live tab, so this build must not set
    ``VITE_STATIC``. It also leaves ``frontend/dist`` in the state ``make demo`` expects,
    which is why it runs again at the end of the export.
    """
    print("building the frontend for screenshots")
    env = {k: v for k, v in _env().items() if k != "VITE_STATIC"}
    subprocess.run(
        ["npm", "run", "build"], cwd=FRONTEND, check=True, env=env, stdout=subprocess.DEVNULL
    )


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def record_api(out: Path) -> list[str]:
    """Record every static tab's GET responses into ``out/api``.

    Returns:
        The routes that could not be recorded, with their reason.
    """
    from fastapi.testclient import TestClient

    from storygit.api.app import create_app
    from storygit.api.deps import build_state
    from storygit.providers.router import build_router

    if not DEMO_DB.exists():
        raise SystemExit(f"no demo story at {DEMO_DB}; run scripts/seed_demo.py first")

    # A copy, because the app opens the database read-write and the committed demo state is
    # a build input rather than scratch space.
    work = out / "_demo.db"
    shutil.copy(DEMO_DB, work)

    state = build_state(work, router=build_router(), results_dir=ROOT / "eval" / "results", seed=7)
    failures: list[str] = []
    # A context manager, because the application attaches its state in a lifespan handler
    # and TestClient only runs lifespan when used as one. Without this every route 500s on
    # a missing attribute, which is a confusing way to learn it.
    with TestClient(create_app(state=state)) as client:
        for route in ROUTES:
            response = client.get(route)
            if response.status_code != 200:
                failures.append(f"{route} -> HTTP {response.status_code}")
                continue
            _write_json(out / (route.lstrip("/") + ".json"), response.json())

        # The gallery's sessions are fetched one at a time by id.
        listing = client.get("/api/gallery")
        if listing.status_code == 200:
            for session in listing.json().get("sessions", []):
                sid = session.get("id") or session.get("name")
                if not sid:
                    continue
                one = client.get(f"/api/gallery/{sid}")
                if one.status_code == 200:
                    _write_json(out / "api" / "gallery" / f"{sid}.json", one.json())
                else:
                    failures.append(f"/api/gallery/{sid} -> HTTP {one.status_code}")

    # The figures the Eval tab renders are files on disk, not API responses.
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for svg in sorted((ROOT / "eval" / "results").glob("*.svg")):
        shutil.copy(svg, figures / svg.name)

    work.unlink(missing_ok=True)
    for sidecar in out.glob("_demo.db-*"):
        sidecar.unlink()
    return failures


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def capture_shots(out: Path, port: int = 8129) -> list[str]:
    """Screenshot the running tool for the static Live tab.

    Returns:
        Names of the shots that could not be taken, with their reason.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ["playwright is not installed; the Live tab will have no images"]

    shots = out / "shots"
    shots.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, "scripts/demo_server.py"],
        cwd=ROOT,
        env={**_env(), "STORYGIT_PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    missing: list[str] = []
    try:
        import time
        import urllib.request

        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{base}/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            return ["the demo server did not come up; the Live tab will have no images"]

        with sync_playwright() as play:
            browser = play.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(f"{base}/#live", wait_until="networkidle")
            page.wait_for_timeout(1500)

            # The three panes, with a beat selected so the right-hand one is populated.
            beat = page.get_by_text("The Hollow Knock at the bus stop", exact=False).first
            if beat.count():
                beat.click()
                page.wait_for_timeout(1200)
            page.screenshot(path=str(shots / "live-tree.png"))

            # The world state at that beat, scrolled so the epistemic column and the open
            # threads are both in frame.
            page.mouse.move(1450, 600)
            page.mouse.wheel(0, 420)
            page.wait_for_timeout(600)
            page.screenshot(path=str(shots / "live-world.png"))

            # The audit over the whole story. It runs layers 1 and 2 with no model call,
            # so it works here with no keys.
            page.mouse.wheel(0, -1200)
            page.wait_for_timeout(400)
            audit = page.get_by_role("button", name="audit", exact=True)
            if audit.count():
                audit.first.click()
                page.wait_for_timeout(6000)
            page.screenshot(path=str(shots / "live-audit.png"))

            # Candidates need provider keys, so the recorded Gallery session is the honest
            # place to photograph them: it is the real interface over real recorded output.
            page.goto(f"{base}/#gallery", wait_until="networkidle")
            page.wait_for_timeout(1200)
            labelled = page.get_by_text("labelled", exact=False).first
            if labelled.count():
                labelled.click()
                page.wait_for_timeout(1500)
            page.screenshot(path=str(shots / "live-candidates.png"))

            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)

    for name, _ in SHOTS:
        if not (shots / f"{name}.png").exists():
            missing.append(f"{name}.png was not captured")
    return missing


def main() -> None:
    """Build the static site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "site"))
    parser.add_argument("--no-shots", action="store_true", help="skip the screenshots")
    args = parser.parse_args()

    out = Path(args.out)
    # Screenshots first, and against a *normal* build. The static build replaces the Live
    # tab with this page, so screenshotting after it would photograph this page's own
    # placeholders -- which is exactly what the first version did.
    shots: list[Path] = []
    problems: list[str] = []
    if not args.no_shots:
        build_frontend_normal()
        problems += capture_shots(Path(args.out + ".shots"))
        shots = sorted((Path(args.out + ".shots") / "shots").glob("*.png"))

    build_frontend(out)
    problems += record_api(out)

    if shots:
        target = out / "shots"
        target.mkdir(parents=True, exist_ok=True)
        for shot in shots:
            shutil.copy(shot, target / shot.name)
        shutil.rmtree(Path(args.out + ".shots"), ignore_errors=True)

    # A dumb file server has no rewrite rule, so the hash-routed shell is the only entry
    # point; 404.html makes a mistyped path land somewhere sensible on hosts that use it.
    shutil.copy(out / "index.html", out / "404.html")
    (out / ".nojekyll").write_text("")

    # Leave the working tree as `make demo` expects it: a normal build, not the static one.
    build_frontend_normal()

    files = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"\nwrote {out} ({files} files)")
    if problems:
        print("\nincomplete:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("every tab has its data; serve it with any static file server")


if __name__ == "__main__":
    main()
