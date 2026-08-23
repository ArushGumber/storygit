"""The HTTP layer: every route's happy path, its failure, and the invariants.

Runs against a `TestClient` with a mock provider, so nothing here touches a network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.mockprovider import MockProvider, canned

from storygit.api.app import create_app
from storygit.api.deps import AppState
from storygit.domain.diff import AddEntity, AddNode, Diff, DiffAuthor
from storygit.domain.ids import IdGenerator, NodeId
from storygit.domain.nodes import Beat, Episode, Scene, Story
from storygit.domain.state import StoryState
from storygit.domain.world import Entity, EntityKind
from storygit.providers.router import Router
from storygit.selection.select import SelectionConfig, Selector
from storygit.store.repository import Repository

BEAT = {
    "title": "The offer",
    "what_happens": "The Warden offers Kael a way out.",
    "audience_learns": "the Warden wants something",
    "audience_feels": "wary",
    "location": "Ashfall",
    "time": "night",
    "produces": [
        {
            "subject": "Kael",
            "predicate": "secret",
            "object": "he can hear the ash",
            "object_is_entity": False,
            "known_by": ["Kael"],
        }
    ],
    "consumes": [],
    "threads_touched": [],
    "new_characters": [],
    "rationale": "Gives Kael something to refuse.",
    "delta_summary": ["the Warden makes an offer"],
}
PROSE = {"text": "Ash fell. Kael counted guards.", "rationale": "short", "delta_summary": ["prose"]}
EXTRACTION = {"facts": [], "new_characters": [], "threads_opened": [], "threads_touched": []}


@pytest.fixture
def api(tmp_path: Path) -> tuple[TestClient, AppState, MockProvider]:
    """A client over a small seeded story, with a mock provider."""
    ids = IdGenerator(seed=5, stream="api")
    repo = Repository.open(tmp_path / "story.db")
    story_id, episode_id, scene_id, beat_id = (ids.node() for _ in range(4))
    kael = ids.entity()

    repo.initialize(StoryState.build(nodes={story_id: Story(id=story_id, title="Ashfall")}))
    repo.commit_diff(
        Diff(
            ops=(
                AddNode(node=Episode(id=episode_id, parent_id=story_id, title="Episode 1")),
                AddNode(node=Scene(id=scene_id, parent_id=episode_id, title="The market")),
                AddNode(
                    node=Beat(
                        id=beat_id,
                        parent_id=scene_id,
                        title="Caught",
                        what_happens="Kael is caught stealing.",
                    )
                ),
                AddEntity(entity=Entity(id=kael, kind=EntityKind.character, name="Kael")),
            ),
            author=DiffAuthor.human,
            intent="seed",
        )
    )

    provider = MockProvider(
        lambda req: canned(
            PROSE
            if req.purpose.startswith("propose.prose")
            else EXTRACTION
            if req.purpose.startswith("extract")
            else dict(BEAT, title=f"b{req.sample_index}")
        )
    )
    results = tmp_path / "results"
    (results / "gallery").mkdir(parents=True)
    (results / "summary.json").write_text(json.dumps({"runs": [], "skipped": []}))
    (results / "acceptance.svg").write_text("<svg/>")

    from storygit.config import Settings

    state = AppState(
        repo=repo,
        router=Router({"gemini": provider, "groq": provider}),
        settings=Settings(),
        results_dir=results,
        seed=5,
    )
    tiny = SelectionConfig(
        n=2, k=2, selector=Selector.topk_temperature, use_judge=False, use_dial=False
    )
    from storygit.engine import Engine
    from storygit.preference.layer import PreferenceLayer

    state.engines["main"] = Engine(
        repo,
        state.router,
        ids=IdGenerator(seed=6, stream="api-engine"),
        selection=tiny,
        use_nli=False,
        preference=PreferenceLayer(enabled=False),
    )

    app = create_app(state=state, frontend_dir=tmp_path / "nonexistent")
    with TestClient(app) as client:
        yield client, state, provider


# --- reads --------------------------------------------------------------------


def test_health_reports_the_lock(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["openrouter_enabled"] is False, "the lock is visible in the interface"
    assert body["nodes"] == 4


def test_tree_returns_every_node_in_reading_order(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    body = client.get("/api/tree").json()
    assert body["branch"] == "main"
    assert [n["node_type"] for n in body["nodes"]] == ["story", "episode", "scene", "beat"]
    assert body["nodes"] == sorted(body["nodes"], key=lambda n: n["seq"])
    assert body["stale_count"] == 0


def test_node_detail_and_404(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    beat = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "beat")
    body = client.get(f"/api/node/{beat['id']}").json()
    assert body["what_happens"] == "Kael is caught stealing."
    assert body["node"]["title"] == "Caught"

    missing = client.get("/api/node/n_nope")
    assert missing.status_code == 404


def test_slice_threads_flags_ledger_authorship(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    beat = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "beat")
    slice_body = client.get(f"/api/slice?node={beat['id']}").json()
    assert "entities" in slice_body and "facts" in slice_body

    assert client.get("/api/threads").json()["threads"] == []
    assert client.get("/api/flags").json()["flags"] == []

    ledger = client.get("/api/ledger").json()
    assert ledger["dial"] == pytest.approx(0.35)
    assert ledger["locks"] == []

    authorship = client.get("/api/authorship").json()
    assert authorship["sentences"] == 0

    assert len(client.get("/api/history").json()["history"]) == 2


# --- the loop -----------------------------------------------------------------


def test_propose_accept_and_the_bible_diff(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    scene = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "scene")
    proposed = client.post(
        "/api/propose",
        json={"node_id": scene["id"], "level": "beat", "intent": "go on"},
    ).json()
    assert len(proposed["candidates"]) == 2
    first = proposed["candidates"][0]
    assert first["axis_label"], "every candidate reaches the writer with its label"
    assert first["delta_summary"], "and with what it would change, in English"
    assert first["op_count"] > 0

    accepted = client.post("/api/action/accept", json={"proposal_id": first["proposal_id"]}).json()
    assert accepted["added"] == 1
    assert accepted["bible_diff"][0].startswith("+ ")

    after = client.get("/api/tree").json()
    assert len(after["nodes"]) == 5, "the beat was committed"


def test_accepting_an_unknown_proposal_is_a_404(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    response = client.post("/api/action/accept", json={"proposal_id": "p_nope"})
    assert response.status_code == 404


def test_an_unknown_level_is_a_422(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    response = client.post("/api/propose", json={"level": "chapter"})
    assert response.status_code == 422


def test_reject_records_the_direction(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    scene = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "scene")
    proposed = client.post("/api/propose", json={"node_id": scene["id"], "level": "beat"}).json()
    client.post(
        "/api/action/reject",
        json={
            "proposal_id": proposed["candidates"][0]["proposal_id"],
            "reason": "no shadowy benefactors",
        },
    )
    ledger = client.get("/api/ledger").json()
    assert "no shadowy benefactors" in ledger["rejected_directions"]


def test_hand_writing_a_beat_goes_through_the_same_path(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    beat = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "beat")
    response = client.post(
        "/api/action/write",
        json={"node_id": beat["id"], "text": "Kael walked into Ashfall at dusk."},
    ).json()
    assert "snapshot_id" in response

    detail = client.get(f"/api/node/{beat['id']}").json()
    assert detail["prose"].startswith("Kael walked")
    assert detail["authorship"]["human"] == 1.0


def test_lock_unlock_and_dismiss_stale(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    beat = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "beat")
    client.post(f"/api/node/{beat['id']}/lock")
    tree = client.get("/api/tree").json()
    locked = next(n for n in tree["nodes"] if n["id"] == beat["id"])
    assert locked["locked"] is True and locked["status"] == "locked"
    assert beat["id"] in client.get("/api/ledger").json()["locks"]

    client.post(f"/api/node/{beat['id']}/unlock")
    assert not client.get("/api/ledger").json()["locks"]

    client.post(f"/api/node/{beat['id']}/dismiss-stale")
    assert client.get("/api/tree").json()["stale_count"] == 0


def test_ledger_controls(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    client.post("/api/ledger/dial", json={"value": 0.8})
    client.post("/api/ledger/style-note", json={"text": "shorter sentences"})
    client.post(
        "/api/ledger/criterion",
        json={"name": "menace", "description": "every scene should threaten", "weight": 1.0},
    )
    ledger = client.get("/api/ledger").json()
    assert ledger["dial"] == pytest.approx(0.8)
    assert ledger["active_style_notes"] == ["shorter sentences"]
    assert ledger["criteria"][0]["name"] == "menace"


def test_striking_a_fact_returns_the_bible_diff(api) -> None:  # type: ignore[no-untyped-def]
    client, state, _ = api
    scene = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "scene")
    proposed = client.post("/api/propose", json={"node_id": scene["id"], "level": "beat"}).json()
    client.post(
        "/api/action/accept", json={"proposal_id": proposed["candidates"][0]["proposal_id"]}
    )
    fact_id = next(iter(state.repo.state("main").facts))

    struck = client.post(f"/api/fact/{fact_id}/strike").json()
    assert struck["removed"] == 1
    assert struck["bible_diff"][0].startswith("- ")
    assert fact_id not in state.repo.state("main").facts

    assert client.post("/api/fact/f_nope/strike").status_code == 404


# --- branches -----------------------------------------------------------------


def test_branch_switch_diff_and_clean_merge(api) -> None:  # type: ignore[no-untyped-def]
    client, state, _ = api
    assert client.post("/api/branch", json={"name": "what-if"}).json()["ok"] is True
    assert client.post("/api/branch", json={"name": "what-if"}).status_code == 409

    branches = client.get("/api/branches").json()
    assert set(branches["branches"]) == {"main", "what-if"}
    assert branches["current"] == "main"

    from storygit.domain.diff import UpdateNode

    beat = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "beat")
    state.repo.commit_diff(
        Diff(ops=(UpdateNode(node_id=NodeId(beat["id"]), fields={"title": "Changed"}),)),
        branch="what-if",
    )

    difference = client.get("/api/branch/diff?a=main&b=what-if").json()
    assert difference["op_count"] >= 1
    assert any("Caught" in line for line in difference["summary"])

    merged = client.post(
        "/api/branch/merge", json={"ours": "main", "theirs": "what-if", "commit": True}
    ).json()
    assert merged["clean"] is True and merged["committed"]
    assert state.repo.state("main").nodes[NodeId(beat["id"])].title == "Changed"

    assert client.post("/api/branch/switch", json={"name": "what-if"}).json()["current"] == (
        "what-if"
    )
    assert client.post("/api/branch/switch", json={"name": "nope"}).status_code == 404


def test_a_merge_conflict_surfaces_rather_than_resolving(api) -> None:  # type: ignore[no-untyped-def]
    client, state, _ = api
    from storygit.domain.diff import UpdateNode

    beat = NodeId(
        next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "beat")["id"]
    )
    client.post("/api/branch", json={"name": "alt"})
    state.repo.commit_diff(
        Diff(ops=(UpdateNode(node_id=beat, fields={"what_happens": "A"}),)), branch="main"
    )
    state.repo.commit_diff(
        Diff(ops=(UpdateNode(node_id=beat, fields={"what_happens": "B"}),)), branch="alt"
    )

    merged = client.post(
        "/api/branch/merge", json={"ours": "main", "theirs": "alt", "commit": True}
    ).json()
    assert merged["clean"] is False
    assert merged["conflicts"][0]["logical_id"] == str(beat)
    assert merged["committed"] is None, "a conflicted merge commits nothing"
    assert state.repo.state("main").nodes[beat].what_happens == "A"


# --- artifacts ----------------------------------------------------------------


def test_gallery_and_eval_routes(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    assert client.get("/api/gallery").json()["sessions"] == []
    assert client.get("/api/gallery/nope").status_code == 404

    summary = client.get("/api/eval/summary").json()
    assert summary["available"] is True

    plots = client.get("/api/eval/plots").json()["plots"]
    assert [p["name"] for p in plots] == ["acceptance.svg"]
    assert plots[0]["caption"], "every figure carries a caption saying what it demonstrates"

    served = client.get(plots[0]["url"])
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("image/svg")
    assert client.get("/api/eval/plot/../../etc/passwd").status_code in (404, 400)


def test_an_unrun_evaluation_is_a_state_not_an_error(tmp_path: Path) -> None:
    from storygit.config import Settings

    repo = Repository.open(tmp_path / "s.db")
    root = Story(id=NodeId("n_r"), title="S")
    repo.initialize(StoryState.build(nodes={root.id: root}))
    state = AppState(
        repo=repo,
        router=Router({}),
        settings=Settings(),
        results_dir=tmp_path / "no-results",
    )
    with TestClient(create_app(state=state, frontend_dir=tmp_path / "nope")) as client:
        body = client.get("/api/eval/summary").json()
        assert body["available"] is False
        assert "eval.run" in body["detail"]


# --- error handling -----------------------------------------------------------


def test_engine_errors_become_the_shared_problem_shape(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    beat = NodeId(
        next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "beat")["id"]
    )
    client.post(f"/api/node/{beat}/lock")

    response = client.post(f"/api/node/{beat}/dismiss-stale")
    assert response.status_code == 409, "a locked node cannot be staled or unstaled"
    body = response.json()
    assert body["kind"] == "LockedNodeError"
    assert "locked" in body["detail"]
    assert "Traceback" not in body["detail"]


def test_a_rate_limit_carries_retry_after(api) -> None:  # type: ignore[no-untyped-def]
    client, state, _ = api
    from storygit.providers.base import RateLimited

    class Exhausted:
        name = "gemini"
        model = "m"

        async def complete(self, request: object) -> object:
            raise RateLimited("all keys cooling", retry_after=45.0)

        async def aclose(self) -> None:
            return None

    state.router.providers["gemini"] = Exhausted()  # type: ignore[assignment]
    state.engines.clear()

    response = client.post("/api/propose", json={"level": "beat"})
    assert response.status_code == 429
    body = response.json()
    assert body["kind"] == "RateLimited"
    assert body["retry_after"] == 45.0, (
        "the interface can wait exactly long enough instead of guessing"
    )
