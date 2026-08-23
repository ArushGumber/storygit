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


def test_a_real_recorded_session_loads_and_replays_over_http(tmp_path: Path) -> None:
    """The Gallery must serve an actual recording, not just an empty list.

    The empty case passed while the route was broken: it imported the evaluation harness,
    which fails at request time from anywhere but the repository root, and every session
    returned a 500. A screenshot caught it; this test is so a screenshot does not have to
    next time.
    """
    from storygit.config import Settings
    from storygit.gallery import Session

    results = Path("eval/results")
    sessions = sorted(p for p in (results / "gallery").glob("*.json") if p.name != "index.json")
    if not sessions:
        pytest.skip("no recorded sessions; run `python -m eval.record_gallery`")

    repo = Repository.open(tmp_path / "s.db")
    root = Story(id=NodeId("n_r"), title="S")
    repo.initialize(StoryState.build(nodes={root.id: root}))
    state = AppState(repo=repo, router=Router({}), settings=Settings(), results_dir=results)

    with TestClient(create_app(state=state, frontend_dir=tmp_path / "nope")) as client:
        index = client.get("/api/gallery").json()["sessions"]
        assert len(index) >= 7, "FABLE asks for seven scenarios"
        assert all(entry["title"] and entry["summary"] for entry in index), (
            "every session says what it demonstrates"
        )

        name = Session.load(sessions[0]).name
        body = client.get(f"/api/gallery/{name}").json()
        assert body["session"]["steps"], "the session has steps"
        assert "replay" in body, "and they resolve against the snapshots they name"
        first = body["replay"][0]
        assert first["tree"], "replay reconstructs the plan tree from the store"

        # Every session in the index loads, not just the first.
        for entry in index:
            response = client.get(f"/api/gallery/{entry['name']}")
            assert response.status_code == 200, f"{entry['name']} failed to load"


def test_a_writer_can_delete_a_rule_the_system_mined(api) -> None:  # type: ignore[no-untyped-def]
    """A mined rule the writer cannot remove is a rule influencing prose without consent.

    The UI has always promised this ("rules you can see and delete"); until this was
    written there was no route behind the promise.
    """
    client, _, _ = api
    assert client.post("/api/ledger/style-note", json={"text": "no adverbs"}).status_code == 200
    assert (
        client.post(
            "/api/ledger/criterion", json={"name": "dread", "description": "unease that builds"}
        ).status_code
        == 200
    )
    ledger = client.get("/api/ledger").json()
    assert "no adverbs" in [n["text"] for n in ledger["style_notes"]]
    assert "dread" in [c["name"] for c in ledger["criteria"]]

    assert (
        client.post("/api/ledger/style-note/remove", json={"text": "no adverbs"}).status_code == 200
    )
    assert client.post("/api/ledger/criterion/remove", json={"name": "dread"}).status_code == 200

    ledger = client.get("/api/ledger").json()
    assert [n["text"] for n in ledger["style_notes"]] == []
    assert [c["name"] for c in ledger["criteria"]] == []


def test_a_duplicate_entity_can_be_merged_and_a_hard_rule_added(api) -> None:  # type: ignore[no-untyped-def]
    """Two operations that existed with no caller anywhere until this was written.

    Alias resolution is deliberately conservative — exact match or create — which means it
    will sometimes leave a duplicate rather than risk welding two characters together. The
    docstring says the writer can then merge them "in one click"; this is the click.
    """
    client, state, _ = api
    engine = state.engine()
    before = engine.state()
    kael = next(e for e in before.entities.values() if e.name == "Kael")

    from storygit.domain.diff import AddEntity, Diff
    from storygit.domain.ids import EntityId
    from storygit.domain.world import Entity, EntityKind

    duplicate = EntityId("e_dupe")
    state.repo.commit_diff(
        Diff(
            ops=(AddEntity(entity=Entity(id=duplicate, kind=EntityKind.character, name="the boy")),)
        )
    )

    response = client.post(
        "/api/entity/merge", json={"source_id": str(duplicate), "target_id": str(kael.id)}
    )
    assert response.status_code == 200
    after = state.engine().state()
    assert duplicate not in after.entities
    assert "the boy" in after.entities[kael.id].aliases

    assert (
        client.post(
            "/api/ledger/hard-constraint", json={"text": "nobody dies off-page"}
        ).status_code
        == 200
    )
    assert "nobody dies off-page" in client.get("/api/ledger").json()["hard_constraints"]


def test_the_whole_story_audit_answers_over_http(api) -> None:  # type: ignore[no-untyped-def]
    """The audit is the check that catches drift no single accept was wrong about.

    Layer 2 is off in this fixture, so this proves the route reaches the engine and the
    engine degrades to layer 1 rather than raising — which is what a deployment without
    the NLI model does.
    """
    client, state, _ = api
    state.engine().use_nli = False
    body = client.get("/api/flags?audit=true").json()
    assert body["flags"] == []
    assert body["by_layer"] == {}, "a clean story has nothing to report per layer"
    assert body["summary"].startswith("audit: 0 flags")


def test_editing_a_plan_candidate_keeps_the_writers_words(api) -> None:  # type: ignore[no-untyped-def]
    """This returned 200 and threw the writer's text away at every plan level.

    `_with_edited_prose` rewrote prose operations and passed everything else through, so an
    edit to an episode, scene or beat committed the model's paragraph instead — while still
    recording the edit signal, so the preference layer learned from a change that never
    happened. A writer found it by re-reading the node.
    """
    client, state, _ = api
    story = next(n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "story")
    proposed = client.post(
        "/api/propose", json={"node_id": story["id"], "level": "episode", "intent": "go on"}
    ).json()
    assert proposed["candidates"], "the mock produced no candidates"
    first = proposed["candidates"][0]

    mine = "Ronnie refuses the money for slightly under one second."
    response = client.post(
        "/api/action/edit", json={"proposal_id": first["proposal_id"], "text": mine}
    )
    assert response.status_code == 200
    episodes = [n for n in client.get("/api/tree").json()["nodes"] if n["node_type"] == "episode"]
    detail = client.get(f"/api/node/{episodes[-1]['id']}").json()
    assert detail["what_happens"] == mine


def test_a_criterion_weight_out_of_range_is_a_422_not_a_500(api) -> None:  # type: ignore[no-untyped-def]
    """A writer reaching for 1.5 to mean "this matters more" got a bare 500 and no body.

    They only noticed because the criterion was missing from the ledger afterwards, which
    is the worst way to find out that the objective you set was never set.
    """
    client, _, _ = api
    response = client.post(
        "/api/ledger/criterion",
        json={"name": "escalating cost", "description": "each lie costs more", "weight": 1.5},
    )
    assert response.status_code == 422
    assert "weight" in response.text
    assert client.get("/api/ledger").json()["criteria"] == []
