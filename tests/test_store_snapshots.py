"""Snapshot immutability, structural sharing, diff round-trip, and branching."""

from __future__ import annotations

import pytest
from tests.conftest import Fixture

from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddNode,
    Diff,
    DiffAuthor,
    SetDial,
    UpdateNode,
)
from storygit.domain.errors import SnapshotNotFoundError
from storygit.domain.ids import IdGenerator, NodeId
from storygit.domain.nodes import Beat, Scene
from storygit.store.branches import structural_diff
from storygit.store.repository import Repository
from storygit.store.snapshots import canonical_json


def test_commit_creates_a_new_snapshot_and_leaves_the_old_one_intact(fixture: Fixture) -> None:
    repo = fixture.repo
    before_id = repo.head()
    before_manifest = repo.snapshots.manifest(before_id)

    repo.commit_diff(
        Diff(
            ops=(UpdateNode(node_id=fixture.beat_a, fields={"title": "Beat A, revised"}),),
            author=DiffAuthor.human,
            intent="rename a beat",
        )
    )
    after_id = repo.head()

    assert after_id != before_id
    assert repo.snapshots.manifest(before_id) == before_manifest
    assert repo.state_at(before_id).nodes[fixture.beat_a].title == "Beat A"
    assert repo.state().nodes[fixture.beat_a].title == "Beat A, revised"


def test_unchanged_objects_are_shared_between_snapshots(fixture: Fixture) -> None:
    repo = fixture.repo
    before = repo.snapshots.manifest(repo.head())
    repo.commit_diff(Diff(ops=(UpdateNode(node_id=fixture.beat_a, fields={"title": "changed"}),)))
    after = repo.snapshots.manifest(repo.head())

    changed = {k for k in before["node"] if before["node"][k] != after["node"][k]}
    assert changed == {str(fixture.beat_a)}
    for key in before["node"].keys() - changed:
        assert before["node"][key] == after["node"][key], "unchanged nodes must reuse their hash"
    assert before["entity"] == after["entity"]
    assert before["fact"] == after["fact"]


def test_snapshot_ids_are_content_derived(repo: Repository, ids: IdGenerator) -> None:
    from storygit.domain.nodes import Story
    from storygit.domain.state import StoryState

    root = Story(id=NodeId("n_root"), title="S")
    first = repo.initialize(StoryState.build(nodes={root.id: root}))

    other = Repository.open(":memory:", clock=repo._clock)
    second = other.initialize(StoryState.build(nodes={root.id: root}))
    assert first == second


def test_structural_diff_round_trips(fixture: Fixture, ids: IdGenerator) -> None:
    repo = fixture.repo
    base_id = repo.head()
    new_scene = NodeId("n_scene2")
    repo.commit_diff(
        Diff(
            ops=(
                AddNode(
                    node=Scene(id=new_scene, parent_id=fixture.episode, title="Scene 2", position=1)
                ),
                AddNode(node=Beat(id=NodeId("n_beat_e"), parent_id=new_scene, title="Beat E")),
                UpdateNode(node_id=fixture.beat_d, fields={"what_happens": "D changes"}),
                SetDial(value=0.8),
            ),
            author=DiffAuthor.human,
            intent="grow the story",
        )
    )
    target_id = repo.head()

    replay = apply(repo.state_at(base_id), repo.diff(base_id, target_id))
    assert repo.snapshots.manifest_of_state(replay) == repo.snapshots.manifest(target_id)


def test_structural_diff_round_trips_on_removals(fixture: Fixture) -> None:
    repo = fixture.repo
    base_id = repo.head()
    repo.commit_diff(Diff(ops=(UpdateNode(node_id=fixture.beat_b, fields={"consumes": []}),)))
    from storygit.domain.diff import RemoveNode

    repo.commit_diff(Diff(ops=(RemoveNode(node_id=fixture.beat_a),)))
    target_id = repo.head()

    replay = apply(repo.state_at(base_id), repo.diff(base_id, target_id))
    assert repo.snapshots.manifest_of_state(replay) == repo.snapshots.manifest(target_id)


def test_structural_diff_round_trips_through_locks(fixture: Fixture) -> None:
    repo = fixture.repo
    from storygit.domain.diff import SetLock

    repo.commit_diff(Diff(ops=(SetLock(node_id=fixture.beat_b),)))
    base_id = repo.head()
    other = repo.state_at(base_id)
    target = apply(
        other,
        Diff(ops=(AddNode(node=Beat(id=NodeId("n_new"), parent_id=fixture.scene, position=9)),)),
    )
    # A locked node changed on the far side: the diff must unlock, change, and relock.
    target = target.evolve(
        nodes={
            **target.nodes,
            fixture.beat_b: target.nodes[fixture.beat_b].model_copy(
                update={"what_happens": "rewritten upstream"}
            ),
        }
    )
    replay = apply(other, structural_diff(other, target))
    assert repo.snapshots.manifest_of_state(replay) == repo.snapshots.manifest_of_state(target)


def test_history_walks_the_parent_chain(fixture: Fixture) -> None:
    history = fixture.repo.history()
    assert [h["intent"] for h in history][-1] == "initialize"
    assert len(history) == 4


def test_missing_branch_raises(repo: Repository) -> None:
    with pytest.raises(SnapshotNotFoundError):
        repo.head("does-not-exist")


def test_canonical_json_is_stable() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'
