"""Branching, divergence, cross-branch diff, and three-way merge."""

from __future__ import annotations

import pytest
from tests.conftest import Fixture

from storygit.domain.apply import apply
from storygit.domain.diff import AddNode, Diff, DiffAuthor, UpdateNode
from storygit.domain.errors import SnapshotNotFoundError
from storygit.domain.ids import NodeId
from storygit.domain.nodes import Beat


def test_branch_starts_at_the_same_state(fixture: Fixture) -> None:
    repo = fixture.repo
    repo.create_branch("what-if")
    assert repo.head("what-if") == repo.head("main")
    assert repo.state("what-if").nodes == repo.state("main").nodes


def test_duplicate_branch_name_raises(fixture: Fixture) -> None:
    fixture.repo.create_branch("alt")
    with pytest.raises(SnapshotNotFoundError):
        fixture.repo.create_branch("alt")


def test_divergent_branches_diff(fixture: Fixture) -> None:
    repo = fixture.repo
    repo.create_branch("alt")
    repo.commit_diff(
        Diff(ops=(UpdateNode(node_id=fixture.beat_a, fields={"title": "main edit"}),)),
        branch="main",
    )
    repo.commit_diff(
        Diff(ops=(UpdateNode(node_id=fixture.beat_d, fields={"title": "alt edit"}),)),
        branch="alt",
    )
    diff = repo.diff_branches("main", "alt")
    changed = {op.node_id for op in diff.ops if isinstance(op, UpdateNode)}
    assert changed == {fixture.beat_a, fixture.beat_d}


def test_merge_combines_disjoint_changes(fixture: Fixture) -> None:
    repo = fixture.repo
    repo.create_branch("alt")
    repo.commit_diff(
        Diff(
            ops=(UpdateNode(node_id=fixture.beat_a, fields={"what_happens": "main version"}),),
            author=DiffAuthor.human,
        ),
        branch="main",
    )
    repo.commit_diff(
        Diff(
            ops=(
                AddNode(node=Beat(id=NodeId("n_alt_beat"), parent_id=fixture.scene, position=7)),
                UpdateNode(node_id=fixture.beat_d, fields={"what_happens": "alt version"}),
            ),
            author=DiffAuthor.human,
        ),
        branch="alt",
    )

    result = repo.merge_branches("main", "alt")
    assert result.clean
    merged = apply(repo.state("main"), result.diff)
    assert merged.nodes[fixture.beat_a].what_happens == "main version"
    assert merged.nodes[fixture.beat_d].what_happens == "alt version"
    assert NodeId("n_alt_beat") in merged.nodes


def test_merge_reports_conflicts_without_guessing(fixture: Fixture) -> None:
    repo = fixture.repo
    repo.create_branch("alt")
    repo.commit_diff(
        Diff(ops=(UpdateNode(node_id=fixture.beat_a, fields={"what_happens": "main version"}),)),
        branch="main",
    )
    repo.commit_diff(
        Diff(ops=(UpdateNode(node_id=fixture.beat_a, fields={"what_happens": "alt version"}),)),
        branch="alt",
    )

    result = repo.merge_branches("main", "alt")
    assert not result.clean
    assert [c.logical_id for c in result.conflicts] == [str(fixture.beat_a)]
    conflict = result.conflicts[0]
    assert conflict.kind == "node"
    assert conflict.ours_hash != conflict.theirs_hash != conflict.base_hash
    # The conflicted object is left exactly as it was on our branch.
    merged = apply(repo.state("main"), result.diff)
    assert merged.nodes[fixture.beat_a].what_happens == "main version"


def test_merge_base_requires_shared_history(fixture: Fixture) -> None:
    from storygit.domain.nodes import Story
    from storygit.domain.state import StoryState
    from storygit.store.repository import Repository

    other = Repository.open(":memory:")
    root = Story(id=NodeId("n_other"), title="Other")
    foreign = other.initialize(StoryState.build(nodes={root.id: root}))
    with pytest.raises(SnapshotNotFoundError):
        fixture.repo.merge_base(fixture.repo.head(), foreign)
