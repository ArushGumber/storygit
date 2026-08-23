"""Branching, divergence, cross-branch diff, and three-way merge."""

from __future__ import annotations

import pytest
from tests.conftest import Fixture

from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddEntity,
    AddFact,
    AddKnows,
    AddNode,
    AddStyleNote,
    Diff,
    DiffAuthor,
    OpenThread,
    RemoveEntity,
    RemoveFact,
    RemoveKnows,
    RemoveNode,
    RemoveThread,
    UpdateNode,
)
from storygit.domain.errors import SnapshotNotFoundError
from storygit.domain.ids import NodeId
from storygit.domain.ledger import StyleNote
from storygit.domain.nodes import Beat
from storygit.domain.threads import Thread
from storygit.domain.world import Entity, EntityKind, Fact, FactSource, Knows, Predicate


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


def test_merge_carries_every_object_kind_including_removals(fixture: Fixture) -> None:
    """Merge is claimed to work at object granularity, not just over the plan tree.

    The state has six kinds of object, and a branch can add or remove any of them.
    This walks all six through a merge in both directions — something taken from
    theirs, and something they deleted — because a merge that silently drops a
    thread or an epistemic edge would be invisible until a much later contradiction.
    """
    repo, ids = fixture.repo, fixture.ids
    thread_t, fact_h = ids.thread(), ids.fact()

    # Everything the alt branch will later remove has to exist in the common ancestor.
    repo.commit_diff(
        Diff(
            ops=(
                OpenThread(
                    thread=Thread(
                        id=thread_t,
                        description="who burned the granary",
                        opened_at_beat=fixture.beat_c,
                        last_touched_beat=fixture.beat_c,
                    )
                ),
                AddFact(
                    fact=Fact(
                        id=fact_h,
                        subject=fixture.kael,
                        predicate=Predicate.trait,
                        object_text="flinches at open flame",
                        valid_from_beat=fixture.beat_c,
                        established_by_beat=fixture.beat_c,
                        source=FactSource.ai,
                    )
                ),
                AddKnows(
                    knows=Knows(
                        character=fixture.kael, fact=fixture.fact_f, since_beat=fixture.beat_c
                    )
                ),
            ),
            author=DiffAuthor.human,
            intent="seed one object of every kind",
        )
    )

    repo.create_branch("alt")
    repo.commit_diff(
        Diff(ops=(UpdateNode(node_id=fixture.beat_a, fields={"what_happens": "main version"}),)),
        branch="main",
    )

    ferryman, fact_i, thread_u = ids.entity(), ids.fact(), ids.thread()
    repo.commit_diff(
        Diff(
            ops=(
                # additions, one per kind
                AddNode(node=Beat(id=NodeId("n_alt_beat"), parent_id=fixture.scene, position=7)),
                AddEntity(
                    entity=Entity(id=ferryman, kind=EntityKind.character, name="the Ferryman")
                ),
                AddFact(
                    fact=Fact(
                        id=fact_i,
                        subject=ferryman,
                        predicate=Predicate.location,
                        object_entity=fixture.ashfall,
                        valid_from_beat=fixture.beat_b,
                        established_by_beat=fixture.beat_b,
                        source=FactSource.ai,
                    )
                ),
                AddKnows(
                    knows=Knows(character=fixture.kael, fact=fact_i, since_beat=fixture.beat_b)
                ),
                OpenThread(
                    thread=Thread(
                        id=thread_u,
                        description="what the Ferryman wants",
                        opened_at_beat=fixture.beat_b,
                        last_touched_beat=fixture.beat_b,
                    )
                ),
                AddStyleNote(note=StyleNote(text="shorter sentences at the turn")),
                # removals, one per kind
                RemoveNode(node_id=fixture.beat_d),
                RemoveEntity(entity_id=fixture.kell),
                RemoveFact(fact_id=fact_h),
                RemoveKnows(character=fixture.kael, fact_id=fixture.fact_f),
                RemoveThread(thread_id=thread_t),
            ),
            author=DiffAuthor.human,
            intent="touch every kind on the branch",
        ),
        branch="alt",
    )

    result = repo.merge_branches("main", "alt")
    assert result.clean
    merged = apply(repo.state("main"), result.diff)

    # Ours is untouched where only we changed it.
    assert merged.nodes[fixture.beat_a].what_happens == "main version"
    # Additions arrive, one kind at a time.
    assert NodeId("n_alt_beat") in merged.nodes
    assert ferryman in merged.entities
    assert fact_i in merged.facts
    assert (fixture.kael, fact_i) in merged.knows
    assert thread_u in merged.threads
    assert any(note.text == "shorter sentences at the turn" for note in merged.ledger.style_notes)
    # Removals arrive too, which is the half a naive union-merge gets wrong.
    assert fixture.beat_d not in merged.nodes
    assert fixture.kell not in merged.entities
    assert fact_h not in merged.facts
    assert (fixture.kael, fixture.fact_f) not in merged.knows
    assert thread_t not in merged.threads
