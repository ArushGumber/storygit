"""The wire format.

These are the only types the frontend knows about, and they are deliberately *not* the
domain models. Two reasons.

**The interface needs different shapes.** A plan-tree node on screen wants its children
inlined and its authorship ratio computed; the domain model has neither, because neither
belongs in the thing that gets hashed into a snapshot.

**The wire format is a contract.** Serializing domain models directly would make every
internal refactor a breaking API change, and would leak fields the frontend has no business
seeing. The mapping functions in this module are the seam, and they are the only place that
has to change when either side moves.

Where a domain model happens to be exactly right — `Fact`, `Thread`, `Flag` — it is reused
rather than duplicated, because a wrapper that adds nothing is a wrapper that will drift.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from storygit.continuity.flags import Flag
from storygit.domain.ids import EntityId, FactId, NodeId, ProposalId
from storygit.domain.ledger import Criterion, StyleNote
from storygit.domain.nodes import BaseNode, Episode, NodeStatus, NodeType, Prose
from storygit.domain.provenance import Authorship, authorship_ratio
from storygit.domain.state import StoryState
from storygit.domain.threads import Thread
from storygit.domain.world import Entity, Fact
from storygit.graph.propagation import StaleMark


class NodeSummary(BaseModel):
    """One node as the plan tree shows it.

    Attributes:
        id: Node id.
        parent_id: Parent, or ``None`` for the root.
        node_type: Which level.
        title: Display label.
        status: Review state, which drives the colour.
        stale_reason: Why it is stale or flagged for review.
        locked: Whether the writer has frozen it.
        position: Order among siblings.
        seq: Global reading order, so the client never has to recompute it.
        has_prose: Whether a prose child exists.
        children: Child ids, in reading order.
        flag_count: Continuity flags currently on it.
    """

    model_config = ConfigDict(frozen=True)

    id: NodeId
    parent_id: NodeId | None
    node_type: NodeType
    title: str
    status: NodeStatus
    stale_reason: str | None = None
    locked: bool = False
    position: int = 0
    seq: int = 0
    has_prose: bool = False
    children: tuple[NodeId, ...] = ()
    flag_count: int = 0


class TreeResponse(BaseModel):
    """The whole plan tree, flat, with the root named.

    Flat rather than nested because the client needs random access by id constantly —
    selecting a node, following a flag citation, highlighting a stale mark — and rebuilding
    an index from a nested structure on every render is wasted work.
    """

    model_config = ConfigDict(frozen=True)

    branch: str
    root_id: NodeId | None
    nodes: tuple[NodeSummary, ...]
    stale_count: int = 0
    review_count: int = 0


class ProseSpan(BaseModel):
    """One provenance span, for rendering authorship in the prose pane."""

    model_config = ConfigDict(frozen=True)

    start: int
    end: int
    source: Authorship
    proposal_id: ProposalId | None = None


class NodeDetail(BaseModel):
    """Everything the centre pane shows for one node."""

    model_config = ConfigDict(frozen=True)

    node: NodeSummary
    what_happens: str = ""
    audience_learns: str = ""
    audience_feels: str = ""
    location: str = ""
    time: str = ""
    prose: str = ""
    spans: tuple[ProseSpan, ...] = ()
    authorship: dict[str, float] = Field(default_factory=dict)
    produces: tuple[FactId, ...] = ()
    consumes: tuple[FactId, ...] = ()
    episode: dict[str, Any] | None = None
    flags: tuple[Flag, ...] = ()


class FactView(BaseModel):
    """A fact, rendered for the world-state pane.

    Carries the English sentence alongside the structured fields, because the pane shows
    the sentence and the client should not be reimplementing the templates.
    """

    model_config = ConfigDict(frozen=True)

    fact: Fact
    sentence: str
    subject_name: str
    established_in: str
    known_by: tuple[str, ...] = ()


class SliceResponse(BaseModel):
    """The world-state pane: entities, facts, knowledge, threads."""

    model_config = ConfigDict(frozen=True)

    beat_id: NodeId | None
    entities: tuple[Entity, ...] = ()
    facts: tuple[FactView, ...] = ()
    threads: tuple[Thread, ...] = ()
    hard_constraints: tuple[str, ...] = ()


class CandidateView(BaseModel):
    """One candidate as the centre pane shows it.

    The diff is sent as its English delta summary rather than as raw operations: the whole
    argument of the design is that the writer reviews *changes* rather than text, and a
    list of typed ops is not a change a novelist can read. The op count is sent too, so the
    interface can say how large the change is.
    """

    model_config = ConfigDict(frozen=True)

    proposal_id: ProposalId
    level: str
    axis_label: str = ""
    rationale: str = ""
    delta_summary: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    text: str = ""
    op_count: int = 0
    stale_preview: int = 0
    flags: tuple[Flag, ...] = ()
    base_quality: float = 0.0
    surprise: float = 0.0
    effective_quality: float = 0.0
    selected: bool = False


class ProposeRequest(BaseModel):
    """Ask for candidates at a node."""

    node_id: NodeId | None = None
    level: str
    intent: str = ""


class ProposeResponse(BaseModel):
    """The candidates, shortlisted first."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[CandidateView, ...] = ()
    shown: tuple[ProposalId, ...] = ()


class ActionRequest(BaseModel):
    """Accept or reject a proposal."""

    proposal_id: ProposalId
    reason: str = ""


class EditRequest(BaseModel):
    """Accept a proposal with the writer's own text."""

    proposal_id: ProposalId
    text: str


class ActionResponse(BaseModel):
    """What changed, for the right-hand pane."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    bible_diff: tuple[str, ...] = ()
    added: int = 0
    ended: int = 0
    removed: int = 0
    marks: tuple[dict[str, Any], ...] = ()
    flags: tuple[Flag, ...] = ()
    extracted: bool = False


class LedgerResponse(BaseModel):
    """The writer's standing instructions, including what was mined from their edits."""

    model_config = ConfigDict(frozen=True)

    dial: float
    locks: tuple[NodeId, ...] = ()
    hard_constraints: tuple[str, ...] = ()
    style_notes: tuple[StyleNote, ...] = ()
    active_style_notes: tuple[str, ...] = ()
    criteria: tuple[Criterion, ...] = ()
    rejected_directions: tuple[str, ...] = ()
    learned: dict[str, Any] = Field(default_factory=dict)


class BranchesResponse(BaseModel):
    """Branch names, their heads, and which one is current."""

    model_config = ConfigDict(frozen=True)

    current: str
    branches: dict[str, str] = Field(default_factory=dict)


class MergeResponse(BaseModel):
    """A merge preview: what would combine, and what needs a decision."""

    model_config = ConfigDict(frozen=True)

    clean: bool
    conflicts: tuple[dict[str, Any], ...] = ()
    summary: tuple[str, ...] = ()
    committed: str | None = None


class AuthorshipResponse(BaseModel):
    """How much of the story each source wrote."""

    model_config = ConfigDict(frozen=True)

    overall: dict[str, float] = Field(default_factory=dict)
    sentences: int = 0
    by_node: dict[str, dict[str, float]] = Field(default_factory=dict)


class ProblemDetail(BaseModel):
    """The error shape every failing route returns.

    One shape for every failure means the client has exactly one error path to handle, and
    the ``kind`` field is the typed engine error's class name — so the interface can tell a
    rate limit (retry) from a locked node (tell the writer) without parsing prose.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    detail: str
    retry_after: float | None = None


# --- mapping ------------------------------------------------------------------


def node_summary(
    state: StoryState, node: BaseNode, *, flag_counts: dict[NodeId, int] | None = None
) -> NodeSummary:
    """Map a domain node onto the wire shape."""
    counts = flag_counts or {}
    return NodeSummary(
        id=node.id,
        parent_id=node.parent_id,
        node_type=node.node_type,
        title=node.title or node.node_type.value,
        status=node.status,
        stale_reason=node.stale_reason,
        locked=state.is_locked(node.id),
        position=node.position,
        seq=state.seq.get(node.id, 0),
        has_prose=node.id in state.prose_of_beat,
        children=state.children.get(node.id, ()),
        flag_count=counts.get(node.id, 0),
    )


def tree_response(state: StoryState, branch: str, *, flags: tuple[Flag, ...] = ()) -> TreeResponse:
    """Build the plan-tree response."""
    counts: dict[NodeId, int] = {}
    for flag in flags:
        if flag.node_id is not None:
            counts[flag.node_id] = counts.get(flag.node_id, 0) + 1
    nodes = sorted(state.nodes.values(), key=lambda n: state.seq.get(n.id, 0))
    summaries = tuple(node_summary(state, node, flag_counts=counts) for node in nodes)
    return TreeResponse(
        branch=branch,
        root_id=state.root_id,
        nodes=summaries,
        stale_count=sum(1 for n in summaries if n.status is NodeStatus.stale),
        review_count=sum(
            1 for n in summaries if n.status is not NodeStatus.stale and n.stale_reason
        ),
    )


def node_detail(state: StoryState, node: BaseNode, *, flags: tuple[Flag, ...] = ()) -> NodeDetail:
    """Build the centre pane's node detail."""
    prose = state.prose_for(node.id) if node.node_type is NodeType.beat else None
    if isinstance(node, Prose):
        prose = node
    spans = prose.spans if prose is not None else ()
    ratio = authorship_ratio(spans)
    episode_fields: dict[str, Any] | None = None
    if isinstance(node, Episode):
        episode_fields = {
            "hook": node.hook,
            "cliffhanger": node.cliffhanger,
            "recap_of_previous": node.recap_of_previous,
            "threads_in": list(node.threads_in),
            "threads_out": list(node.threads_out),
            "episode_goals": list(node.episode_goals),
            "target_length": node.target_length,
        }
    return NodeDetail(
        node=node_summary(state, node),
        what_happens=node.what_happens,
        audience_learns=node.audience_learns,
        audience_feels=node.audience_feels,
        location=node.location,
        time=node.time,
        prose=prose.text if prose is not None else "",
        spans=tuple(
            ProseSpan(start=s.start, end=s.end, source=s.source, proposal_id=s.proposal_id)
            for s in spans
        ),
        authorship={k.value: round(v, 3) for k, v in ratio.items()},
        produces=tuple(getattr(node, "produces", ())),
        consumes=tuple(getattr(node, "consumes", ())),
        episode=episode_fields,
        flags=tuple(f for f in flags if f.node_id == node.id),
    )


def fact_view(state: StoryState, fact: Fact) -> FactView:
    """Map a fact onto the world-state pane's shape."""
    names = state.entity_names()
    established = state.nodes.get(fact.established_by_beat)
    knowers = tuple(
        names.get(character, str(character))
        for (character, fact_id) in sorted(state.knows)
        if fact_id == fact.id
    )
    return FactView(
        fact=fact,
        sentence=fact.sentence(names),
        subject_name=names.get(fact.subject, str(fact.subject)),
        established_in=(
            (established.title or established.node_type.value)
            if established is not None
            else "a removed beat"
        ),
        known_by=knowers,
    )


def slice_response(
    state: StoryState, entity_ids: set[EntityId], at_beat: NodeId | None
) -> SliceResponse:
    """Build the world-state pane."""
    from storygit.graph.dependency import hard_constraints

    beats = state.beats_in_order()
    target = at_beat or (beats[-1].id if beats else None)
    facts: tuple[Fact, ...] = ()
    if target is not None:
        facts = tuple(
            f
            for f in state.facts_valid_at(target)
            if not entity_ids or f.subject in entity_ids or f.object_entity in entity_ids
        )
    entities = tuple(
        state.entities[e] for e in sorted(entity_ids or set(state.entities)) if e in state.entities
    )
    from storygit.domain.threads import ThreadStatus

    return SliceResponse(
        beat_id=target,
        entities=entities,
        facts=tuple(fact_view(state, f) for f in facts),
        threads=tuple(
            t
            for t in sorted(state.threads.values(), key=lambda t: t.id)
            if t.status is ThreadStatus.open
        ),
        hard_constraints=tuple(hard_constraints(state)),
    )


def candidate_view(candidate: Any) -> CandidateView:
    """Map an engine candidate onto the wire shape."""
    from storygit.selection.select import candidate_text

    proposal = candidate.proposal
    return CandidateView(
        proposal_id=proposal.id,
        level=proposal.level.value,
        axis_label=candidate.axis_label,
        rationale=proposal.rationale,
        delta_summary=proposal.delta_summary,
        notes=proposal.notes,
        text=candidate_text(proposal),
        op_count=len(proposal.diff),
        stale_preview=proposal.stale_preview,
        flags=candidate.flags,
        base_quality=round(candidate.base_quality, 4),
        surprise=round(candidate.surprise, 4),
        effective_quality=round(candidate.effective_quality, 4),
        selected=candidate.selected,
    )


def mark_dict(mark: StaleMark) -> dict[str, Any]:
    """Map a stale mark onto the wire shape."""
    return {
        "node_id": mark.node_id,
        "kind": mark.kind.value,
        "reason": mark.reason,
        "origin_beat": mark.origin_beat,
        "origin_fact": mark.origin_fact,
    }


class RevisionPreviewResponse(BaseModel):
    """What a revision or removal would cost, before it is committed.

    The blast radius shown *before* the writer commits is what makes an accepted node
    revisable rather than frozen -- the same contract striking a fact already had.
    """

    model_config = ConfigDict(frozen=True)

    removed_nodes: tuple[str, ...] = ()
    removed_facts: tuple[str, ...] = ()
    marks: tuple[dict[str, Any], ...] = ()
    lines: tuple[str, ...] = ()
    summary: str = ""


def revision_preview(preview: Any) -> RevisionPreviewResponse:
    """Map an engine revision preview onto the wire shape."""
    parts: list[str] = []
    if preview.removed_nodes:
        parts.append(f"removes {len(preview.removed_nodes)} node(s)")
    if preview.removed_facts:
        parts.append(f"invalidates {len(preview.removed_facts)} fact(s)")
    if preview.marks:
        parts.append(f"marks {len(preview.marks)} node(s) for review")
    return RevisionPreviewResponse(
        removed_nodes=tuple(str(n) for n in preview.removed_nodes),
        removed_facts=tuple(str(f) for f in preview.removed_facts),
        marks=tuple(mark_dict(m) for m in preview.marks),
        lines=tuple(preview.bible_diff.lines),
        summary=", ".join(parts) or "changes this node only",
    )


def action_response(result: Any) -> ActionResponse:
    """Map an accept result onto the wire shape."""
    counts = result.bible_diff.counts()
    return ActionResponse(
        snapshot_id=str(result.snapshot_id),
        bible_diff=result.bible_diff.lines,
        added=counts["added"],
        ended=counts["ended"],
        removed=counts["removed"],
        marks=tuple(mark_dict(m) for m in result.marks),
        flags=result.flags,
        extracted=result.extracted,
    )


def authorship_response(state: StoryState) -> AuthorshipResponse:
    """Compute authorship ratios across the whole story."""
    from storygit.domain.provenance import merge_ratios

    per_node: dict[str, dict[str, float]] = {}
    ratios: list[tuple[dict[Authorship, float], int]] = []
    total = 0
    for beat in state.beats_in_order():
        prose = state.prose_for(beat.id)
        if prose is None or not prose.spans:
            continue
        ratio = authorship_ratio(prose.spans)
        sentences = max(span.end for span in prose.spans)
        total += sentences
        ratios.append((ratio, sentences))
        per_node[str(beat.id)] = {k.value: round(v, 3) for k, v in ratio.items()}
    overall = merge_ratios(ratios)
    return AuthorshipResponse(
        overall={k.value: round(v, 3) for k, v in overall.items()},
        sentences=total,
        by_node=per_node,
    )


def ledger_response(state: StoryState, learned: dict[str, Any]) -> LedgerResponse:
    """Build the writer-ledger pane."""
    ledger = state.ledger
    return LedgerResponse(
        dial=ledger.dial,
        locks=tuple(sorted(ledger.locks)),
        hard_constraints=ledger.hard_constraints,
        style_notes=ledger.style_notes,
        active_style_notes=tuple(n.text for n in ledger.active_style_notes()),
        criteria=ledger.criteria,
        rejected_directions=tuple(r.text for r in ledger.rejected_directions),
        learned=learned,
    )


__all__ = [
    "ActionRequest",
    "ActionResponse",
    "AuthorshipResponse",
    "BranchesResponse",
    "CandidateView",
    "EditRequest",
    "FactView",
    "LedgerResponse",
    "MergeResponse",
    "NodeDetail",
    "NodeSummary",
    "ProblemDetail",
    "ProposeRequest",
    "ProposeResponse",
    "SliceResponse",
    "TreeResponse",
    "action_response",
    "authorship_response",
    "candidate_view",
    "fact_view",
    "ledger_response",
    "mark_dict",
    "node_detail",
    "node_summary",
    "slice_response",
    "tree_response",
]
