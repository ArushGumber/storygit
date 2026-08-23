"""The engine: the loop the whole system exists to run.

    human intent -> AI proposal -> human action -> system update -> AI proposal

Everything else in this package is a component of that loop; this is the loop itself.
It holds no state of its own — the repository owns the story, the signal store owns the
history of writer actions — so it can be constructed per request in chunk 6 without
anything going stale.

Three things happen on every accept, in this order, and the order matters:

1. The diff is committed. Nothing has been rewritten; the story simply has one more
   version.
2. If the accepted change wrote prose, facts are extracted from it, so the world graph
   stays populated even for text the writer typed themselves.
3. Propagation marks what the change affected, as its own commit. The writer sees a
   bible diff (facts added, ended, struck) and a list of what is now stale.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from storygit.agents.propose import Proposal, Proposer
from storygit.agents.schemas import Level
from storygit.domain.diff import (
    AddCriterion,
    AddNode,
    AddRejectedDirection,
    AddStyleNote,
    ClearLock,
    Diff,
    DiffAuthor,
    Op,
    SetDial,
    SetLock,
    SetNodeStatus,
    SetProse,
)
from storygit.domain.ids import IdGenerator, NodeId, ProposalId, SnapshotId
from storygit.domain.ledger import Criterion, StyleNote
from storygit.domain.nodes import NodeStatus, Prose
from storygit.domain.provenance import Authorship, ProvenanceSpan
from storygit.domain.state import StoryState
from storygit.domain.world import Fact
from storygit.graph.propagation import StaleMark, marks_to_diff, propagate_change
from storygit.providers.router import Router
from storygit.store.branches import DEFAULT_BRANCH
from storygit.store.repository import Repository
from storygit.store.signals import Signal, SignalKind, SignalStore


class BibleDiff(BaseModel):
    """What a change did to the world graph, in the writer's terms.

    Shown on every accept so that "what did I just agree to" has an answer that is about
    the story rather than about JSON.

    Attributes:
        added: Facts newly asserted.
        ended: Facts whose validity was closed off by this change.
        removed: Facts struck outright.
        lines: The same information as sentences, ready to render.
    """

    model_config = ConfigDict(frozen=True)

    added: tuple[Fact, ...] = ()
    ended: tuple[Fact, ...] = ()
    removed: tuple[Fact, ...] = ()
    lines: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether the change touched no facts at all."""
        return not (self.added or self.ended or self.removed)


class AcceptResult(BaseModel):
    """Everything the writer should see after accepting a proposal.

    Attributes:
        snapshot_id: The commit the accepted diff produced.
        propagation_snapshot_id: The follow-up commit recording the stale marks, if any.
        bible_diff: What changed in the world graph.
        marks: What is now stale or wants review.
        extracted: Whether facts were extracted from newly written prose.
    """

    model_config = ConfigDict(frozen=True)

    snapshot_id: SnapshotId
    propagation_snapshot_id: SnapshotId | None = None
    bible_diff: BibleDiff = BibleDiff()
    marks: tuple[StaleMark, ...] = ()
    extracted: bool = False


class Engine:
    """Proposes, accepts, rejects, and edits — the writer-facing verbs.

    Attributes:
        repo: The story repository.
        router: Model access.
        proposer: Generation.
        signals: The learning signal store.
    """

    def __init__(
        self,
        repository: Repository,
        router: Router,
        *,
        ids: IdGenerator | None = None,
        branch: str = DEFAULT_BRANCH,
    ) -> None:
        """Wire the engine.

        Args:
            repository: The story repository. Owns all state.
            router: Model access.
            ids: Id generator; seed it to make a whole session reproducible.
            branch: The branch this engine reads and writes.
        """
        self.repo = repository
        self.router = router
        self.branch = branch
        self.ids = ids or IdGenerator()
        self.proposer = Proposer(router, self.ids)
        self.signals = SignalStore(repository.conn)
        self._pending: dict[ProposalId, Proposal] = {}

    # --- reads ------------------------------------------------------------------

    def state(self) -> StoryState:
        """Current state of this engine's branch."""
        return self.repo.state(self.branch)

    def pending(self, proposal_id: ProposalId) -> Proposal | None:
        """A proposal this engine generated and is still holding."""
        return self._pending.get(proposal_id)

    # --- propose ----------------------------------------------------------------

    async def propose_at(
        self,
        level: Level,
        *,
        node_id: NodeId | None = None,
        intent: str = "",
        k: int = 3,
    ) -> list[Proposal]:
        """Generate candidates at a level, and remember them pending a decision.

        Args:
            level: What to propose.
            node_id: The node to attach to; defaults per level.
            intent: The writer's instruction.
            k: How many candidates.

        Returns:
            The candidates, each already carrying its diff, rationale, delta summary,
            and the count of nodes it would mark stale.
        """
        proposals = await self.proposer.propose(
            self.state(), level, target_node_id=node_id, intent=intent, k=k
        )
        for proposal in proposals:
            self._pending[proposal.id] = proposal
        return proposals

    # --- actions ----------------------------------------------------------------

    async def accept(
        self,
        proposal_id: ProposalId,
        *,
        shown_with: tuple[ProposalId, ...] = (),
    ) -> AcceptResult:
        """Commit a proposal, extract any new facts, and propagate.

        Args:
            proposal_id: The proposal to accept.
            shown_with: The other candidates it was shown alongside, recorded so the
                preference layer can learn from the comparison rather than the choice.

        Returns:
            The bible diff, the stale marks, and the snapshots produced.

        Raises:
            KeyError: If the proposal is not one this engine is holding.
        """
        proposal = self._require(proposal_id)
        before = self.state()
        snapshot_id = self.repo.commit_diff(
            proposal.diff, branch=self.branch, intent=proposal.diff.intent
        )
        after = self.repo.state(self.branch)

        extracted = False
        prose_beat = self._prose_beat(proposal, after)
        if prose_beat is not None:
            prose = after.prose_for(prose_beat)
            if prose is not None and prose.text.strip():
                extraction = await self.proposer.extract(after, prose_beat, prose.text)
                if len(extraction):
                    snapshot_id = self.repo.commit_diff(extraction, branch=self.branch)
                    after = self.repo.state(self.branch)
                    extracted = True

        self.signals.record(
            Signal(
                kind=SignalKind.accept,
                branch=self.branch,
                node_id=proposal.target_node_id,
                proposal_id=proposal.id,
                shown_with=tuple(p for p in shown_with if p != proposal.id),
                payload={"level": proposal.level.value, "axis": proposal.axis},
            )
        )

        marks = propagate_change(before, after)
        propagation_snapshot: SnapshotId | None = None
        if marks:
            mark_diff = marks_to_diff(after, marks)
            if len(mark_diff):
                propagation_snapshot = self.repo.commit_diff(mark_diff, branch=self.branch)

        self._pending.pop(proposal_id, None)
        return AcceptResult(
            snapshot_id=snapshot_id,
            propagation_snapshot_id=propagation_snapshot,
            bible_diff=bible_diff(before, self.repo.state(self.branch)),
            marks=tuple(marks),
            extracted=extracted,
        )

    def reject(self, proposal_id: ProposalId, *, reason: str = "") -> None:
        """Record a rejection, and add it to the ledger as a direction to avoid.

        Rejections are not discarded: they become negative exemplars for retrieval and
        negative examples for the preference head, and the writer's stated reason
        becomes a standing instruction they can see and delete.

        Args:
            proposal_id: The proposal being rejected.
            reason: The writer's reason, if they gave one.

        Raises:
            KeyError: If the proposal is not one this engine is holding.
        """
        proposal = self._require(proposal_id)
        self.signals.record(
            Signal(
                kind=SignalKind.reject,
                branch=self.branch,
                node_id=proposal.target_node_id,
                proposal_id=proposal.id,
                payload={
                    "level": proposal.level.value,
                    "axis": proposal.axis,
                    "reason": reason,
                    "rationale": proposal.rationale,
                },
            )
        )
        if reason:
            self.repo.commit_diff(
                Diff(
                    ops=(AddRejectedDirection(text=reason, proposal_id=proposal.id),),
                    author=DiffAuthor.human,
                    intent="record a rejected direction",
                ),
                branch=self.branch,
            )
        self._pending.pop(proposal_id, None)

    async def edit(self, proposal_id: ProposalId, new_text: str) -> AcceptResult:
        """Accept a proposal with the writer's own words in place of the model's.

        The before/after pair is the most informative signal the system gets: it says
        not just that the writer wanted something different, but *what* different looks
        like. Chunk 4 mines exactly these pairs into style rules and an edit-direction
        vector.

        Args:
            proposal_id: The proposal being edited.
            new_text: The writer's replacement prose.

        Returns:
            The same result an accept produces.

        Raises:
            KeyError: If the proposal is not one this engine is holding.
        """
        proposal = self._require(proposal_id)
        original = _prose_text(proposal)
        edited = _with_edited_prose(proposal, new_text)
        self._pending[proposal.id] = edited
        self.signals.record(
            Signal(
                kind=SignalKind.edit,
                branch=self.branch,
                node_id=proposal.target_node_id,
                proposal_id=proposal.id,
                payload={
                    "level": proposal.level.value,
                    "before": original,
                    "after": new_text,
                },
            )
        )
        return await self.accept(proposal.id)

    async def write_beat(self, beat_id: NodeId, text: str) -> AcceptResult:
        """The hand-written path: the writer writes, the system only reads.

        No generation happens. The prose is stored as human-authored, facts are
        extracted from it, and propagation runs — so a hand-written beat participates in
        continuity exactly like a generated one.

        Args:
            beat_id: The beat being written.
            text: The writer's prose.

        Returns:
            The bible diff and marks, as for any accept.
        """
        before = self.state()
        sentences = max(1, text.count(".") + text.count("?") + text.count("!"))
        span = ProvenanceSpan(start=0, end=sentences, source=Authorship.human)
        existing = before.prose_of_beat.get(beat_id)
        ops: tuple[Op, ...]
        if existing is not None:
            ops = (SetProse(node_id=existing, text=text, spans=(span,)),)
        else:
            ops = (
                AddNode(
                    node=Prose(
                        id=self.ids.node(),
                        parent_id=beat_id,
                        title="prose",
                        text=text,
                        spans=(span,),
                    )
                ),
            )
        snapshot_id = self.repo.commit_diff(
            Diff(ops=ops, author=DiffAuthor.human, intent="hand-written prose"),
            branch=self.branch,
        )
        after = self.repo.state(self.branch)

        extraction = await self.proposer.extract(after, beat_id, text)
        extracted = False
        if len(extraction):
            snapshot_id = self.repo.commit_diff(extraction, branch=self.branch)
            after = self.repo.state(self.branch)
            extracted = True

        marks = propagate_change(before, after)
        propagation_snapshot: SnapshotId | None = None
        if marks:
            mark_diff = marks_to_diff(after, marks)
            if len(mark_diff):
                propagation_snapshot = self.repo.commit_diff(mark_diff, branch=self.branch)

        return AcceptResult(
            snapshot_id=snapshot_id,
            propagation_snapshot_id=propagation_snapshot,
            bible_diff=bible_diff(before, self.repo.state(self.branch)),
            marks=tuple(marks),
            extracted=extracted,
        )

    # --- writer ledger and node controls ---------------------------------------

    def lock(self, node_id: NodeId) -> SnapshotId:
        """Freeze a node. Propagation skips it; its facts become hard constraints."""
        self.signals.record(Signal(kind=SignalKind.lock, branch=self.branch, node_id=node_id))
        return self.repo.commit_diff(
            Diff(ops=(SetLock(node_id=node_id),), author=DiffAuthor.human, intent="lock"),
            branch=self.branch,
        )

    def unlock(self, node_id: NodeId) -> SnapshotId:
        """Unfreeze a node."""
        self.signals.record(Signal(kind=SignalKind.unlock, branch=self.branch, node_id=node_id))
        return self.repo.commit_diff(
            Diff(ops=(ClearLock(node_id=node_id),), author=DiffAuthor.human, intent="unlock"),
            branch=self.branch,
        )

    def dismiss_stale(self, node_id: NodeId) -> SnapshotId:
        """The writer's "it still works" — clears a stale mark without regenerating."""
        self.signals.record(
            Signal(kind=SignalKind.dismiss_stale, branch=self.branch, node_id=node_id)
        )
        return self.repo.commit_diff(
            Diff(
                ops=(
                    SetNodeStatus(node_id=node_id, status=NodeStatus.accepted, stale_reason=None),
                ),
                author=DiffAuthor.human,
                intent="dismiss stale mark",
            ),
            branch=self.branch,
        )

    def set_dial(self, value: float) -> SnapshotId:
        """Move the coherent-to-surprising dial."""
        self.signals.record(
            Signal(kind=SignalKind.dial_moved, branch=self.branch, payload={"value": value})
        )
        return self.repo.commit_diff(
            Diff(ops=(SetDial(value=value),), author=DiffAuthor.human, intent="move the dial"),
            branch=self.branch,
        )

    def add_style_note(self, text: str) -> SnapshotId:
        """Add a writer-written style rule."""
        self.signals.record(
            Signal(kind=SignalKind.style_note, branch=self.branch, payload={"text": text})
        )
        return self.repo.commit_diff(
            Diff(
                ops=(AddStyleNote(note=StyleNote(text=text)),),
                author=DiffAuthor.human,
                intent="add a style note",
            ),
            branch=self.branch,
        )

    def add_criterion(self, name: str, description: str, weight: float = 1.0) -> SnapshotId:
        """Add a writer-defined scoring axis, in the writer's own words."""
        self.signals.record(
            Signal(
                kind=SignalKind.criterion_added,
                branch=self.branch,
                payload={"name": name, "description": description},
            )
        )
        return self.repo.commit_diff(
            Diff(
                ops=(
                    AddCriterion(
                        criterion=Criterion(name=name, description=description, weight=weight)
                    ),
                ),
                author=DiffAuthor.human,
                intent="add a criterion",
            ),
            branch=self.branch,
        )

    # --- internals --------------------------------------------------------------

    def _require(self, proposal_id: ProposalId) -> Proposal:
        proposal = self._pending.get(proposal_id)
        if proposal is None:
            raise KeyError(f"no pending proposal {proposal_id}")
        return proposal

    def _prose_beat(self, proposal: Proposal, state: StoryState) -> NodeId | None:
        """The beat whose prose this proposal wrote, if it wrote any."""
        if proposal.level is not Level.prose:
            return None
        target = proposal.target_node_id
        if target is not None and target in state.prose_of_beat:
            return target
        return None


def bible_diff(before: StoryState, after: StoryState) -> BibleDiff:
    """What changed in the world graph between two states.

    Args:
        before: State before the change.
        after: State after it.

    Returns:
        Facts added, ended, and struck, plus writer-readable lines.
    """
    names = after.entity_names() or before.entity_names()
    added = tuple(after.facts[f] for f in sorted(after.facts.keys() - before.facts.keys()))
    removed = tuple(before.facts[f] for f in sorted(before.facts.keys() - after.facts.keys()))
    ended = tuple(
        after.facts[f]
        for f in sorted(before.facts.keys() & after.facts.keys())
        if before.facts[f].valid_until_beat is None and after.facts[f].valid_until_beat is not None
    )
    lines = [
        *(f"+ {fact.sentence(names)}" for fact in added),
        *(f"~ {fact.sentence(names)} (no longer true from here)" for fact in ended),
        *(f"- {fact.sentence(names)}" for fact in removed),
    ]
    return BibleDiff(added=added, ended=ended, removed=removed, lines=tuple(lines))


def _prose_text(proposal: Proposal) -> str:
    """The prose a proposal would write, if any."""
    for op in proposal.diff.ops:
        text = getattr(op, "text", None)
        if isinstance(text, str):
            return text
        node = getattr(op, "node", None)
        if isinstance(node, Prose):
            return node.text
    return ""


def _with_edited_prose(proposal: Proposal, new_text: str) -> Proposal:
    """Rebuild a proposal with the writer's text, marked as an edit of AI output."""
    sentences = max(1, new_text.count(".") + new_text.count("?") + new_text.count("!"))
    span = ProvenanceSpan(
        start=0,
        end=sentences,
        source=Authorship.ai_edited_by_human,
        proposal_id=proposal.id,
    )
    ops = []
    for op in proposal.diff.ops:
        node = getattr(op, "node", None)
        if isinstance(node, Prose):
            ops.append(
                op.model_copy(
                    update={"node": node.model_copy(update={"text": new_text, "spans": (span,)})}
                )
            )
        elif isinstance(op, SetProse):
            ops.append(op.model_copy(update={"text": new_text, "spans": (span,)}))
        else:
            ops.append(op)
    return proposal.model_copy(
        update={
            "diff": proposal.diff.model_copy(update={"ops": tuple(ops), "author": DiffAuthor.human})
        }
    )
