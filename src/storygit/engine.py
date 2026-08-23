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

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from storygit.agents.propose import Proposal, Proposer
from storygit.agents.schemas import Level
from storygit.continuity import layer1, layer2_nli
from storygit.continuity.audit import AuditReport, run_audit
from storygit.continuity.bible_diff import BibleDiff, recheck_after_strike, strike
from storygit.continuity.bible_diff import compute as compute_bible_diff
from storygit.continuity.flags import Flag, sort_flags
from storygit.domain.diff import (
    AddCriterion,
    AddHardConstraint,
    AddNode,
    AddRejectedDirection,
    AddStyleNote,
    ClearLock,
    Diff,
    DiffAuthor,
    MergeEntities,
    Op,
    RemoveCriterion,
    RemoveNode,
    RemoveStyleNote,
    SetDial,
    SetLock,
    SetNodeStatus,
    SetProse,
    UpdateNode,
)
from storygit.domain.errors import (
    InvalidStructureError,
    LockedNodeError,
    UnknownNodeError,
)
from storygit.domain.ids import EntityId, FactId, IdGenerator, NodeId, ProposalId, SnapshotId
from storygit.domain.ledger import Criterion, StyleNote
from storygit.domain.nodes import NodeStatus, Prose
from storygit.domain.provenance import Authorship, ProvenanceSpan
from storygit.domain.state import StoryState
from storygit.graph.dependency import EdgeProvider
from storygit.graph.propagation import (
    StaleMark,
    marks_to_diff,
    propagate_change,
)
from storygit.graph.propagation import preview as propagation_preview
from storygit.preference.features import FeatureVector
from storygit.preference.layer import PreferenceLayer
from storygit.preference.signals import SignalReader
from storygit.preference.state import PreferenceStateStore
from storygit.providers.router import Router
from storygit.selection.select import (
    Candidate,
    CandidateSelector,
    SelectionConfig,
    candidate_text,
)
from storygit.store.branches import DEFAULT_BRANCH
from storygit.store.repository import Repository
from storygit.store.signals import Signal, SignalKind, SignalStore


class AcceptResult(BaseModel):
    """Everything the writer should see after accepting a proposal.

    Attributes:
        snapshot_id: The commit the accepted diff produced.
        propagation_snapshot_id: The follow-up commit recording the stale marks, if any.
        bible_diff: What changed in the world graph.
        marks: What is now stale or wants review.
        extracted: Whether facts were extracted from newly written prose.
        flags: Continuity flags on the new state, hard first.
    """

    model_config = ConfigDict(frozen=True)

    snapshot_id: SnapshotId
    propagation_snapshot_id: SnapshotId | None = None
    bible_diff: BibleDiff = BibleDiff()
    marks: tuple[StaleMark, ...] = ()
    extracted: bool = False
    flags: tuple[Flag, ...] = ()


def _swap_node_id(op: Op, old: NodeId, new: NodeId) -> Op:
    """Point every node reference in an op at ``new`` instead of ``old``.

    A regenerated proposal was written against a node id that will never be created, and
    the facts, entity knowledge and dependency edges it produces all cite that id -- as
    ``established_by_beat``, ``valid_from_beat``, ``since_beat``, and so on. Committing
    them unrewritten would put facts in the graph established by a beat that does not
    exist, which is exactly the dangling-reference class of bug that makes ``valid_at``
    silently answer "true forever".

    Walks the model's own fields rather than naming them, so an op added later is covered
    without anyone remembering to come back here.

    Args:
        op: The operation to rewrite.
        old: The id the proposal generated and that will not be created.
        new: The id of the node that already exists.

    Returns:
        The op, with references swapped.
    """

    def swap(value: Any) -> tuple[Any, bool]:
        if isinstance(value, str) and value == str(old):
            return type(value)(new), True
        if isinstance(value, BaseModel):
            inner = {}
            for name in type(value).model_fields:
                replacement, hit = swap(getattr(value, name))
                if hit:
                    inner[name] = replacement
            return (value.model_copy(update=inner), True) if inner else (value, False)
        if isinstance(value, (list, tuple)):
            walked = [swap(item) for item in value]
            if any(hit for _, hit in walked):
                return type(value)(item for item, _ in walked), True
            return value, False
        if isinstance(value, dict):
            mapped = {k: swap(v) for k, v in value.items()}
            if any(hit for _, hit in mapped.values()):
                return {k: v for k, (v, _) in mapped.items()}, True
            return value, False
        return value, False

    updates = {}
    for name in type(op).model_fields:
        if name == "op":
            continue
        replacement, hit = swap(getattr(op, name))
        if hit:
            updates[name] = replacement
    return op.model_copy(update=updates) if updates else op


class RevisionPreview(BaseModel):
    """What revising or removing an accepted node would cost, before committing it.

    Attributes:
        removed_nodes: Nodes that would cease to exist.
        removed_facts: Facts that would cease to exist with them.
        marks: What propagation would mark stale or send for review, each with its reason
            and the fact that caused it.
        bible_diff: The world-graph change, in the writer's own sentences.
    """

    model_config = ConfigDict(frozen=True)

    removed_nodes: tuple[NodeId, ...] = ()
    removed_facts: tuple[FactId, ...] = ()
    marks: tuple[StaleMark, ...] = ()
    bible_diff: BibleDiff = BibleDiff()


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
        selection: SelectionConfig | None = None,
        use_nli: bool = True,
        edge_provider: EdgeProvider | None = None,
        preference: PreferenceLayer | None = None,
    ) -> None:
        """Wire the engine.

        Args:
            repository: The story repository. Owns all state.
            router: Model access.
            ids: Id generator; seed it to make a whole session reproducible.
            branch: The branch this engine reads and writes.
            selection: Candidate-selection tunables (n, k, selector, dial, judge).
            use_nli: Whether accepts run layer 2. The ablation runs turn it off.
            edge_provider: Optional soft-edge provider for propagation. ``None`` means
                declared dependencies only, which is the default and the honest one.
            preference: The preference layer. Omitted means load whatever this branch has
                learned; pass a disabled layer for the no-preference ablation.
        """
        self.repo = repository
        self.router = router
        self.branch = branch
        self.ids = ids or IdGenerator()
        self.proposer = Proposer(router, self.ids)
        self.signals = SignalStore(repository.conn)
        self.selector = CandidateSelector(self.proposer, router, selection)
        self.use_nli = use_nli
        self.edge_provider = edge_provider
        self.preference_state = PreferenceStateStore(repository.conn)
        self.preference = preference or PreferenceLayer.load(self.preference_state, branch)
        self._reader = SignalReader(self.signals, branch=branch)
        self._pending: dict[ProposalId, Proposal] = {}
        self._last_candidates: list[Candidate] = []
        # Features and text are captured when a candidate is *shown*, because both depend
        # on the state at that moment and cannot be reconstructed afterwards.
        self._features: dict[str, FeatureVector] = {}
        self._texts: dict[str, str] = {}

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
    ) -> list[Candidate]:
        """Generate labelled, checked, diverse candidates and hold them for a decision.

        The full pipeline: axis-conditioned sampling, continuity checking on each
        candidate's would-be facts, judging, the dial, and MMR or DPP selection. Every
        candidate comes back — selected ones first — because the evaluation measures
        diversity over the whole set and chunk 4 needs the unselected ones as the
        negative side of a preference pair.

        Args:
            level: What to propose.
            node_id: The node to attach to; defaults per level.
            intent: The writer's instruction.

        Returns:
            Candidates, shortlisted ones first.
        """
        # The writer's own accepted sentences, and the ones they turned down, retrieved
        # against this intent. Empty for a new writer and when the layer is disabled, so
        # the no-preference ablation sends byte-identical prompts.
        exemplars = self.preference.exemplars.render(intent, enabled=self.preference.enabled)
        # Which feature slot each writer-defined criterion occupies is a property of the
        # ledger, and the layer is persisted independently of any repository, so the
        # engine is the one that knows both and hands the order over.
        self.preference.criterion_order = tuple(c.name for c in self.state().ledger.criteria)
        candidates = await self.selector.select(
            self.state(),
            level,
            target_node_id=node_id,
            intent=intent,
            exemplars=exemplars,
            quality_fn=self._preference_quality,
        )
        features = self.preference.features_for(candidates)
        for candidate, vector in zip(candidates, features, strict=True):
            key = str(candidate.proposal.id)
            self._pending[candidate.proposal.id] = candidate.proposal
            self._features[key] = vector
            self._texts[key] = candidate_text(candidate.proposal)
        self._last_candidates = candidates
        return candidates

    async def _preference_quality(self, candidates: Sequence[Candidate]) -> list[float] | None:
        """Base quality from the learned head, or ``None`` to defer to the judge.

        This is the chunk 3 seam. When the layer is disabled, or has not yet seen a single
        comparison, it returns ``None`` and selection ranks by the judge exactly as it did
        before the head existed — which is what makes the no-preference ablation exact
        rather than merely similar.
        """
        if not self.preference.enabled or self.preference.state.pairs_seen < 1:
            return None
        return self.preference.score(candidates)

    def shown(self) -> tuple[ProposalId, ...]:
        """Ids of the candidates last shortlisted, for recording preferences."""
        return tuple(c.proposal.id for c in self._last_candidates if c.selected)

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

        marks = propagate_change(before, after, edge_provider=self.edge_provider)
        propagation_snapshot: SnapshotId | None = None
        if marks:
            mark_diff = marks_to_diff(after, marks)
            if len(mark_diff):
                propagation_snapshot = self.repo.commit_diff(mark_diff, branch=self.branch)

        self._pending.pop(proposal_id, None)
        self._learn(human_prose=self._human_prose(after))
        final = self.repo.state(self.branch)
        return AcceptResult(
            snapshot_id=snapshot_id,
            propagation_snapshot_id=propagation_snapshot,
            bible_diff=compute_bible_diff(before, final),
            marks=tuple(marks),
            extracted=extracted,
            flags=tuple(self.check(before, final)),
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
        self._learn(human_prose=self._human_prose(self.state()))

    async def edit(
        self,
        proposal_id: ProposalId,
        new_text: str,
        *,
        shown_with: tuple[ProposalId, ...] = (),
    ) -> AcceptResult:
        """Accept a proposal with the writer's own words in place of the model's.

        The before/after pair is the most informative signal the system gets: it says
        not just that the writer wanted something different, but *what* different looks
        like. Chunk 4 mines exactly these pairs into style rules and an edit-direction
        vector.

        ``shown_with`` matters as much here as on a clean accept, and for longer it was
        missing: the edit path called ``accept`` with no shown set, so an edited win
        produced no preference pair at all. Which made ``PreferencePair.was_edited`` and
        ``EDITED_WIN_WEIGHT`` -- a documented mechanism, with an argument behind it --
        unreachable from the only edit path the product has. The most informative signal
        the system gets was contributing the least to the head.

        Args:
            proposal_id: The proposal being edited.
            new_text: The writer's replacement prose.
            shown_with: The other candidates it was shown alongside. An edited winner is
                still a winner, weighted at half a clean accept.

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
        return await self.accept(proposal.id, shown_with=shown_with)

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

        marks = propagate_change(before, after, edge_provider=self.edge_provider)
        propagation_snapshot: SnapshotId | None = None
        if marks:
            mark_diff = marks_to_diff(after, marks)
            if len(mark_diff):
                propagation_snapshot = self.repo.commit_diff(mark_diff, branch=self.branch)

        final = self.repo.state(self.branch)
        return AcceptResult(
            snapshot_id=snapshot_id,
            propagation_snapshot_id=propagation_snapshot,
            bible_diff=compute_bible_diff(before, final),
            marks=tuple(marks),
            extracted=extracted,
            flags=tuple(self.check(before, final)),
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
        """The writer's "it still works" — clears a stale mark without regenerating.

        The status only changes when it was actually ``stale``. Propagation also writes a
        reason onto nodes it marks ``review`` or ``maybe_affected`` *without* changing their
        status, so a draft can carry a reason -- and dismissing it used to mark that draft
        accepted, granting a review the writer never gave. Dismissing a mark means "this
        still works", not "I approve of this".
        """
        self.signals.record(
            Signal(kind=SignalKind.dismiss_stale, branch=self.branch, node_id=node_id)
        )
        node = self.repo.state(self.branch).nodes.get(node_id)
        if node is not None and node.status is NodeStatus.locked:
            # apply()'s guard exempts SetNodeStatus(status=locked), which is what preserving
            # the node's own status would produce here -- so the refusal has to be explicit
            # or dismissing a mark on a locked node silently succeeds. A locked node is
            # never marked in the first place; there is nothing to dismiss.
            raise LockedNodeError(f"{node_id} is locked; unlock it before changing its status")
        status = (
            NodeStatus.accepted if node is None or node.status is NodeStatus.stale else node.status
        )
        return self.repo.commit_diff(
            Diff(
                ops=(SetNodeStatus(node_id=node_id, status=status, stale_reason=None),),
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

    def add_hard_constraint(self, text: str) -> SnapshotId:
        """Add a rule generation must never break.

        Harder than a style note: hard constraints go into every prompt built from this
        point on, alongside the facts of every locked node, and they are not soft
        preferences the ranking can outvote.
        """
        self.signals.record(
            Signal(kind=SignalKind.hard_constraint, branch=self.branch, payload={"text": text})
        )
        return self.repo.commit_diff(
            Diff(
                ops=(AddHardConstraint(text=text),),
                author=DiffAuthor.human,
                intent="add a hard constraint",
            ),
            branch=self.branch,
        )

    def merge_entities(self, source_id: EntityId, target_id: EntityId) -> SnapshotId:
        """Fold one entity into another, keeping the source's names as aliases.

        Alias resolution is deliberately conservative — it matches exactly or creates a
        new entity — which means it will sometimes create a duplicate rather than risk
        welding two characters together. This is the writer's correction for that case,
        and it is the reason the conservative rule is safe to have.

        Every fact whose subject or object was the source is re-pointed at the target, and
        the source's epistemic edges move with it, so nothing is left referring to an
        entity that no longer exists.
        """
        self.signals.record(
            Signal(
                kind=SignalKind.entities_merged,
                branch=self.branch,
                payload={"source": str(source_id), "target": str(target_id)},
            )
        )
        return self.repo.commit_diff(
            Diff(
                ops=(MergeEntities(source_id=source_id, target_id=target_id),),
                author=DiffAuthor.human,
                intent="merge two entities the resolver kept apart",
            ),
            branch=self.branch,
        )

    def remove_style_note(self, text: str) -> SnapshotId:
        """Delete a style rule, including one the system mined from the writer's edits.

        A mined rule the writer cannot remove is a rule influencing generation without
        consent, which is the failure the whole ledger is designed to avoid. Removal is
        itself a signal: it says the system read the writer wrong.
        """
        self.signals.record(
            Signal(kind=SignalKind.style_note_removed, branch=self.branch, payload={"text": text})
        )
        return self.repo.commit_diff(
            Diff(
                ops=(RemoveStyleNote(text=text),),
                author=DiffAuthor.human,
                intent="remove a style note",
            ),
            branch=self.branch,
        )

    def remove_criterion(self, name: str) -> SnapshotId:
        """Delete a writer-defined scoring axis.

        The judge stops scoring candidates on it and the preference head stops seeing it
        as a feature, from the next proposal onward.
        """
        self.signals.record(
            Signal(kind=SignalKind.criterion_removed, branch=self.branch, payload={"name": name})
        )
        return self.repo.commit_diff(
            Diff(
                ops=(RemoveCriterion(name=name),),
                author=DiffAuthor.human,
                intent="remove a criterion",
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

    # --- continuity -------------------------------------------------------------

    def check(self, before: StoryState, after: StoryState) -> list[Flag]:
        """Run layers 1 and 2 over what a change introduced.

        Incremental by design: only the facts the change added are checked, against
        everything already true. The whole-graph pass is :meth:`audit`, which is where
        slow drift gets caught.

        Args:
            before: State before the change.
            after: State after it.

        Returns:
            Flags, hard first.
        """
        new_facts = set(after.facts) - set(before.facts)
        flags = layer1.check_new_facts(after, new_facts)
        if self.use_nli and new_facts:
            flags.extend(layer2_nli.check(after, fact_ids=new_facts))
        return sort_flags(flags)

    def audit(self) -> AuditReport:
        """Walk the whole graph through layers 1 and 2, plus the thread ledger."""
        return run_audit(self.state(), use_nli=self.use_nli)

    def strike_fact(
        self, fact_id: FactId, *, at_beat: NodeId | None = None
    ) -> tuple[SnapshotId, list[Flag], tuple[StaleMark, ...]]:
        """The writer striking a fact from the bible.

        Args:
            fact_id: The fact to strike.
            at_beat: End the fact here rather than deleting it. Ending preserves the
                history earlier beats depend on; deleting does not, and propagation will
                mark them.

        Returns:
            ``(snapshot, flags, marks)`` — layer 1 re-runs on whatever depended on it.
        """
        before = self.state()
        diff = strike(before, fact_id, at_beat=at_beat)
        self.signals.record(
            Signal(
                kind=SignalKind.strike_fact,
                branch=self.branch,
                payload={"fact_id": str(fact_id), "at_beat": str(at_beat) if at_beat else None},
            )
        )
        snapshot_id = self.repo.commit_diff(diff, branch=self.branch)
        after = self.repo.state(self.branch)
        marks = propagate_change(before, after, edge_provider=self.edge_provider)
        if marks:
            mark_diff = marks_to_diff(after, marks)
            if len(mark_diff):
                snapshot_id = self.repo.commit_diff(mark_diff, branch=self.branch)
                after = self.repo.state(self.branch)
        return snapshot_id, recheck_after_strike(after, fact_id), tuple(marks)

    # --- revising what was already accepted --------------------------------------

    async def regenerate_at(self, node_id: NodeId, *, intent: str = "") -> list[Candidate]:
        """Propose replacements *for* a node, not siblings beside it.

        This used to call ``propose_at(level, node_id=parent)``, which is the call that
        adds a new child under that parent -- so accepting a regenerated episode one gave
        you two episode ones. The writer who used the tool hit exactly that. The generation
        is the same; what changes is where the result lands, so each candidate's diff is
        rewritten to update the node in place and to attach anything it produces to the
        node that already exists.

        Args:
            node_id: The node being regenerated. Must not be the story root, which has no
                level to propose at.
            intent: Why, in the writer's words.

        Returns:
            Candidates whose diffs replace this node rather than adding one.

        Raises:
            UnknownNodeError: If the node does not exist.
            LockedNodeError: If it is locked.
            InvalidStructureError: If it is the story root.
        """
        state = self.state()
        node = state.nodes.get(node_id)
        if node is None:
            raise UnknownNodeError(f"no node {node_id}")
        if node_id in state.ledger.locks:
            raise LockedNodeError(f"{node_id} is locked; unlock it before regenerating it")
        if node.parent_id is None or node.node_type.value not in set(Level):
            raise InvalidStructureError(
                f"{node_id} is the story root; regeneration works on an episode, scene, "
                "beat or prose node"
            )

        candidates = await self.propose_at(
            Level(node.node_type.value), node_id=node.parent_id, intent=intent
        )
        replaced = [
            candidate.model_copy(update={"proposal": self._retarget(candidate.proposal, node_id)})
            for candidate in candidates
        ]
        # The pending map is keyed by proposal id, which the rewrite preserves, so the
        # retargeted proposal has to replace the one propose_at stored or accept() would
        # commit the sibling-adding version.
        for candidate in replaced:
            self._pending[candidate.proposal.id] = candidate.proposal
        self._last_candidates = replaced
        return replaced

    def _retarget(self, proposal: Proposal, node_id: NodeId) -> Proposal:
        """Rewrite a proposal's diff to replace ``node_id`` instead of adding a sibling.

        The generated node's own ``AddNode`` becomes an ``UpdateNode`` carrying its
        authored fields -- everything except identity and position, which belong to the
        node that is already in the tree and to its place in the reading order. Anything
        else the proposal produces (facts, entities, threads, dependency edges) is
        retargeted from the id that was never created to the one that exists.

        Args:
            proposal: The proposal as generated.
            node_id: The node it should replace.

        Returns:
            A proposal whose diff updates in place.
        """
        structural = {"id", "parent_id", "position", "node_type", "status", "stale_reason"}
        generated: NodeId | None = None
        ops: list[Op] = []
        for op in proposal.diff.ops:
            if isinstance(op, AddNode) and generated is None and op.node.parent_id is not None:
                generated = op.node.id
                fields = {
                    name: value
                    for name, value in op.node.model_dump().items()
                    if name not in structural
                }
                ops.append(UpdateNode(node_id=node_id, fields=fields))
            else:
                ops.append(op)

        if generated is not None:
            ops = [_swap_node_id(op, generated, node_id) for op in ops]

        return proposal.model_copy(
            update={
                "target_node_id": node_id,
                "diff": proposal.diff.model_copy(
                    update={"ops": tuple(ops), "author": DiffAuthor.ai}
                ),
            }
        )

    def preview_revision(self, diff: Diff) -> RevisionPreview:
        """What a structural change would cost, before it is committed.

        The whole argument for this operation is that the writer sees the blast radius
        first. Striking a fact already works this way; accepting a plan node did not, and
        the writer who used the tool said so plainly -- they stopped trusting accept and
        over-deliberated on every candidate, because acceptance was irreversible. A tool
        whose main verb cannot be undone makes its user slow, which is the opposite of what
        this one is for.

        Args:
            diff: The candidate change.

        Returns:
            The nodes and facts that would go, and the marks propagation would leave.
        """
        before = self.state()
        after = self.repo.preview_apply(diff, branch=self.branch)
        marks = propagation_preview(before, diff, edge_provider=self.edge_provider)
        return RevisionPreview(
            removed_nodes=tuple(sorted(before.nodes.keys() - after.nodes.keys())),
            removed_facts=tuple(sorted(before.facts.keys() - after.facts.keys())),
            marks=tuple(marks),
            bible_diff=compute_bible_diff(before, after),
        )

    def revise(self, node_id: NodeId, fields: dict[str, Any]) -> AcceptResult:
        """Change an accepted node's own fields, marking whatever depended on it.

        The same semantics as striking a fact, for the same reason: this is the writer
        acting, so the system does not refuse, and it does not silently rewrite anything
        either -- it *marks*. A revised beat's prose is not regenerated; it is flagged as
        no longer following from what is above it, and the writer decides.

        Args:
            node_id: The node to revise.
            fields: Field names to new values, validated against the node's model.

        Returns:
            The same shape an accept produces, so the interface renders it identically.

        Raises:
            LockedNodeError: If the node is locked. A lock means what it says; unlock
                first. Refusing here rather than in ``apply`` gives the writer the sentence
                that tells them what to do.
            UnknownNodeError: If the node does not exist.
        """
        before = self.state()
        node = before.nodes.get(node_id)
        if node is None:
            raise UnknownNodeError(f"no node {node_id}")
        if node_id in before.ledger.locks:
            raise LockedNodeError(f"{node_id} is locked; unlock it before revising it")

        diff = Diff(
            ops=(UpdateNode(node_id=node_id, fields=dict(fields)),),
            author=DiffAuthor.human,
            intent=f"revise {node.title or node_id}",
        )
        self.signals.record(
            Signal(
                kind=SignalKind.revise,
                branch=self.branch,
                node_id=node_id,
                payload={"fields": sorted(fields)},
            )
        )
        return self._commit_and_propagate(before, diff)

    def remove(self, node_id: NodeId, *, recursive: bool = True) -> AcceptResult:
        """Delete a node and its subtree, marking whatever depended on it.

        Args:
            node_id: The node to remove.
            recursive: Take its children with it. False refuses when it has any.

        Returns:
            The same shape an accept produces.

        Raises:
            LockedNodeError: If the node is locked.
            UnknownNodeError: If the node does not exist.
        """
        before = self.state()
        node = before.nodes.get(node_id)
        if node is None:
            raise UnknownNodeError(f"no node {node_id}")
        if node_id in before.ledger.locks:
            raise LockedNodeError(f"{node_id} is locked; unlock it before removing it")

        diff = Diff(
            ops=(RemoveNode(node_id=node_id, recursive=recursive),),
            author=DiffAuthor.human,
            intent=f"remove {node.title or node_id}",
        )
        self.signals.record(Signal(kind=SignalKind.remove, branch=self.branch, node_id=node_id))
        return self._commit_and_propagate(before, diff)

    def _commit_and_propagate(self, before: StoryState, diff: Diff) -> AcceptResult:
        """Commit a writer-authored structural diff and record what it made stale."""
        snapshot_id = self.repo.commit_diff(diff, branch=self.branch)
        after = self.repo.state(self.branch)
        marks = propagate_change(before, after, edge_provider=self.edge_provider)
        propagation_snapshot: SnapshotId | None = None
        if marks:
            mark_diff = marks_to_diff(after, marks)
            if len(mark_diff):
                propagation_snapshot = self.repo.commit_diff(mark_diff, branch=self.branch)
                after = self.repo.state(self.branch)
        return AcceptResult(
            snapshot_id=snapshot_id,
            propagation_snapshot_id=propagation_snapshot,
            bible_diff=compute_bible_diff(before, after),
            marks=tuple(marks),
            flags=sort_flags(layer1.check_state(after)),
        )

    # --- learning ---------------------------------------------------------------

    async def mine_edits(self) -> SnapshotId | None:
        """Read the writer's edits into style rules and put them in the ledger.

        Kept separate from ``learn`` because it costs a model call per edit, so it runs on
        demand (after a session, or from the interface) rather than after every accept.

        Returns:
            The snapshot recording the new rules, or ``None`` if nothing was mined.
        """
        from storygit.preference import edit_mining

        pairs = self._reader.edit_pairs()
        if not pairs:
            return None
        rules = await edit_mining.mine(self.router, pairs)
        diff = edit_mining.to_diff(self.state().ledger, rules)
        if not len(diff):
            return None
        return self.repo.commit_diff(diff, branch=self.branch)

    def _learn(self, *, human_prose: Sequence[str] = ()) -> None:
        """Refit the preference layer and persist it. Milliseconds, so it runs inline."""
        if not self.preference.enabled:
            return
        self.preference.learn(
            self._reader,
            self._features,
            texts_by_proposal=self._texts,
            human_prose=human_prose,
        )
        self.preference.save(self.preference_state, self.branch)

    def _human_prose(self, state: StoryState) -> list[str]:
        """Prose the writer wrote or edited — the anchors for the voice model."""
        out: list[str] = []
        for beat in state.beats_in_order():
            prose = state.prose_for(beat.id)
            if prose is None or not prose.text.strip():
                continue
            if any(
                span.source in (Authorship.human, Authorship.ai_edited_by_human)
                for span in prose.spans
            ):
                out.append(prose.text)
        return out

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
    """Rebuild a proposal with the writer's text, marked as an edit of AI output.

    At prose level the writer's words replace the prose. At a plan level there is no prose
    to replace, so they replace the node's summary --- the sentence the writer was actually
    shown and actually edited.

    This used to rewrite prose operations only and pass everything else through untouched,
    which meant an edit to an episode, scene or beat returned ``200`` and silently threw the
    writer's words away. Worse, the edit *signal* was still recorded, so the preference
    layer learned from a change that never happened. A writer who edits a plan node is
    editing the one field they can see.
    """
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
        elif node is not None and hasattr(node, "what_happens"):
            ops.append(
                op.model_copy(update={"node": node.model_copy(update={"what_happens": new_text})})
            )
        else:
            ops.append(op)
    return proposal.model_copy(
        update={
            "diff": proposal.diff.model_copy(update={"ops": tuple(ops), "author": DiffAuthor.human})
        }
    )
