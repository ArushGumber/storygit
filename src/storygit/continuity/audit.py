"""The periodic bible audit: check the whole graph, not just the last change.

Per-accept checking is incremental — it looks at the facts a change introduced. That is
right for the interactive loop and it can miss the slow kind of drift: a contradiction
assembled over ten episodes where no single accept was wrong. The audit walks everything
in entity-sized batches, runs layers 1 and 2 over all of it, and samples layer 3.

It also answers a question no per-fact check can: which threads are being dropped. A
thread opened in episode 2 and untouched since episode 3 is not a contradiction, so
nothing else in the system would ever mention it — and for a serial it is the most
expensive kind of mistake.

Callable from the evaluation harness and, in chunk 6, from the interface.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from storygit.continuity import layer1, layer2_nli
from storygit.continuity.flags import Flag, FlagKind, Severity, sort_flags
from storygit.domain.ids import EntityId
from storygit.domain.state import StoryState
from storygit.domain.threads import ThreadStatus

STALE_THREAD_BEATS = 12
"""How many beats a thread may go untouched before the audit mentions it."""


class AuditReport(BaseModel):
    """The result of a full-graph audit.

    Attributes:
        flags: Everything found, hard first.
        entities_checked: How many entities were walked.
        facts_checked: How many facts were considered.
        pairs_scored: How many pairs went through the cross-encoder.
        by_layer: Flag counts per layer.
    """

    model_config = ConfigDict(frozen=True)

    flags: tuple[Flag, ...] = ()
    entities_checked: int = 0
    facts_checked: int = 0
    pairs_scored: int = 0
    by_layer: dict[str, int] = {}

    def summary(self) -> str:
        """A one-line summary for logs."""
        hard = sum(1 for f in self.flags if f.is_hard)
        return (
            f"audit: {len(self.flags)} flags ({hard} hard) over "
            f"{self.entities_checked} entities / {self.facts_checked} facts; "
            f"{self.pairs_scored} pairs scored"
        )


def run_audit(
    state: StoryState,
    *,
    use_nli: bool = True,
    nli_threshold: float = layer2_nli.CONTRADICTION_THRESHOLD,
    max_pairs: int = 400,
) -> AuditReport:
    """Walk the whole graph through layers 1 and 2, plus the thread ledger.

    Args:
        state: The state to audit.
        use_nli: Whether to run layer 2. Off makes the audit instant and deterministic,
            which is what the ablation runs want.
        nli_threshold: Contradiction probability to flag at.
        max_pairs: Cap on cross-encoder pairs.

    Returns:
        The report.
    """
    flags: list[Flag] = list(layer1.check_state(state))
    pairs_scored = 0
    if use_nli:
        pairs = layer2_nli.candidate_pairs(state)
        pairs_scored = min(len(pairs), max_pairs)
        flags.extend(layer2_nli.check(state, threshold=nli_threshold, max_pairs=max_pairs))
    flags.extend(check_dropped_threads(state))

    by_layer: dict[str, int] = {}
    for flag in flags:
        key = f"layer{flag.layer}"
        by_layer[key] = by_layer.get(key, 0) + 1

    entities: set[EntityId] = set(state.entities)
    return AuditReport(
        flags=tuple(sort_flags(flags)),
        entities_checked=len(entities),
        facts_checked=len(state.facts),
        pairs_scored=pairs_scored,
        by_layer=by_layer,
    )


def check_dropped_threads(state: StoryState, *, max_gap: int = STALE_THREAD_BEATS) -> list[Flag]:
    """Flag open threads nobody has touched for a while.

    Args:
        state: The state to check.
        max_gap: How many beats of silence before a thread is mentioned.

    Returns:
        Soft flags, one per neglected thread, citing where it was opened.
    """
    beats = state.beats_in_order()
    if not beats:
        return []
    latest = state.seq.get(beats[-1].id, 0)
    flags: list[Flag] = []
    for thread in sorted(state.threads.values(), key=lambda t: t.id):
        if thread.status is not ThreadStatus.open:
            continue
        touched = state.seq.get(thread.last_touched_beat, 0)
        gap = latest - touched
        if gap < max_gap:
            continue
        opened = state.nodes.get(thread.opened_at_beat)
        where = (opened.title or opened.node_type.value) if opened else "an earlier beat"
        flags.append(
            Flag(
                kind=FlagKind.dropped_thread,
                severity=Severity.soft,
                layer=1,
                message=(
                    f"“{thread.description}” has been open since {where} and nothing has "
                    f"touched it for {gap} beats. Pay it off, advance it, or drop it "
                    "deliberately."
                ),
                node_id=thread.opened_at_beat,
                established_by=thread.opened_at_beat,
                subgraph=(str(thread.id),),
            )
        )
    return flags
