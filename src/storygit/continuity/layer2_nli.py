"""Layer 2: natural-language inference on the facts layer 1 cannot decide.

Layer 1 handles the closed predicate vocabulary by equality. That leaves two residues, and
both of them are genuinely linguistic:

* **``note`` facts.** The free-text escape hatch. "The ash responds to grief, not to will"
  and "Kael can command the ash whenever he chooses" contradict each other, and no
  string comparison will ever say so.
* **Borderline layer-1 hits.** Same subject and predicate, different object strings, on a
  multi-valued predicate. "wants to find his sister" and "wants to forget his sister" are
  a contradiction; "wants to find his sister" and "wants to reach the harbour" are a
  character with two goals.

A DeBERTa cross-encoder fine-tuned on MNLI scores each pair for entailment, neutrality,
and contradiction. It runs locally on CPU, costs nothing, and takes a few milliseconds a
pair. Because it is a model making a judgement about meaning, everything it produces is a
**soft** flag: shown differently, never blocking, and quoting both sentences so the writer
can see exactly what was compared.
"""

from __future__ import annotations

from storygit.continuity.flags import Flag, FlagKind, Severity
from storygit.domain.ids import FactId, NodeId
from storygit.domain.state import StoryState
from storygit.domain.world import (
    ACCUMULATIVE_PREDICATES,
    SINGLE_VALUED_PREDICATES,
    Fact,
    Predicate,
)

CONTRADICTION_THRESHOLD = 0.35
"""Contradiction probability above which a pair becomes a soft flag.

Calibrated against the local checkpoint rather than guessed. Measured on three pairs:

===================================================  ==============
pair                                                 P(contradiction)
===================================================  ==============
"can command the ash" / "never able to affect it"    1.00
"can command the ash" / "the ash never obeys anyone" 0.43
"can command the ash" / "does what Kael tells it"    0.00
===================================================  ==============

The middle row is the interesting one: it *is* a contradiction, but a universal statement
against a specific one, and the model is only moderately confident. A 0.5 threshold would
miss it. A 0.35 threshold catches it while leaving an enormous margin above the paraphrase
at 0.00.

This is a default, not a constant of nature. The evaluation sweeps it and reports the
precision/recall curve, so the trade-off is a number rather than an assertion.
"""


def candidate_pairs(
    state: StoryState,
    *,
    fact_ids: set[FactId] | None = None,
) -> list[tuple[Fact, Fact]]:
    """Fact pairs worth sending to the cross-encoder.

    Only pairs that can be simultaneously true are considered, and only those layer 1
    could not decide. Everything else is either already flagged or provably fine, and
    sending it would waste the budget and add false positives.

    Args:
        state: The state to search.
        fact_ids: Restrict to pairs involving these facts; ``None`` considers all.

    Returns:
        Ordered, de-duplicated pairs.
    """
    pairs: list[tuple[Fact, Fact]] = []
    seen: set[tuple[FactId, FactId]] = set()
    by_subject: dict[tuple[object, Predicate], list[Fact]] = {}
    for fact in state.facts.values():
        by_subject.setdefault((fact.subject, fact.predicate), []).append(fact)

    for (_, predicate), facts in by_subject.items():
        if predicate in SINGLE_VALUED_PREDICATES:
            continue  # layer 1 decided these by equality
        if predicate in ACCUMULATIVE_PREDICATES:
            continue  # a character may hold two things, two scars, or two secrets
        ordered = sorted(facts, key=lambda f: (state.seq.get(f.valid_from_beat, 0), f.id))
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if first.object.strip().lower() == second.object.strip().lower():
                    continue
                if fact_ids is not None and not ({first.id, second.id} & fact_ids):
                    continue
                later = (
                    second
                    if state.seq.get(second.valid_from_beat, 0)
                    >= state.seq.get(first.valid_from_beat, 0)
                    else first
                )
                if not state.valid_at(first, later.valid_from_beat):
                    continue  # the earlier one was properly ended; not simultaneous
                key = (first.id, second.id)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((first, second))
    return pairs


def check(
    state: StoryState,
    *,
    fact_ids: set[FactId] | None = None,
    threshold: float = CONTRADICTION_THRESHOLD,
    max_pairs: int = 200,
) -> list[Flag]:
    """Run the cross-encoder over the borderline pairs and flag contradictions.

    Args:
        state: The state to check.
        fact_ids: Restrict to pairs involving these facts.
        threshold: Contradiction probability above which to flag.
        max_pairs: Safety cap, so a large story cannot turn one accept into a minute of
            inference. Pairs are ordered by story position, so the cap drops the newest.

    Returns:
        Soft flags, each quoting both sentences and citing the establishing beat.
    """
    pairs = candidate_pairs(state, fact_ids=fact_ids)
    if not pairs:
        return []
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    from storygit.providers.local import nli_scores

    names = state.entity_names()
    sentences = [(a.sentence(names), b.sentence(names)) for a, b in pairs]
    scores = nli_scores(sentences)

    flags: list[Flag] = []
    for (first, second), (premise, hypothesis), score in zip(pairs, sentences, scores, strict=True):
        probability = float(score.get("contradiction", 0.0))
        if probability < threshold:
            continue
        flags.append(
            Flag(
                kind=FlagKind.contradiction,
                severity=Severity.soft,
                layer=2,
                message=(
                    f"These may contradict each other: “{premise}” (established in "
                    f"{_label(state, first.established_by_beat)}) and “{hypothesis}” "
                    f"(established in {_label(state, second.established_by_beat)})."
                ),
                node_id=second.established_by_beat,
                fact_ids=(first.id, second.id),
                established_by=first.established_by_beat,
                entity_ids=(first.subject,),
                subgraph=(str(first.id), str(second.id)),
                score=probability,
            )
        )
    return flags


def _label(state: StoryState, node_id: NodeId | None) -> str:
    if node_id is None:
        return "an earlier beat"
    node = state.nodes.get(node_id)
    if node is None:
        return "a removed beat"
    return node.title or f"{node.node_type.value} {node_id}"
