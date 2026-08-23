"""Layer 1: deterministic contradiction checks. No model, by construction.

This module imports nothing from ``providers/`` and cannot call a language model. That is
a design constraint, not an accident: the checks it performs are ones where a
probabilistic answer would be strictly worse than an exact one. Two locations valid at the
same beat is a contradiction, full stop, and asking a model to confirm it would add cost,
latency, and a false-negative rate to a question that arithmetic already answers.

What it catches:

* **Single-valued predicate conflicts.** A character is in one place, is alive or not,
  belongs to one faction. Two simultaneously valid values is a contradiction. Multi-valued
  predicates (goals, traits, possessions) are not checked this way — two goals is a
  character, not a bug.
* **Dead actors.** A character established as dead who then does something.
* **Possession conflicts.** Two holders of the same object at the same beat.
* **Epistemic violations.** A beat that relies on a fact, performed by a character who has
  not been told it. This is the one readers notice most, and it is invisible to any check
  that only looks at what is *true*.

What it deliberately does not flag: a properly invalidated fact being replaced (that is a
change, not a contradiction), and a fact re-established with the same value (that is
repetition, not a conflict).
"""

from __future__ import annotations

from storygit.continuity.flags import Flag, FlagKind, Severity
from storygit.domain.ids import EntityId, FactId, NodeId
from storygit.domain.state import StoryState
from storygit.domain.world import (
    SINGLE_VALUED_PREDICATES,
    EntityKind,
    Fact,
    Predicate,
)

DEAD_VALUES = frozenset({"dead", "deceased", "killed", "false", "no"})
"""Object values of the ``alive`` predicate that mean the character is not alive."""


def check_state(state: StoryState, *, beats: tuple[NodeId, ...] | None = None) -> list[Flag]:
    """Run every layer-1 check over a state.

    Args:
        state: The state to check.
        beats: Restrict to these beats; ``None`` checks the whole story.

    Returns:
        Hard flags, each citing the beat that established the fact being contradicted.
    """
    targets = beats if beats is not None else tuple(b.id for b in state.beats_in_order())
    flags: list[Flag] = []
    flags.extend(check_conflicting_values(state, targets))
    flags.extend(check_dead_actors(state, targets))
    flags.extend(check_possession(state, targets))
    flags.extend(check_epistemic(state, targets))
    return flags


def check_new_facts(state: StoryState, fact_ids: set[FactId]) -> list[Flag]:
    """Check only the facts a change introduced, against everything already true.

    This is the hot path: it runs on every candidate before the writer sees it, so it
    touches only the beats the new facts concern rather than the whole story.

    Args:
        state: The state *after* the change would be applied.
        fact_ids: The facts the change introduces.

    Returns:
        Hard flags concerning those facts.
    """
    beats = {state.facts[fid].established_by_beat for fid in fact_ids if fid in state.facts}
    return [
        flag
        for flag in check_state(state, beats=tuple(sorted(beats)))
        if not fact_ids or set(flag.fact_ids) & fact_ids or flag.kind is FlagKind.epistemic
    ]


# --- individual checks --------------------------------------------------------


def check_conflicting_values(state: StoryState, beats: tuple[NodeId, ...]) -> list[Flag]:
    """Two different values of a single-valued predicate, valid at the same beat."""
    names = state.entity_names()
    flags: list[Flag] = []
    for beat_id in beats:
        for (subject, predicate), fact_ids in state.facts_by_key.items():
            if predicate not in SINGLE_VALUED_PREDICATES:
                continue
            valid = [
                state.facts[fid] for fid in fact_ids if state.valid_at(state.facts[fid], beat_id)
            ]
            if len(valid) < 2:
                continue
            distinct = {fact.object for fact in valid}
            if len(distinct) < 2:
                # Re-establishing the same value is repetition, not a contradiction.
                continue
            earlier, later = valid[0], valid[-1]
            flags.append(
                Flag(
                    kind=FlagKind.contradiction,
                    severity=Severity.hard,
                    layer=1,
                    message=(
                        f"{names.get(subject, subject)} has two different "
                        f"{predicate.value} values at the same point: "
                        f"“{earlier.sentence(names)}” (established in "
                        f"{_label(state, earlier.established_by_beat)}) and "
                        f"“{later.sentence(names)}” (established in "
                        f"{_label(state, later.established_by_beat)}). "
                        "If this is a change, end the earlier one where the new one begins."
                    ),
                    node_id=beat_id,
                    fact_ids=tuple(f.id for f in valid),
                    established_by=earlier.established_by_beat,
                    entity_ids=(subject,),
                    subgraph=tuple(str(f.id) for f in valid),
                )
            )
    return _dedupe(flags)


def check_dead_actors(state: StoryState, beats: tuple[NodeId, ...]) -> list[Flag]:
    """A character established as dead who then does something."""
    names = state.entity_names()
    flags: list[Flag] = []
    dead_from: dict[EntityId, Fact] = {}
    for fact in state.facts.values():
        if fact.predicate is Predicate.alive and fact.object.strip().lower() in DEAD_VALUES:
            current = dead_from.get(fact.subject)
            if current is None or _seq(state, fact.valid_from_beat) < _seq(
                state, current.valid_from_beat
            ):
                dead_from[fact.subject] = fact

    for beat_id in beats:
        beat = state.nodes.get(beat_id)
        if beat is None:
            continue
        actors = _actors_of(state, beat_id)
        for actor in sorted(actors):
            death = dead_from.get(actor)
            if death is None or not state.valid_at(death, beat_id):
                continue
            if _seq(state, death.valid_from_beat) >= _seq(state, beat_id):
                continue
            flags.append(
                Flag(
                    kind=FlagKind.dead_actor,
                    severity=Severity.hard,
                    layer=1,
                    message=(
                        f"{names.get(actor, actor)} acts here, but was established as "
                        f"{death.object} in {_label(state, death.established_by_beat)} "
                        "and has not been brought back."
                    ),
                    node_id=beat_id,
                    fact_ids=(death.id,),
                    established_by=death.established_by_beat,
                    entity_ids=(actor,),
                    subgraph=(str(beat_id), str(death.id)),
                )
            )
    return _dedupe(flags)


def check_possession(state: StoryState, beats: tuple[NodeId, ...]) -> list[Flag]:
    """Two characters holding the same object at the same beat."""
    names = state.entity_names()
    flags: list[Flag] = []
    for beat_id in beats:
        holders: dict[str, list[Fact]] = {}
        for fact in state.facts.values():
            if fact.predicate is not Predicate.possesses:
                continue
            if not state.valid_at(fact, beat_id):
                continue
            holders.setdefault(fact.object.strip().lower(), []).append(fact)
        for item, facts in sorted(holders.items()):
            subjects = {fact.subject for fact in facts}
            if len(subjects) < 2:
                continue
            ordered = sorted(facts, key=lambda f: _seq(state, f.valid_from_beat))
            who = ", ".join(sorted(names.get(s, s) for s in subjects))
            flags.append(
                Flag(
                    kind=FlagKind.possession,
                    severity=Severity.hard,
                    layer=1,
                    message=(
                        f"{who} all hold “{item}” at the same point. It was first "
                        f"established in {_label(state, ordered[0].established_by_beat)}; "
                        "end that possession where the new one begins."
                    ),
                    node_id=beat_id,
                    fact_ids=tuple(f.id for f in ordered),
                    established_by=ordered[0].established_by_beat,
                    entity_ids=tuple(sorted(subjects)),
                    subgraph=tuple(str(f.id) for f in ordered),
                )
            )
    return _dedupe(flags)


def check_epistemic(state: StoryState, beats: tuple[NodeId, ...]) -> list[Flag]:
    """A character acting on a fact they have not been told.

    A beat declares the facts it relies on (``consumes``). The characters *acting* in the
    beat are those it establishes facts about. If an acting character has no epistemic
    edge to a consumed fact dated at or before this beat, they are acting on information
    they do not have — and the flag says whether they learn it later, or never.
    """
    names = state.entity_names()
    flags: list[Flag] = []
    for beat_id in beats:
        beat = state.nodes.get(beat_id)
        consumed: tuple[FactId, ...] = tuple(getattr(beat, "consumes", ()))
        if not consumed:
            continue
        actors = _actors_of(state, beat_id)
        for fact_id in consumed:
            fact = state.facts.get(fact_id)
            if fact is None:
                continue
            # Only knowledge-bearing facts are gated. Where a character is standing is
            # not something they need to have been told.
            if fact.predicate not in (Predicate.secret, Predicate.note, Predicate.goal):
                continue
            for actor in sorted(actors):
                entity = state.entities.get(actor)
                if entity is None or entity.kind is not EntityKind.character:
                    continue
                if actor == fact.subject:
                    continue  # you know your own secrets and goals
                if state.knows_at(actor, fact_id, beat_id):
                    continue
                edge = state.knows.get((actor, fact_id))
                if edge is not None:
                    detail = f"learns it in {_label(state, edge.since_beat)}"
                else:
                    detail = "never learns it anywhere in the story"
                flags.append(
                    Flag(
                        kind=FlagKind.epistemic,
                        severity=Severity.hard,
                        layer=1,
                        message=(
                            f"{names.get(actor, actor)} acts on “{fact.sentence(names)}” "
                            f"here, but does not know it yet — {detail}. It was "
                            f"established in {_label(state, fact.established_by_beat)}."
                        ),
                        node_id=beat_id,
                        fact_ids=(fact_id,),
                        established_by=fact.established_by_beat,
                        entity_ids=(actor, fact.subject),
                        subgraph=(str(beat_id), str(fact_id)),
                    )
                )
    return _dedupe(flags)


# --- helpers ------------------------------------------------------------------


def _actors_of(state: StoryState, beat_id: NodeId) -> set[EntityId]:
    """Characters doing something in a beat — the subjects of the facts it establishes."""
    beat = state.nodes.get(beat_id)
    if beat is None:
        return set()
    actors: set[EntityId] = set()
    for fact_id in getattr(beat, "produces", ()):
        fact = state.facts.get(fact_id)
        if fact is None:
            continue
        entity = state.entities.get(fact.subject)
        if entity is not None and entity.kind is EntityKind.character:
            actors.add(fact.subject)
    return actors


def _seq(state: StoryState, node_id: NodeId | None) -> int:
    if node_id is None:
        return -1
    return state.seq.get(node_id, -1)


def _label(state: StoryState, node_id: NodeId | None) -> str:
    if node_id is None:
        return "an earlier beat"
    node = state.nodes.get(node_id)
    if node is None:
        return "a removed beat"
    return node.title or f"{node.node_type.value} {node_id}"


def _dedupe(flags: list[Flag]) -> list[Flag]:
    """Collapse flags that repeat the same finding at different beats.

    A location conflict spanning six beats is one problem, not six lines in the interface.
    """
    seen: dict[tuple[str, str, tuple[FactId, ...]], Flag] = {}
    for flag in flags:
        key = (flag.kind.value, str(flag.entity_ids), flag.fact_ids)
        if key not in seen:
            seen[key] = flag
    return list(seen.values())
