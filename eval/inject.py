"""Ground truth by construction.

Every recall number the continuity checker claims needs a denominator, and the only way to
have one is to put contradictions into the story yourself and count how many come back.
This module builds those, one function per class of error, plus the two other ground truths
the evaluation needs: known fact sets for extraction recall, and scripted edits whose true
downstream impact is known because the dependency edges were written by hand.

Every injection returns both the modified state and a description of what was injected, so
a metric can be computed without anyone having to remember what was done.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddEntity,
    AddFact,
    AddKnows,
    AddNode,
    Diff,
    DiffAuthor,
    UpdateNode,
)
from storygit.domain.ids import EntityId, FactId, IdGenerator, NodeId
from storygit.domain.nodes import Beat, Episode, Scene, Story
from storygit.domain.state import StoryState
from storygit.domain.world import Entity, EntityKind, Fact, Knows, Predicate


class ContradictionClass(StrEnum):
    """The kinds of contradiction the evaluation injects."""

    location = "location"
    death = "death"
    possession = "possession"
    epistemic = "epistemic"
    note = "note"


class Injection(BaseModel):
    """One injected contradiction and the facts that constitute it.

    Attributes:
        kind: Which class it is.
        fact_ids: The facts involved. A checker is credited with catching it when a flag
            mentions any of them.
        node_id: Where it shows up.
        description: What was done, in English, for the report.
        expected_layer: Which layer *should* catch it. Reported alongside which layer
            actually did — a layer-2 catch of a layer-1 case means layer 1 has a hole.
    """

    model_config = ConfigDict(frozen=True)

    kind: ContradictionClass
    fact_ids: tuple[FactId, ...]
    node_id: NodeId | None = None
    description: str = ""
    expected_layer: int = 1


class Scenario(BaseModel):
    """A story plus what was injected into it.

    Attributes:
        injections: Everything injected, for scoring.
        beats: Beat ids in reading order, for convenience.
    """

    model_config = ConfigDict(frozen=True)

    injections: tuple[Injection, ...] = ()
    beats: tuple[NodeId, ...] = ()


def clean_story(ids: IdGenerator, *, beats: int = 6) -> tuple[StoryState, Scenario]:
    """A small, internally consistent story to inject into.

    Args:
        ids: Id generator.
        beats: How many beats to create.

    Returns:
        The state and an empty scenario carrying the beat ids.
    """
    story_id, episode_id, scene_id = ids.node(), ids.node(), ids.node()
    beat_ids = [ids.node() for _ in range(beats)]
    kael, mara, ashfall, kell, token = (ids.entity() for _ in range(5))

    state = StoryState.build(
        nodes={story_id: Story(id=story_id, title="Ashfall", seed="A powerless orphan...")}
    )
    ops = [
        AddNode(node=Episode(id=episode_id, parent_id=story_id, title="Episode 1")),
        AddNode(node=Scene(id=scene_id, parent_id=episode_id, title="Scene 1")),
        *(
            AddNode(
                node=Beat(
                    id=beat_id,
                    parent_id=scene_id,
                    title=f"Beat {index + 1}",
                    position=index,
                    what_happens=f"Something happens in beat {index + 1}.",
                )
            )
            for index, beat_id in enumerate(beat_ids)
        ),
        AddEntity(entity=Entity(id=kael, kind=EntityKind.character, name="Kael")),
        AddEntity(entity=Entity(id=mara, kind=EntityKind.character, name="Mara")),
        AddEntity(entity=Entity(id=ashfall, kind=EntityKind.place, name="Ashfall")),
        AddEntity(entity=Entity(id=kell, kind=EntityKind.place, name="Kell")),
        AddEntity(entity=Entity(id=token, kind=EntityKind.object, name="the Warden's token")),
    ]
    state = apply(state, Diff(ops=tuple(ops), author=DiffAuthor.human, intent="clean story"))

    base_facts = [
        Fact(
            id=ids.fact(),
            subject=kael,
            predicate=Predicate.location,
            object_entity=ashfall,
            valid_from_beat=beat_ids[0],
            established_by_beat=beat_ids[0],
        ),
        Fact(
            id=ids.fact(),
            subject=kael,
            predicate=Predicate.alive,
            object_text="alive",
            valid_from_beat=beat_ids[0],
            established_by_beat=beat_ids[0],
        ),
        Fact(
            id=ids.fact(),
            subject=kael,
            predicate=Predicate.possesses,
            object_text="the Warden's token",
            valid_from_beat=beat_ids[1],
            established_by_beat=beat_ids[1],
        ),
        Fact(
            id=ids.fact(),
            subject=kael,
            predicate=Predicate.secret,
            object_text="the ash answers him",
            valid_from_beat=beat_ids[1],
            established_by_beat=beat_ids[1],
        ),
    ]
    state = apply(
        state,
        Diff(ops=tuple(AddFact(fact=f) for f in base_facts), author=DiffAuthor.ai),
    )
    return state, Scenario(beats=tuple(beat_ids))


def _entity(state: StoryState, name: str) -> EntityId:
    entity = state.entity_by_name(name)
    if entity is None:  # pragma: no cover - fixtures always contain these
        raise KeyError(name)
    return entity.id


def inject_location(
    state: StoryState, scenario: Scenario, ids: IdGenerator
) -> tuple[StoryState, Injection]:
    """Put Kael in two places at once, without ending the first."""
    kael, kell = _entity(state, "Kael"), _entity(state, "Kell")
    at = scenario.beats[3]
    fact = Fact(
        id=ids.fact(),
        subject=kael,
        predicate=Predicate.location,
        object_entity=kell,
        valid_from_beat=at,
        established_by_beat=at,
    )
    return (
        apply(state, Diff(ops=(AddFact(fact=fact),), author=DiffAuthor.ai)),
        Injection(
            kind=ContradictionClass.location,
            fact_ids=(fact.id,),
            node_id=at,
            description="Kael is in Kell while still established in Ashfall",
            expected_layer=1,
        ),
    )


def inject_death(
    state: StoryState, scenario: Scenario, ids: IdGenerator
) -> tuple[StoryState, Injection]:
    """Kill Kael, then have him act two beats later."""
    kael = _entity(state, "Kael")
    death_at, act_at = scenario.beats[2], scenario.beats[4]
    death = Fact(
        id=ids.fact(),
        subject=kael,
        predicate=Predicate.alive,
        object_text="dead",
        valid_from_beat=death_at,
        established_by_beat=death_at,
    )
    acting = Fact(
        id=ids.fact(),
        subject=kael,
        predicate=Predicate.goal,
        object_text="reach the harbour",
        valid_from_beat=act_at,
        established_by_beat=act_at,
    )
    return (
        apply(
            state,
            Diff(ops=(AddFact(fact=death), AddFact(fact=acting)), author=DiffAuthor.ai),
        ),
        Injection(
            kind=ContradictionClass.death,
            fact_ids=(death.id, acting.id),
            node_id=act_at,
            description="Kael dies in beat 3 and acts in beat 5",
            expected_layer=1,
        ),
    )


def inject_possession(
    state: StoryState, scenario: Scenario, ids: IdGenerator
) -> tuple[StoryState, Injection]:
    """Give Mara the token Kael is already holding."""
    mara = _entity(state, "Mara")
    at = scenario.beats[3]
    fact = Fact(
        id=ids.fact(),
        subject=mara,
        predicate=Predicate.possesses,
        object_text="the Warden's token",
        valid_from_beat=at,
        established_by_beat=at,
    )
    return (
        apply(state, Diff(ops=(AddFact(fact=fact),), author=DiffAuthor.ai)),
        Injection(
            kind=ContradictionClass.possession,
            fact_ids=(fact.id,),
            node_id=at,
            description="Mara holds the token Kael already has",
            expected_layer=1,
        ),
    )


def inject_epistemic(
    state: StoryState, scenario: Scenario, ids: IdGenerator
) -> tuple[StoryState, Injection]:
    """Have Mara act on Kael's secret without ever being told it."""
    mara = _entity(state, "Mara")
    secret = next(f for f in state.facts.values() if f.predicate is Predicate.secret)
    at = scenario.beats[4]
    acting = Fact(
        id=ids.fact(),
        subject=mara,
        predicate=Predicate.goal,
        object_text="use what Kael can do",
        valid_from_beat=at,
        established_by_beat=at,
    )
    return (
        apply(
            state,
            Diff(
                ops=(
                    AddFact(fact=acting),
                    UpdateNode(node_id=at, fields={"consumes": [str(secret.id)]}),
                ),
                author=DiffAuthor.ai,
            ),
        ),
        Injection(
            kind=ContradictionClass.epistemic,
            fact_ids=(secret.id, acting.id),
            node_id=at,
            description="Mara acts on Kael's secret and is never told it",
            expected_layer=1,
        ),
    )


def inject_note(
    state: StoryState, scenario: Scenario, ids: IdGenerator
) -> tuple[StoryState, Injection]:
    """Two free-text notes that contradict each other. Only layer 2 can decide this."""
    kael = _entity(state, "Kael")
    first_at, second_at = scenario.beats[1], scenario.beats[4]
    first = Fact(
        id=ids.fact(),
        subject=kael,
        predicate=Predicate.note,
        object_text="can command the ash whenever he chooses",
        valid_from_beat=first_at,
        established_by_beat=first_at,
    )
    second = Fact(
        id=ids.fact(),
        subject=kael,
        predicate=Predicate.note,
        object_text="has never been able to affect the ash at all",
        valid_from_beat=second_at,
        established_by_beat=second_at,
    )
    return (
        apply(
            state,
            Diff(ops=(AddFact(fact=first), AddFact(fact=second)), author=DiffAuthor.ai),
        ),
        Injection(
            kind=ContradictionClass.note,
            fact_ids=(first.id, second.id),
            node_id=second_at,
            description="two free-text notes about the ash contradict each other",
            expected_layer=2,
        ),
    )


INJECTORS = {
    ContradictionClass.location: inject_location,
    ContradictionClass.death: inject_death,
    ContradictionClass.possession: inject_possession,
    ContradictionClass.epistemic: inject_epistemic,
    ContradictionClass.note: inject_note,
}
"""One injector per contradiction class."""


def build_scenario(
    *,
    classes: tuple[ContradictionClass, ...] = tuple(ContradictionClass),
    seed: int = 0,
) -> tuple[StoryState, Scenario]:
    """A story with one contradiction of each requested class.

    The injections are applied *sequentially to one story*, so with more than one class
    they can and do interact: ``inject_location`` puts Kael in Kell at beat 3 and
    ``inject_possession`` puts the token on Mara at the same beat, while ``inject_death``
    kills Kael two beats earlier. ``metrics.checker_recall`` credits a catch when any flag
    mentions any of an injection's facts, so cross-talk between classes inflates recall.

    Isolation is therefore the caller's job, and it is one line: pass a single class.
    ``offline.checker_layers`` builds one story per class for exactly this reason. This
    function's multi-class form is for building a story that has several things wrong with
    it, not for measuring per-class recall.

    Args:
        classes: Which classes to inject.
        seed: Id-generator seed.

    Returns:
        The state and the scenario describing what is in it.
    """
    ids = IdGenerator(seed=seed, stream="inject")
    state, scenario = clean_story(ids)
    injections: list[Injection] = []
    for cls in classes:
        state, injection = INJECTORS[cls](state, scenario, ids)
        injections.append(injection)
    return state, scenario.model_copy(update={"injections": tuple(injections)})


# --- extraction ground truth --------------------------------------------------


class ExtractionCase(BaseModel):
    """A passage with the facts it establishes, written by hand.

    Attributes:
        text: The prose.
        facts: ``(subject, predicate, object)`` triples the passage establishes.
        knows: ``(character, fact index)`` pairs the passage establishes as known.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    facts: tuple[tuple[str, Predicate, str], ...]
    knows: tuple[tuple[str, int], ...] = ()


EXTRACTION_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase(
        text=(
            "Kael pressed his blistered hand against his coat and said nothing. The Warden "
            "watched him the whole way across the square, and did not look away once."
        ),
        facts=(
            ("Kael", Predicate.injury, "blistered hand"),
            ("Warden", Predicate.goal, "watch Kael"),
        ),
    ),
    ExtractionCase(
        text=(
            'Mara took the token from the table and put it in her coat. "He does not know '
            'I have it," she said, and Kael, standing behind her, heard every word.'
        ),
        facts=(("Mara", Predicate.possesses, "the token"),),
        knows=(("Kael", 0),),
    ),
    ExtractionCase(
        text=(
            "The harbour was closed. Kael had known that since morning, and had come "
            "anyway, because the alternative was the market and the market had the Warden "
            "in it."
        ),
        facts=(("Kael", Predicate.location, "the harbour"),),
    ),
)
"""Hand-built passages with known fact sets. Small on purpose: every one is checked by eye,
and a ground truth nobody has read is not a ground truth."""


# --- staleness ground truth ---------------------------------------------------


class StaleCase(BaseModel):
    """A scripted edit whose true downstream impact is known by construction.

    Attributes:
        changed_fact: The fact the edit changes.
        truly_affected: Beats that genuinely depend on it, transitively.
        undeclared: Beats that genuinely depend on it but did **not** declare the
            dependency. These are the only beats the soft-edge ablation could add over
            declared edges, so a sweep that omits them cannot show any recall gain and
            would be an unfair test of the idea.
        unaffected: Beats that do not depend on it, and must not be marked.
    """

    model_config = ConfigDict(frozen=True)

    changed_fact: FactId
    truly_affected: tuple[NodeId, ...]
    undeclared: tuple[NodeId, ...] = ()
    unaffected: tuple[NodeId, ...] = ()

    @property
    def all_affected(self) -> tuple[NodeId, ...]:
        """Every beat that genuinely depends on the change, declared or not."""
        return (*self.truly_affected, *self.undeclared)


def build_stale_case(*, seed: int = 0) -> tuple[StoryState, StaleCase]:
    """A story with a hand-written dependency chain and a known blast radius.

    Beat 1 establishes a fact; beats 2 and 3 form a consuming chain; beats 4-6 are
    unrelated. Because the edges are written here rather than inferred, the true answer is
    known, which is what makes stale-prediction precision and recall computable.

    Args:
        seed: Id-generator seed.

    Returns:
        The state and the case.
    """
    ids = IdGenerator(seed=seed, stream="stale")
    state, scenario = clean_story(ids, beats=6)
    beats = scenario.beats
    kael = _entity(state, "Kael")

    root_fact = next(
        f
        for f in state.facts.values()
        if f.predicate is Predicate.location and f.established_by_beat == beats[0]
    )
    derived = Fact(
        id=ids.fact(),
        subject=kael,
        predicate=Predicate.goal,
        object_text="leave the city before dark",
        valid_from_beat=beats[1],
        established_by_beat=beats[1],
    )
    state = apply(
        state,
        Diff(
            ops=(
                AddFact(fact=derived),
                # Beats 2 and 3 declare their dependency. Beat 4 depends on the same fact
                # in substance -- it is about Kael being in Ashfall -- but says so only in
                # its prose, which is exactly the case the soft-edge ablation exists for.
                UpdateNode(node_id=beats[1], fields={"consumes": [str(root_fact.id)]}),
                UpdateNode(node_id=beats[2], fields={"consumes": [str(derived.id)]}),
                UpdateNode(
                    node_id=beats[0],
                    fields={
                        "what_happens": "Kael waits in the Ashfall market, watching the gate.",
                        "location": "Ashfall",
                    },
                ),
                UpdateNode(
                    node_id=beats[3],
                    fields={
                        "what_happens": (
                            "Kael crosses the Ashfall market again, keeping to the gate side."
                        ),
                        "location": "Ashfall",
                    },
                ),
                UpdateNode(
                    node_id=beats[4],
                    fields={
                        "what_happens": "Mara counts the boats in the harbour at Kell.",
                        "location": "Kell",
                    },
                ),
                UpdateNode(
                    node_id=beats[5],
                    fields={
                        "what_happens": "The Warden reads a letter she was not meant to see.",
                        "location": "Kell",
                    },
                ),
            ),
            author=DiffAuthor.human,
            intent="declare the dependency chain, and leave one undeclared",
        ),
    )
    return state, StaleCase(
        changed_fact=root_fact.id,
        truly_affected=(beats[1], beats[2]),
        undeclared=(beats[3],),
        unaffected=tuple(beats[4:]),
    )


def knows_edge(state: StoryState, character: str, fact_id: FactId, at: NodeId) -> Diff:
    """A diff teaching a character a fact, for building negative controls.

    The negative controls matter as much as the injections: a checker that flags everything
    has perfect recall and is useless, so every class needs a nearby case that must *not*
    fire.
    """
    return Diff(
        ops=(
            AddKnows(knows=Knows(character=_entity(state, character), fact=fact_id, since_beat=at)),
        ),
        author=DiffAuthor.human,
        intent=f"{character} is told",
    )
