"""Every prompt in the system, in one file, named and documented.

Two rules hold everywhere here.

**Prompts are fed slices, never state.** The input is always an entity-scoped
:class:`~storygit.graph.slices.StateSlice` rendered compactly, plus the writer's intent
and their standing instructions. Nothing serializes the whole story: it would not fit,
it would cost a fortune, and the extra context measurably makes the output vaguer.

**The writer's constraints are non-negotiable and come last.** Hard constraints (their
rules plus every fact a locked node established) are placed at the end of the system
message, where instruction-following is strongest, and are phrased as prohibitions.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from storygit.agents.schemas import Level
from storygit.domain.world import Predicate
from storygit.graph.slices import StateSlice
from storygit.providers.base import Message, Role

PREDICATE_LIST = ", ".join(p.value for p in Predicate)

SYSTEM_BASE = """\
You are a story architect working for a human writer on a serialized audio-fiction \
series. The writer is the author; you are proposing options for them to accept, edit, or \
reject. You never have the last word.

House rules:
- Be specific. Named people, concrete places, real consequences. No "a mysterious figure".
- Serve the story that exists, not the story you would have written.
- Do not restate what is already established; add.
- Do not resolve everything. A serial lives on unanswered questions.
- Never contradict an established fact or a hard constraint.
"""

FACT_RULES = f"""\
When you list the facts a beat establishes, use only these predicates: {PREDICATE_LIST}.
Use `note` only when nothing else fits. Set `object_is_entity` to true when the object \
names a character, place, object, or faction, and false when it is a literal value.

`known_by` is important and easy to get wrong: list only characters who actually learn \
the fact in this beat. A fact can be true without anyone present knowing it. A later beat \
where a character acts on something they were never told is the single most noticeable \
continuity error in serialized fiction, and this field is what prevents it.
"""


def _schema_block(model_cls: type[BaseModel]) -> str:
    """Render a response model's JSON Schema for inclusion in a prompt."""
    return json.dumps(model_cls.model_json_schema(), indent=2)


def system_message(slice_: StateSlice, *, extra: str = "") -> Message:
    """Build the system turn: house rules, the writer's rules, then prohibitions.

    Args:
        slice_: The state slice, for style notes, criteria, and hard constraints.
        extra: Level-specific instructions appended after the house rules.

    Returns:
        The system message.
    """
    parts = [SYSTEM_BASE]
    if extra:
        parts.append(extra.strip())
    if slice_.style_notes:
        rules = "\n".join(f"- {note.text}" for note in slice_.style_notes)
        parts.append(f"The writer's style notes (follow these):\n{rules}")
    if slice_.criteria:
        criteria = "\n".join(f"- {c.name}: {c.description}" for c in slice_.criteria)
        parts.append(
            "The writer judges work on their own criteria. Optimise for these:\n" + criteria
        )
    if slice_.hard_constraints:
        constraints = "\n".join(f"- {line}" for line in slice_.hard_constraints)
        parts.append(
            "HARD CONSTRAINTS. These are not preferences. Violating any of them makes the "
            f"proposal useless:\n{constraints}"
        )
    return Message(role=Role.system, content="\n\n".join(parts))


def _user_message(
    slice_: StateSlice, intent: str, task: str, model_cls: type[BaseModel]
) -> Message:
    """Build the user turn: current state, the writer's intent, the task, the schema."""
    rendered = slice_.render()
    blocks = []
    if rendered:
        blocks.append(f"CURRENT STATE\n{rendered}")
    if intent:
        blocks.append(f"THE WRITER'S INTENT\n{intent}")
    blocks.append(f"YOUR TASK\n{task}")
    blocks.append(
        "Reply with JSON only, matching this schema exactly. No commentary, no code "
        f"fences.\n{_schema_block(model_cls)}"
    )
    return Message(role=Role.user, content="\n\n".join(blocks))


# --- one builder per level ----------------------------------------------------


def premise_prompt(
    slice_: StateSlice, seed: str, intent: str, model_cls: type[BaseModel]
) -> tuple[Message, Message]:
    """Develop the writer's one-line seed into a premise, a question, and a cast.

    Called by :meth:`storygit.agents.propose.Proposer.propose` at
    :attr:`~storygit.agents.schemas.Level.premise`.

    Args:
        slice_: State slice (nearly empty at this point).
        seed: The writer's original one-line idea.
        intent: What the writer asked for.
        model_cls: The response schema.

    Returns:
        The messages for the call.
    """
    extra = (
        "You are at the very start. Do not plan episodes yet; the writer settles the "
        "premise first. Give the story an existential question — the thing it is "
        "actually about — and two to four characters with something to want."
    )
    task = (
        f"The writer's seed is:\n  {seed}\n\n"
        "Develop it into a premise of two or three sentences, an existential question, "
        "a working title, and the opening cast."
    )
    return (system_message(slice_, extra=extra), _user_message(slice_, intent, task, model_cls))


def episode_prompt(
    slice_: StateSlice, intent: str, model_cls: type[BaseModel], position: int
) -> tuple[Message, Message]:
    """Propose one episode, with its hook, cliffhanger, recap, and threads.

    Args:
        slice_: State slice around the end of the story so far.
        intent: What the writer asked for.
        model_cls: The response schema.
        position: 1-based index of the episode being proposed.

    Returns:
        The messages for the call.
    """
    extra = (
        "This is serialized audio fiction. Retention is the product. Every episode needs "
        "a hook in its first minute, a cliffhanger at its end, and enough recap that a "
        "listener returning after a week is not lost. Advance at least one open thread "
        "and leave at least one open."
    )
    task = (
        f"Propose episode {position}. Say what happens, what the audience learns, how "
        "they should feel, and where and when it takes place. Then give the hook, the "
        "cliffhanger, the recap, and which threads you open, advance, and close (by id "
        "for existing ones)."
    )
    return (system_message(slice_, extra=extra), _user_message(slice_, intent, task, model_cls))


def scene_prompt(
    slice_: StateSlice, intent: str, model_cls: type[BaseModel], episode_title: str
) -> tuple[Message, Message]:
    """Propose one scene inside an episode.

    Args:
        slice_: State slice around the episode.
        intent: What the writer asked for.
        model_cls: The response schema.
        episode_title: The parent episode, for context.

    Returns:
        The messages for the call.
    """
    extra = (
        "A scene is one continuous unit of place and time. It should turn: something is "
        "different at the end of it than at the start."
    )
    task = (
        f"Propose the next scene of “{episode_title}”. Say what happens, what the "
        "audience learns, how they should feel, and where and when it takes place."
    )
    return (system_message(slice_, extra=extra), _user_message(slice_, intent, task, model_cls))


def beat_prompt(
    slice_: StateSlice, intent: str, model_cls: type[BaseModel], scene_title: str
) -> tuple[Message, Message]:
    """Propose one beat, including the facts it establishes and relies on.

    This is the prompt that maintains the dependency graph, which is why the fact rules
    are stated at length here and nowhere else.

    Args:
        slice_: State slice around the scene.
        intent: What the writer asked for.
        model_cls: The response schema.
        scene_title: The parent scene, for context.

    Returns:
        The messages for the call.
    """
    extra = (
        "A beat is the smallest unit of change: one action, one revelation, one reversal.\n\n"
        + FACT_RULES
    )
    task = (
        f"Propose the next beat of “{scene_title}”. Then list, precisely, the facts it "
        "establishes and the ids of the already-established facts it relies on."
    )
    return (system_message(slice_, extra=extra), _user_message(slice_, intent, task, model_cls))


def prose_prompt(
    slice_: StateSlice, intent: str, model_cls: type[BaseModel], beat: str
) -> tuple[Message, Message]:
    """Write the prose for a beat.

    Args:
        slice_: State slice around the beat, including recent prose for voice.
        intent: What the writer asked for.
        model_cls: The response schema.
        beat: What the beat is meant to do.

    Returns:
        The messages for the call.
    """
    extra = (
        "Write the scene, do not summarize it. This is audio: it will be heard, not read, "
        "so favour speech and concrete action over interior description, and never write "
        "a sentence a listener would have to re-read. Match the voice of the recent prose "
        "if any is shown."
    )
    task = f"Write the prose for this beat:\n  {beat}\n\nThen say why you made your choices."
    return (system_message(slice_, extra=extra), _user_message(slice_, intent, task, model_cls))


def extraction_prompt(
    slice_: StateSlice, text: str, model_cls: type[BaseModel]
) -> tuple[Message, Message]:
    """Pull facts, new entities, and thread movements out of accepted prose.

    Runs on the cheap model: this is high-volume structured reading, not writing. It is
    what keeps the world graph populated when the writer hand-writes a beat.

    Args:
        slice_: State slice, so known entities resolve to their canonical names.
        text: The prose to read.
        model_cls: The response schema.

    Returns:
        The messages for the call.
    """
    system = Message(
        role=Role.system,
        content=(
            "You extract structured facts from fiction. You are a reader, not a writer: "
            "report only what the passage actually establishes, never what it implies or "
            "what would make a better story. If a detail is ambiguous, leave it out.\n\n"
            + FACT_RULES
        ),
    )
    rendered = slice_.render()
    blocks = []
    if rendered:
        blocks.append(f"WHAT IS ALREADY KNOWN\n{rendered}")
    blocks.append(f"PASSAGE\n{text}")
    blocks.append(
        "List the facts this passage establishes, any entity appearing for the first "
        "time, any new open question it raises, and the ids of open threads it advances."
    )
    blocks.append(
        "Reply with JSON only, matching this schema exactly. No commentary, no code "
        f"fences.\n{_schema_block(model_cls)}"
    )
    return (system, Message(role=Role.user, content="\n\n".join(blocks)))


PROMPT_FOR_LEVEL = {
    Level.premise: premise_prompt,
    Level.episode: episode_prompt,
    Level.scene: scene_prompt,
    Level.beat: beat_prompt,
    Level.prose: prose_prompt,
}
"""Level to its prompt builder. Signatures differ; :mod:`propose` dispatches explicitly."""
