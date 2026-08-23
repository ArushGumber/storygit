"""The plan tree: Story, Episode, Scene, Beat, Prose.

Every node carries the same four planning questions Fabula found writers actually
use — what happens, what the audience learns, how they feel, where and when — plus
a status that drives the whole review loop. Nodes are frozen: changing one means
producing a new object, which is what makes the content-addressed store work.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.ids import FactId, NodeId, ThreadId
from storygit.domain.provenance import ProvenanceSpan


class NodeType(StrEnum):
    """Level of the plan tree a node sits at."""

    story = "story"
    episode = "episode"
    scene = "scene"
    beat = "beat"
    prose = "prose"


class NodeStatus(StrEnum):
    """Review state of a node.

    ``draft`` is AI-proposed and unreviewed, ``accepted`` has been approved by the
    writer, ``locked`` is frozen (propagation never touches it and its facts become
    hard constraints), ``stale`` means something it depends on changed.
    """

    draft = "draft"
    accepted = "accepted"
    locked = "locked"
    stale = "stale"


class BaseNode(BaseModel):
    """Fields shared by every level of the plan tree.

    Attributes:
        id: Stable identifier; also the logical key in a snapshot manifest.
        parent_id: Parent node, ``None`` only for the story root.
        node_type: Discriminator; fixed per subclass.
        title: Short human label shown in the plan tree.
        what_happens: Plot content of this node.
        audience_learns: New information the audience receives here.
        audience_feels: Intended emotional effect.
        location: Where it takes place, as free text.
        time: When it takes place, as free text.
        status: Review state.
        stale_reason: Why this node was marked stale or flagged for review.
        position: Index among its siblings; defines reading order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: NodeId
    parent_id: NodeId | None = None
    node_type: NodeType
    title: str = ""
    what_happens: str = ""
    audience_learns: str = ""
    audience_feels: str = ""
    location: str = ""
    time: str = ""
    status: NodeStatus = NodeStatus.draft
    stale_reason: str | None = None
    position: int = Field(default=0, ge=0)


class Story(BaseNode):
    """Root of the plan tree.

    Attributes:
        seed: The writer's original one-line intent.
        premise: The developed premise.
        existential_question: The question the story exists to ask, per Fabula's
            premise level.
    """

    node_type: Literal[NodeType.story] = NodeType.story
    seed: str = ""
    premise: str = ""
    existential_question: str = ""


class Episode(BaseNode):
    """One serialized episode: the unit a platform ships and listeners return to.

    Attributes:
        hook: Opening pull that earns the first minute.
        cliffhanger: Closing pull that earns the next episode.
        recap_of_previous: What a returning listener needs re-established.
        threads_in: Open threads carried in from earlier episodes.
        threads_out: Threads left open at the end of this episode.
        episode_goals: Writer-set goals for the episode; never AI-set.
        target_length: Target length in words, ``None`` when unset.
    """

    node_type: Literal[NodeType.episode] = NodeType.episode
    hook: str = ""
    cliffhanger: str = ""
    recap_of_previous: str = ""
    threads_in: tuple[ThreadId, ...] = ()
    threads_out: tuple[ThreadId, ...] = ()
    episode_goals: tuple[str, ...] = ()
    target_length: int | None = None


class Scene(BaseNode):
    """A continuous unit of place and time inside an episode."""

    node_type: Literal[NodeType.scene] = NodeType.scene


class Beat(BaseNode):
    """The smallest planned unit, and the atom of dependency tracking.

    ``produces`` and ``consumes`` are what make propagation deterministic: an edit
    to a beat can only affect beats that consume a fact this beat produced.

    Attributes:
        produces: Facts established here.
        consumes: Facts this beat relies on being true.
    """

    node_type: Literal[NodeType.beat] = NodeType.beat
    produces: tuple[FactId, ...] = ()
    consumes: tuple[FactId, ...] = ()


class Prose(BaseNode):
    """Written text for a beat, with per-sentence provenance.

    Attributes:
        text: The prose itself.
        spans: Provenance spans over sentence indices of ``text``.
    """

    node_type: Literal[NodeType.prose] = NodeType.prose
    text: str = ""
    spans: tuple[ProvenanceSpan, ...] = ()


AnyNode = Annotated[
    Story | Episode | Scene | Beat | Prose,
    Field(discriminator="node_type"),
]
"""Discriminated union of every node kind, keyed on ``node_type``."""

NODE_CLASSES: dict[NodeType, type[BaseNode]] = {
    NodeType.story: Story,
    NodeType.episode: Episode,
    NodeType.scene: Scene,
    NodeType.beat: Beat,
    NodeType.prose: Prose,
}
"""Lookup used when rehydrating a node from its stored JSON payload."""

ALLOWED_CHILDREN: dict[NodeType, frozenset[NodeType]] = {
    NodeType.story: frozenset({NodeType.episode}),
    NodeType.episode: frozenset({NodeType.scene}),
    NodeType.scene: frozenset({NodeType.beat}),
    NodeType.beat: frozenset({NodeType.prose}),
    NodeType.prose: frozenset(),
}
"""The tree's shape rule, enforced on every structural op."""


def node_from_payload(kind: str, payload: dict[str, object]) -> BaseNode:
    """Rebuild a node from a stored payload.

    Args:
        kind: The node type name as stored.
        payload: The node's JSON payload.

    Returns:
        The node, as its concrete subclass.

    Raises:
        ValueError: If ``kind`` is not a known node type.
    """
    try:
        cls = NODE_CLASSES[NodeType(kind)]
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown node type {kind!r}") from exc
    return cls.model_validate(payload)
