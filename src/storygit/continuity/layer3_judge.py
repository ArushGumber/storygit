"""Layer 3: a language model's opinion, clearly labelled as one.

Layers 1 and 2 answer questions about facts. Layer 3 answers the questions no graph can:
does this character's behaviour follow from what they want, and does the tone belong to
this story? Both are judgements, so everything this layer produces is a **soft** flag, and
its numeric scores feed candidate ranking rather than gating anything.

Two design choices are worth defending.

**Argue before rating.** The judge is required to write one sentence of reasoning
*before* it emits a number. This is Fabula's own finding — their auto-rater's agreement
with human preference improved when it argued first — and it matches the wider result that
a score produced after an explanation is better calibrated than one produced before it.

**The writer's criteria are in the prompt.** Alongside fixed narratology axes, the judge
scores whatever the writer has said they care about, in the writer's own words. That is
what stops the system optimizing its objective while the writer watches: the objective is
partly theirs.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from storygit.agents.schemas import BoundedModel
from storygit.continuity.flags import Flag, FlagKind, Severity
from storygit.domain.ids import NodeId
from storygit.graph.slices import StateSlice
from storygit.providers.base import LLMRequest, Message, Role, SchemaParseError
from storygit.providers.router import Router

MAX_ARGUMENT = 400
MAX_NOTE = 240


class CriterionScore(BoundedModel):
    """One score on one axis, with the argument that produced it.

    ``BoundedModel`` rather than ``BaseModel``, for the reason its own docstring gives: the
    provider does not enforce ``maxLength``, so a bounded string field that *rejects*
    throws away a good answer over punctuation. This is the highest-volume call in the
    pipeline -- six per node -- against a field the prompt explicitly asks the model to
    fill with a sentence of prose, so it is the exact case ``BoundedModel`` was built for.
    And the failure was invisible: ``judge()`` catches the parse error and returns a
    neutral 0.5, so a run where every verdict failed to validate looked identical to a run
    where none did.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(max_length=60, description="The axis being scored.")
    argument: str = Field(
        default="",
        max_length=MAX_ARGUMENT,
        description="One sentence of reasoning. Write this BEFORE deciding the score.",
    )
    score: float = Field(default=3.0, ge=1.0, le=5.0, description="1 to 5.")


class JudgeVerdict(BoundedModel):
    """What the soft judge returns for one candidate.

    Bounded rather than rejecting, for the same reason as :class:`CriterionScore`.
    """

    model_config = ConfigDict(extra="ignore")

    scores: list[CriterionScore] = Field(
        default_factory=list, max_length=12, description="One entry per axis, in order."
    )
    motivation_concern: str = Field(
        default="",
        max_length=MAX_NOTE,
        description=(
            "If a character does something their established wants do not support, say so "
            "in one sentence. Empty if there is no concern."
        ),
    )
    tone_concern: str = Field(
        default="",
        max_length=MAX_NOTE,
        description=(
            "If the register breaks from the surrounding story, say so in one sentence. "
            "Empty if there is no concern."
        ),
    )


FIXED_CRITERIA = (
    ("momentum", "Does this move the story, or does it restate what we already knew?"),
    ("specificity", "Concrete people, places, and consequences rather than gestures at them."),
    ("consequence", "Does something here cost something, or change what is possible next?"),
    ("voice", "Does it sound like this story, rather than like competent generic fiction?"),
)
"""The narratology axes every candidate is scored on, whatever the writer adds."""


def build_request(
    slice_: StateSlice,
    candidate_text: str,
    *,
    sample_index: int = 0,
) -> LLMRequest:
    """Build the judging call for one candidate.

    Args:
        slice_: The state slice the candidate was generated against.
        candidate_text: The candidate, rendered as text.
        sample_index: Distinguishes cache entries when several candidates are judged.

    Returns:
        The request.
    """
    axes = [f"- {name}: {question}" for name, question in FIXED_CRITERIA]
    axes += [f"- {c.name}: {c.description}" for c in slice_.criteria]
    system = Message(
        role=Role.system,
        content=(
            "You are a script editor giving a writer a second opinion. You are not the "
            "author and you do not rewrite anything.\n\n"
            "For each axis: write one sentence arguing the case FIRST, then give a score "
            "from 1 to 5. Deciding the number before you have argued it produces worse "
            "scores, so do it in that order.\n\n"
            "Then, only if there is something real to say, note a motivation concern "
            "(a character doing something their established wants do not support) and a "
            "tone concern (a break in register). Leave them empty otherwise; inventing a "
            "concern to look thorough is worse than saying nothing.\n\n"
            "Axes:\n" + "\n".join(axes)
        ),
    )
    user = Message(
        role=Role.user,
        content=(
            f"CURRENT STATE\n{slice_.render()}\n\n"
            f"CANDIDATE\n{candidate_text}\n\n"
            "Reply with JSON only, matching the schema. No commentary, no code fences."
        ),
    )
    return LLMRequest(
        messages=(system, user),
        purpose="judge.soft",
        temperature=0.0,
        max_tokens=1200,
        json_schema=JudgeVerdict.model_json_schema(),
        sample_index=sample_index,
    )


async def judge(
    router: Router,
    slice_: StateSlice,
    candidate_text: str,
    *,
    node_id: NodeId | None = None,
    sample_index: int = 0,
) -> tuple[float, list[Flag], dict[str, float]]:
    """Score a candidate and return any soft concerns.

    Args:
        router: Model access.
        slice_: The slice the candidate was generated against.
        candidate_text: The candidate, as text.
        node_id: The node the candidate targets, for flag attribution.
        sample_index: Cache discriminator.

    Returns:
        ``(quality, flags, sub_scores)`` where quality is the mean score rescaled into
        ``[0, 1]``, and ``sub_scores`` is the per-axis breakdown the preference head uses
        as features in chunk 4. On a model or parse failure the quality is a neutral 0.5
        and the flag list is empty: a judge that cannot be reached must not silently
        rank every candidate as bad.
    """
    from storygit.agents.parse import complete_structured

    request = build_request(slice_, candidate_text, sample_index=sample_index)
    try:
        verdict, _ = await complete_structured(router, request, JudgeVerdict)
    except SchemaParseError:
        return 0.5, [], {}

    sub_scores = {s.name: float(s.score) for s in verdict.scores if s.name}
    quality = (sum(sub_scores.values()) / len(sub_scores) - 1.0) / 4.0 if sub_scores else 0.5

    flags: list[Flag] = []
    if verdict.motivation_concern.strip():
        flags.append(
            Flag(
                kind=FlagKind.motivation,
                severity=Severity.soft,
                layer=3,
                message=verdict.motivation_concern.strip(),
                node_id=node_id,
                score=quality,
            )
        )
    if verdict.tone_concern.strip():
        flags.append(
            Flag(
                kind=FlagKind.tone,
                severity=Severity.soft,
                layer=3,
                message=verdict.tone_concern.strip(),
                node_id=node_id,
                score=quality,
            )
        )
    return max(0.0, min(1.0, quality)), flags, sub_scores
