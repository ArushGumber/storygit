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


class ConstraintViolation(BoundedModel):
    """One hard constraint a candidate breaks, quoted so the citation is checkable.

    Quoting rather than referencing by index is deliberate: a writer reading "violates
    hard constraint 2" has to go and count, and a flag that costs work to understand is a
    flag they learn to skip. The quote also makes the citation *checkable* -- a constraint
    the judge invented does not appear in the ledger, and the caller drops it.
    """

    model_config = ConfigDict(extra="ignore")

    constraint: str = Field(
        max_length=MAX_NOTE, description="The constraint, quoted exactly as it was given."
    )
    how: str = Field(
        default="",
        max_length=MAX_NOTE,
        description="One sentence: what in the candidate breaks it.",
    )


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
    constraint_violations: list[ConstraintViolation] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Every hard constraint this candidate breaks. Quote the constraint exactly as "
            "it was given. Empty if it breaks none, which is the normal case."
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
    # Hard constraints reach the *generation* prompt already, and reaching a prompt is not
    # the same as being enforced -- the writer who added "Present day. No ten-shilling
    # notes, no shillings, no Austin Cambridges" was then offered four pounds ten, a Morris
    # Minor and three-shilling bits. Their sentence for it is the right one: a hard
    # constraint that is not checked anywhere is a style note with a more confident name.
    # Checking rides this call rather than adding one per constraint per candidate, which
    # would multiply the most expensive stage of the pipeline by the size of the ledger.
    constraint_block = ""
    if slice_.hard_constraints:
        listed = "\n".join(f"- {line}" for line in slice_.hard_constraints)
        constraint_block = (
            "\n\nHARD CONSTRAINTS. These are the writer's, and they are not suggestions. "
            "For each one, check the candidate against it. Report only the ones it "
            "actually breaks, quoting the constraint exactly:\n" + listed
        )
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
            "Axes:\n" + "\n".join(axes) + constraint_block
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
    # Only constraints the writer actually set. The judge quoting something that is not in
    # the ledger is the judge inventing a rule, and a flag citing a constraint the writer
    # cannot find is worse than no flag -- it teaches them the citations are decorative.
    declared = {c.strip(): c.strip() for c in slice_.hard_constraints}
    lowered = {c.lower(): c for c in declared}
    for violation in verdict.constraint_violations:
        quoted = violation.constraint.strip()
        matched = declared.get(quoted) or lowered.get(quoted.lower())
        if matched is None:
            continue
        detail = violation.how.strip()
        flags.append(
            Flag(
                kind=FlagKind.hard_constraint,
                severity=Severity.soft,
                layer=3,
                message=f"violates hard constraint: \u201c{matched}\u201d"
                + (f" \u2014 {detail}" if detail else ""),
                node_id=node_id,
                score=quality,
            )
        )

    return max(0.0, min(1.0, quality)), flags, sub_scores
