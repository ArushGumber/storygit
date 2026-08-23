"""The feature vector every candidate is scored on.

Exactly the list from the design, no more and no less:

1. judge sub-scores on the fixed narratology criteria,
2. judge sub-scores on the writer's own criteria, one slot each,
3. continuity score (from the checker's flags),
4. voice cosine (contrastive head, chunk 4),
5. edit-direction projection,
6. length,
7. dialogue ratio.

Two things are worth defending. The features are **few and interpretable** — a dozen or so
numbers, each of which a human can name — because the preference head has to learn from
ten or twenty comparisons, not ten thousand, and a high-dimensional head would simply
memorize. And they are **the same features the simulated writers' hidden weights are
defined over**, which is what makes the evaluation's central claim measurable: if the
fitted weights correlate with the hidden ones, the machinery recovers taste.

The writer's own criteria get **one slot each**, not one shared average. Averaging them
made the head structurally unable to learn that a writer weights *menace* twice as heavily
as *warmth*, which is the single thing writer-defined criteria exist to express; a writer
who defines two criteria and cares about one of them was indistinguishable from a writer
who cared equally about both.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

FIXED_CRITERIA = ("momentum", "specificity", "consequence", "voice")
"""The narratology axes the judge always scores. Mirrors ``layer3_judge.FIXED_CRITERIA``."""

MAX_CRITERION_SLOTS = 4
"""How many writer-defined criteria the head can weight independently.

A fixed width is what lets the head be a plain vector: the feature space cannot change
shape every time a writer adds a criterion, because the persisted weights are indexed by
position. Four is chosen against the data regime rather than the interface — the head is
fitted on tens of comparisons, and every extra slot is a parameter competing for them. A
writer may define more than four criteria; the judge scores all of them and they all reach
the prompt, but only the first four are separately *weighted*. Slot order is creation
order, so a slot's meaning never moves under a persisted model.
"""

CRITERION_SLOTS = tuple(f"criterion_{i + 1}" for i in range(MAX_CRITERION_SLOTS))
"""The per-criterion feature names, in slot order."""

BASE_FEATURES = (
    "judge_momentum",
    "judge_specificity",
    "judge_consequence",
    "judge_voice",
    *CRITERION_SLOTS,
    "continuity",
    "voice_cosine",
    "edit_direction",
    "length",
    "dialogue_ratio",
)
"""Feature names in a fixed order. The order is part of the persisted model."""

TARGET_WORDS = 220.0
"""Length is scored against a target rather than raw, so "longer" is not "better"."""

# Terminal punctuation, then any closing quote or bracket, then whitespace. The closing
# quote has to be part of the separator rather than the lookbehind, because Python's
# lookbehind must be fixed width -- and getting this wrong merges every line of dialogue
# with the sentence after it, which is precisely the case the feature exists to measure.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"\u201d\u2019')\]]*\s+")


class FeatureVector(BaseModel):
    """One candidate's features, named.

    Attributes:
        values: Feature name to value, all roughly in ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True)

    values: dict[str, float]

    def as_list(self, names: tuple[str, ...] = BASE_FEATURES) -> list[float]:
        """The vector in a fixed order, zero-filled for anything missing."""
        return [float(self.values.get(name, 0.0)) for name in names]

    def __sub__(self, other: FeatureVector) -> list[float]:
        """Element-wise difference — what Bradley-Terry is fitted on."""
        mine, theirs = self.as_list(), other.as_list()
        return [a - b for a, b in zip(mine, theirs, strict=True)]


def dialogue_ratio(text: str) -> float:
    """Fraction of sentences that contain quoted speech.

    A blunt proxy for a real stylistic axis: some writers want scenes carried by dialogue,
    others by action and interiority. Counting quotation marks is crude and it separates
    those two modes reliably, which is all a feature needs to do.

    Args:
        text: The prose.

    Returns:
        A value in ``[0, 1]``.
    """
    if not text.strip():
        return 0.0
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return 0.0
    quoted = sum(1 for s in sentences if '"' in s or "“" in s or "’’" in s)
    return quoted / len(sentences)


def length_score(text: str, target: float = TARGET_WORDS) -> float:
    """How close the text is to a target length, in ``[0, 1]``.

    Peaks at the target and falls off either side, so the head can learn "this writer
    likes them shorter" as a signed weight rather than having "longer is better" baked in.

    Args:
        text: The prose.
        target: Target word count.

    Returns:
        1.0 at the target, decaying towards 0.
    """
    words = len(text.split())
    if words == 0:
        return 0.0
    ratio = words / target
    return float(1.0 / (1.0 + abs(ratio - 1.0)))


def continuity_score(hard_flags: int, soft_flags: int) -> float:
    """Turn flag counts into a score in ``[0, 1]``.

    Hard flags cost much more than soft ones, and neither is fatal — the writer decides.

    Args:
        hard_flags: Number of deterministic contradictions.
        soft_flags: Number of model opinions.

    Returns:
        1.0 when clean, decaying with flags.
    """
    return float(max(0.0, 1.0 - 0.35 * hard_flags - 0.08 * soft_flags))


def criterion_slots(scores: dict[str, float], order: Sequence[str]) -> dict[str, float]:
    """Place writer-criterion scores into their fixed slots.

    Args:
        scores: Judge scores on the writer's criteria, by name, on a 1-5 scale.
        order: The writer's criteria in creation order. A criterion's slot is its
            position here, so adding a criterion never renumbers the existing ones.

    Returns:
        One entry per slot. An unfilled slot is **0.0**, not the neutral 0.5 a missing
        judge score gets: "this writer has not defined a third criterion" is a fact about
        the writer, and a fixed constant carries no gradient into the head.
    """
    scaled = {k: float(max(0.0, min(1.0, (v - 1.0) / 4.0))) for k, v in scores.items()}
    out: dict[str, float] = dict.fromkeys(CRITERION_SLOTS, 0.0)
    for index, name in enumerate(order[:MAX_CRITERION_SLOTS]):
        # A defined criterion the judge did not score is neutral, like any missing score.
        out[CRITERION_SLOTS[index]] = scaled.get(name, 0.5)
    return out


def build(
    *,
    judge_scores: dict[str, float] | None = None,
    writer_criteria_scores: dict[str, float] | None = None,
    criterion_order: Sequence[str] = (),
    hard_flags: int = 0,
    soft_flags: int = 0,
    voice_cosine: float = 0.0,
    edit_direction: float = 0.0,
    text: str = "",
) -> FeatureVector:
    """Assemble a candidate's feature vector.

    Args:
        judge_scores: Per-axis judge scores on a 1-5 scale.
        writer_criteria_scores: Judge scores on the writer's own criteria, 1-5.
        criterion_order: The writer's criteria in creation order, which fixes which slot
            each one occupies. Empty means the writer has defined none.
        hard_flags: Deterministic contradictions found.
        soft_flags: Soft flags found.
        voice_cosine: Cosine to the writer-voice centroid, already in ``[0, 1]``.
        edit_direction: Projection onto the mean edit direction, already in ``[0, 1]``.
        text: The candidate's prose, for length and dialogue ratio.

    Returns:
        The feature vector.
    """
    judge = judge_scores or {}
    writer = writer_criteria_scores or {}
    # Falling back to the scored names keeps callers that never had an order (the gallery
    # scripts, ad-hoc tests) producing the same slots they would have with one.
    order = tuple(criterion_order) or tuple(writer)

    def scaled(name: str) -> float:
        raw = judge.get(name)
        if raw is None:
            return 0.5  # a missing judge score is neutral, not bad
        return float(max(0.0, min(1.0, (raw - 1.0) / 4.0)))

    return FeatureVector(
        values={
            "judge_momentum": scaled("momentum"),
            "judge_specificity": scaled("specificity"),
            "judge_consequence": scaled("consequence"),
            "judge_voice": scaled("voice"),
            **criterion_slots(writer, order),
            "continuity": continuity_score(hard_flags, soft_flags),
            "voice_cosine": float(max(0.0, min(1.0, voice_cosine))),
            "edit_direction": float(max(0.0, min(1.0, edit_direction))),
            "length": length_score(text),
            "dialogue_ratio": dialogue_ratio(text),
        }
    )


def from_candidate(
    candidate: object,
    *,
    voice_cosine: float = 0.0,
    edit_direction: float = 0.0,
    criterion_order: Sequence[str] = (),
) -> FeatureVector:
    """Build features from a :class:`storygit.selection.select.Candidate`.

    Args:
        candidate: The candidate.
        voice_cosine: Voice-model score, if one has been fitted.
        edit_direction: Edit-direction projection, if one exists.
        criterion_order: The writer's criteria in creation order.

    Returns:
        The feature vector.
    """
    from storygit.selection.select import candidate_text

    scores = dict(getattr(candidate, "judge_scores", {}) or {})
    fixed = {k: v for k, v in scores.items() if k in FIXED_CRITERIA}
    writer = {k: v for k, v in scores.items() if k not in FIXED_CRITERIA}
    flags = getattr(candidate, "flags", ())
    return build(
        judge_scores=fixed,
        writer_criteria_scores=writer,
        criterion_order=criterion_order,
        hard_flags=sum(1 for f in flags if f.is_hard),
        soft_flags=sum(1 for f in flags if not f.is_hard),
        voice_cosine=voice_cosine,
        edit_direction=edit_direction,
        text=candidate_text(candidate.proposal),  # type: ignore[attr-defined]
    )
