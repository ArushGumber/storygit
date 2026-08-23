"""Simulated writers: hidden weight vectors, style quirks, and forbidden moves.

A human study is out of scope, so the evaluation needs writers it can run thousands of
decisions through. The design that makes this honest rather than circular is that each
persona's preferences are **a hidden weight vector over exactly the feature space the
preference head is fitted on**. That turns the central question into a measurable one:
after N decisions, does the fitted weight vector correlate with the hidden one? Not "did
acceptance go up" — which any degenerate system can achieve by proposing the same thing
repeatedly — but "did the machinery recover the taste it was shown".

**What this cannot measure**, stated here because it is the first thing an interviewer
should ask about: real taste, fatigue, trust, whether the prose is any good, or whether a
listener would press next episode. A persona is a linear functional plus noise. It cannot
be surprised, cannot change its mind, and cannot tell you the system is exhausting to use.
Everything measured here is a property of the *machinery*, and that is the only claim made.

The personas are deliberately a different module from `preference/pretrain.py`'s
proto-personas. The prior is fitted on those; the system is measured on these. Letting the
two touch would be marking one's own homework.
"""

from __future__ import annotations

import random
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from storygit.preference.features import BASE_FEATURES, FeatureVector


class ForbiddenMove(StrEnum):
    """Things a persona will always reject, regardless of score.

    Real writers have absolutes — a line they will not cross whatever else is on offer —
    and a purely linear preference model cannot express one. These are the hard veto, and
    they matter for the evaluation because they are the only signal the system cannot
    learn by moving weights around: it has to learn them as a rejected direction.
    """

    kill_the_mentor = "kill_the_mentor"
    flashback = "flashback"
    prophecy = "prophecy"
    romance_subplot = "romance_subplot"


FORBIDDEN_MARKERS: dict[ForbiddenMove, tuple[str, ...]] = {
    ForbiddenMove.kill_the_mentor: ("dies", "death of", "kills the", "murdered", "corpse"),
    ForbiddenMove.flashback: ("years earlier", "flashback", "remembered the day", "as a child"),
    ForbiddenMove.prophecy: ("prophecy", "foretold", "chosen one", "destined"),
    ForbiddenMove.romance_subplot: ("kissed", "in love", "lovers", "romance"),
}
"""Crude surface markers. A persona is a simulation, not a reader; this is enough to make
the veto fire reliably and is honest about being a proxy."""


class Persona(BaseModel):
    """One simulated writer.

    Attributes:
        name: Label used in results and plots.
        weights: Hidden preference weights over ``BASE_FEATURES``. Never visible to the
            engine — the evaluation asserts this.
        accept_threshold: Score below which nothing is accepted, in ``[0, 1]``.
        edit_threshold: Score below the accept threshold but above this triggers an edit
            rather than a rejection.
        noise: Decision noise, so choices are not perfectly separable.
        forbidden: Moves that are always rejected.
        target_words: Preferred prose length.
        dialogue_appetite: Preferred dialogue ratio in ``[0, 1]``.
        lock_probability: Chance of locking a node after accepting it.
        dial: Where this writer keeps the coherent-to-surprising dial.
        style_notes: Notes this writer states explicitly early in a run.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    weights: dict[str, float]
    accept_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    edit_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    noise: float = Field(default=0.08, ge=0.0)
    forbidden: tuple[ForbiddenMove, ...] = ()
    target_words: int = 200
    dialogue_appetite: float = 0.3
    lock_probability: float = 0.05
    dial: float = 0.35
    style_notes: tuple[str, ...] = ()

    def score(self, features: FeatureVector, rng: random.Random) -> float:
        """This persona's private opinion of a candidate.

        Args:
            features: The candidate's features.
            rng: Decision noise source.

        Returns:
            A score in ``[0, 1]``.
        """
        total = sum(self.weights.get(k, 0.0) * v for k, v in features.values.items())
        scale = sum(abs(w) for w in self.weights.values()) or 1.0
        return max(0.0, min(1.0, total / scale + rng.gauss(0.0, self.noise)))

    def vetoes(self, text: str) -> ForbiddenMove | None:
        """Whether the candidate crosses one of this writer's absolute lines.

        Args:
            text: The candidate's text.

        Returns:
            The move that was crossed, or ``None``.
        """
        lowered = text.lower()
        for move in self.forbidden:
            if any(marker in lowered for marker in FORBIDDEN_MARKERS[move]):
                return move
        return None

    def edit_instruction(self) -> str:
        """What this writer would tell a copy editor.

        Deliberately says nothing about the hidden weights: the edit model is told the
        *style*, never the objective, so the engine cannot learn the answer from text the
        persona produced.
        """
        parts = [f"Rewrite to about {self.target_words} words."]
        if self.dialogue_appetite > 0.5:
            parts.append("Carry it with dialogue; cut narration.")
        elif self.dialogue_appetite < 0.2:
            parts.append("Cut the dialogue; carry it with action and observation.")
        parts.extend(self.style_notes)
        return " ".join(parts)


def _weights(**overrides: float) -> dict[str, float]:
    """A weight vector at zero except where named."""
    return {**dict.fromkeys(BASE_FEATURES, 0.0), **overrides}


SERIALIST = Persona(
    name="the Serialist",
    weights=_weights(
        judge_momentum=1.0,
        judge_consequence=0.9,
        judge_specificity=0.4,
        continuity=0.8,
        writer_criteria=0.7,
        length=0.2,
        dialogue_ratio=0.3,
    ),
    accept_threshold=0.50,
    forbidden=(ForbiddenMove.flashback,),
    target_words=240,
    dialogue_appetite=0.45,
    dial=0.45,
    style_notes=("End on a question, not an answer.",),
)
"""Writes for retention: momentum and consequence above all, no flashbacks (they stall)."""

MINIMALIST = Persona(
    name="the Minimalist",
    weights=_weights(
        judge_specificity=1.0,
        judge_voice=0.7,
        continuity=0.6,
        length=-0.9,
        judge_momentum=0.3,
        dialogue_ratio=0.2,
    ),
    accept_threshold=0.55,
    edit_threshold=0.35,
    forbidden=(ForbiddenMove.prophecy,),
    target_words=110,
    dialogue_appetite=0.25,
    dial=0.25,
    style_notes=("Shorter sentences. No adverbs.",),
)
"""Short, concrete, exposition-averse. The negative weight on length is the point."""

MAXIMALIST = Persona(
    name="the Maximalist",
    weights=_weights(
        judge_voice=1.0,
        judge_specificity=0.6,
        voice_cosine=0.8,
        length=0.6,
        continuity=0.4,
        judge_consequence=0.5,
        dialogue_ratio=-0.3,
    ),
    accept_threshold=0.52,
    forbidden=(ForbiddenMove.romance_subplot,),
    target_words=340,
    dialogue_appetite=0.10,
    dial=0.70,
    style_notes=("Stay inside the character's head. Interiority over incident.",),
)
"""Interiority and subversion; wants the dial high and the dialogue low."""

CONTROLLER = Persona(
    name="the Controller",
    weights=_weights(
        continuity=1.0,
        judge_consequence=0.7,
        writer_criteria=0.9,
        judge_specificity=0.5,
        edit_direction=0.6,
        length=-0.2,
    ),
    accept_threshold=0.68,
    edit_threshold=0.30,
    noise=0.05,
    forbidden=(ForbiddenMove.kill_the_mentor, ForbiddenMove.prophecy),
    target_words=190,
    dialogue_appetite=0.35,
    lock_probability=0.45,
    dial=0.15,
    style_notes=("Nothing contradicts what is already on the page.",),
)
"""Accepts rarely, edits heavily, locks constantly. The hardest writer to satisfy, and the
one whose behaviour most stresses propagation and the lock semantics."""

PERSONAS: dict[str, Persona] = {p.name: p for p in (SERIALIST, MINIMALIST, MAXIMALIST, CONTROLLER)}
"""All personas by name."""


def get(name: str) -> Persona:
    """Look a persona up by name.

    Args:
        name: The persona's name.

    Returns:
        The persona.

    Raises:
        KeyError: If no such persona exists.
    """
    return PERSONAS[name]
