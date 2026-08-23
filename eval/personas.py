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

from storygit.preference.features import BASE_FEATURES, CRITERION_SLOTS, FeatureVector


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
        accept_quantile: How picky this writer is, as a quantile of the scores they would
            typically give. 0.6 means "accepts roughly the top 40% of what they see".
        edit_quantile: Below the accept quantile but above this, they rewrite it rather
            than rejecting it.
        noise: Decision noise, so choices are not perfectly separable.
        forbidden: Moves that are always rejected.
        hand_write_probability: How often this writer discards the suggestion at prose
            level and writes the beat themselves. Real writers do; the interface has the
            path ("write it myself"); and until this existed no live run ever produced it,
            which is why the voice model was untrained in every run ever recorded. It
            wants anchors — prose the *writer* wrote or rewrote — and generated prose the
            writer merely accepted is a positive example, not an anchor.
        prose_polish_probability: How often this writer accepts a *prose* candidate but in
            their own words rather than the model's. Writers polish prose and do not polish
            an outline the same way, so this applies at the prose level only. It is the one
            action that produces a before/after pair, which is what the edit-direction
            feature is computed from and a second source of voice anchors.
        criteria: Writer-defined scoring axes as ``(name, description)``, in the order
            this writer would have created them. The run adds them to the ledger, so the
            judge scores every candidate on them and each one lands in its own feature
            slot. Before this existed the personas defined none, which meant the collapsed
            ``writer_criteria`` feature was the constant 0.5 in every live run and the two
            personas that carried weight on it were weighting a constant.
        target_words: Preferred prose length.
        dialogue_appetite: Preferred dialogue ratio in ``[0, 1]``.
        lock_probability: Chance of locking a node after accepting it.
        dial: Where this writer keeps the coherent-to-surprising dial.
        style_notes: Notes this writer states explicitly early in a run.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    weights: dict[str, float]
    accept_quantile: float = Field(default=0.55, ge=0.0, le=1.0)
    edit_quantile: float = Field(default=0.25, ge=0.0, le=1.0)
    noise: float = Field(default=0.08, ge=0.0)
    forbidden: tuple[ForbiddenMove, ...] = ()
    target_words: int = 200
    dialogue_appetite: float = 0.3
    lock_probability: float = 0.05
    dial: float = 0.35
    style_notes: tuple[str, ...] = ()
    criteria: tuple[tuple[str, str], ...] = ()
    hand_write_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    prose_polish_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    def thresholds(self) -> tuple[float, float]:
        """The accept and edit thresholds, in score units.

        Computed as quantiles of this persona's *own* score distribution over a reference
        sample of candidate feature vectors, rather than set as raw numbers.

        The raw thresholds this replaced are worth recording as a mistake. 0.55/0.40
        sounded reasonable and sat far below where scores actually land: the first full run
        accepted 33 of 33 candidates, so the acceptance curve was flat at 1.0 and measured
        nothing. The cause is structural — a persona's score is a weighted mean of features
        that are themselves mostly in the upper half of their range (continuity is 1.0 on a
        clean story, judge scores cluster near 0.7), so it lands around 0.8 and any
        threshold below that accepts everything.

        Quantiles fix it for every persona at once, adjust to each weight vector (the
        Minimalist's large negative weight on length shifts its whole distribution), and
        are interpretable in the write-up: "the Controller accepts roughly the top 8% of
        what it is shown".

        Returns:
            ``(accept_threshold, edit_threshold)`` in score units.
        """
        cached = _THRESHOLDS.get(self.name)
        if cached is not None:
            return cached
        rng = random.Random(20260823)
        scores = sorted(
            max(
                self._raw_score(reference_features(rng, criteria=len(self.criteria)))
                for _ in range(BEST_OF)
            )
            for _ in range(REFERENCE_SAMPLES)
        )

        def quantile(q: float) -> float:
            index = min(len(scores) - 1, max(0, int(q * len(scores))))
            return scores[index]

        pair = (quantile(self.accept_quantile), quantile(self.edit_quantile))
        _THRESHOLDS[self.name] = pair
        return pair

    def _raw_score(self, features: FeatureVector) -> float:
        """The noiseless score, used for threshold calibration."""
        total = sum(self.weights.get(k, 0.0) * v for k, v in features.values.items())
        scale = sum(abs(w) for w in self.weights.values()) or 1.0
        return total / scale

    def score(self, features: FeatureVector, rng: random.Random) -> float:
        """This persona's private opinion of a candidate.

        Args:
            features: The candidate's features.
            rng: Decision noise source.

        Returns:
            A score in ``[0, 1]``.
        """
        return max(0.0, min(1.0, self._raw_score(features) + rng.gauss(0.0, self.noise)))

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


REFERENCE_SAMPLES = 4000
"""How many synthetic candidates the threshold calibration draws."""

REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "judge_momentum": (0.60, 1.00),
    "judge_specificity": (0.60, 1.00),
    "judge_consequence": (0.55, 1.00),
    "judge_voice": (0.55, 1.00),
    "criterion_1": (0.55, 0.98),
    "criterion_2": (0.55, 0.98),
    "criterion_3": (0.55, 0.98),
    "criterion_4": (0.55, 0.98),
    "continuity": (0.65, 1.00),
    "voice_cosine": (0.45, 0.58),
    "edit_direction": (0.42, 0.60),
    "length": (0.40, 1.00),
    "dialogue_ratio": (0.00, 0.75),
}
"""Ranges fitted to a measured run, not guessed.

The first version used plausible-looking ranges centred near 0.5 and put every threshold
about 0.15 too low, so the first full run accepted 33 of 33 candidates. Real candidate
features sit high: continuity is 1.0 on a clean story, an untrained voice model returns a
flat 0.5, and the judge rarely scores below 3 out of 5. These ranges reproduce the score
distribution actually observed (p10 0.67, p50 0.80, p90 0.90 for the Serialist).

The quantile a persona is defined by is the design parameter — how picky this writer is.
The acceptance rate that results is a *measurement*, and the two are deliberately not
forced to agree.
"""

BEST_OF = 3
"""Calibrate on the best of this many draws, because that is what a decision faces.

A writer does not judge a random candidate; they judge the best of the three they were
shown. Calibrating on a single draw would put the threshold well below where the decision
actually happens.
"""

_THRESHOLDS: dict[str, tuple[float, float]] = {}


def reference_features(rng: random.Random, *, criteria: int = 0) -> FeatureVector:
    """A synthetic candidate drawn from the ranges real candidates occupy.

    Args:
        rng: Draw source.
        criteria: How many criteria the writer has defined. Slots beyond that are exactly
            zero in a real run, and drawing them from a range would calibrate the
            thresholds against candidates that cannot occur.
    """
    values = {name: rng.uniform(*REFERENCE_RANGES.get(name, (0.0, 1.0))) for name in BASE_FEATURES}
    for index, slot in enumerate(CRITERION_SLOTS):
        if index >= criteria:
            values[slot] = 0.0
    return FeatureVector(values=values)


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
        criterion_1=0.9,
        criterion_2=0.3,
        length=0.2,
        dialogue_ratio=0.3,
    ),
    criteria=(
        ("pull", "Does the end of this make me need the next one?"),
        ("payoff", "Does it move an open thread towards being answered?"),
    ),
    accept_quantile=0.35,
    edit_quantile=0.10,
    hand_write_probability=0.20,
    prose_polish_probability=0.3,
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
        criterion_1=0.8,
        criterion_2=0.2,
    ),
    criteria=(
        ("restraint", "Is anything here said twice, or said at all when it could be shown?"),
        ("concreteness", "Nouns a reader can see, rather than gestures at them."),
    ),
    accept_quantile=0.65,
    edit_quantile=0.20,
    hand_write_probability=0.35,
    prose_polish_probability=0.45,
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
        criterion_1=0.7,
        criterion_2=0.4,
    ),
    criteria=(
        ("interiority", "Are we inside someone, or watching from outside?"),
        ("texture", "Does the world have weather, smell, and wear on it?"),
    ),
    accept_quantile=0.55,
    edit_quantile=0.15,
    hand_write_probability=0.15,
    prose_polish_probability=0.25,
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
        criterion_1=1.0,
        criterion_2=0.15,
        judge_specificity=0.5,
        edit_direction=0.6,
        length=-0.2,
    ),
    criteria=(
        ("no loose ends", "Does this leave anything unexplained that was not meant to be?"),
        ("earned", "Has the story paid for what happens here?"),
    ),
    accept_quantile=0.92,
    edit_quantile=0.15,
    hand_write_probability=0.30,
    prose_polish_probability=0.4,
    noise=0.05,
    forbidden=(ForbiddenMove.kill_the_mentor, ForbiddenMove.prophecy),
    target_words=190,
    dialogue_appetite=0.35,
    lock_probability=0.45,
    dial=0.15,
    style_notes=("Nothing contradicts what is already on the page.",),
)
"""Accepts rarely, edits heavily, locks constantly. Taking only the top ~8% of what they
are shown makes this the hardest writer to satisfy, and the one whose behaviour most
stresses propagation, the lock semantics, and the edit-mining path."""

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
