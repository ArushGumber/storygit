"""``PreferenceLayer``: the four learners behind one interface.

The engine sees one object with two methods — ``score`` and ``learn`` — and the four
learners underneath it are an implementation detail. That matters because they have very
different data appetites: exemplar retrieval works from the first accepted paragraph, the
edit-direction vector from the first edit, the Bradley-Terry head from about ten
comparisons, and the voice model wants a few dozen. Hiding them behind one interface lets
each contribute as soon as it has enough and stay silent until then.

The ablation switch is the same interface: ``enabled=False`` makes ``score`` return
``None``, which makes selection fall back to the judge score — byte-identically to chunk 3.
"""

from __future__ import annotations

from collections.abc import Sequence

from storygit.preference.bandit import Arm, BanditState, ThompsonBandit
from storygit.preference.bt_head import BTWeights, fit
from storygit.preference.exemplars import ExemplarPool
from storygit.preference.features import FeatureVector, from_candidate
from storygit.preference.signals import EditPair, PreferencePair, SignalReader
from storygit.preference.state import PreferenceState, PreferenceStateStore
from storygit.preference.voice import VoiceModel, train

EDITED_WIN_WEIGHT = 0.5
"""How much a win counts when the writer had to edit it first. Half of a clean accept."""

MIN_PAIRS_TO_FIT = 4
"""Below this the head stays at its prior. Four points cannot outvote a population."""

MIN_ANCHORS_TO_TRAIN_VOICE = 4
"""Below this the voice model stays untrained and scores everything 0.5."""


class PreferenceLayer:
    """Scores candidates from what the writer has done, and learns from what they do.

    Attributes:
        state: The current learned state.
        enabled: When False, the layer contributes nothing — the ablation switch.
    """

    def __init__(
        self,
        state: PreferenceState | None = None,
        *,
        enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        """Create the layer.

        Args:
            state: Learned state to start from; a fresh one if omitted.
            enabled: Whether the layer contributes at all.
            seed: Fixes bandit sampling and voice-model initialization.
        """
        self.state = state or PreferenceState()
        self.enabled = enabled
        self._seed = seed
        self._bandit = ThompsonBandit(self.state.bandit, seed=seed)
        self._exemplars = ExemplarPool()

    # --- scoring ----------------------------------------------------------------

    @property
    def exemplars(self) -> ExemplarPool:
        """The retrieval pool used for prompt exemplars."""
        return self._exemplars

    def set_exemplars(self, positives: Sequence[str], negatives: Sequence[str]) -> None:
        """Replace the exemplar pools."""
        self._exemplars = ExemplarPool(positives, negatives)

    def features_for(self, candidates: Sequence[object]) -> list[FeatureVector]:
        """Build feature vectors for a candidate set.

        Voice and edit-direction scores are computed over the whole set at once, because
        both are relative measures — the direction projection is min-max scaled across the
        candidates on offer.

        Args:
            candidates: :class:`storygit.selection.select.Candidate` objects.

        Returns:
            One feature vector per candidate, in order.
        """
        if not candidates:
            return []
        from storygit.selection.select import candidate_text

        texts = [candidate_text(c.proposal) for c in candidates]  # type: ignore[attr-defined]
        voice_scores = self.state.voice.score(texts)
        direction_scores = self.state.voice.direction_scores(texts)
        return [
            from_candidate(
                candidate,
                voice_cosine=voice_scores[i],
                edit_direction=direction_scores[i],
            )
            for i, candidate in enumerate(candidates)
        ]

    def score(self, candidates: Sequence[object]) -> list[float] | None:
        """Rank candidates by the learned head.

        Args:
            candidates: The candidate set.

        Returns:
            One score per candidate in ``[0, 1]``, or ``None`` when the layer is disabled
            — which is what makes the ablation exact rather than approximate.
        """
        if not self.enabled or not candidates:
            return None
        features = self.features_for(candidates)
        raw = [self.state.weights.score(f) for f in features]
        low, high = min(raw), max(raw)
        if high - low < 1e-9:
            return [0.5] * len(raw)
        return [(value - low) / (high - low) for value in raw]

    def compose(
        self, ranked: list[int], surprising: list[int], k: int = 3
    ) -> tuple[list[int], Arm]:
        """Let the bandit decide whether to spend a slot on exploration."""
        if not self.enabled:
            return ranked[:k], Arm.safe
        return self._bandit.compose(ranked, surprising, k=k)

    def reward(self, arm: Arm, reward: float) -> None:
        """Record what the writer did with the composition that was shown."""
        self._bandit.update(arm, reward)
        self.state = self.state.model_copy(update={"bandit": self._bandit.state})

    # --- learning ---------------------------------------------------------------

    def learn(
        self,
        reader: SignalReader,
        features_by_proposal: dict[str, FeatureVector],
        *,
        texts_by_proposal: dict[str, str] | None = None,
        human_prose: Sequence[str] = (),
    ) -> PreferenceState:
        """Refit everything from the accumulated signals.

        Cheap enough (milliseconds) to run after every accept, so the next candidate set
        already reflects the last decision. That immediacy is most of the point: a writer
        who corrects the same thing twice and sees no change stops correcting.

        Args:
            reader: Reads preference pairs and edit pairs out of the signal store.
            features_by_proposal: Feature vectors for every proposal ever shown, by id.
                The engine caches these when it shows a candidate set; they cannot be
                recomputed later because they depend on the state at that moment.
            texts_by_proposal: Candidate text by proposal id, for the exemplar pools and
                voice training.
            human_prose: Prose the writer wrote or edited, the anchors for the voice model.

        Returns:
            The updated state, also stored on ``self``.
        """
        if not self.enabled:
            return self.state

        texts = texts_by_proposal or {}
        pairs = reader.preference_pairs()
        edits = reader.edit_pairs()

        weights = self._fit_head(pairs, features_by_proposal)
        voice = self._train_voice(reader, edits, texts, human_prose)
        self._refresh_exemplars(reader, texts, human_prose)

        self.state = self.state.model_copy(
            update={
                "weights": weights,
                "voice": voice,
                "bandit": self._bandit.state,
                "pairs_seen": len(pairs),
                "edits_seen": len(edits),
            }
        )
        return self.state

    def _fit_head(
        self,
        pairs: Sequence[PreferencePair],
        features_by_proposal: dict[str, FeatureVector],
    ) -> BTWeights:
        """Fit Bradley-Terry on the comparisons we have features for."""
        usable: list[tuple[FeatureVector, FeatureVector]] = []
        sample_weights: list[float] = []
        for pair in pairs:
            winner = features_by_proposal.get(str(pair.winner))
            loser = features_by_proposal.get(str(pair.loser))
            if winner is None or loser is None:
                continue
            usable.append((winner, loser))
            sample_weights.append(EDITED_WIN_WEIGHT if pair.was_edited else 1.0)

        if len(usable) < MIN_PAIRS_TO_FIT:
            return self.state.weights
        return fit(
            usable,
            prior=self.state.weights,
            l2=2.0,
            sample_weights=sample_weights,
        )

    def _train_voice(
        self,
        reader: SignalReader,
        edits: Sequence[EditPair],
        texts: dict[str, str],
        human_prose: Sequence[str],
    ) -> VoiceModel:
        """Retrain the contrastive head and recompute the edit direction."""
        anchors = [*human_prose, *(e.after for e in edits)]
        positives = [texts[str(p)] for p in reader.accepted_ids() if str(p) in texts]
        negatives = [
            texts[str(p)]
            for p in (*reader.rejected_ids(), *reader.unchosen_ids())
            if str(p) in texts
        ]
        negatives.extend(e.before for e in edits)

        if len(anchors) < MIN_ANCHORS_TO_TRAIN_VOICE or not positives or not negatives:
            # Not enough to train a head, but the edit direction is meaningful from one
            # pair, so compute that and leave the projection untrained.
            from storygit.preference.voice import edit_direction

            return VoiceModel(
                direction=edit_direction([e.before for e in edits], [e.after for e in edits]),
                edits_seen=len(edits),
            )
        return train(
            anchors,
            positives,
            negatives,
            edit_before=[e.before for e in edits],
            edit_after=[e.after for e in edits],
            seed=self._seed or 0,
        )

    def _refresh_exemplars(
        self,
        reader: SignalReader,
        texts: dict[str, str],
        human_prose: Sequence[str],
    ) -> None:
        """Rebuild the retrieval pools from accepted, human, and rejected work."""
        positives = [texts[str(p)] for p in reader.accepted_ids() if str(p) in texts]
        positives.extend(human_prose)
        negatives = [texts[str(p)] for p in reader.rejected_ids() if str(p) in texts]
        self._exemplars = ExemplarPool(positives, negatives)

    # --- persistence ------------------------------------------------------------

    def save(self, store: PreferenceStateStore, branch: str) -> None:
        """Persist the learned state for a branch."""
        store.save(branch, self.state)

    @staticmethod
    def load(
        store: PreferenceStateStore,
        branch: str,
        *,
        enabled: bool = True,
        seed: int | None = None,
    ) -> PreferenceLayer:
        """Load a branch's learned state.

        Args:
            store: The state store.
            branch: Branch name.
            enabled: Whether the layer contributes.
            seed: Fixes sampling.

        Returns:
            A layer, freshly initialized if the branch has no state yet.
        """
        return PreferenceLayer(store.load(branch), enabled=enabled, seed=seed)

    @staticmethod
    def with_prior(*, enabled: bool = True, seed: int | None = None) -> PreferenceLayer:
        """A layer starting from the pretrained population prior.

        Args:
            enabled: Whether the layer contributes.
            seed: Fixes sampling.

        Returns:
            A layer whose head starts at the prior rather than at uniform weights.
        """
        from storygit.preference.pretrain import fit_prior

        return PreferenceLayer(PreferenceState(weights=fit_prior()), enabled=enabled, seed=seed)


__all__ = [
    "Arm",
    "BanditState",
    "PreferenceLayer",
    "PreferenceState",
    "PreferenceStateStore",
]
