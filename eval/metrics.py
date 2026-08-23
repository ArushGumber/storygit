"""Every metric the evaluation reports, one function each.

Each takes plain data and returns plain numbers, so every one is unit-testable on
hand-built input with a known answer — which is the point. A metrics module that can only
be exercised by running the whole harness is a metrics module nobody checks.

Where a metric could flatter the system, it is defined to avoid that:

* Acceptance rate counts edit-then-accept as an accept, because a writer who had to rewrite
  it still used it — but edit *distance* is reported beside it, so "acceptance went up"
  cannot hide "and they rewrote every word".
* Checker recall is reported per layer, with the false-positive rate on a clean story
  alongside. A checker that flags everything has perfect recall and is worthless.
* Stale prediction reports precision *and* recall across a threshold sweep, because the
  soft-edge ablation trades one for the other and a single number would hide the trade.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict


class Curve(BaseModel):
    """A named series, ready to plot.

    Attributes:
        label: Series name.
        x: X values.
        y: Y values.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    x: tuple[float, ...]
    y: tuple[float, ...]


class PrecisionRecall(BaseModel):
    """Precision, recall, and F1 for one configuration.

    Attributes:
        label: What was measured.
        true_positives: Correctly flagged.
        false_positives: Flagged and should not have been.
        false_negatives: Missed.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Of what was flagged, how much was real."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Of what was real, how much was flagged."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


# --- writer-facing curves -----------------------------------------------------


def rolling(values: Sequence[float], window: int = 10) -> list[float]:
    """Rolling mean, with a shrinking window at the start rather than a gap.

    A 15-episode run produces perhaps sixty decisions. Dropping the first ``window`` of
    them to get a clean rolling mean would throw away the part where learning is fastest.

    Args:
        values: The series.
        window: Window size.

    Returns:
        A series of the same length.
    """
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def acceptance_curve(kinds: Sequence[str], *, window: int = 10) -> list[float]:
    """Rolling acceptance rate over decisions.

    Args:
        kinds: ``accept`` / ``edit`` / ``reject`` per decision, in order.
        window: Rolling window.

    Returns:
        Rolling fraction of decisions that ended in the writer using something.
    """
    return rolling([1.0 if k in ("accept", "edit") else 0.0 for k in kinds], window)


def edit_distance_curve(
    kinds: Sequence[str], distances: Sequence[float], *, window: int = 10
) -> list[float]:
    """Rolling mean edit distance over the decisions where the writer edited.

    Reported beside the acceptance curve on purpose. Acceptance rising while edit distance
    stays flat means the writer is rewriting just as much and merely accepting more — which
    is not the system improving.

    Args:
        kinds: Decision kinds, in order.
        distances: Edit distance per decision (0 where none).
        window: Rolling window.

    Returns:
        Rolling mean distance, carrying the last value where no edit occurred.
    """
    seen: list[float] = []
    out: list[float] = []
    for kind, distance in zip(kinds, distances, strict=True):
        if kind == "edit":
            seen.append(distance)
        window_values = seen[-window:] if seen else []
        out.append(sum(window_values) / len(window_values) if window_values else 0.0)
    return out


# --- continuity checker -------------------------------------------------------


def checker_recall(
    injected: Sequence[object],
    flags: Sequence[object],
    *,
    layer: int | None = None,
) -> PrecisionRecall:
    """Recall over injected contradictions, optionally restricted to one layer.

    An injection counts as caught when any flag mentions one of its facts. False positives
    are flags that mention no injected fact — on a story that is otherwise clean by
    construction, those are genuine false alarms.

    Args:
        injected: :class:`eval.inject.Injection` objects.
        flags: :class:`storygit.continuity.flags.Flag` objects.
        layer: Restrict to flags from this layer.

    Returns:
        Precision, recall, and F1.
    """
    considered = [f for f in flags if layer is None or getattr(f, "layer", 0) == layer]
    flagged_facts: set[str] = set()
    for flag in considered:
        flagged_facts.update(str(f) for f in getattr(flag, "fact_ids", ()))

    caught = 0
    for injection in injected:
        wanted = {str(f) for f in getattr(injection, "fact_ids", ())}
        if wanted & flagged_facts:
            caught += 1

    injected_facts = {str(f) for injection in injected for f in getattr(injection, "fact_ids", ())}
    false_positives = sum(
        1
        for flag in considered
        if not ({str(f) for f in getattr(flag, "fact_ids", ())} & injected_facts)
    )
    return PrecisionRecall(
        label=f"layer {layer}" if layer is not None else "all layers",
        true_positives=caught,
        false_positives=false_positives,
        false_negatives=len(injected) - caught,
    )


def false_positive_rate(flags: Sequence[object], beats: int) -> float:
    """Flags per beat on a story with nothing wrong with it.

    The number that decides whether a checker is usable. A writer who sees a flag on every
    third beat that turns out to be nothing stops reading flags, and then the checker's
    recall no longer matters.

    Args:
        flags: Flags raised on a clean story.
        beats: How many beats it has.

    Returns:
        Flags per beat.
    """
    return len(flags) / beats if beats else 0.0


# --- extraction ---------------------------------------------------------------


def extraction_scores(
    expected: Sequence[tuple[str, str, str]],
    extracted: Sequence[tuple[str, str, str]],
) -> PrecisionRecall:
    """Precision and recall of fact extraction against a hand-built truth set.

    Matching is on ``(subject, predicate)`` with a loose object check: every content word
    of the shorter object must appear in the longer one, so "a blistered left hand" matches
    a truth of "blistered hand". Exact string matching would measure phrasing rather than
    extraction, and substring matching is not enough — an inserted adjective breaks it.

    Args:
        expected: Ground-truth ``(subject, predicate, object)`` triples, lowercased.
        extracted: What the model returned, same shape.

    Returns:
        Precision, recall, and F1.
    """

    def normalize(triple: tuple[str, str, str]) -> tuple[str, str, str]:
        a, b, c = triple
        return (a.strip().lower(), b.strip().lower(), c.strip().lower())

    truth = [normalize(t) for t in expected]
    got = [normalize(t) for t in extracted]

    matched_truth: set[int] = set()
    matched_got: set[int] = set()
    for got_index, (subject, predicate, obj) in enumerate(got):
        for truth_index, (t_subject, t_predicate, t_obj) in enumerate(truth):
            if truth_index in matched_truth:
                continue
            if subject != t_subject or predicate != t_predicate:
                continue
            if _objects_match(obj, t_obj):
                matched_truth.add(truth_index)
                matched_got.add(got_index)
                break

    return PrecisionRecall(
        label="extraction",
        true_positives=len(matched_truth),
        false_positives=len(got) - len(matched_got),
        false_negatives=len(truth) - len(matched_truth),
    )


# --- propagation --------------------------------------------------------------


STOPWORDS = frozenset({"a", "an", "the", "of", "his", "her", "their", "its", "to", "is"})
"""Words ignored when matching an extracted object against the truth."""


def _objects_match(a: str, b: str) -> bool:
    """Whether two object strings describe the same thing, loosely.

    Every content word of the shorter one must appear in the longer one. Loose enough that
    an inserted adjective is not a miss, strict enough that two different objects do not
    match each other.
    """
    tokens_a = {t for t in a.split() if t not in STOPWORDS}
    tokens_b = {t for t in b.split() if t not in STOPWORDS}
    if not tokens_a or not tokens_b:
        return a == b
    shorter, longer = (
        (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    )
    return shorter <= longer


def stale_scores(
    predicted: Sequence[str],
    truly_affected: Sequence[str],
    unaffected: Sequence[str],
    *,
    label: str = "declared",
) -> PrecisionRecall:
    """Precision and recall of staleness prediction against a known blast radius.

    Args:
        predicted: Node ids the system marked.
        truly_affected: Node ids that genuinely depend on the change.
        unaffected: Node ids that do not.
        label: What configuration produced this.

    Returns:
        Precision, recall, and F1.
    """
    marked = set(predicted)
    truth = set(truly_affected)
    clean = set(unaffected)
    return PrecisionRecall(
        label=label,
        true_positives=len(marked & truth),
        false_positives=len(marked & clean),
        false_negatives=len(truth - marked),
    )


# --- diversity ----------------------------------------------------------------


def mean_pairwise_distance(vectors: Sequence[Sequence[float]]) -> float:
    """Mean cosine distance between every pair, the diversity number.

    Args:
        vectors: Unit-norm embeddings of the shown candidates.

    Returns:
        Mean of ``1 - cosine`` over all pairs; 0.0 for fewer than two.
    """
    if len(vectors) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
            total += 1.0 - dot
            pairs += 1
    return total / pairs if pairs else 0.0


def diversity_quality_point(
    vectors: Sequence[Sequence[float]], qualities: Sequence[float], label: str
) -> dict[str, Any]:
    """One point on the diversity-versus-quality plot.

    Both numbers together, because either alone is trivially gameable: maximum diversity is
    three random candidates, and maximum quality is the same good candidate three times.

    Args:
        vectors: Embeddings of the shown candidates.
        qualities: Their quality scores.
        label: Which selector produced them.

    Returns:
        ``{"label", "diversity", "quality"}``.
    """
    return {
        "label": label,
        "diversity": mean_pairwise_distance(vectors),
        "quality": sum(qualities) / len(qualities) if qualities else 0.0,
    }


# --- learning -----------------------------------------------------------------


def retry_rescues(levels: list[str], kinds: list[str]) -> dict[str, int]:
    """How often a rejected set was rescued by the one informed re-proposal.

    A rejection is followed by exactly one retry at the same level. If that retry is
    accepted or edited, the decision was rescued; if it is rejected too, the writer meant
    it. Reported because the retry is the fix for the truncations, and "the runs stopped
    truncating" is a weaker claim than "the retry rescued N decisions that would otherwise
    have been lost".

    Args:
        levels: Level per decision, in order.
        kinds: ``accept`` / ``edit`` / ``reject`` per decision, in order.

    Returns:
        ``{"rejected_sets": ..., "rescued": ..., "stood": ...}``.
    """
    rejected = rescued = stood = 0
    index = 0
    while index < len(kinds):
        if kinds[index] != "reject":
            index += 1
            continue
        rejected += 1
        follows = index + 1 < len(kinds) and levels[index + 1] == levels[index]
        if follows and kinds[index + 1] != "reject":
            rescued += 1
            index += 2
        elif follows and kinds[index + 1] == "reject":
            stood += 1
            index += 2
        else:
            stood += 1
            index += 1
    return {"rejected_sets": rejected, "rescued": rescued, "stood": stood}


def unexercised_features(
    matrices: list[list[dict[str, float]]], *, tolerance: float = 1e-9
) -> list[str]:
    """Features that never varied within any candidate set the writer was shown.

    A feature constant across every set carries **no gradient**, so the fitted head keeps
    whatever the population prior said about it. In session that is harmless — a constant
    column cannot reorder anything the writer sees — and it is the reason it goes unnoticed.
    It stops being harmless the moment the head is applied to candidates drawn from a
    different distribution, which is exactly what the cross-persona probe does: the weight
    is then a prior artifact asserting a preference this writer never expressed.

    Reported per run so the claim is checkable in the artifact rather than asserted in
    prose.

    Args:
        matrices: One list of candidate feature dicts per decision.
        tolerance: Spread below which a feature counts as constant.

    Returns:
        Feature names, sorted, that never varied within a set.
    """
    names: set[str] = set()
    for shown in matrices:
        names |= set().union(*(set(f) for f in shown)) if shown else set()
    constant = []
    for name in sorted(names):
        spread = 0.0
        for shown in matrices:
            values = [f.get(name, 0.0) for f in shown]
            if values:
                spread = max(spread, max(values) - min(values))
        if spread <= tolerance:
            constant.append(name)
    return constant


def weight_recovery(fitted: dict[str, float], hidden: dict[str, float]) -> float:
    """Pearson correlation between the fitted and hidden weight vectors.

    The sharpest question the evaluation can ask, and the reason the personas are defined
    over the head's own feature space: not "did acceptance rise" but "did the machinery
    recover the taste it was shown".

    Args:
        fitted: What the head learned.
        hidden: The persona's true weights.

    Returns:
        Pearson r, or 0.0 if either vector is constant.
    """
    return _pearson(fitted, hidden, sorted(set(fitted) | set(hidden)))


def weight_recovery_on_axes_the_writer_has(
    fitted: dict[str, float], hidden: dict[str, float]
) -> float:
    """Weight recovery over the coordinates the writer actually weights.

    The headline figure is a thirteen-dimensional Pearson r, and five or six of a persona's
    thirteen weights are *exactly* zero -- including the two criterion slots no persona
    fills, which are constant 0.0 in every recorded decision. Shrinkage drives the fitted
    values on those same dead coordinates towards zero, so it mechanically improves
    agreement on dimensions nobody chose anything about. That is a real effect and it is
    not learning, and a reader who notices it before the author does has found something.

    So both are reported. This one asks the narrower and more honest question: over the
    axes this writer actually has an opinion on, how close is the fitted vector? It cannot
    be flattered by getting the zeros right, because the zeros are not in it.

    Args:
        fitted: What the head learned.
        hidden: The persona's true weights.

    Returns:
        Pearson r over the non-zero hidden coordinates, or 0.0 if there are fewer than two.
    """
    names = sorted(n for n, w in hidden.items() if w != 0.0)
    return _pearson(fitted, hidden, names) if len(names) >= 2 else 0.0


def _pearson(fitted: dict[str, float], hidden: dict[str, float], names: list[str]) -> float:
    """Pearson r between two weight dictionaries over a fixed coordinate list."""
    a = [fitted.get(n, 0.0) for n in names]
    b = [hidden.get(n, 0.0) for n in names]
    n = len(names)
    if n < 2:
        return 0.0
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a < 1e-12 or var_b < 1e-12:
        return 0.0
    return float(cov / (var_a**0.5 * var_b**0.5))


# --- cost ---------------------------------------------------------------------


def cost_per_action(call_summary: dict[str, Any], actions: int) -> dict[str, float]:
    """Tokens and dollars per writer decision.

    Dollars are zero during development, so the number that matters is the would-be cost
    from the price table — which is what makes the chunk 7 rerun on a paid model comparable
    rather than a separate experiment.

    Args:
        call_summary: Output of ``CallLog.summary()``.
        actions: How many writer decisions the run contained.

    Returns:
        Per-action tokens, dollars, calls, and the cache hit rate.
    """
    if not actions:
        return {"tokens": 0.0, "usd": 0.0, "calls": 0.0, "cache_hit_rate": 0.0}
    total_tokens = float(call_summary.get("total_tokens", 0) or 0)
    cost = float(call_summary.get("cost_usd", 0.0) or 0.0)
    calls = float(call_summary.get("calls", 0) or 0)
    return {
        "tokens": total_tokens / actions,
        "usd": cost / actions,
        "calls": calls / actions,
        "cache_hit_rate": float(call_summary.get("cache_hit_rate", 0.0) or 0.0),
    }
