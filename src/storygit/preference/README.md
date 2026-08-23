# `storygit.preference`

Learning the writer's taste from what they accept, reject, and edit. Four learners with
very different data appetites, behind one interface.

## Why four learners and not one

A writer produces about twenty comparisons in a session. That is nowhere near enough to
fit one model that does everything, so instead each signal is exploited by the cheapest
mechanism that can use it, and each contributes as soon as it has enough:

| Learner | File | Needs | Contributes |
|---|---|---|---|
| Exemplar retrieval | `exemplars.py` | 1 accepted paragraph | Their own sentences in the prompt |
| Edit direction | `voice.py` | 1 edit | A direction to score candidates against |
| Bradley-Terry head | `bt_head.py` | ~10 comparisons | The ranking |
| Edit mining | `edit_mining.py` | 2 similar edits | A standing style rule |
| Voice model | `voice.py` | ~a few dozen texts | "Sounds like this writer" |
| Bandit | `bandit.py` | ~20 shown sets | How much to explore |

`layer.py` is the façade. The engine sees `score(candidates)` and `learn(...)`, and the
learners underneath are an implementation detail.

## The signal is a comparison, not a rating

Every accept is recorded with `shown_with`: the other candidates on screen at the time.
"The writer liked A" is weak; "the writer preferred A to B and C, having seen all three" is
a comparison, and comparisons are what Bradley-Terry is defined on.

Pairs are only formed **within one shown set**. Comparing a candidate accepted on Monday
against one rejected in a different scene on Friday would compare two different questions.

An edited-then-accepted win is weaker evidence than a clean accept, and is down-weighted to
0.5 when fitting.

## Bradley-Terry, in one line

`P(a ≻ b) = σ(w·(f_a − f_b))` — logistic regression on feature differences, with the
intercept vanishing because it cancels. Convex, milliseconds in NumPy, and the learned `w`
is directly readable: a positive weight on `dialogue_ratio` means this writer likes
dialogue. That readability is not decoration — it is what lets the interface say *"you have
been choosing for specificity and against length."*

It is the same construction as the reward model in RLHF, minus the policy-gradient half.

## The feature vector is exactly the designed list

Ten features: four judge sub-scores on fixed narratology axes, the writer's own criteria,
continuity, voice cosine, edit-direction projection, length, and dialogue ratio. Few and
interpretable, because the head has to learn from tens of comparisons and a
high-dimensional head would memorize.

They are also **the same features the simulated writers' hidden weights are defined over**,
which is what makes the evaluation's central claim measurable: if the fitted weights
correlate with the hidden ones, the machinery recovers taste.

## The voice model

A judge can score whether a passage is well constructed. It cannot say whether it sounds
like *this writer* — that is an identity, not a quality, and it is different for everyone.

Freeze bge-small, train a 384→128 linear projection on top with InfoNCE. Anchors are prose
the writer wrote or edited; positives are what they accepted; negatives are what they
rejected and the pre-edit versions of what they changed. A frozen encoder and a small head,
because a writer produces a few thousand words in a session and fine-tuning an encoder on
that would destroy it.

**Edit direction vectors** are the cheap twin: the mean of `embed(after) − embed(before)`
over edit pairs, pointing from what the model writes towards what the writer wants. No
training at all, so it works from the first edit — long before the head has data.

## The bandit

A ranker that always shows its top three converges: the writer only sees what the model
already believes, so the model never learns it was wrong. Two arms over the *composition*
of the shown set — three safe, or two safe plus one high-surprise — chosen by Thompson
sampling.

Why Thompson over ε-greedy: ε-greedy explores at a fixed rate forever and explores
uniformly. Thompson's exploration rate falls out of the posterior width, with no schedule
to tune. The self-test (`python -m eval.bandit_selftest`, arms at 0.40 and 0.70) shows it:

```
pulls: {'safe': 16, 'explore': 384}
posterior means: safe=0.389 explore=0.707   (true 0.40 / 0.70)
pseudo-regret: thompson 4.8, epsilon-greedy 6.3 and still climbing
first half 4.5 -> second half 0.3
```

Thompson pays its exploration cost early and then flattens. ε-greedy keeps paying forever.

## Pretraining, and its honest limitation

Twenty comparisons cannot fit a model from scratch, so the head starts from a prior fitted
across synthetic proto-personas (`pretrain.py`) and adapts from there.

**The prior cannot contain real human taste** — nothing in that data ever met a reader.
What it can contain is the structure of the problem: that features have consistent signs,
that the signal is noisy, that some axes matter more. The bet is that the *adaptation
machinery* transfers even though the values do not, and that bet is measured, not asserted:

```
prior 0.775  vs  uniform 0.645  after a fresh writer's first 10 decisions   (+13pp)
```

If that lift were zero the prior would be worthless and should be dropped.

## Everything degrades gracefully at zero data

A brand new writer gets: an empty exemplar block, the head at its prior, the voice model
scoring everything 0.5, the edit direction absent, and uniform bandit posteriors. Nothing
crashes and — importantly — nothing skews. An unfitted head returns `None` rather than a
confident-looking guess, so selection ranks by the judge exactly as it did before the head
existed.

That is also the ablation switch: `PreferenceLayer(enabled=False)` reproduces chunk 3
**identically**, which is tested by comparing candidate ids, axes, selection, and effective
qualities against the selector called directly.

## Mined rules are never invisible

Every rule mined from an edit lands in the writer ledger, is shown in the interface, and
can be deleted. A rule enters prompts only after being observed twice — one idiosyncratic
edit is not a standing instruction. Rules are deduplicated by embedding cosine, so "shorter
sentences" and "keep the sentences short" bump one counter instead of filling the ledger
with near-duplicates.

## Where learned state lives

A `preference_state` table in the story's own SQLite file, keyed by branch. One file, so a
story stays a single copyable artifact; and branch-scoped, so taste learned on a what-if
branch cannot leak back to main.
