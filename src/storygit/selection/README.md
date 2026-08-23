# `storygit.selection`

Turning `n` samples into `k` candidates a writer can actually choose between.

## The problem this solves

Sampling three continuations from one prompt gives you three phrasings of the same idea.
Fabula's answer was to penalize similarity against previous suggestions, which produces
variants that *differ* but not variants that are **nameably** different — so reviewing
three costs three times as much as reviewing one, and their writers reported it was
cheaper to write from scratch. That is Fabula finding P1, and it is an interface problem
disguised as a sampling problem.

Here every candidate is generated under a named instruction, and the name travels with it
to the screen. The writer reads *raise the stakes / slow down / subvert the expectation*
and picks a direction. Diversity stops being a property of the embedding space and becomes
a property of the choice.

## The pipeline

```
axes -> parallel sample -> embed -> quality -> continuity -> dial -> MMR/DPP -> k shown
```

| File | What it does |
|---|---|
| `axes.py` | The seven conditioning axes as data, with deterministic rotation per node. |
| `embed.py` | bge-small embeddings and similarity kernels, thin enough that the algorithms can be tested on synthetic geometry. |
| `mmr.py` | `argmax λ·quality − (1−λ)·max_sim_to_selected`. |
| `dpp.py` | Greedy MAP over a quality-weighted DPP kernel, plus the top-k baseline. |
| `dial.py` | Coherent ↔ surprising, as a reweighting against the greedy continuation. |
| `select.py` | The façade: `CandidateSelector.select(...) -> list[Candidate]`. |

## MMR versus DPP versus top-k

All three share one signature, so switching is one config value and the evaluation can
ablate it honestly.

**MMR** is the heuristic: subtract a similarity penalty, tune λ. It works, it is fast, and
the λ is a knob nobody can justify from first principles.

**DPP** is the principled version of the same instinct. With kernel
`L = diag(q)·S·diag(q)`, `det(L_S)` for a subset is the squared volume of the
parallelepiped those quality-scaled vectors span. Volume is large when the vectors are
long (high quality) *and* spread out (dissimilar), and collapses to zero when two are
parallel. So "maximize the determinant" is exactly "pick a good, non-redundant set" — with
no hand-tuned trade-off. Exact MAP is NP-hard in general; at n=6, k=3 greedy is trivial and
usually exact.

**`topk_temperature`** is the ablation baseline: sample hot, keep the best, and let all
three be the same idea. The evaluation compares mean pairwise distance at matched quality.

## The dial

Fabula's writers asked for an "absurdity dial", and the request is deeper than it sounds:
it asks to control the *objective*, not the output. A writer at 3am on episode 40 wants
safe continuations; the same writer opening a new arc wants to be surprised.

Take the greedy temperature-0 continuation as "what this model would obviously do next".
Each candidate's surprise is its embedding distance from that. Then

```
effective = (1 − d)·quality + d·surprise
```

At `d=0` the ranking is pure quality; at `d=1` pure distance-from-obvious. Both terms are
min-max rescaled first, because a judge score and a cosine distance arrive on unrelated
scales and blending them raw would let whichever has the wider range silently dominate.
The greedy continuation costs one extra cached call per node, so moving the dial is free.

## Candidates carry their own flags

`select()` runs layer 1 of the continuity checker on each candidate's *would-be* facts —
the diff is applied to a scratch state and checked there — so a candidate that contradicts
the bible arrives with the contradiction already attached. Hard flags push a candidate down
the ranking (×0.6 each) but never remove it: a contradiction can be exactly what the writer
wants, because characters lie and narrators are unreliable.

Every sampled candidate is returned, shortlisted ones first. The unselected ones matter:
the evaluation measures diversity over the whole set, and the preference layer needs them
as the negative side of a preference pair.

## Worked example

```python
selector = CandidateSelector(proposer, router, SelectionConfig(n=6, k=3))
candidates = await selector.select(
    state, Level.beat, target_node_id=scene_id, intent="Kael's ability shows itself"
)

for c in candidates:
    print(c.selected, c.axis_label, c.base_quality, c.surprise, c.effective_quality)
# True  raise the stakes           0.88  0.23  0.88
# True  introduce a complication   0.75  0.28  0.70
# True  slow down                  0.81  0.17  0.59
# False subvert the expectation    0.69  0.16  0.27
# ...
```
