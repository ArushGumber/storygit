# 2026-08-23, pre-final-pass

The four-persona live run from the **fix pass**, kept because the final pass changed the
learner and the feature computation underneath it. These numbers are not comparable with
the current ones, and the paper does not compare them.

What changed between this run and the current one:

- **The learner.** Fitting now shrinks feature directions this writer's data cannot
  identify towards zero instead of leaving them at the population prior. Decided by an
  offline A/B against the held-out probe (agreement 0.426 → 0.565), with the decision rule
  written down before the number existed. In this run every unidentified direction still
  carried whatever the prior asserted about it.
- **Three features that were silently dead.** `dialogue_ratio` counted only double
  quotation marks while the model writes speech in single ones, so it read exactly 0.0
  here despite the prose being full of dialogue. `voice_cosine` and `edit_direction` sat at
  a flat 0.5 because the voice model had never been trained in any run — it needs prose the
  writer wrote or rewrote, and no simulated writer used those paths.
- **The writers.** Personas now hand-write some beats and polish prose in their own words,
  which is what produces the anchors and the before/after pairs those two features need. A
  hand-write counts against acceptance; a polish counts for it. Neither existed here, so
  the acceptance rates are not measuring quite the same thing.
- **The probe fixture.** 16 points chosen by argmax on discrimination; now 24 chosen by a
  floor plus a stratified draw, because argmax concentrates on near-ties and maximises the
  variance of every reading. The old fixture also had zero within-point spread on the five
  features the shrinkage flag touches, which made it provably unable to answer the question
  it was later asked.

`arush/logs/final_pass.md` has the full account.
