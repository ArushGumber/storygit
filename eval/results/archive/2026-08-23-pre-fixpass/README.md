# 2026-08-23, pre-fix-pass

The four-persona live run made **before** the fix pass, kept because deleting a result you
have already reported is not a thing to do.

It is not comparable with the current numbers, and the paper does not compare them. Between
this run and the current one the *system* changed, not the reporting:

- **The feature space.** The writer's criteria collapsed into one averaged score; they now
  get four separate slots. Weight recovery here is measured in a 10-feature space and
  currently in a 13-feature one, so the two numbers are not the same quantity.
- **The personas define criteria.** In this run they defined none, which means the
  collapsed `writer_criteria` feature was the constant 0.5 throughout and the two personas
  carrying weight on it were weighting a constant.
- **A rejected set now gets one informed retry.** Here, one unlucky candidate set ended a
  run: two of the four stopped at episode 3, one of them on a margin of 0.0001.
- **Rejection reasons reach the prompt.** They did not in this run.
- **Per-persona cost figures were wrong.** One call log served all four personas and each
  snapshotted the running total, so `tokens_per_action` here is inflated by up to 4x for
  every persona after the first.

`arush/logs/fix_pass.md` has the autopsy that led to each of those.
