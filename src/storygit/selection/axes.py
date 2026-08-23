"""The seven conditioning axes, as data.

Fabula sampled several candidates and kept them apart by penalizing similarity to previous
suggestions. That produces variants that differ, but not variants a writer can *choose
between* — the difference is not nameable, so reviewing three of them costs three times as
much as reviewing one, and writers reported it was cheaper to write from scratch.

Here each candidate is generated under a named instruction, and the name travels with it
all the way to the screen. The writer reads "raise the stakes / slow down / subvert the
expectation" and picks a direction rather than adjudicating three paragraphs. Diversity
stops being a property of the embedding space and becomes a property of the interface.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Axis(BaseModel):
    """One named direction a candidate can be generated along.

    Attributes:
        key: Stable identifier, stored on the proposal and in the signal log.
        label: What the writer sees on the candidate.
        fragment: The instruction appended to the prompt for this candidate.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    fragment: str


AXES: tuple[Axis, ...] = (
    Axis(
        key="raise_stakes",
        label="raise the stakes",
        fragment=(
            "Raise the stakes. Make the cost of failure concrete and personal, and make it "
            "land in this beat rather than being promised for later."
        ),
    ),
    Axis(
        key="slow_down",
        label="slow down",
        fragment=(
            "Slow down. Take the moment the story just skipped and stay in it. Less plot, "
            "more of what it is actually like to be there."
        ),
    ),
    Axis(
        key="subvert",
        label="subvert the expectation",
        fragment=(
            "Subvert the obvious next move. Work out what the audience is bracing for, then "
            "do the thing that is more interesting and still earned by what came before."
        ),
    ),
    Axis(
        key="shift_pov",
        label="shift the point of view",
        fragment=(
            "Shift the point of view to another character present. Same events, someone "
            "else's read on them, and let that reveal something the first view could not."
        ),
    ),
    Axis(
        key="interiority",
        label="deepen the interiority",
        fragment=(
            "Go inward. What the character wants here versus what they will admit to "
            "wanting. Keep it in behaviour and speech, not in stated feeling."
        ),
    ),
    Axis(
        key="complication",
        label="introduce a complication",
        fragment=(
            "Introduce a complication that makes the current plan impossible as stated. "
            "It must come from something already established, not from nowhere."
        ),
    ),
    Axis(
        key="pay_off_thread",
        label="pay off an open thread",
        fragment=(
            "Pay off one of the open threads listed above, or move it decisively. Name "
            "which one you are advancing and what it now costs."
        ),
    ),
)
"""The seven axes from the design. Order is stable; rotation is deterministic."""

AXES_BY_KEY: dict[str, Axis] = {axis.key: axis for axis in AXES}


def rotate(n: int, *, offset: int = 0) -> tuple[Axis, ...]:
    """Pick ``n`` axes, rotating deterministically.

    Rotating by a per-node offset rather than sampling randomly means that asking for
    candidates twice at the same node gives the same axes (so the cache works and runs
    reproduce), while different nodes across a session still see every axis.

    Args:
        n: How many axes to return. Values above the number of axes wrap.
        offset: Where to start in the ring; callers derive it from the node id.

    Returns:
        ``n`` axes.
    """
    if n <= 0:
        return ()
    start = offset % len(AXES)
    return tuple(AXES[(start + i) % len(AXES)] for i in range(n))


def offset_for(node_id: str) -> int:
    """A stable rotation offset for a node.

    Args:
        node_id: The node candidates are being generated for.

    Returns:
        A deterministic offset in ``[0, len(AXES))``.
    """
    return sum(ord(c) for c in node_id) % len(AXES)
