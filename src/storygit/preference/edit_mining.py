"""Turning edits into style rules the writer can see and delete.

A writer who shortens every paragraph is telling the system something, but only in a form
it cannot act on. This module reads each before/after pair with a cheap model and turns it
into one to three plain-language rules — "shorter sentences", "less exposition", "more
sensory detail" — which go into the writer ledger and, once corroborated, into prompts.

Three constraints shape the design.

**Rules must be corroborated before they act.** One idiosyncratic edit is not a standing
instruction. A rule enters prompts only after it has been observed twice, which is what
``StyleNote.count`` and ``WriterLedger.active_style_notes`` implement.

**Rules must be deduplicated semantically.** "shorter sentences" and "keep the sentences
short" are the same rule, and letting both accumulate would mean the ledger fills with
near-duplicates and the corroboration threshold never fires. Dedup is by embedding cosine.

**Rules must never be invisible.** Every mined rule lands in the ledger, is shown in the
interface, and can be deleted. A system that quietly rewrites its own prompts based on
inferred preferences is doing something the writer did not agree to.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.diff import AddStyleNote, Diff, DiffAuthor, Op
from storygit.domain.ledger import StyleNote, StyleNoteSource, WriterLedger
from storygit.preference.signals import EditPair
from storygit.providers.base import LLMRequest, Message, Role, SchemaParseError
from storygit.providers.router import Router

DEDUP_THRESHOLD = 0.82
"""Cosine above which two rule texts are treated as the same rule."""

MINED_THRESHOLD = 2
"""How many observations a mined rule needs before it may enter a prompt."""


class MinedRules(BaseModel):
    """What the model extracted from one edit."""

    model_config = ConfigDict(extra="ignore")

    rules: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "One to three short imperative rules describing the change, each under ten "
            "words. Describe the writer's preference, not the specific edit."
        ),
    )


def build_request(pair: EditPair, *, sample_index: int = 0) -> LLMRequest:
    """Build the mining call for one edit pair.

    Args:
        pair: The before/after pair.
        sample_index: Cache discriminator.

    Returns:
        The request, tagged ``classify.edit`` so it routes to the cheap model.
    """
    system = Message(
        role=Role.system,
        content=(
            "A writer edited a passage an AI drafted. Say what their edit reveals about "
            "how they want prose written, as one to three short imperative rules.\n\n"
            "Describe the preference, not the edit. 'shorter sentences' is a rule; "
            "'changed paragraph two' is not. If the edit reveals nothing general, return "
            "an empty list — inventing a rule to look useful is worse than saying nothing, "
            "because these become standing instructions."
        ),
    )
    user = Message(
        role=Role.user,
        content=(
            f"BEFORE (the AI wrote this)\n{pair.before}\n\n"
            f"AFTER (the writer changed it to this)\n{pair.after}\n\n"
            'Reply with JSON only: {"rules": ["..."]}'
        ),
    )
    return LLMRequest(
        messages=(system, user),
        purpose="classify.edit",
        temperature=0.0,
        max_tokens=300,
        json_schema=MinedRules.model_json_schema(),
        sample_index=sample_index,
    )


async def mine(router: Router, pairs: Sequence[EditPair]) -> list[str]:
    """Extract style rules from a batch of edit pairs.

    Args:
        router: Model access.
        pairs: The edits to read.

    Returns:
        Rule texts, in order, with duplicates left in — deduplication happens against the
        ledger in :func:`to_diff`, where the existing rules are known.
    """
    import asyncio

    from storygit.agents.parse import complete_structured

    if not pairs:
        return []

    async def one(index: int, pair: EditPair) -> list[str]:
        try:
            result, _ = await complete_structured(
                router, build_request(pair, sample_index=index), MinedRules
            )
        except SchemaParseError:
            return []
        return [r.strip() for r in result.rules if r.strip()]

    batches = await asyncio.gather(*(one(i, p) for i, p in enumerate(pairs)))
    return [rule for batch in batches for rule in batch]


def dedupe(rules: Sequence[str], *, threshold: float = DEDUP_THRESHOLD) -> list[str]:
    """Collapse rules that say the same thing.

    Args:
        rules: Candidate rule texts.
        threshold: Cosine above which two rules are the same.

    Returns:
        One representative per cluster, first occurrence kept.
    """
    cleaned = [r.strip() for r in rules if r.strip()]
    if len(cleaned) < 2:
        return cleaned
    from storygit.selection.embed import embed_texts, similarity

    matrix = similarity(embed_texts(cleaned))
    kept: list[int] = []
    for index in range(len(cleaned)):
        if all(float(matrix[index][other]) < threshold for other in kept):
            kept.append(index)
    return [cleaned[i] for i in kept]


def to_diff(
    ledger: WriterLedger,
    rules: Sequence[str],
    *,
    threshold: float = DEDUP_THRESHOLD,
) -> Diff:
    """Turn mined rules into ledger operations, merging with what is already there.

    A rule that matches an existing one bumps its count (which is what moves it towards
    the corroboration threshold); a genuinely new rule is added at count 1.

    Args:
        ledger: The current ledger.
        rules: Freshly mined rule texts.
        threshold: Cosine for matching against existing rules.

    Returns:
        A ``system``-authored diff of style-note operations.
    """
    fresh = dedupe(rules, threshold=threshold)
    if not fresh:
        return Diff(author=DiffAuthor.system, intent="mine style rules")

    existing = [note.text for note in ledger.style_notes]
    ops: list[Op] = []
    if existing:
        from storygit.selection.embed import embed_texts

        vectors = embed_texts([*existing, *fresh])
        existing_vectors = vectors[: len(existing)]
        for offset, rule in enumerate(fresh):
            similarities = existing_vectors @ vectors[len(existing) + offset]
            best = int(similarities.argmax())
            if float(similarities[best]) >= threshold:
                # Same rule, said differently: bump the existing one's count instead of
                # adding a near-duplicate the writer would have to read twice.
                ops.append(
                    AddStyleNote(
                        note=StyleNote(text=existing[best], source=StyleNoteSource.mined, count=1)
                    )
                )
            else:
                ops.append(AddStyleNote(note=StyleNote(text=rule, source=StyleNoteSource.mined)))
    else:
        ops.extend(
            AddStyleNote(note=StyleNote(text=rule, source=StyleNoteSource.mined)) for rule in fresh
        )

    return Diff(
        ops=tuple(ops),
        author=DiffAuthor.system,
        intent=f"mine {len(ops)} style rule(s) from the writer's edits",
    )
