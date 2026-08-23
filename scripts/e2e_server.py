#!/usr/bin/env python
"""Boot the API with a canned provider, for the end-to-end smoke test.

Uses a mock provider rather than the real one so the smoke test needs no network, no keys,
and no quota — it is testing the stack, not the model. Everything else is the production
path: a real socket, the real app factory, and the built frontend served from disk.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import uvicorn

from storygit.api.app import create_app
from storygit.api.deps import build_state
from storygit.domain.ids import IdGenerator
from storygit.engine import Engine
from storygit.preference.layer import PreferenceLayer
from storygit.providers.base import LLMRequest, LLMResponse
from storygit.providers.router import Router
from storygit.selection.select import SelectionConfig, Selector

BEAT = {
    "title": "The offer",
    "what_happens": "The Warden offers Kael a way out of the city.",
    "audience_learns": "the Warden wants something from him",
    "audience_feels": "wary",
    "location": "Ashfall",
    "time": "night",
    "produces": [
        {
            "subject": "Kael",
            "predicate": "secret",
            "object": "he can hear the ash",
            "object_is_entity": False,
            "known_by": ["Kael"],
        }
    ],
    "consumes": [],
    "threads_touched": [],
    "new_characters": [],
    "rationale": "Gives Kael something to refuse.",
    "delta_summary": ["the Warden makes an offer"],
}

VARIANTS = (
    {
        "title": "The offer",
        "what_happens": "The Warden offers Kael a way out of the city, for a price.",
        "audience_learns": "the Warden wants something from him",
        "rationale": "Gives Kael something he can refuse, which is a decision.",
        "delta_summary": ["the Warden makes an offer", "opens: what the Warden wants"],
    },
    {
        "title": "What the ash says",
        "what_happens": "Kael hears the ash name a street he has never been to.",
        "audience_learns": "his ability is specific, and it is not his to control",
        "rationale": "Turns the ability from a power into a problem.",
        "delta_summary": ["Kael's ability shows itself", "establishes: the ash names places"],
    },
    {
        "title": "The guard who knows",
        "what_happens": "A guard calls Kael by a name his mother used.",
        "audience_learns": "somebody in the city has been watching him",
        "rationale": "Raises the stakes without spending the Warden.",
        "delta_summary": ["a guard recognises Kael", "opens: who has been watching"],
    },
)
"""Three distinct beats, so a screenshot shows the tool choosing rather than repeating.

The canned provider is not trying to be a model. It is trying to make the interface render
the thing it renders in real use: options that differ, summaries of different lengths, and
scores that are not all the same number. A screenshot of three identical cards proves the
layout and nothing about the layout under real content.
"""

JUDGE_SCORES = ((4.5, 4.0, 4.0, 3.5), (3.5, 4.5, 3.0, 4.0), (3.0, 3.0, 4.5, 3.5))
"""Per-variant scores on momentum, specificity, consequence, voice."""

TITLES = {
    "propose.episode": ("Ashfall", "The long night", "What the ash knows"),
    "propose.scene": ("The market", "Under the wall", "The Warden's room"),
    "propose.beat": ("The offer", "What the ash says", "The guard who knows"),
}
"""Titles per level, so the plan tree does not read as the same node four times deep."""

PROSE = (
    "Ash came down over the market. Kael counted three guards and kept walking.",
    "The Warden did not look up. He let the silence do the asking, and Kael let it.",
    "Somewhere under the wall a bell went, twice, and every head in the street turned.",
)
"""Three drafts. They differ in length and in rhythm so the interface has something to
show: word counts that are not identical, and a dial distance that is not zero."""


def _variant(request: LLMRequest, count: int) -> int:
    """Pick a variant from the request itself, not from ``sample_index`` alone.

    Axis-conditioned sampling sends one call per axis, each with its own prompt and each
    with ``sample_index`` 0, so keying only on the index would hand every axis the same
    answer — three identical candidates, which is exactly what a selector is supposed to
    avoid and exactly the bug a screenshot of three identical cards hides.
    """
    prompt = "".join(message.content for message in request.messages)
    digest = hashlib.sha256(prompt.encode()).digest()
    return (request.sample_index + digest[0]) % count


class CannedProvider:
    """Returns a valid, varying beat for every request. No network, no keys."""

    name = "gemini"
    model = "canned"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Serve a canned response shaped for whichever level was asked for."""
        if request.purpose.startswith("extract"):
            payload: dict[str, object] = {
                "facts": [],
                "new_characters": [],
                "threads_opened": [],
                "threads_touched": [],
            }
        elif request.purpose.startswith("judge"):
            index = _variant(request, len(JUDGE_SCORES))
            names = ("momentum", "specificity", "consequence", "voice")
            payload = {
                "scores": [
                    {"name": name, "argument": f"{name} reads as intended here.", "score": score}
                    for name, score in zip(names, JUDGE_SCORES[index], strict=True)
                ],
                "motivation_concern": "",
                "tone_concern": "",
            }
        elif request.purpose.startswith("propose.prose"):
            index = _variant(request, len(PROSE))
            payload = {
                "text": PROSE[index],
                "rationale": ("short sentences", "let the silence carry it", "widen the frame")[
                    index
                ],
                "delta_summary": ["writes the beat"],
            }
        else:
            index = _variant(request, len(VARIANTS))
            payload = dict(BEAT, **VARIANTS[index])
            titles = TITLES.get(request.purpose)
            if titles:
                payload["title"] = titles[index]
        return LLMResponse(
            text=json.dumps(payload),
            model=self.model,
            provider=self.name,
            prompt_tokens=10,
            completion_tokens=10,
        )

    async def aclose(self) -> None:
        """Nothing to close."""


def main() -> None:
    """Boot the app on the configured port."""
    provider = CannedProvider()
    state = build_state(
        os.environ.get("STORYGIT_DB", "e2e.db"),
        router=Router({"gemini": provider, "groq": provider}),
        results_dir=Path("eval/results"),
        seed=99,
    )
    # The smoke test wants the stack, fast and deterministic; the screenshot audit wants
    # the interface under content that varies the way real content does — different scores
    # per card, a surprise number that is not zero, and a selector that actually chooses.
    shot = os.environ.get("STORYGIT_SHOT") == "1"
    selection = (
        SelectionConfig(n=3, k=3, selector=Selector.mmr, use_judge=True, use_dial=True)
        if shot
        else SelectionConfig(
            n=3, k=3, selector=Selector.topk_temperature, use_judge=False, use_dial=False
        )
    )
    state.engines["main"] = Engine(
        state.repo,
        state.router,
        ids=IdGenerator(seed=99, stream="e2e"),
        selection=selection,
        use_nli=False,
        preference=PreferenceLayer(enabled=shot),
    )
    uvicorn.run(
        create_app(state=state, frontend_dir="frontend/dist"),
        host="127.0.0.1",
        port=int(os.environ.get("STORYGIT_PORT", "8123")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
