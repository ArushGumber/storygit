#!/usr/bin/env python
"""Boot the API with a canned provider, for the end-to-end smoke test.

Uses a mock provider rather than the real one so the smoke test needs no network, no keys,
and no quota — it is testing the stack, not the model. Everything else is the production
path: a real socket, the real app factory, and the built frontend served from disk.
"""

from __future__ import annotations

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


class CannedProvider:
    """Returns the same valid beat for every request. No network, no keys."""

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
        elif request.purpose.startswith("propose.prose"):
            payload = {
                "text": "Ash came down over the market. Kael counted three guards.",
                "rationale": "short sentences",
                "delta_summary": ["writes the beat"],
            }
        else:
            payload = dict(BEAT, title=f"Option {request.sample_index + 1}")
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
    state.engines["main"] = Engine(
        state.repo,
        state.router,
        ids=IdGenerator(seed=99, stream="e2e"),
        selection=SelectionConfig(
            n=3, k=3, selector=Selector.topk_temperature, use_judge=False, use_dial=False
        ),
        use_nli=False,
        preference=PreferenceLayer(enabled=False),
    )
    uvicorn.run(
        create_app(state=state, frontend_dir="frontend/dist"),
        host="127.0.0.1",
        port=int(os.environ.get("STORYGIT_PORT", "8123")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
