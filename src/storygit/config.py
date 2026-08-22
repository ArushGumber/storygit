"""Settings, loaded from the workspace-level ``.env`` and never from anywhere else.

The keys live one level above the repository so they cannot be committed by accident.
Nothing in this module is ever logged: the call logger records a *key index*, never a
key, and :meth:`Settings.redact` exists so that any accidental interpolation of a secret
into an error message can be scrubbed before it is written down.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_env_file() -> Path:
    """Locate the workspace ``.env``.

    Walks up from this file looking for a ``.env`` outside the repository, so the same
    code works from the repo root, from ``eval/``, and from a test run.

    Returns:
        The path to use; may not exist, in which case settings fall back to the
        process environment.
    """
    override = os.environ.get("STORYGIT_ENV_FILE")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return here.parents[2] / ".env"


class Settings(BaseSettings):
    """Everything configurable, with the defaults the project actually runs on.

    Attributes:
        groq_api_key: Groq key, used for extraction and classification.
        groq_model: Groq model id.
        gemini_model: Primary Gemini model.
        gemini_fallback_models: Comma-separated fallbacks, tried in order when every
            key is cooling on the primary model.
        gemini_keys: The six free-tier keys, rotated round-robin.
        openrouter_api_key: Present but unusable until ``openrouter_enabled`` is
            exactly ``"true"``.
        openrouter_enabled: The lock. Anything but the exact string ``"true"`` means
            every OpenRouter call raises before a request is built.
        openrouter_budget_usd: In-code hard cap, independent of the account cap.
        cache_path: Disk cache database. Lives outside the repo.
        request_timeout_s: Per-request timeout.
        max_retries: Attempts per provider call before giving up on a key.
    """

    model_config = SettingsConfigDict(
        env_file=_default_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.1-8b-instant"

    gemini_model: str = "gemini-2.0-flash"
    gemini_fallback_models: str = ""
    gemini_key_1: SecretStr = SecretStr("")
    gemini_key_2: SecretStr = SecretStr("")
    gemini_key_3: SecretStr = SecretStr("")
    gemini_key_4: SecretStr = SecretStr("")
    gemini_key_5: SecretStr = SecretStr("")
    gemini_key_6: SecretStr = SecretStr("")

    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_enabled: str = "false"
    openrouter_budget_usd: float = 12.0
    openrouter_model: str = "anthropic/claude-3.5-sonnet"

    cache_path: str = Field(default="")
    request_timeout_s: float = 90.0
    max_retries: int = 2

    @property
    def gemini_keys(self) -> list[str]:
        """The configured Gemini keys, in rotation order, skipping blanks."""
        raw = [
            self.gemini_key_1,
            self.gemini_key_2,
            self.gemini_key_3,
            self.gemini_key_4,
            self.gemini_key_5,
            self.gemini_key_6,
        ]
        return [key.get_secret_value() for key in raw if key.get_secret_value()]

    @property
    def gemini_fallbacks(self) -> list[str]:
        """Fallback models, in the order they should be tried."""
        return [m.strip() for m in self.gemini_fallback_models.split(",") if m.strip()]

    @property
    def openrouter_is_enabled(self) -> bool:
        """The lock, evaluated strictly.

        Only the exact lowercase string ``"true"`` unlocks OpenRouter. ``"True"``,
        ``"1"``, ``"yes"`` and everything else keep it locked, deliberately: a lock that
        can be tripped by a typo is not a lock.
        """
        return self.openrouter_enabled == "true"

    @property
    def resolved_cache_path(self) -> Path:
        """Where the LLM disk cache lives.

        Defaults to ``<workspace>/.storygit_cache/llmcache.db`` — outside the repository,
        so a cache full of model output is never committed.
        """
        if self.cache_path:
            return Path(self.cache_path)
        return _default_env_file().parent / ".storygit_cache" / "llmcache.db"

    def secrets(self) -> list[str]:
        """Every secret value currently loaded, for scrubbing log lines."""
        values = [
            self.groq_api_key.get_secret_value(),
            self.openrouter_api_key.get_secret_value(),
            *self.gemini_keys,
        ]
        return [v for v in values if v]

    def redact(self, text: str) -> str:
        """Replace any loaded secret appearing in ``text`` with a placeholder.

        Args:
            text: A message that may have interpolated a key (an httpx error carrying a
                URL with a query parameter, for instance).

        Returns:
            The message with every known secret replaced by ``<redacted>``.
        """
        for secret in self.secrets():
            if secret and secret in text:
                text = text.replace(secret, "<redacted>")
        return text


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process.

    Returns:
        The cached :class:`Settings`.
    """
    return Settings()
