"""OpenAI-compatible Groq client with multi-key rotation and rate-limit failover.

Keys are read from the environment (GROQ_API_KEYS) or a git-ignored local file
(keys.local.txt). They are NEVER hard-coded and NEVER committed.
Free keys (no credit card): https://console.groq.com
"""
from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

from openai import APIError, OpenAI, RateLimitError

from core.config import GROQ_MODEL, GROQ_STT_MODEL

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _load_keys() -> list[str]:
    raw = os.getenv("GROQ_API_KEYS", "")
    keys = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
    if not keys:
        local = Path("keys.local.txt")
        if local.exists():
            keys = [ln.strip() for ln in local.read_text().splitlines() if ln.strip()]
    if not keys:
        raise RuntimeError(
            "No API keys found. Copy .env.example to .env and set GROQ_API_KEYS, "
            "or create keys.local.txt (one key per line; git-ignored)."
        )
    seen, unique = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


class RotatingGroqClient:
    """Rotates across several free-tier keys, failing over on HTTP 429 / API errors."""

    def __init__(self, model: str = GROQ_MODEL, max_retries: int | None = None):
        self._keys = _load_keys()
        self._cycle = itertools.cycle(range(len(self._keys)))
        self._idx = next(self._cycle)
        self.model = model
        self.max_retries = max_retries or (len(self._keys) * 2)

    def _client(self) -> OpenAI:
        return OpenAI(api_key=self._keys[self._idx], base_url=GROQ_BASE_URL)

    def chat(self, messages: list[dict], temperature: float = 0.4, **kwargs) -> str:
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                resp = self._client().chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature, **kwargs
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APIError) as err:
                last_err = err
                self._idx = next(self._cycle)
                time.sleep(1)
        raise RuntimeError(f"All {len(self._keys)} keys exhausted or failing: {last_err}")

    def transcribe(self, audio_bytes: bytes, filename: str = "speech.wav",
                   model: str | None = None) -> str:
        """Speech-to-text via Groq Whisper (used for the voice modality)."""
        import io
        model = model or GROQ_STT_MODEL
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                bio = io.BytesIO(audio_bytes)
                bio.name = filename
                resp = self._client().audio.transcriptions.create(model=model, file=bio)
                return resp.text
            except (RateLimitError, APIError) as err:
                last_err = err
                self._idx = next(self._cycle)
                time.sleep(1)
        raise RuntimeError(f"Transcription failed on all keys: {last_err}")