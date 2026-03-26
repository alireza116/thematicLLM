"""
Unified LLM client that works with:
  - OpenAI (and any OpenAI-compatible API: Groq, Together, Mistral, Ollama, etc.)
  - Anthropic
  - Google Gemini (via google-genai)

API keys are loaded automatically from the .env file in the project root.
They can also be passed explicitly or set as environment variables.

Usage:
    # OpenAI
    client = LLMClient(provider="openai", model="gpt-4o")

    # Anthropic
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")

    # Gemini
    client = LLMClient(provider="gemini", model="gemini-3-flash-preview")

    # Any OpenAI-compatible endpoint (e.g. Groq, local Ollama)
    client = LLMClient(provider="openai", model="llama3-8b-8192",
                       api_key="gsk_...", base_url="https://api.groq.com/openai/v1")

    response = client.complete([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations
import json
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file).
# Does nothing if the file doesn't exist, so environment variables already
# set in the shell take precedence over .env values.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ---------------------------------------------------------------------------
# Rate limiter — sliding-window RPM + TPM enforcement
# ---------------------------------------------------------------------------

class _RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Tracks request timestamps in a rolling 60-second window and blocks
    (with a short sleep) until making a new call would stay within limits.
    """

    def __init__(self, rpm: int, tpm: Optional[int] = None):
        self.rpm = rpm
        self.tpm = tpm
        self._request_times: deque = deque()
        self._token_counts: deque = deque()   # (timestamp, n_tokens)
        self._lock = threading.Lock()

    def wait(self, estimated_tokens: int = 0) -> None:
        with self._lock:
            now = time.monotonic()
            window = 60.0

            # Evict entries older than the window
            while self._request_times and now - self._request_times[0] >= window:
                self._request_times.popleft()
            while self._token_counts and now - self._token_counts[0][0] >= window:
                self._token_counts.popleft()

            # Wait until RPM headroom exists
            while len(self._request_times) >= self.rpm:
                oldest = self._request_times[0]
                sleep_for = window - (now - oldest) + 0.05
                time.sleep(max(sleep_for, 0.05))
                now = time.monotonic()
                while self._request_times and now - self._request_times[0] >= window:
                    self._request_times.popleft()

            # Wait until TPM headroom exists (if configured)
            if self.tpm and estimated_tokens:
                current_tokens = sum(t for _, t in self._token_counts)
                while current_tokens + estimated_tokens > self.tpm:
                    oldest_ts = self._token_counts[0][0]
                    sleep_for = window - (now - oldest_ts) + 0.05
                    time.sleep(max(sleep_for, 0.05))
                    now = time.monotonic()
                    while self._token_counts and now - self._token_counts[0][0] >= window:
                        self._token_counts.popleft()
                    current_tokens = sum(t for _, t in self._token_counts)

            # Record this request
            self._request_times.append(now)
            if estimated_tokens:
                self._token_counts.append((now, estimated_tokens))


# Default RPM limits per provider (conservative free-tier values)
_DEFAULT_RPM = {
    "gemini":    15,    # Gemini free tier: 15 RPM
    "openai":   500,    # OpenAI tier-1: 500 RPM
    "anthropic":  50,   # Anthropic tier-1: 50 RPM
}

# Default TPM limits per provider (set to None to disable)
_DEFAULT_TPM = {
    "gemini":    1_000_000,   # Gemini free tier: 1M TPM
    "openai":    None,
    "anthropic": None,
}


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES = 6
_BASE_BACKOFF = 2.0   # seconds; doubles each retry, plus jitter


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True for any provider's rate-limit / quota error."""
    name = type(exc).__name__
    msg  = str(exc).lower()
    if "ratelimit" in name.lower():
        return True
    if any(k in msg for k in ("rate limit", "rate_limit", "429", "quota", "resource exhausted",
                               "too many requests", "overloaded")):
        return True
    return False


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for transient server errors worth retrying."""
    msg = str(exc).lower()
    return _is_rate_limit_error(exc) or any(
        k in msg for k in ("500", "502", "503", "504", "timeout", "connection",
                            "server error", "internal error", "unavailable")
    )


def _retry_delay(attempt: int, exc: Exception) -> float:
    """
    Compute wait time for retry attempt (0-indexed).
    Reads Retry-After header when available, otherwise exponential backoff + jitter.
    """
    # Some SDKs expose the retry-after value on the exception
    for attr in ("retry_after", "retry_after_ms"):
        val = getattr(exc, attr, None)
        if val is not None:
            return float(val) / 1000 if attr.endswith("_ms") else float(val)

    backoff = _BASE_BACKOFF * (2 ** attempt)
    jitter  = random.uniform(0, backoff * 0.2)
    return min(backoff + jitter, 120.0)   # cap at 2 minutes


class LLMClient:
    SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini")

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None,
        max_retries: int = _MAX_RETRIES,
        **kwargs,
    ):
        """
        Args:
            rpm:         Requests per minute limit. Defaults to a safe value per
                         provider (gemini=15, openai=500, anthropic=50).
                         Set to 0 to disable rate limiting.
            tpm:         Tokens per minute limit (optional, provider-specific default).
            max_retries: Max retry attempts on rate-limit / transient errors.
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose from {self.SUPPORTED_PROVIDERS}."
            )
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = self._build_client(provider, api_key, base_url, **kwargs)
        self._api_key = api_key

        # Rate limiter — use provided values or provider defaults
        effective_rpm = rpm if rpm is not None else _DEFAULT_RPM.get(provider, 60)
        effective_tpm = tpm if tpm is not None else _DEFAULT_TPM.get(provider)
        if effective_rpm > 0:
            self._rate_limiter: Optional[_RateLimiter] = _RateLimiter(
                rpm=effective_rpm, tpm=effective_tpm
            )
        else:
            self._rate_limiter = None

    # ------------------------------------------------------------------

    def _build_client(self, provider, api_key, base_url, **kwargs):
        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Install openai: pip install openai")
            return OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url,
                **kwargs,
            )
        elif provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
            return anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
                **kwargs,
            )
        elif provider == "gemini":
            try:
                from google import genai
            except ImportError:
                raise ImportError("Install google-genai: pip install google-genai")
            return genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        json_mode: bool = False,
        system: Optional[str] = None,
    ) -> str:
        """
        Call the LLM and return the text response.

        Automatically:
          - Enforces the RPM/TPM rate limit (sleeps if needed).
          - Retries on rate-limit and transient server errors with
            exponential backoff + jitter (up to max_retries attempts).

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."} dicts.
            json_mode: Request JSON output (provider-specific enforcement).
            system: System prompt (used differently per provider).

        Returns:
            String response from the model.
        """
        # Rough token estimate for TPM tracking (input only; good enough for throttling)
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        if system:
            estimated_tokens += len(system) // 4

        for attempt in range(self.max_retries + 1):
            # Wait for rate-limit headroom before each attempt
            if self._rate_limiter:
                self._rate_limiter.wait(estimated_tokens)

            try:
                if self.provider == "openai":
                    return self._complete_openai(messages, json_mode, system)
                elif self.provider == "anthropic":
                    return self._complete_anthropic(messages, json_mode, system)
                elif self.provider == "gemini":
                    return self._complete_gemini(messages, json_mode, system)

            except Exception as exc:
                if attempt < self.max_retries and _is_retryable_error(exc):
                    delay = _retry_delay(attempt, exc)
                    kind  = "rate limit" if _is_rate_limit_error(exc) else "server error"
                    print(
                        f"  [LLMClient] {kind} on attempt {attempt + 1}/{self.max_retries} "
                        f"— retrying in {delay:.1f}s … ({type(exc).__name__})"
                    )
                    time.sleep(delay)
                else:
                    raise

    def _complete_openai(self, messages, json_mode, system):
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        kwargs = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _complete_anthropic(self, messages, json_mode, system):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system
        elif json_mode:
            kwargs["system"] = (
                "You must respond with valid JSON only. Do not include any text "
                "outside the JSON object."
            )

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _complete_gemini(self, messages, json_mode, system):
        from google.genai import types

        system_instruction = None
        if system:
            system_instruction = system
        elif json_mode:
            system_instruction = (
                "You must respond with valid JSON only. "
                "Do not include any text outside the JSON object."
            )

        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json" if json_mode else None,
        )

        # Convert shared {"role", "content"} format to Gemini Content objects.
        # Gemini uses "model" instead of "assistant".
        contents = [
            types.Content(
                role="model" if msg["role"] == "assistant" else msg["role"],
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in messages
        ]

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text

    # ------------------------------------------------------------------

    def complete_json(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> dict:
        """Call the LLM and parse the response as JSON.

        Tries three strategies in order:
        1. Direct parse of the raw response.
        2. Strip markdown code fences then parse.
        3. Extract the first {...} or [...] block and parse it.
        """
        text = self.complete(messages, json_mode=True, system=system)
        text = text.strip()

        # Strategy 1: direct
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip code fences
        cleaned = text
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: extract first JSON object or array
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = cleaned.find(start_char)
            end = cleaned.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass

        raise json.JSONDecodeError(
            f"Could not extract JSON from model response: {text[:200]}", text, 0
        )
