"""LLM client supporting Anthropic (Claude) and Google Gemini, with friendly error handling."""
from typing import Iterator

from . import config


class LLMError(Exception):
    """Raised for any LLM/API failure, with a user-presentable message."""


def normalize_messages(messages: list[dict]) -> list[dict]:
    """APIs require alternating roles starting with 'user'. Merge repeats, drop leading assistant turns."""
    out: list[dict] = []
    for m in messages:
        if not m["content"].strip():
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


class LLMClient:
    def __init__(self, model: str = None, provider: str = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL
        self._client = None

    # ------------------------------------------------------------ public API
    def complete(self, system: str, messages: list[dict], max_tokens: int = 500,
                 temperature: float = 0.0) -> str:
        try:
            if self.provider == "gemini":
                return self._gemini_complete(system, messages, max_tokens, temperature)
            return self._anthropic_complete(system, messages, max_tokens, temperature)
        except LLMError:
            raise
        except Exception as e:
            raise self._friendly(e) from e

    def stream(self, system: str, messages: list[dict], max_tokens: int = None,
               temperature: float = None) -> Iterator[str]:
        max_tokens = max_tokens or config.MAX_RESPONSE_TOKENS
        temperature = config.RESPONSE_TEMPERATURE if temperature is None else temperature
        try:
            if self.provider == "gemini":
                yield from self._gemini_stream(system, messages, max_tokens, temperature)
            else:
                yield from self._anthropic_stream(system, messages, max_tokens, temperature)
        except LLMError:
            raise
        except Exception as e:
            raise self._friendly(e) from e

    # ------------------------------------------------------------ Anthropic
    def _anthropic(self):
        if self._client is None:
            if not config.ANTHROPIC_API_KEY:
                raise LLMError(config.missing_key_message())
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=2, timeout=60)
        return self._client

    def _anthropic_complete(self, system, messages, max_tokens, temperature) -> str:
        resp = self._anthropic().messages.create(
            model=self.model, system=system, messages=normalize_messages(messages),
            max_tokens=max_tokens, temperature=temperature,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _anthropic_stream(self, system, messages, max_tokens, temperature) -> Iterator[str]:
        with self._anthropic().messages.stream(
            model=self.model, system=system, messages=normalize_messages(messages),
            max_tokens=max_tokens, temperature=temperature,
        ) as s:
            yield from s.text_stream

    # ------------------------------------------------------------ Gemini
    def _gemini(self):
        if self._client is None:
            if not config.GEMINI_API_KEY:
                raise LLMError(config.missing_key_message())
            from google import genai

            self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        return self._client

    @staticmethod
    def _gemini_request(system, messages, max_tokens, temperature):
        from google.genai import types

        contents = [
            types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])])
            for m in normalize_messages(messages)
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens, temperature=temperature,
        )
        return contents, cfg

    def _gemini_complete(self, system, messages, max_tokens, temperature) -> str:
        contents, cfg = self._gemini_request(system, messages, max_tokens, temperature)
        resp = self._gemini().models.generate_content(model=self.model, contents=contents, config=cfg)
        return resp.text or ""

    def _gemini_stream(self, system, messages, max_tokens, temperature) -> Iterator[str]:
        contents, cfg = self._gemini_request(system, messages, max_tokens, temperature)
        for chunk in self._gemini().models.generate_content_stream(
            model=self.model, contents=contents, config=cfg
        ):
            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------ errors
    @staticmethod
    def _friendly(e: Exception) -> LLMError:
        # Gemini (google.genai.errors.APIError exposes .code and .message)
        code = getattr(e, "code", None)
        if type(e).__module__.startswith("google.genai") and isinstance(code, int):
            msg = str(getattr(e, "message", "") or e)
            if code in (401, 403) or (code == 400 and "api key" in msg.lower()):
                return LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY.")
            if code == 429:
                return LLMError("Gemini rate limit or quota reached. Please wait a moment and try again.")
            if code == 404:
                # return LLMError(f"Gemini model '{config.LLM_MODEL}' was not found. Check LLM_MODEL.")
                return LLMError(f"Gemini model '{msg}")

            return LLMError(f"The Gemini API returned an error ({code}). Please try again.")

        # Anthropic
        try:
            import anthropic

            if isinstance(e, anthropic.AuthenticationError):
                return LLMError("The API key was rejected. Check ANTHROPIC_API_KEY.")
            if isinstance(e, anthropic.RateLimitError):
                return LLMError("Rate limit reached. Please wait a moment and try again.")
            if isinstance(e, anthropic.APIConnectionError):
                return LLMError("Could not reach the LLM API. Check your internet connection.")
            if isinstance(e, anthropic.APIStatusError):
                return LLMError(f"The LLM API returned an error ({e.status_code}). Please try again.")
        except ImportError:
            pass

        if type(e).__module__.startswith("httpx") or isinstance(e, (ConnectionError, TimeoutError)):
            return LLMError("Could not reach the LLM API. Check your internet connection.")
        return LLMError(f"Unexpected LLM error: {e}")
