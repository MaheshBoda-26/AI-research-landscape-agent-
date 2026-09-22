"""The LLM client: one interface, two OpenAI-compatible backends.

NVIDIA NIM and OpenRouter are both reachable through the OpenAI SDK by changing
``base_url``, but they do **not** agree on how structured output is requested:

* NIM's documented path is ``extra_body={"nvext": {"guided_json": <schema>}}``.
  It does not use OpenAI's ``response_format``.
* OpenRouter uses ``response_format={"type": "json_schema", ...}`` and needs
  ``provider.require_parameters`` set, otherwise it will happily route to a
  provider that ignores the schema entirely.

Rather than trust either mechanism, every response is validated against the
Pydantic schema and a bounded repair loop is run on failure. That is not
belt-and-braces: constrained decoding can still emit JSON that is *syntactically*
fine and *semantically* wrong, and OpenRouter's schema support is known to be
inconsistent across providers.

``complete_json`` never raises for a bad model response. It returns ``None`` and
lets the caller decide whether that is fatal. Extraction degrades per paper;
cluster labeling falls back to a placeholder; neither should take down a run.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from config import Settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: Response text sometimes arrives inside a markdown fence despite instructions.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

#: HTTP statuses worth retrying: rate limits and transient upstream failures.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, ignoring braces inside strings.

    A plain ``text[text.find('{'):text.rfind('}')+1]`` breaks as soon as the
    payload contains a brace inside a string, which extracted abstracts routinely
    do (inline JSON snippets, LaTeX, code fragments).
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def candidate_json_payloads(raw: str) -> list[str]:
    """Extraction candidates for a model response, most-likely first."""
    if not raw:
        return []
    candidates: list[str] = []
    stripped = raw.strip()
    candidates.append(stripped)
    for match in _FENCE_RE.findall(raw):
        candidates.append(match.strip())
    block = _first_json_object(raw)
    if block:
        candidates.append(block)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def parse_into(raw: str, schema: type[SchemaT]) -> SchemaT:
    """Parse a raw model response into ``schema``, or raise ``ValidationError``.

    Raises the *last* validation error so the repair loop can feed a useful
    message back to the model.
    """
    payloads = candidate_json_payloads(raw)
    if not payloads:
        raise ValidationError.from_exception_data(
            schema.__name__,
            [{"type": "value_error", "loc": (), "input": raw, "ctx": {"error": "empty response"}}],
        )
    last_error: Exception | None = None
    for payload in payloads:
        try:
            return schema.model_validate_json(payload)
        except ValidationError as exc:
            last_error = exc
        except (ValueError, TypeError) as exc:
            last_error = exc
    if isinstance(last_error, ValidationError):
        raise last_error
    raise ValidationError.from_exception_data(
        schema.__name__,
        [
            {
                "type": "value_error",
                "loc": (),
                "input": raw,
                "ctx": {"error": str(last_error or "unparseable response")},
            }
        ],
    )


class LLMError(RuntimeError):
    """Raised for transport-level failures the caller may want to see."""


class LLMClient:
    """Synchronous, thread-safe structured-output client.

    Synchronous because the pipeline is: the arXiv client is blocking and model
    inference is blocking, so the whole run happens in a worker thread and
    concurrency comes from a thread pool plus the semaphore here.
    """

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            if not settings.llm_api_key:
                raise LLMError(
                    "No API key configured for LLM_PROVIDER="
                    f"{settings.llm_provider}. Set the matching key in .env."
                )
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=float(settings.llm_timeout_seconds),
                max_retries=0,  # retries are handled here so they can be logged
            )
        # Bounds concurrent requests to the provider.
        self._semaphore = threading.BoundedSemaphore(max(1, settings.llm_concurrency))
        self.calls = 0
        self.failures = 0

    # -- introspection ------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return self.settings.llm_model

    # -- request shaping ---------------------------------------------------- #

    def _response_format(self, schema: type[BaseModel]) -> dict[str, Any] | None:
        """Backend-specific structured-output request.

        ``strict`` is deliberately False for OpenRouter. Strict mode requires
        every property to be listed in ``required``, which would misrepresent
        genuinely optional extraction fields (a paper may have no stated
        limitations). Validation and repair cover the gap instead.
        """
        if self.settings.llm_provider == "openrouter":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": False,
                    "schema": schema.model_json_schema(),
                },
            }
        return None

    def _extra_body(self, schema: type[BaseModel]) -> dict[str, Any]:
        if self.settings.llm_provider == "nim":
            # vLLM's guided decoding. On NIM this is the documented path for
            # structured output; response_format is not.
            return {"nvext": {"guided_json": schema.model_json_schema()}}
        # Only route to providers that actually honour the schema.
        return {"provider": {"require_parameters": True}}

    # -- main entry point --------------------------------------------------- #

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        max_repairs: int | None = None,
        temperature: float = 0.0,
    ) -> SchemaT | None:
        """Return a validated ``schema`` instance, or ``None`` on failure."""
        repairs = self.settings.llm_max_repairs if max_repairs is None else max_repairs
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for attempt in range(repairs + 1):
            try:
                raw = self._chat(messages, schema=schema, temperature=temperature)
            except LLMError as exc:
                logger.warning("LLM call failed (%s); giving up on this request", exc)
                self.failures += 1
                return None

            if raw is None:
                self.failures += 1
                return None

            try:
                return parse_into(raw, schema)
            except ValidationError as exc:
                if attempt >= repairs:
                    logger.warning(
                        "Discarding response after %d repair attempts: %s",
                        repairs,
                        _brief(exc),
                    )
                    self.failures += 1
                    return None
                logger.info("Repairing response (attempt %d): %s", attempt + 1, _brief(exc))
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response did not match the required JSON schema.\n"
                            f"Validation error: {_brief(exc)}\n"
                            "Reply with corrected JSON only. No prose, no code fences."
                        ),
                    }
                )
        return None  # pragma: no cover - loop always returns

    # -- transport ---------------------------------------------------------- #

    def _chat(self, messages: list[dict[str, str]], *, schema: type[BaseModel], temperature: float) -> str | None:
        request: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        response_format = self._response_format(schema)
        if response_format is not None:
            request["response_format"] = response_format
        request["extra_body"] = self._extra_body(schema)

        last_error: Exception | None = None
        for attempt in range(4):
            with self._semaphore:
                try:
                    self.calls += 1
                    response = self._client.chat.completions.create(**request)
                except Exception as exc:  # noqa: BLE001 - transport varies by provider
                    status = getattr(exc, "status_code", None)
                    if status is not None and status not in _RETRYABLE_STATUS:
                        raise LLMError(f"HTTP {status}: {exc}") from exc
                    last_error = exc
                else:
                    try:
                        content = response.choices[0].message.content
                    except (AttributeError, IndexError, KeyError) as exc:
                        raise LLMError(f"Malformed completion shape: {exc}") from exc
                    return content or None

            # Exponential backoff with jitter, so parallel callers do not
            # synchronise their retries against a throttled provider.
            delay = min(30.0, (2**attempt) * 1.0) * (0.5 + random.random())
            logger.info("Retrying LLM call in %.1fs (attempt %d): %s", delay, attempt + 1, last_error)
            time.sleep(delay)

        raise LLMError(f"LLM request failed after retries: {last_error}")


def _brief(exc: Exception, limit: int = 400) -> str:
    text = json.dumps(exc.errors()[:3], default=str) if isinstance(exc, ValidationError) else str(exc)
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["LLMClient", "LLMError", "candidate_json_payloads", "parse_into"]
