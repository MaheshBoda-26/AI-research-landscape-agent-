"""Tests for the LLM client.

No network. The OpenAI SDK object is injected, so these tests assert the exact
request that would be sent for each backend — which matters because NIM and
OpenRouter do not accept the same structured-output parameters, and getting that
wrong silently degrades every extraction to a parse failure.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field, ValidationError

from config import Settings
from llm.client import LLMClient, candidate_json_payloads, parse_into


class Verdict(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=10.0)


VALID = '{"label": "relevant", "score": 8.5}'


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("no scripted response left")
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def fake_openai(responses) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(responses)))


def status_error(code: int) -> Exception:
    exc = RuntimeError(f"provider said {code}")
    exc.status_code = code  # type: ignore[attr-defined]
    return exc


@pytest.fixture
def nim_settings() -> Settings:
    return replace(
        Settings.from_env(), llm_provider="nim", llm_model="test/model", llm_max_repairs=2
    )


@pytest.fixture
def openrouter_settings() -> Settings:
    return replace(
        Settings.from_env(),
        llm_provider="openrouter",
        llm_model="test/model",
        llm_max_repairs=2,
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retry backoff must not make the suite slow."""
    monkeypatch.setattr("llm.client.time.sleep", lambda _seconds: None)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def test_plain_json_parses():
    assert parse_into(VALID, Verdict).label == "relevant"


def test_fenced_json_parses():
    assert parse_into(f"```json\n{VALID}\n```", Verdict).score == 8.5


def test_bare_fence_parses():
    assert parse_into(f"```\n{VALID}\n```", Verdict).label == "relevant"


def test_prose_wrapped_json_parses():
    assert parse_into(f"Sure! Here you go:\n{VALID}\nHope that helps.", Verdict).label == "relevant"


def test_json_containing_braces_inside_a_string_parses():
    """Abstracts routinely contain braces; naive slicing breaks on them."""
    raw = '{"label": "a {nested} brace and an escaped \\" quote", "score": 1.0}'
    assert parse_into(raw, Verdict).score == 1.0


def test_a_fields_wrapped_response_parses():
    """Some chat models nest the requested object under a wrapper key.

    Observed live from this catalogue's chat-tuned model: asked for
    ``{label, score}`` it replied ``{"fields": {"label": ..., "score": ...}}``.
    """
    assert parse_into('{"fields": {"label": "x", "score": 2.0}}', Verdict).score == 2.0


def test_a_result_wrapped_response_parses():
    assert parse_into('{"result": ' + VALID + "}", Verdict).label == "relevant"


def test_a_partially_wrapped_response_still_fails_loudly():
    """A wrapper whose contents are wrong must not validate."""
    with pytest.raises(ValidationError):
        parse_into('{"fields": {"label": "x", "score": 99.0}}', Verdict)


def test_a_wrapper_with_extra_siblings_does_not_lose_them_silently():
    """When the wrapper's inner object fails, the error is a validation error."""
    with pytest.raises(ValidationError):
        parse_into('{"fields": {"label": "x"}}', Verdict)


def test_invalid_json_raises_validation_error():
    with pytest.raises(ValidationError):
        parse_into("not json at all", Verdict)


def test_schema_violating_json_raises_validation_error():
    with pytest.raises(ValidationError):
        parse_into('{"label": "x", "score": 99.0}', Verdict)


def test_empty_response_raises_validation_error():
    with pytest.raises(ValidationError):
        parse_into("", Verdict)


def test_payload_candidates_are_deduplicated():
    raw = f"{VALID}\n{VALID}"
    assert len(candidate_json_payloads(raw)) >= 1
    assert len(set(candidate_json_payloads(raw))) == len(candidate_json_payloads(raw))


def test_payload_candidates_of_empty_input_is_empty():
    assert candidate_json_payloads("") == []


# --------------------------------------------------------------------------- #
# Request shaping per backend
# --------------------------------------------------------------------------- #


def test_nim_uses_nvext_guided_json_and_not_response_format(nim_settings):
    client = LLMClient(nim_settings, client=fake_openai([VALID]))
    client.complete_json(system="s", user="u", schema=Verdict)

    request = client._client.chat.completions.requests[0]
    assert "guided_json" in request["extra_body"]["nvext"]
    assert request["extra_body"]["nvext"]["guided_json"]["properties"].keys() >= {"label", "score"}
    # NIM's documented structured-output path does not use response_format.
    assert "response_format" not in request


def test_openrouter_uses_response_format_and_require_parameters(openrouter_settings):
    client = LLMClient(openrouter_settings, client=fake_openai([VALID]))
    client.complete_json(system="s", user="u", schema=Verdict)

    request = client._client.chat.completions.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["name"] == "Verdict"
    # Without require_parameters, OpenRouter may route to a provider that
    # ignores the schema entirely.
    assert request["extra_body"]["provider"]["require_parameters"] is True
    assert "nvext" not in request["extra_body"]


def test_openrouter_strict_is_off_because_fields_are_genuinely_optional(openrouter_settings):
    """strict=True demands every property be required, which would misrepresent
    optional extraction fields (many abstracts state no limitations)."""
    client = LLMClient(openrouter_settings, client=fake_openai([VALID]))
    client.complete_json(system="s", user="u", schema=Verdict)
    request = client._client.chat.completions.requests[0]
    assert request["response_format"]["json_schema"]["strict"] is False


def test_model_and_messages_are_passed_through(nim_settings):
    client = LLMClient(nim_settings, client=fake_openai([VALID]))
    client.complete_json(system="SYS", user="USR", schema=Verdict, temperature=0.3)

    request = client._client.chat.completions.requests[0]
    assert request["model"] == "test/model"
    assert request["temperature"] == 0.3
    assert request["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


# --------------------------------------------------------------------------- #
# Repair loop
# --------------------------------------------------------------------------- #


def test_valid_first_response_costs_one_call(nim_settings):
    client = LLMClient(nim_settings, client=fake_openai([VALID]))
    assert client.complete_json(system="s", user="u", schema=Verdict) is not None
    assert client.calls == 1


def test_invalid_response_is_repaired(nim_settings):
    client = LLMClient(nim_settings, client=fake_openai(['{"label": "x"}', VALID]))
    result = client.complete_json(system="s", user="u", schema=Verdict)

    assert result is not None and result.score == 8.5
    requests = client._client.chat.completions.requests
    assert len(requests) == 2
    # The repair turn must show the model its own bad output and the error.
    second_messages = requests[1]["messages"]
    assert second_messages[-2] == {"role": "assistant", "content": '{"label": "x"}'}
    assert "schema" in second_messages[-1]["content"].lower()


def test_repair_loop_gives_up_after_max_repairs(nim_settings):
    settings = replace(nim_settings, llm_max_repairs=1)
    client = LLMClient(settings, client=fake_openai(["bad", "still bad"]))
    assert client.complete_json(system="s", user="u", schema=Verdict) is None
    assert client.calls == 2
    assert client.failures == 1


def test_max_repairs_zero_means_one_attempt(nim_settings):
    settings = replace(nim_settings, llm_max_repairs=0)
    client = LLMClient(settings, client=fake_openai(["bad"]))
    assert client.complete_json(system="s", user="u", schema=Verdict) is None
    assert client.calls == 1


def test_empty_content_is_a_failure_not_a_crash(nim_settings):
    client = LLMClient(nim_settings, client=fake_openai(["", VALID]))
    # Empty content still goes through the repair loop, which can rescue it.
    assert client.complete_json(system="s", user="u", schema=Verdict) is not None


# --------------------------------------------------------------------------- #
# Transport errors
# --------------------------------------------------------------------------- #


def test_non_retryable_status_is_not_retried(nim_settings):
    client = LLMClient(nim_settings, client=fake_openai([status_error(400)]))
    assert client.complete_json(system="s", user="u", schema=Verdict) is None
    assert client.calls == 1


def test_retryable_status_is_retried_then_succeeds(nim_settings):
    client = LLMClient(nim_settings, client=fake_openai([status_error(429), VALID]))
    assert client.complete_json(system="s", user="u", schema=Verdict) is not None
    assert client.calls == 2


def test_retries_are_bounded(nim_settings):
    client = LLMClient(
        nim_settings, client=fake_openai([status_error(503)] * 8)
    )
    assert client.complete_json(system="s", user="u", schema=Verdict) is None
    assert client.calls == 4  # four attempts, then give up


def test_error_without_a_status_code_is_retried(nim_settings):
    """Connection resets arrive without a status code."""
    client = LLMClient(nim_settings, client=fake_openai([RuntimeError("connection reset"), VALID]))
    assert client.complete_json(system="s", user="u", schema=Verdict) is not None


def test_malformed_completion_shape_does_not_crash(nim_settings):
    broken = SimpleNamespace(choices=[])
    client = LLMClient(nim_settings, client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: broken))))
    assert client.complete_json(system="s", user="u", schema=Verdict) is None


def test_constructing_without_a_key_names_the_variable():
    from llm.client import LLMError

    settings = replace(Settings.from_env(), nvidia_api_key="")
    with pytest.raises(LLMError) as excinfo:
        LLMClient(settings)
    assert "NVIDIA_API_KEY" in str(excinfo.value)


def test_model_name_is_exposed_for_cache_keying(nim_settings):
    assert LLMClient(nim_settings, client=fake_openai([])).model_name == "test/model"
