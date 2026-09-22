"""The interface pipeline stages depend on.

Stages accept ``JSONCompleter`` rather than a concrete client so that tests can
substitute a fake without patching ``openai``, and so that the pipeline stays
runnable with no API key at all (stages degrade instead of crashing).
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class JSONCompleter(Protocol):
    """A model that can be asked for JSON conforming to a Pydantic schema."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model, recorded with cached output."""
        ...

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        max_repairs: int | None = None,
        temperature: float = 0.0,
    ) -> SchemaT | None:
        """Return a validated instance of ``schema``, or ``None`` on failure.

        Implementations must never raise for a malformed model response: the
        caller decides whether a missing value is fatal (extraction degrades per
        paper) or merely a lost nicety (cluster labeling falls back to a
        placeholder).
        """
        ...
