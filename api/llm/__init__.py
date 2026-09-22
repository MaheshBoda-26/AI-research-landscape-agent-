"""LLM access: one client, two OpenAI-compatible backends (NIM, OpenRouter)."""

from llm.protocol import JSONCompleter

__all__ = ["JSONCompleter"]
