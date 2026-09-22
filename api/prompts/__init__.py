"""Prompt builders and their versions.

``PROMPT_VERSION`` for extraction lives in the environment (``PROMPT_VERSION``,
default ``extract_v1``) because it is the cache key for ``paper_extractions``.
Bump it whenever the extraction prompt *or* the ``PaperExtraction`` schema
changes; extractions are then recomputed without touching embeddings or layout.
"""
