"""
tests/conftest.py — Shared pytest fixtures.

FakeLLMProvider lets the whole agentic pipeline be exercised end-to-end
without any network calls or real API keys -- the security/governance
tests need to run in any environment, including CI with no secrets
configured.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# rag_engine.py (imported transitively by agents.tools) requires
# QDRANT_URL/QDRANT_API_KEY at import time -- load .env before any test
# module pulls that chain in, exactly like app.py does at startup.
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import pytest

from llm_providers import LLMProvider


class FakeLLMProvider(LLMProvider):
    """Deterministic canned-answer provider for tests. Never makes a
    network call."""

    name = "fake"

    def __init__(self, canned_answer: str = "This is a test answer [Source: doc, Chunk 0]."):
        self.canned_answer = canned_answer
        self.calls = []

    def generate(self, system: str, messages: list, max_tokens: int = 1024) -> str:
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        return self.canned_answer


@pytest.fixture
def fake_provider():
    return FakeLLMProvider()


@pytest.fixture
def sample_chunks():
    return [
        {"text": "The sky is blue because of Rayleigh scattering.", "doc_name": "physics", "chunk_index": 0, "score": 0.91},
        {"text": "Water boils at 100 degrees Celsius at sea level.", "doc_name": "physics", "chunk_index": 1, "score": 0.85},
    ]
