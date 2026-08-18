"""
llm_providers.py — LLM provider abstraction
===============================================
    LLMProvider (ABC)
    ├── GroqProvider       -- existing production provider, default
    ├── AnthropicProvider  -- Claude via the official anthropic SDK
    └── OpenAIProvider     -- GPT via the official openai SDK

Business logic (agents/analysis_agent.py) talks only to the LLMProvider
interface, so switching models is a config change (LLM_PROVIDER env
var), never a code change. API keys are always read from environment
variables -- never hardcoded, never logged.
"""

import os
from abc import ABC, abstractmethod

import config


class LLMProvider(ABC):
    """A provider turns (system prompt, message history) into a
    completion string. Implementations must not perform any security
    filtering themselves -- that's the guardrail's job, kept separate
    on purpose (see security/__init__.py)."""

    name: str = "base"

    @abstractmethod
    def generate(self, system: str, messages: list, max_tokens: int = 1024) -> str:
        """messages: list of {"role": "user"|"assistant", "content": str}"""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Wraps the Groq client exactly as rag_engine.py used to call it
    directly -- preserves the existing deployed behavior byte-for-byte."""

    name = "groq"

    def __init__(self, model: str = None, api_key: str = None):
        from groq import Groq

        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        api_key = api_key or os.environ["GROQ_API_KEY"]
        self._client = Groq(api_key=api_key)

    def generate(self, system: str, messages: list, max_tokens: int = 1024) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Claude via the official `anthropic` SDK. Default model is
    claude-opus-5; override with ANTHROPIC_MODEL."""

    name = "anthropic"

    def __init__(self, model: str = None, api_key: str = None):
        import anthropic

        self.model = model or config.ANTHROPIC_MODEL
        api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system: str, messages: list, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIProvider(LLMProvider):
    """GPT via the official `openai` SDK. Default model is gpt-4o-mini;
    override with OPENAI_MODEL."""

    name = "openai"

    def __init__(self, model: str = None, api_key: str = None):
        from openai import OpenAI

        self.model = model or config.OPENAI_MODEL
        api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._client = OpenAI(api_key=api_key)

    def generate(self, system: str, messages: list, max_tokens: int = 1024) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        return response.choices[0].message.content


_PROVIDERS = {
    "groq": GroqProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider(name: str = None) -> LLMProvider:
    """Instantiate the configured provider. Defaults to LLM_PROVIDER
    (env var, defaults to "groq" -- preserves current deployed
    behavior for anyone who hasn't opted into a different provider)."""
    name = (name or config.LLM_PROVIDER).lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Valid options: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()
