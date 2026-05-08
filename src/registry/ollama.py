"""
Ollama registry helpers.

This module is intentionally the only place that imports `langchain_ollama`.
Importing the rest of the project should not require Ollama/LangChain deps
unless you explicitly use the Ollama backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL_ID = "nemotron3-4b"


@dataclass(frozen=True)
class OllamaModel:
    id: str
    ollama_name: str
    description: str = ""
    base_url: str = DEFAULT_BASE_URL
    reasoning: bool = False
    options: dict[str, Any] = field(default_factory=dict)


MODELS: dict[str, OllamaModel] = {
    "nemotron3-4b": OllamaModel(
        id="nemotron3",
        ollama_name="nemotron-3-nano:4b",
        description="Nemotron-family agent/reasoning default (Nano 4B).",
        options={
            "temperature": 0.2,
            "num_ctx": 16384,
            "top_p": 0.95,
        },
    ),
    "gemma4": OllamaModel(
        id="gemma4",
        ollama_name="gemma4:e4b",
        description="Gemma 4 default tag; supports long context and agent-style use.",
        reasoning=True,
        options={
            "temperature": 0.15,
            "num_ctx": 65536,
            "top_p": 0.9,
        },
    ),
    "gemma4-e4b-no-reasoning": OllamaModel(
        id="gemma4-e4b-no-reasoning",
        ollama_name="gemma4:e4b",
        description="Gemma 4 default tag; supports long context and agent-style use.",
        reasoning=False,
        options={
            "temperature": 0.15,
            "num_ctx": 65536,
            "top_p": 0.9,
        },
    ),
}


def get_model(model_id: str):
    """
    Build an Ollama chat model.

    Returns a `langchain_ollama.ChatOllama` instance.
    """

    try:
        from langchain_ollama import ChatOllama  # local import by design
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Ollama backend requires `langchain_ollama`. Install it to use get_model()."
        ) from e

    try:
        profile = MODELS[model_id]
    except KeyError as e:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"Unknown model_id {model_id!r}; expected one of: {known}") from e

    return ChatOllama(
        model=profile.ollama_name,
        reasoning=profile.reasoning,
        **profile.options,
    )


def list_models() -> list[str]:
    return sorted(MODELS.keys())


def default_model_id() -> str:
    return DEFAULT_MODEL_ID

