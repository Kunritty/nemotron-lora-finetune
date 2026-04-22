"""
Ollama model registry: keep each profile's ollama_name aligned with `ollama list` after pulls.
Example pulls (adjust tags if you use different variants):
  ollama pull nemotron-3-nano:4b
  ollama pull gemma4:latest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_ollama import ChatOllama

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


def get_model(model_id: str) -> ChatOllama:
    try:
        return ChatOllama(
            model=MODELS[model_id].ollama_name,
            reasoning=MODELS[model_id].reasoning,
            **MODELS[model_id].options,
        )
    except KeyError as e:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"Unknown model_id {model_id!r}; expected one of: {known}") from e


def list_models() -> list[str]:
    return sorted(MODELS.keys())


def default_model_id() -> str:
    return DEFAULT_MODEL_ID


