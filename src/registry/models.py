"""
Model helpers.

This file intentionally avoids importing Ollama/LangChain at import time.
If you need the Ollama backend, use :mod:`registry.ollama` (or call the
compatibility wrappers here, which import lazily).
"""

from __future__ import annotations

from typing import Any

from .transformers_chat import TransformersChatLLM

def get_model(model_id: str):
    """Compatibility wrapper for the Ollama backend (lazy import)."""

    from .ollama import get_model as _get_model

    return _get_model(model_id)


def list_models() -> list[str]:
    """Compatibility wrapper for the Ollama backend (lazy import)."""

    from .ollama import list_models as _list_models

    return _list_models()


def default_model_id() -> str:
    """Compatibility wrapper for the Ollama backend (lazy import)."""

    from .ollama import default_model_id as _default_model_id

    return _default_model_id()


def get_transformers_chat_model(
    pretrained_model_name_or_path: str,
    *,
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.95,
    device_map: str | dict[str, Any] | None = "auto",
    torch_dtype: Any = "auto",
    trust_remote_code: bool = False,
) -> TransformersChatLLM:
    """Load a Hugging Face causal LM for use with :class:`chain.reasoning_chain.ReasoningChain`."""
    return TransformersChatLLM(
        pretrained_model_name_or_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

