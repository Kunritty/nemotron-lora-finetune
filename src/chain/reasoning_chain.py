import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from langchain_core.messages import BaseMessage

from .data_types import DataPoint, Entry
from .pipeline import ChainState, MessageStep, PhaseResult, default_message_steps

logger = logging.getLogger(__name__)


class SupportsInvokeMessages(Protocol):
    """Anything compatible with LangChain chat models (``invoke([BaseMessage, ...])``)."""

    def invoke(self, messages: list[BaseMessage], config: Any | None = None) -> Any: ...


def _normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def _extract_reasoning(response: object) -> str:
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict):
        v = additional.get("reasoning_content")
        return "" if v is None else str(v)
    return ""


def _phase_block(name: str, reasoning: str, content: str) -> str:
    lines = [f"=== {name} ==="]
    if (reasoning or "").strip():
        lines.append((reasoning or "").strip())
    out = (content or "").strip()
    if out:
        lines.append("[output]\n" + out)
    return "\n\n".join(lines).strip()


def _now_ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ReasoningChain:
    """
    Sequentially runs :class:`MessageStep` s (``build_messages`` → model ``invoke`` → append phase).

    ``model`` may be :class:`langchain_ollama.ChatOllama`, :class:`registry.hf_transformers_chat.TransformersChatLLM`,
    or any object matching :class:`SupportsInvokeMessages`.

    Pass ``steps=`` to define the pipeline, or omit it and set ``system_prompt`` /
    ``verify_system_prompt`` / ``improve_system_prompt`` to use
    :func:`~pipeline.default_message_steps`.

    With ``verbose=True``, each :meth:`run` logs phase progress at INFO on this
    module's logger and includes timestamps (configure root logging or
    ``chain.reasoning_chain`` to see it).
    """

    def __init__(
        self,
        model: SupportsInvokeMessages,
        *,
        steps: Sequence[MessageStep] | None = None,
        system_prompt: str | None = None,
        verify_system_prompt: str | None = None,
        improve_system_prompt: str | None = None,
        verbose: bool = False,
    ):
        self._model = model
        self._verbose = verbose
        if steps is not None:
            self._steps = tuple(steps)
        else:
            self._steps = default_message_steps(
                system_prompt=system_prompt,
                verify_system_prompt=verify_system_prompt,
                improve_system_prompt=improve_system_prompt,
            )
        if not self._steps:
            raise ValueError("ReasoningChain requires at least one MessageStep")

    @property
    def steps(self) -> tuple[MessageStep, ...]:
        return self._steps

    def _vlog(self, msg: str, *args: object) -> None:
        if self._verbose:
            logger.info("[%s] " + msg, _now_ts(), *args)

    def run(self, entry: Entry) -> DataPoint:
        n = len(self._steps)
        self._vlog("ReasoningChain.run start id=%s phases=%d", entry.id, n)
        state = ChainState(entry=entry)
        for i, step in enumerate(self._steps, start=1):
            self._vlog("ReasoningChain.run phase %d/%d: %s", i, n, step.name)
            messages = step.build_messages(state)
            response = self._model.invoke(messages)
            state.phases.append(
                PhaseResult(
                    name=step.name,
                    reasoning=_extract_reasoning(response),
                    content=str(getattr(response, "content", "")),
                )
            )
            self._vlog("ReasoningChain.run finished phase %d/%d: %s", i, n, step.name)

        last = state.last()
        final_answer = last.content
        answer = entry.answer
        category = entry.category
        prompt = entry.prompt

        reasoning = "\n\n".join(
            _phase_block(p.name, p.reasoning, p.content) for p in state.phases
        )

        is_correct = _normalize_answer(final_answer) == _normalize_answer(answer)
        confidence = 1.0 if is_correct else 0.0

        return DataPoint(
            id=entry.id,
            category=category,
            prompt=prompt,
            answer=answer,
            reasoning=reasoning,
            final_answer=final_answer,
            confidence=confidence,
            is_correct=is_correct,
        )
