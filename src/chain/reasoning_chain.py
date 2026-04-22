from collections.abc import Sequence

from langchain_ollama import ChatOllama

from .data_types import DataPoint, Entry
from .pipeline import ChainState, MessageStep, PhaseResult, default_message_steps


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


class ReasoningChain:
    """
    Sequentially runs :class:`MessageStep` s (``build_messages`` → model ``invoke`` → append phase).

    Pass ``steps=`` to define the pipeline, or omit it and set ``system_prompt`` /
    ``verify_system_prompt`` / ``improve_system_prompt`` to use
    :func:`~pipeline.default_message_steps`.
    """

    def __init__(
        self,
        model: ChatOllama,
        *,
        steps: Sequence[MessageStep] | None = None,
        system_prompt: str | None = None,
        verify_system_prompt: str | None = None,
        improve_system_prompt: str | None = None,
    ):
        self._model = model
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

    def run(self, entry: Entry) -> DataPoint:
        state = ChainState(entry=entry)
        for step in self._steps:
            messages = step.build_messages(state)
            response = self._model.invoke(messages)
            state.phases.append(
                PhaseResult(
                    name=step.name,
                    reasoning=_extract_reasoning(response),
                    content=str(getattr(response, "content", "")),
                )
            )

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
