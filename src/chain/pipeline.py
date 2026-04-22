"""Pluggable message steps and shared state for :class:`ReasoningChain`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage

from .data_types import Entry
from .prompts import (
    DEFAULT_IMPROVE_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VERIFY_SYSTEM_PROMPT,
    build_improve_messages,
    build_solve_messages,
    build_verify_messages,
    compose_system_prompt,
)


@dataclass
class PhaseResult:
    name: str
    reasoning: str
    content: str


@dataclass
class ChainState:
    """Read-only :attr:`entry` plus prior phases; each step may read this and add one phase when run."""

    entry: Entry
    phases: list[PhaseResult] = field(default_factory=list)

    def last(self) -> PhaseResult:
        if not self.phases:
            raise RuntimeError("no phases in state")
        return self.phases[-1]

    def by_name(self, name: str) -> PhaseResult:
        for p in reversed(self.phases):
            if p.name == name:
                return p
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class MessageStep:
    """A named turn: build messages from current state, then the runner appends a :class:`PhaseResult`."""

    name: str
    build_messages: Callable[[ChainState], list[BaseMessage]]


def _solve_messages(system: str | None) -> Callable[[ChainState], list[BaseMessage]]:
    def build(state: ChainState) -> list[BaseMessage]:
        return build_solve_messages(state.entry.prompt, system_prompt=system)

    return build


def _verify_messages(
    system: str | None, *, from_solve: str
) -> Callable[[ChainState], list[BaseMessage]]:
    def build(state: ChainState) -> list[BaseMessage]:
        sol = state.by_name(from_solve)
        return build_verify_messages(
            state.entry.prompt,
            sol.content,
            sol.reasoning,
            system_prompt=system,
        )

    return build


def _improve_messages(
    system: str | None, *, from_solve: str, from_verify: str
) -> Callable[[ChainState], list[BaseMessage]]:
    def build(state: ChainState) -> list[BaseMessage]:
        sol = state.by_name(from_solve)
        ver = state.by_name(from_verify)
        return build_improve_messages(
            state.entry.prompt,
            sol.content,
            ver.content,
            system_prompt=system,
        )

    return build


def make_solve_step(
    name: str = "SOLVE",
    *,
    system_prompt: str | None = None,
) -> MessageStep:
    system = compose_system_prompt(DEFAULT_SYSTEM_PROMPT, system_prompt)
    return MessageStep(name=name, build_messages=_solve_messages(system))


def make_verify_step(
    name: str = "VERIFY",
    *,
    system_prompt: str | None = None,
    from_solve: str = "SOLVE",
) -> MessageStep:
    system = compose_system_prompt(
        DEFAULT_VERIFY_SYSTEM_PROMPT, system_prompt
    )
    return MessageStep(
        name=name, build_messages=_verify_messages(system, from_solve=from_solve)
    )


def make_improve_step(
    name: str = "IMPROVE",
    *,
    system_prompt: str | None = None,
    from_solve: str = "SOLVE",
    from_verify: str = "VERIFY",
) -> MessageStep:
    system = compose_system_prompt(
        DEFAULT_IMPROVE_SYSTEM_PROMPT, system_prompt
    )
    return MessageStep(
        name=name,
        build_messages=_improve_messages(
            system, from_solve=from_solve, from_verify=from_verify
        ),
    )


def default_message_steps(
    *,
    system_prompt: str | None = None,
    verify_system_prompt: str | None = None,
    improve_system_prompt: str | None = None,
) -> tuple[MessageStep, MessageStep, MessageStep]:
    return (
        make_solve_step(system_prompt=system_prompt),
        make_verify_step(system_prompt=verify_system_prompt),
        make_improve_step(system_prompt=improve_system_prompt),
    )
