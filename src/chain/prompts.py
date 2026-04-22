"""Build chat messages: optional system text plus the competition problem as the user turn."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# Optional baseline; combine with :func:`compose_system_prompt` or pass your own string.
DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant . Your objective is to solve the following puzzle and 
provide logical reasoning steps 
"""

DEFAULT_VERIFY_SYSTEM_PROMPT = """
You are a strict verifier. Given a puzzle and a candidate answer (and the solver's reasoning if any), check
whether the answer matches the problem's requirements, whether the logic is sound, and whether anything is missing.
Be concise. If you find issues, state what is wrong and what must change.
"""

DEFAULT_IMPROVE_SYSTEM_PROMPT = """
You are an editor. Using the problem, the first draft, and the verification feedback, produce a clear final
answer. Fix any errors the verifier identified. Keep reasoning legible. Your last line should be the final
answer the grader can compare, clearly separated from the rest.
"""


def compose_system_prompt(*segments: str | None) -> str | None:
    """
    Join non-empty segments with blank lines. Returns ``None`` if nothing to send.

    Use this to stack a default instruction, category hints, formatting rules, etc.
    """
    parts = [str(s).strip() for s in segments if s is not None and str(s).strip()]
    if not parts:
        return None
    return "\n\n".join(parts)


def _messages_with_optional_system(
    system_prompt: str | None, user_content: str
) -> list[BaseMessage]:
    text = (system_prompt or "").strip()
    messages: list[BaseMessage] = []
    if text:
        messages.append(SystemMessage(content=text))
    messages.append(HumanMessage(content=user_content))
    return messages


def build_solve_messages(
    user_content: str,
    *,
    system_prompt: str | None = None,
) -> list[BaseMessage]:
    """Initial solve phase: system (optional) + problem as the user message."""
    return _messages_with_optional_system(system_prompt, user_content)


def build_reasoning_messages(
    user_content: str,
    *,
    system_prompt: str | None = None,
) -> list[BaseMessage]:
    """
    One system message (if ``system_prompt`` is non-empty) plus the problem as the user message.
    Same as :func:`build_solve_messages`. Kept for backward compatibility.
    """
    return build_solve_messages(user_content, system_prompt=system_prompt)


def build_verify_messages(
    problem: str,
    candidate_answer: str,
    candidate_reasoning: str,
    *,
    system_prompt: str | None = None,
) -> list[BaseMessage]:
    """Verification phase: system + problem, draft answer, and solve-phase reasoning."""
    user = (
        "Original problem:\n"
        f"{problem}\n\n"
        "Candidate final answer (from the solver):\n"
        f"{candidate_answer}\n\n"
        "Reasoning from the solve phase (may be empty):\n"
        f"{candidate_reasoning or '(none)'}\n\n"
        "Verify this solution. If it is wrong or incomplete, say what is wrong and what should be done."
    )
    return _messages_with_optional_system(system_prompt, user)


def build_improve_messages(
    problem: str,
    draft_answer: str,
    verify_feedback: str,
    *,
    system_prompt: str | None = None,
) -> list[BaseMessage]:
    """Clean-up / improvement phase: system + problem, draft, and verifier output."""
    user = (
        "Original problem:\n"
        f"{problem}\n\n"
        "First draft answer (from the solve step):\n"
        f"{draft_answer}\n\n"
        "Verification feedback:\n"
        f"{verify_feedback}\n\n"
        "Integrate the feedback: correct mistakes, clarify reasoning, and give a polished final response. "
        "End with the final short answer the problem asks for, on its own if possible."
    )
    return _messages_with_optional_system(system_prompt, user)

