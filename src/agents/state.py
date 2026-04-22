from typing import TypedDict, List

class Record(TypedDict):
    attempt: int
    reasoning_steps: List[str]
    final_answer: str
    feedback: str

class SolveState(TypedDict):
    problem: str
    true_answer: str
    final_answer: str
    reasoning_steps: List[str]
    attempts: int
    history: List[Record]
    is_solved: bool
    