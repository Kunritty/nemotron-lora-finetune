from pydantic import BaseModel


class DataPoint(BaseModel):
    id: str
    category: str | None = None
    prompt: str
    answer: str
    reasoning: str 
    final_answer: str
    confidence: float
    is_correct: bool

class Entry(BaseModel):
    id: str
    prompt: str
    answer: str
    category: str | None = None