from typing import List
from pydantic import BaseModel


class Utterance(BaseModel):
    role: str
    en_text: str


class Transcript(BaseModel):
    interaction_transcript: List[Utterance]
