from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=20)
    document_id: str | None = None
    filename: str | None = None


class HighlightSpan(BaseModel):
    start: int
    end: int
    text: str
    source_url: str | None = None
    score: float
    kind: str = "plagiarism"
    explanation: str | None = None


class AISpan(BaseModel):
    start: int
    end: int
    text: str
    score: float
    explanation: str


class SourceResult(BaseModel):
    url: str | None = None
    title: str | None = None
    score: float
    contribution: float
    matched_text: str | None = None
    matched_spans: int = 0
    evidence: list[str] = []
    source_type: str = "web"


class AnalyzeResponse(BaseModel):
    plagiarism_score: float
    ai_score: float
    highlighted_text_spans: list[HighlightSpan]
    ai_highlighted_spans: list[AISpan]
    source_urls: list[SourceResult]
    ai_explanation: str
    chunks_analyzed: int
    metadata: dict[str, Any]
