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


class SourceResult(BaseModel):
    url: str | None = None
    title: str | None = None
    score: float
    contribution: float
    matched_text: str | None = None


class AnalyzeResponse(BaseModel):
    plagiarism_score: float
    ai_score: float
    highlighted_text_spans: list[HighlightSpan]
    source_urls: list[SourceResult]
    ai_explanation: str
    chunks_analyzed: int
    metadata: dict[str, Any]

