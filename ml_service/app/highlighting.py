from __future__ import annotations

from difflib import SequenceMatcher
import re

from .schemas import HighlightSpan


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0).lower(), match.start(), match.end()) for match in re.finditer(r"\b\w+\b", text)]


def lcs_highlights(
    source_text: str,
    candidate_text: str,
    offset: int,
    source_url: str | None,
    score: float,
    min_words: int = 6,
) -> list[HighlightSpan]:
    source_words = _word_spans(source_text)
    candidate_words = _word_spans(candidate_text)
    if not source_words or not candidate_words:
        return []

    matcher = SequenceMatcher(
        None,
        [word for word, _start, _end in source_words],
        [word for word, _start, _end in candidate_words],
        autojunk=False,
    )

    spans: list[HighlightSpan] = []
    for match in matcher.get_matching_blocks():
        if match.size < min_words:
            continue
        start = source_words[match.a][1]
        end = source_words[match.a + match.size - 1][2]
        highlighted = source_text[start:end]
        spans.append(
            HighlightSpan(
                start=offset + start,
                end=offset + end,
                text=highlighted,
                source_url=source_url,
                score=round(score * 100, 2),
            )
        )
    return merge_spans(spans)


def lcs_coverage(source_text: str, candidate_text: str, min_words: int = 6) -> float:
    source_words = _word_spans(source_text)
    candidate_words = _word_spans(candidate_text)
    if not source_words or not candidate_words:
        return 0.0

    matcher = SequenceMatcher(
        None,
        [word for word, _start, _end in source_words],
        [word for word, _start, _end in candidate_words],
        autojunk=False,
    )
    matched = sum(match.size for match in matcher.get_matching_blocks() if match.size >= min_words)
    return matched / len(source_words)


def merge_spans(spans: list[HighlightSpan], gap: int = 16) -> list[HighlightSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged = [ordered[0]]

    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end + gap and span.source_url == last.source_url:
            last.end = max(last.end, span.end)
            if len(span.text) > len(last.text):
                last.text = span.text
            last.score = max(last.score, span.score)
        else:
            merged.append(span)
    return merged
