from __future__ import annotations

from collections import defaultdict

from .ai_classifier import roberta_ai_probability
from .chunking import TextChunk, chunk_text
from .config import get_settings
from .cross_encoder import rerank
from .embeddings import embed_texts
from .ensemble import explain_ai, predict_ai_score
from .highlighting import lcs_coverage, lcs_highlights, merge_spans
from .perplexity import perplexity
from .preprocessing import clean_text
from .retrieval import RetrievalCandidate, get_store, records_from_texts
from .schemas import AISpan, AnalyzeResponse, HighlightSpan, SourceResult
from .similarity import combined_similarity, ngram_overlap
from .stylometry import extract_stylometric_features
from .web_search import web_candidates_for_chunk


async def _augment_from_web(chunk: TextChunk) -> list[RetrievalCandidate]:
    settings = get_settings()
    scraped = await web_candidates_for_chunk(chunk.text, limit=settings.max_web_results)
    if not scraped:
        return []

    candidates: list[RetrievalCandidate] = []
    for page in scraped[: settings.max_web_results]:
        page_chunks = chunk_text(page["text"], chunk_words=250, overlap_words=50)[:8]
        if not page_chunks:
            continue

        vectors = embed_texts([item.text for item in page_chunks], batch_size=16)
        store = get_store()
        store.add(
            vectors,
            records_from_texts([item.text for item in page_chunks], page["url"], page.get("title")),
            persist=True,
        )

        for page_chunk in page_chunks:
            candidates.append(
                RetrievalCandidate(
                    text=page_chunk.text,
                    score=0.0,
                    source_url=page["url"],
                    title=page.get("title"),
                    metadata={"kind": "web", "query": page.get("query"), "queries": page.get("queries", [])},
                )
            )
    return candidates


def _candidate_source_type(candidate: RetrievalCandidate) -> str:
    if candidate.source_url:
        return "web"
    metadata = candidate.metadata or {}
    return metadata.get("kind") or "local"


def _evaluate_candidates(chunk: TextChunk, vector, candidates: list[RetrievalCandidate]) -> tuple[list[dict], list[HighlightSpan]]:
    ranked = rerank(chunk.text, candidates[:25], top_n=6)
    candidate_vectors = embed_texts([candidate.text for candidate in ranked], batch_size=16) if ranked else []
    matches: list[dict] = []
    highlights: list[HighlightSpan] = []

    for candidate, candidate_vector in zip(ranked, candidate_vectors):
        cosine = float(vector @ candidate_vector)
        overlap = ngram_overlap(chunk.text, candidate.text)
        lcs_signal = lcs_coverage(chunk.text, candidate.text)
        score = combined_similarity(cosine, overlap, candidate.score, lcs_signal)
        if score < 0.55 or (overlap < 0.08 and lcs_signal < 0.12):
            continue

        match_highlights = lcs_highlights(
            source_text=chunk.text,
            candidate_text=candidate.text,
            offset=chunk.start_char,
            source_url=candidate.source_url,
            score=score,
        )
        if not match_highlights:
            continue

        for span in match_highlights:
            span.kind = "plagiarism"
            span.explanation = f"{round(score * 100, 1)}% combined match; n-gram overlap {round(overlap * 100, 1)}%; aligned text coverage {round(lcs_signal * 100, 1)}%."

        metadata = candidate.metadata or {}
        matches.append(
            {
                "chunk_id": chunk.id,
                "score": score,
                "source_url": candidate.source_url,
                "title": candidate.title,
                "matched_text": candidate.text[:900],
                "spans": len(match_highlights),
                "evidence": [span.text[:240] for span in match_highlights[:3]],
                "source_type": _candidate_source_type(candidate),
                "query": metadata.get("query"),
                "overlap": overlap,
                "lcs_coverage": lcs_signal,
                "cosine": cosine,
            }
        )
        highlights.extend(match_highlights)

    return matches, highlights


async def plagiarism_pipeline(text: str) -> tuple[float, list[HighlightSpan], list[SourceResult], dict]:
    settings = get_settings()
    chunks = chunk_text(text, chunk_words=250, overlap_words=50)
    if not chunks:
        return 0.0, [], [], {"retrieval": "no chunks"}

    chunk_vectors = embed_texts([chunk.text for chunk in chunks], batch_size=16)
    store = get_store()
    retrieved = store.search(chunk_vectors, top_k=settings.top_k)

    all_matches: list[dict] = []
    highlights: list[HighlightSpan] = []
    web_triggered = 0
    web_searched_chunks = 0

    for chunk, vector, candidates in zip(chunks, chunk_vectors, retrieved):
        best_initial = max([candidate.score for candidate in candidates], default=0.0)
        expanded_candidates = list(candidates)
        if best_initial < settings.web_trigger_threshold:
            web_triggered += 1
            web_searched_chunks += 1
            expanded_candidates.extend(await _augment_from_web(chunk))

        chunk_matches, chunk_highlights = _evaluate_candidates(chunk, vector, expanded_candidates)
        if not chunk_matches and settings.serpapi_api_key and best_initial >= settings.web_trigger_threshold:
            web_triggered += 1
            web_searched_chunks += 1
            web_candidates = await _augment_from_web(chunk)
            chunk_matches, chunk_highlights = _evaluate_candidates(chunk, vector, web_candidates)

        all_matches.extend(chunk_matches)
        highlights.extend(chunk_highlights)

    chunk_scores = defaultdict(float)
    for match in all_matches:
        chunk_scores[match["chunk_id"]] = max(chunk_scores[match["chunk_id"]], match["score"])

    plagiarism_score = (sum(chunk_scores.values()) / max(len(chunks), 1)) * 100
    plagiarism_score = round(max(0.0, min(100.0, plagiarism_score)), 2)

    by_source: dict[str, dict] = {}
    total = sum(match["score"] for match in all_matches) or 1.0
    for match in sorted(all_matches, key=lambda item: item["score"], reverse=True):
        key = match["source_url"] or "local-corpus"
        if key not in by_source:
            by_source[key] = {
                "url": match["source_url"],
                "title": match["title"] or "Local corpus",
                "score": match["score"],
                "contribution": 0.0,
                "matched_text": match["matched_text"],
                "matched_spans": 0,
                "evidence": [],
                "source_type": match["source_type"],
            }
        by_source[key]["contribution"] += match["score"] / total
        by_source[key]["score"] = max(by_source[key]["score"], match["score"])
        by_source[key]["matched_spans"] += match["spans"]
        by_source[key]["evidence"].extend(match["evidence"])

    sources = [
        SourceResult(
            url=item["url"],
            title=item["title"],
            score=round(item["score"] * 100, 2),
            contribution=round(item["contribution"] * 100, 2),
            matched_text=item["matched_text"],
            matched_spans=item["matched_spans"],
            evidence=item["evidence"][:5],
            source_type=item["source_type"],
        )
        for item in sorted(by_source.values(), key=lambda row: row["contribution"], reverse=True)[:10]
    ]

    metadata = {
        "chunks": len(chunks),
        "matches": len(all_matches),
        "web_search_triggered_for_chunks": web_triggered,
        "web_searched_chunks": web_searched_chunks,
        "web_search_enabled": bool(settings.serpapi_api_key),
    }
    return plagiarism_score, merge_spans(highlights), sources, metadata


def ai_detection_pipeline(text: str) -> tuple[float, str, dict, list[AISpan]]:
    chunks = chunk_text(text, chunk_words=250, overlap_words=50)
    chunks = chunks or [TextChunk(id=0, text=text, start_char=0, end_char=len(text), word_count=len(text.split()))]

    chunk_features = []
    chunk_scores = []
    ai_spans: list[AISpan] = []
    for chunk in chunks:
        style = extract_stylometric_features(chunk.text)
        features = {
            **style,
            "roberta_score": roberta_ai_probability(chunk.text),
            "perplexity": perplexity(chunk.text),
        }
        score = predict_ai_score(features)
        chunk_features.append(features)
        chunk_scores.append(score)
        if score >= 0.62:
            reasons = []
            if features["roberta_score"] >= 0.65:
                reasons.append("RoBERTa classifier confidence is high")
            if features["perplexity"] and features["perplexity"] < 65:
                reasons.append("low GPT-2 perplexity")
            if features["sentence_length_variance"] < 30:
                reasons.append("uniform sentence rhythm")
            ai_spans.append(
                AISpan(
                    start=chunk.start_char,
                    end=chunk.end_char,
                    text=chunk.text,
                    score=round(score * 100, 2),
                    explanation=", ".join(reasons) or "ensemble AI probability is elevated",
                )
            )

    avg_features = {
        key: sum(item.get(key, 0.0) for item in chunk_features) / len(chunk_features)
        for key in chunk_features[0]
    }
    ai_score = round((sum(chunk_scores) / len(chunk_scores)) * 100, 2)
    explanation = explain_ai(avg_features, ai_score / 100)
    return ai_score, explanation, avg_features, ai_spans


async def analyze_document(text: str, document_id: str | None = None, filename: str | None = None) -> AnalyzeResponse:
    cleaned = clean_text(text)
    plagiarism_score, highlights, sources, plagiarism_metadata = await plagiarism_pipeline(cleaned)
    ai_score, explanation, ai_features, ai_spans = ai_detection_pipeline(cleaned)

    chunks_count = len(chunk_text(cleaned, chunk_words=250, overlap_words=50))
    return AnalyzeResponse(
        plagiarism_score=plagiarism_score,
        ai_score=ai_score,
        highlighted_text_spans=highlights,
        ai_highlighted_spans=ai_spans,
        source_urls=sources,
        ai_explanation=explanation,
        chunks_analyzed=chunks_count,
        metadata={
            "document_id": document_id,
            "filename": filename,
            "plagiarism": plagiarism_metadata,
            "ai_features": ai_features,
            "device": get_settings().device,
        },
    )
