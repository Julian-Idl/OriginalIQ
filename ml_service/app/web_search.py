from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import httpx

from .config import get_settings
from .preprocessing import clean_text, sentence_segment


def build_queries(chunk_text: str, max_queries: int = 2) -> list[str]:
    sentences = sentence_segment(chunk_text)
    ranked = sorted(sentences, key=lambda item: len(item.split()), reverse=True)
    queries = []
    for sentence in ranked[:max_queries]:
        compact = re.sub(r"\s+", " ", sentence).strip()
        if len(compact) > 220:
            compact = compact[:220].rsplit(" ", 1)[0]
        if len(compact.split()) >= 8:
            queries.append(f'"{compact}"')
    if not queries:
        queries.append(f'"{chunk_text[:180].strip()}"')
    return queries


async def serpapi_search(query: str, limit: int = 5) -> list[dict]:
    settings = get_settings()
    if not settings.serpapi_api_key:
        return []

    params = {
        "engine": "google",
        "q": query,
        "api_key": settings.serpapi_api_key,
        "num": limit,
    }
    async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
        response = await client.get("https://serpapi.com/search.json", params=params)
        response.raise_for_status()
        data = response.json()

    results = []
    for item in data.get("organic_results", [])[:limit]:
        link = item.get("link")
        if not link:
            continue
        results.append(
            {
                "url": link,
                "title": item.get("title") or urlparse(link).netloc,
                "snippet": item.get("snippet"),
            }
        )
    return results


async def scrape_url(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 plagiarism-detector/1.0 (+local research project)",
        "Accept": "text/html,application/xhtml+xml",
    }
    async with httpx.AsyncClient(timeout=25, follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        element.decompose()
    text = soup.get_text("\n")
    return clean_text(text, remove_references=False)


async def web_candidates_for_chunk(chunk_text: str, limit: int = 5) -> list[dict]:
    seen_urls: set[str] = set()
    results: list[dict] = []

    for query in build_queries(chunk_text):
        for result in await serpapi_search(query, limit=limit):
            if result["url"] in seen_urls:
                continue
            seen_urls.add(result["url"])
            results.append(result)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    scraped: list[dict] = []
    for result in results:
        try:
            text = await scrape_url(result["url"])
        except Exception as error:
            result["error"] = str(error)
            text = ""
        if text:
            scraped.append({**result, "text": text})
    return scraped

