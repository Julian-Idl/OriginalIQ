from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import fitz
import requests


ARXIV_CATEGORIES = [
    "cs.CL",
    "cs.AI",
    "cs.LG",
    "stat.ML",
    "math.ST",
    "physics.comp-ph",
    "q-bio.QM",
    "econ.EM",
    "eess.SP",
    "astro-ph.IM",
]

PUBMED_TOPICS = [
    "machine learning medicine",
    "cancer immunotherapy",
    "neuroscience cognition",
    "genomics population study",
    "public health epidemiology",
    "clinical trial meta analysis",
    "medical imaging diagnosis",
    "drug discovery computational biology",
]

HEADERS = {
    "User-Agent": "plagiarism-detector-research-corpus/1.0 (local academic project)"
}


def clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def pdf_text(content: bytes, max_pages: int = 10) -> str:
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            pages = [pdf[index].get_text("text") for index in range(min(max_pages, pdf.page_count))]
        return clean_text("\n".join(pages))
    except Exception:
        return ""


def save_paper(output_dir: Path, rows: list[dict[str, str]], source: str, paper_id: int, text: str, title: str, url: str) -> int:
    text = clean_text(text)[:20000]
    if len(text.split()) < 100:
        return paper_id

    filename = f"{source}_{paper_id:06d}.txt"
    (output_dir / filename).write_text(text, encoding="utf-8")
    rows.append(
        {
            "paper_id": str(paper_id),
            "source": source,
            "filename": filename,
            "title": title[:500],
            "url": url,
            "character_count": str(len(text)),
        }
    )
    return paper_id + 1


def arxiv_entries(category: str, count: int) -> list[dict[str, str]]:
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=cat:{quote_plus(category)}"
        f"&start=0&max_results={count}&sortBy=submittedDate&sortOrder=descending"
    )
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = []

    for entry in root.findall("atom:entry", ns):
        title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
        page_url = clean_text(entry.findtext("atom:id", default="", namespaces=ns))
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        entries.append({"title": title, "summary": summary, "page_url": page_url, "pdf_url": pdf_url})
    return entries


def download_arxiv(output_dir: Path, rows: list[dict[str, str]], total: int, pdf_limit: int, paper_id: int) -> int:
    per_category = max(total // len(ARXIV_CATEGORIES), 1)
    pdf_downloaded = 0

    for category in ARXIV_CATEGORIES:
        print(f"Fetching arXiv category {category}...")
        for entry in arxiv_entries(category, per_category):
            full_text = ""
            if entry["pdf_url"] and pdf_downloaded < pdf_limit:
                try:
                    pdf = requests.get(entry["pdf_url"], headers=HEADERS, timeout=90)
                    pdf.raise_for_status()
                    full_text = pdf_text(pdf.content)
                    pdf_downloaded += 1
                    time.sleep(0.2)
                except Exception as error:
                    print(f"PDF failed: {entry['pdf_url']} ({error})")

            text = "\n\n".join(part for part in [entry["title"], entry["summary"], full_text] if part)
            next_id = save_paper(output_dir, rows, "arxiv", paper_id, text, entry["title"], entry["page_url"])
            paper_id = next_id
        time.sleep(3.0)

    return paper_id


def pubmed_ids(topic: str, count: int) -> list[str]:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={quote_plus(topic)}&retmode=json&retmax={count}&sort=relevance"
    )
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def pubmed_records(ids: list[str]) -> list[dict[str, str]]:
    if not ids:
        return []
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={','.join(ids)}&retmode=xml"
    )
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = ""
        title_node = article.find(".//ArticleTitle")
        if title_node is not None:
            title = clean_text(" ".join(title_node.itertext()))
        abstracts = []
        for abstract_text in article.findall(".//AbstractText"):
            abstracts.append(clean_text(" ".join(abstract_text.itertext())))
        abstract = clean_text(" ".join(abstracts))
        records.append(
            {
                "title": title,
                "abstract": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            }
        )
    return records


def download_pubmed(output_dir: Path, rows: list[dict[str, str]], total: int, paper_id: int) -> int:
    per_topic = max(total // len(PUBMED_TOPICS), 1)
    for topic in PUBMED_TOPICS:
        print(f"Fetching PubMed topic {topic}...")
        ids = pubmed_ids(topic, per_topic)
        time.sleep(0.35)
        for record in pubmed_records(ids):
            text = "\n\n".join(part for part in [record["title"], record["abstract"]] if part)
            paper_id = save_paper(output_dir, rows, "pubmed", paper_id, text, record["title"], record["url"])
        time.sleep(0.35)
    return paper_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a broad research-paper corpus from arXiv and PubMed.")
    parser.add_argument("--arxiv-samples", type=int, default=1200)
    parser.add_argument("--pubmed-samples", type=int, default=1200)
    parser.add_argument("--arxiv-pdf-limit", type=int, default=300)
    parser.add_argument("--output-dir", default="data/raw/corpus/research_papers")
    parser.add_argument("--metadata", default="data/processed/research_corpus_metadata.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    paper_id = 0
    paper_id = download_arxiv(output_dir, rows, args.arxiv_samples, args.arxiv_pdf_limit, paper_id)
    paper_id = download_pubmed(output_dir, rows, args.pubmed_samples, paper_id)

    with open(args.metadata, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "source", "filename", "title", "url", "character_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"papers_saved": len(rows), "output_dir": str(output_dir), "metadata": args.metadata}, indent=2))


if __name__ == "__main__":
    main()
