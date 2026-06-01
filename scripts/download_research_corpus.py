from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import fitz
import requests


ARXIV_DOMAINS = {
    "computer_science": ["cs.AI", "cs.CL", "cs.CV", "cs.DB", "cs.DC", "cs.HC", "cs.IR", "cs.LG", "cs.SE"],
    "mathematics": ["math.AG", "math.AP", "math.CO", "math.DS", "math.FA", "math.NA", "math.NT", "math.PR", "math.ST"],
    "physics": ["physics.app-ph", "physics.comp-ph", "physics.data-an", "physics.flu-dyn", "physics.optics", "hep-ph", "quant-ph"],
    "statistics": ["stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.TH"],
    "electrical_engineering": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"],
    "quantitative_biology": ["q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.NC", "q-bio.QM", "q-bio.TO"],
    "quantitative_finance": ["q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.RM", "q-fin.ST"],
    "economics": ["econ.EM", "econ.GN", "econ.TH"],
    "astro_physics": ["astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE", "astro-ph.IM", "astro-ph.SR"],
    "condensed_matter": ["cond-mat.mtrl-sci", "cond-mat.soft", "cond-mat.stat-mech", "cond-mat.str-el", "cond-mat.supr-con"],
    "nonlinear_sciences": ["nlin.AO", "nlin.CD", "nlin.CG", "nlin.PS", "nlin.SI"],
    "general_relativity": ["gr-qc"],
    "high_energy_physics": ["hep-ex", "hep-lat", "hep-ph", "hep-th"],
    "nuclear_science": ["nucl-ex", "nucl-th"],
    "systems_control": ["eess.SY", "cs.SY", "math.OC"],
    "information_theory": ["cs.IT", "math.IT"],
    "machine_learning": ["cs.LG", "stat.ML", "cs.AI"],
    "natural_language_processing": ["cs.CL", "cs.IR", "cs.AI"],
    "computer_vision": ["cs.CV", "eess.IV", "cs.LG"],
    "robotics": ["cs.RO", "cs.AI", "eess.SY"],
}

DEFAULT_DOMAINS = list(ARXIV_DOMAINS.keys())

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
    "User-Agent": "OriginalIQ-research-corpus/1.0 (local academic project; respectful arXiv API use)"
}

SESSION = requests.Session()


def request_with_backoff(url: str, *, headers: dict[str, str], timeout: int, attempts: int = 6) -> requests.Response:
    delay = 5.0
    for attempt in range(1, attempts + 1):
        response = SESSION.get(url, headers=headers, timeout=timeout)
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response

        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                sleep_for = float(retry_after)
            except ValueError:
                sleep_for = delay
        else:
            sleep_for = delay

        sleep_for = min(sleep_for, 120.0) + random.uniform(0.0, 1.5)
        print(f"Rate limited or transient error on {url}. Sleeping {sleep_for:.1f}s before retry {attempt}/{attempts}.")
        time.sleep(sleep_for)
        delay = min(delay * 2.0, 120.0)

    response.raise_for_status()
    return response


def clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def pdf_text(content: bytes, max_pages: int | None = None) -> str:
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            page_count = pdf.page_count if max_pages is None else min(max_pages, pdf.page_count)
            pages = [pdf[index].get_text("text") for index in range(page_count)]
        return clean_text("\n".join(pages))
    except Exception:
        return ""


def load_existing(metadata_path: Path, output_dir: Path) -> tuple[list[dict[str, str]], set[str], int]:
    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    max_id = -1

    if metadata_path.exists():
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
                if row.get("url"):
                    seen_urls.add(row["url"])
                try:
                    max_id = max(max_id, int(row.get("paper_id", "-1")))
                except ValueError:
                    pass

    if max_id < 0:
        for path in output_dir.glob("*.txt"):
            match = re.search(r"_(\d+)\.txt$", path.name)
            if match:
                max_id = max(max_id, int(match.group(1)))

    return rows, seen_urls, max_id + 1


def write_metadata(metadata_path: Path, rows: list[dict[str, str]]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "source", "domain", "filename", "title", "url", "character_count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_paper(
    output_dir: Path,
    rows: list[dict[str, str]],
    seen_urls: set[str],
    source: str,
    domain: str,
    paper_id: int,
    text: str,
    title: str,
    url: str,
    max_chars: int = 0,
) -> tuple[int, bool]:
    if url in seen_urls:
        return paper_id, False

    text = clean_text(text)
    if max_chars > 0:
        text = text[:max_chars]
    if len(text.split()) < 1000:
        return paper_id, False

    filename = f"{source}_{slug(domain)}_{paper_id:07d}.txt"
    (output_dir / filename).write_text(text, encoding="utf-8")
    rows.append(
        {
            "paper_id": str(paper_id),
            "source": source,
            "domain": domain,
            "filename": filename,
            "title": title[:500],
            "url": url,
            "character_count": str(len(text)),
        }
    )
    seen_urls.add(url)
    return paper_id + 1, True


def arxiv_entries(category: str, count: int, start: int) -> list[dict[str, str]]:
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=cat:{quote_plus(category)}"
        f"&start={start}&max_results={count}&sortBy=submittedDate&sortOrder=descending"
    )
    response = request_with_backoff(url, headers=HEADERS, timeout=60)
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


def download_arxiv_domain(
    output_dir: Path,
    rows: list[dict[str, str]],
    seen_urls: set[str],
    domain: str,
    target: int,
    pdf_limit: int,
    paper_id: int,
    batch_size: int,
    request_delay: float,
    pdf_delay: float,
    start_offset: int,
    metadata_path: Path,
    max_pdf_pages: int | None,
    max_chars: int,
) -> tuple[int, int]:
    categories = ARXIV_DOMAINS[domain]
    existing = sum(1 for row in rows if row.get("source") == "arxiv" and row.get("domain") == domain)
    saved = existing
    category_index = 0
    offset_round = start_offset

    print(f"\n=== Domain: {domain} target={target} existing={existing} categories={','.join(categories)} ===")
    if saved >= target:
        print(f"Skipping {domain}: already have {saved}/{target} papers.")
        return paper_id, 0

    while saved < target:
        category = categories[category_index % len(categories)]
        start = offset_round * batch_size
        print(f"Fetching {domain} / {category} start={start} saved={saved}/{target}")
        entries = arxiv_entries(category, batch_size, start)
        if not entries:
            print(f"No more entries for {category} at start={start}")
            category_index += 1
            if category_index % len(categories) == 0:
                offset_round += 1
            if offset_round > start_offset + 1000:
                break
            continue

        for entry in entries:
            if saved >= target:
                break
            if entry["page_url"] in seen_urls:
                continue
            if not entry["pdf_url"]:
                continue

            try:
                pdf = request_with_backoff(entry["pdf_url"], headers=HEADERS, timeout=120)
                full_text = pdf_text(pdf.content, max_pages=max_pdf_pages)
                time.sleep(pdf_delay)
            except Exception as error:
                print(f"PDF failed: {entry['pdf_url']} ({error})")
                continue

            text = "\n\n".join(part for part in [entry["title"], full_text] if part)
            paper_id, did_save = save_paper(
                output_dir,
                rows,
                seen_urls,
                "arxiv",
                domain,
                paper_id,
                text,
                entry["title"],
                entry["page_url"],
                max_chars=max_chars,
            )
            if did_save:
                saved += 1
                if saved % 100 == 0:
                    write_metadata(metadata_path, rows)
                    print(f"Saved checkpoint: {domain} {saved}/{target}")

        category_index += 1
        if category_index % len(categories) == 0:
            offset_round += 1
        time.sleep(request_delay)

    write_metadata(metadata_path, rows)
    return paper_id, saved - existing


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


def download_pubmed(
    output_dir: Path,
    rows: list[dict[str, str]],
    seen_urls: set[str],
    total: int,
    paper_id: int,
    metadata_path: Path,
) -> tuple[int, int]:
    saved = 0
    per_topic = max(total // len(PUBMED_TOPICS), 1)
    for topic in PUBMED_TOPICS:
        print(f"Fetching PubMed topic {topic}...")
        ids = pubmed_ids(topic, per_topic)
        time.sleep(0.35)
        for record in pubmed_records(ids):
            text = "\n\n".join(part for part in [record["title"], record["abstract"]] if part)
            paper_id, did_save = save_paper(output_dir, rows, seen_urls, "pubmed", "biomedicine", paper_id, text, record["title"], record["url"])
            if did_save:
                saved += 1
        write_metadata(metadata_path, rows)
        time.sleep(0.35)
    return paper_id, saved


def parse_domains(value: str) -> list[str]:
    if value == "default20":
        return DEFAULT_DOMAINS
    domains = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [domain for domain in domains if domain not in ARXIV_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown domains: {unknown}. Available: {sorted(ARXIV_DOMAINS)}")
    return domains


def main() -> None:
    parser = argparse.ArgumentParser(description="Append domain-balanced full-text research papers from arXiv and PubMed.")
    parser.add_argument("--domains", default="default20", help="Comma-separated domain keys, or default20.")
    parser.add_argument("--per-domain", type=int, default=1000)
    parser.add_argument(
        "--arxiv-pdf-limit-per-domain",
        type=int,
        default=0,
        help="Deprecated; full-text mode now downloads PDFs for every saved arXiv paper.",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--request-delay", type=float, default=7.5, help="Delay between arXiv API calls.")
    parser.add_argument("--pdf-delay", type=float, default=1.2, help="Delay between arXiv PDF requests.")
    parser.add_argument("--max-pdf-pages", type=int, default=0, help="Maximum PDF pages to extract; 0 means all pages.")
    parser.add_argument("--max-chars", type=int, default=0, help="Maximum characters to save per paper; 0 means no truncation.")
    parser.add_argument("--start-offset", type=int, default=0, help="Use a larger offset to skip newer arXiv pages.")
    parser.add_argument("--pubmed-samples", type=int, default=0, help="Optional additional PubMed abstracts.")
    parser.add_argument("--output-dir", default="data/raw/corpus/research_papers")
    parser.add_argument("--metadata", default="data/processed/research_corpus_metadata.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata)
    max_pdf_pages = args.max_pdf_pages if args.max_pdf_pages > 0 else None

    rows, seen_urls, paper_id = load_existing(metadata_path, output_dir)
    domains = parse_domains(args.domains)
    summary = {"starting_rows": len(rows), "domains": {}, "pubmed_saved": 0}

    for domain in domains:
        paper_id, saved = download_arxiv_domain(
            output_dir=output_dir,
            rows=rows,
            seen_urls=seen_urls,
            domain=domain,
            target=args.per_domain,
            pdf_limit=args.arxiv_pdf_limit_per_domain,
            paper_id=paper_id,
            batch_size=args.batch_size,
            request_delay=args.request_delay,
            pdf_delay=args.pdf_delay,
            start_offset=args.start_offset,
            metadata_path=metadata_path,
            max_pdf_pages=max_pdf_pages,
            max_chars=args.max_chars,
        )
        summary["domains"][domain] = saved

    if args.pubmed_samples:
        paper_id, pubmed_saved = download_pubmed(output_dir, rows, seen_urls, args.pubmed_samples, paper_id, metadata_path)
        summary["pubmed_saved"] = pubmed_saved

    write_metadata(metadata_path, rows)
    summary["ending_rows"] = len(rows)
    summary["new_rows"] = len(rows) - summary["starting_rows"]
    summary["output_dir"] = str(output_dir)
    summary["metadata"] = str(metadata_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
