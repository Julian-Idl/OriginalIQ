from __future__ import annotations

import argparse
from pathlib import Path

from app.chunking import chunk_text
from app.embeddings import embed_texts
from app.preprocessing import extract_text_from_upload
from app.retrieval import get_store, records_from_texts


SUPPORTED = {".txt", ".pdf", ".docx"}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or extend the FAISS plagiarism corpus index.")
    parser.add_argument("--input-dir", default="data/raw/corpus")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    store = get_store()
    pending_texts: list[str] = []
    pending_records: list[dict] = []

    def flush():
        if not pending_texts:
            return
        vectors = embed_texts(pending_texts, batch_size=args.batch_size)
        store.add(vectors, pending_records, persist=True)
        pending_texts.clear()
        pending_records.clear()

    for path in iter_files(root):
        content = path.read_bytes()
        text = extract_text_from_upload(path.name, None, content)
        chunks = chunk_text(text, chunk_words=250, overlap_words=50)
        for chunk in chunks:
            pending_texts.append(chunk.text)
            pending_records.extend(records_from_texts([chunk.text], None, path.name))
            if len(pending_texts) >= args.batch_size:
                flush()

    flush()
    store.save()


if __name__ == "__main__":
    main()

