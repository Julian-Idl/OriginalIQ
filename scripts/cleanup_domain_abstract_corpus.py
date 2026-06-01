from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove the domain-balanced abstract-only arXiv corpus run.")
    parser.add_argument("--metadata", default="data/processed/research_corpus_metadata.csv")
    parser.add_argument("--corpus-dir", default="data/raw/corpus/research_papers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    corpus_dir = Path(args.corpus_dir)

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    keep_rows: list[dict[str, str]] = []
    remove_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("source") == "arxiv" and row.get("domain"):
            remove_rows.append(row)
        else:
            keep_rows.append(row)

    missing = 0
    deleted = 0
    for row in remove_rows:
        filename = row.get("filename", "")
        if not filename:
            continue
        path = corpus_dir / filename
        if not path.exists():
            missing += 1
            continue
        if not args.dry_run:
            path.unlink()
        deleted += 1

    if not args.dry_run:
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["paper_id", "source", "domain", "filename", "title", "url", "character_count"],
            )
            writer.writeheader()
            writer.writerows(keep_rows)

    print(
        {
            "metadata_rows_before": len(rows),
            "metadata_rows_after": len(keep_rows),
            "rows_removed": len(remove_rows),
            "files_deleted": deleted,
            "files_missing": missing,
            "dry_run": args.dry_run,
        }
    )


if __name__ == "__main__":
    main()
