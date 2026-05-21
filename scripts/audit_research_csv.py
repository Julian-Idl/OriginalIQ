from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path("data/processed/research_ai_human_balanced.csv")
    print(f"exists={path.exists()} bytes={path.stat().st_size if path.exists() else 0}")
    if not path.exists():
        return

    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        print("fields=", reader.fieldnames)
        for row in reader:
            rows.append(row)

    labels = Counter(row.get("label") for row in rows)
    lengths = [len(row.get("text", "").split()) for row in rows]
    source_counts = Counter(row.get("source_file") for row in rows)

    print(f"rows={len(rows)}")
    print(f"labels={dict(labels)}")
    print(f"empty_text={sum(not row.get('text','').strip() for row in rows)}")
    print(f"empty_source={sum(not row.get('source_file','').strip() for row in rows)}")
    print(
        "word_stats="
        f"min={min(lengths)} "
        f"median={statistics.median(lengths)} "
        f"mean={round(statistics.mean(lengths), 1)} "
        f"max={max(lengths)}"
    )
    print(f"short_lt_20={sum(x < 20 for x in lengths)}")
    print(f"short_lt_50={sum(x < 50 for x in lengths)}")
    print(f"unique_sources={len(set(row.get('source_file') for row in rows))}")
    print(f"source_rows={dict(Counter(source_counts.values()))}")

    for label in sorted(set(row.get("label") for row in rows)):
        subset = [len(row["text"].split()) for row in rows if row.get("label") == label]
        if not subset:
            continue
        print(
            f"label_{label}_stats="
            f"n={len(subset)} "
            f"min={min(subset)} "
            f"median={statistics.median(subset)} "
            f"mean={round(statistics.mean(subset), 1)} "
            f"max={max(subset)}"
        )

    print("samples:")
    for row in rows[:2] + rows[-2:]:
        print("---")
        print(f"label={row.get('label')} source={row.get('source_file')} words={len(row.get('text','').split())}")
        print(row.get("text", "")[:500].replace("\n", " "))


if __name__ == "__main__":
    main()
