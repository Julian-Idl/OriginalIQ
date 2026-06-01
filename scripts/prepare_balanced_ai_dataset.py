from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re


def is_good_prose(text: str, min_words: int) -> bool:
    words = text.split()
    if len(words) < min_words:
        return False
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return False
    alpha_ratio = sum(char.isalpha() for char in chars) / len(chars)
    digit_ratio = sum(char.isdigit() for char in chars) / len(chars)
    math_ratio = sum(char in "=<>±√∑∫∞≈≡κτλβδϕηζ¯′″" for char in chars) / len(chars)
    numeric_run = len(re.findall(r"(?:\b\d+(?:\.\d+)?\b\s*){8,}", text))
    return alpha_ratio >= 0.62 and digit_ratio <= 0.14 and math_ratio <= 0.04 and numeric_run == 0


def word_window(text: str, target_words: int) -> str:
    words = text.split()
    if len(words) <= target_words:
        return text
    target_words = max(160, min(target_words, 300))
    start = min(max(len(words) // 5, 0), max(len(words) - target_words, 0))
    return " ".join(words[start : start + target_words])


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance AI/human research dataset lengths by source pair.")
    parser.add_argument("--input", default="data/processed/research_ai_human.csv")
    parser.add_argument("--output", default="data/processed/research_ai_human_balanced.csv")
    parser.add_argument("--min-words", type=int, default=140)
    parser.add_argument("--target-words", type=int, default=240)
    args = parser.parse_args()

    by_source: dict[str, dict[str, str]] = defaultdict(dict)
    with open(args.input, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_source[row["source_file"]][row["label"]] = row["text"]

    rows = []
    for source_file, pair in sorted(by_source.items()):
        human = pair.get("0", "").strip()
        ai = pair.get("1", "").strip()
        if not human or not ai:
            continue
        if not is_good_prose(ai, args.min_words):
            continue
        ai_words = min(max(len(ai.split()) + 20, args.target_words), 300)
        human_excerpt = word_window(human, ai_words)
        if not is_good_prose(human_excerpt, args.min_words):
            continue
        rows.append({"text": human_excerpt, "label": 0, "source_file": source_file})
        rows.append({"text": ai, "label": 1, "source_file": source_file})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote={len(rows)} sources={len(rows) // 2} output={args.output}")


if __name__ == "__main__":
    main()
