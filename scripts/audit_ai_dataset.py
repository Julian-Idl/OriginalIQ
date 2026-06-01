from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


MATH_CHARS = set("=<>+-*/^_{}[]()±√∑∫∞≈≡κτλβδϕηζ¯′″")


def stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p10": 0, "median": 0, "p90": 0, "max": 0}
    ordered = sorted(values)

    def percentile(position: float) -> float:
        index = min(int(position * (len(ordered) - 1)), len(ordered) - 1)
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "max": float(ordered[-1]),
    }


def quality_flags(text: str) -> list[str]:
    words = text.split()
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return ["empty"]
    alpha_ratio = sum(char.isalpha() for char in chars) / len(chars)
    digit_ratio = sum(char.isdigit() for char in chars) / len(chars)
    math_ratio = sum(char in MATH_CHARS for char in chars) / len(chars)
    flags = []
    if len(words) < 120:
        flags.append("short")
    if alpha_ratio < 0.62:
        flags.append("low_alpha")
    if digit_ratio > 0.14:
        flags.append("digit_heavy")
    if math_ratio > 0.08:
        flags.append("math_heavy")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OriginalIQ AI/human CSV quality.")
    parser.add_argument("--input", default="data/processed/research_ai_human_balanced.csv")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    path = Path(args.input)
    label_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    words_by_label: dict[str, list[int]] = {"0": [], "1": []}
    flagged_examples: list[dict[str, str]] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("label", "")
            text = row.get("text", "")
            label_counts[label] += 1
            words_by_label.setdefault(label, []).append(len(text.split()))
            flags = quality_flags(text)
            for flag in flags:
                flag_counts[f"label_{label}_{flag}"] += 1
            if flags and len(flagged_examples) < args.examples:
                flagged_examples.append(
                    {
                        "label": label,
                        "source_file": row.get("source_file", ""),
                        "flags": ",".join(flags),
                        "words": str(len(text.split())),
                        "text": text[:260].replace("\n", " "),
                    }
                )

    print({"path": str(path), "label_counts": dict(label_counts)})
    print({"word_stats": {label: stats(values) for label, values in words_by_label.items()}})
    print({"flag_counts": dict(flag_counts)})
    if flagged_examples:
        print("flagged_examples:")
        for example in flagged_examples:
            print(example)


if __name__ == "__main__":
    main()
