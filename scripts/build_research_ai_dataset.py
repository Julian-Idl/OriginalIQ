from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def text_quality(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    alpha = sum(char.isalpha() for char in chars) / len(chars)
    digits = sum(char.isdigit() for char in chars) / len(chars)
    math_symbols = sum(char in "=<>±√∑∫∞≈≡κτλβδϕηζ¯′″" for char in chars) / len(chars)
    sentence_marks = len(re.findall(r"[.!?]", text))
    word_count = len(text.split())
    sentence_density = min(sentence_marks / max(word_count / 25, 1), 1.0)
    return alpha + (0.25 * sentence_density) - digits - (1.5 * math_symbols)


def is_good_prose(text: str, min_words: int, max_math_ratio: float = 0.035) -> bool:
    words = text.split()
    if len(words) < min_words:
        return False
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return False
    alpha_ratio = sum(char.isalpha() for char in chars) / len(chars)
    digit_ratio = sum(char.isdigit() for char in chars) / len(chars)
    math_ratio = sum(char in "=<>±√∑∫∞≈≡κτλβδϕηζ¯′″" for char in chars) / len(chars)
    punctuation_runs = len(re.findall(r"(?:\b\d+(?:\.\d+)?\b\s*){8,}", text))
    return alpha_ratio >= 0.62 and digit_ratio <= 0.14 and math_ratio <= max_math_ratio and punctuation_runs == 0


def word_window(text: str, target_words: int, source_name: str, min_words: int) -> str:
    words = text.split()
    if len(words) <= target_words:
        return text

    candidates: list[str] = []
    seed = sum(ord(char) for char in source_name)
    max_start = max(len(words) - target_words, 0)
    lower = min(max(len(words) // 8, 0), max_start)
    upper = min(max(len(words) * 4 // 5, lower), max_start)
    span = max(upper - lower, 1)

    for index in range(12):
        start = lower + ((seed + index * 997) % span)
        candidates.append(" ".join(words[start : start + target_words]))

    prose_candidates = [candidate for candidate in candidates if is_good_prose(candidate, min_words)]
    if prose_candidates:
        return max(prose_candidates, key=text_quality)
    return max(candidates, key=text_quality)


def load_texts(corpus_dir: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if len(text.split()) < 300:
            continue
        samples.append((path.name, text))
    return samples


def generate_batch(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    max_new_tokens: int,
    min_new_tokens: int,
    num_candidates: int,
) -> list[list[str]]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=0.72,
            top_p=0.88,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
            num_return_sequences=num_candidates,
        )

    decoded = [tokenizer.decode(output_ids, skip_special_tokens=True).strip() for output_ids in outputs]
    grouped = []
    for index in range(0, len(decoded), num_candidates):
        grouped.append(decoded[index : index + num_candidates])
    return grouped


def best_generation(candidates: list[str], min_words: int) -> str:
    good = [candidate for candidate in candidates if is_good_prose(candidate, min_words, max_math_ratio=0.02)]
    if not good:
        return ""
    return max(good, key=text_quality)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a research-paper AI vs human dataset.")
    parser.add_argument("--corpus-dir", default="data/raw/corpus/research_papers")
    parser.add_argument("--output", default="data/processed/research_ai_human.csv")
    parser.add_argument("--generator", default="google/flan-t5-small")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--human-words", type=int, default=260, help="Human excerpt length written to CSV.")
    parser.add_argument("--min-human-words", type=int, default=160)
    parser.add_argument("--min-ai-words", type=int, default=120)
    parser.add_argument("--ai-max-new-tokens", type=int, default=300)
    parser.add_argument("--ai-min-new-tokens", type=int, default=160)
    parser.add_argument("--num-candidates", type=int, default=2, help="Sample multiple AI rewrites and keep the cleanest.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output CSV instead of appending new sources.")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    samples = load_texts(corpus_dir)[: args.limit]
    if not samples:
        raise RuntimeError("No usable corpus samples found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.generator)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.generator).to(device)
    model.eval()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    output_path = Path(args.output)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                existing.add(row.get("source_file", ""))

    mode = "a" if output_path.exists() else "w"
    with open(output_path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "source_file"])
        if mode == "w":
            writer.writeheader()

        batch_prompts = []
        batch_rows = []
        written = 0
        for filename, human_text in samples:
            if filename in existing:
                continue
            human_excerpt = word_window(human_text, args.human_words, filename, args.min_human_words)
            if not is_good_prose(human_excerpt, args.min_human_words):
                continue
            prompt = (
                "Rewrite the following research paper excerpt as a coherent academic paragraph of 170 to 230 words. "
                "Preserve the technical meaning and vocabulary, avoid lists, equations, tables, and citations, "
                "and write in fluent prose:\n\n"
                f"{human_excerpt}"
            )
            batch_prompts.append(prompt)
            batch_rows.append((filename, human_excerpt))

            if len(batch_prompts) < args.batch_size:
                continue

            generated_groups = generate_batch(
                model,
                tokenizer,
                device,
                batch_prompts,
                args.ai_max_new_tokens,
                args.ai_min_new_tokens,
                args.num_candidates,
            )

            for (source_file, human_text), candidates in zip(batch_rows, generated_groups):
                generated = best_generation(candidates, args.min_ai_words)
                if not generated:
                    continue
                writer.writerow({"text": human_text, "label": 0, "source_file": source_file})
                writer.writerow({"text": generated, "label": 1, "source_file": source_file})
                written += 2

            handle.flush()
            batch_prompts.clear()
            batch_rows.clear()
            print(f"Wrote {written} rows so far")

        if batch_prompts:
            generated_groups = generate_batch(
                model,
                tokenizer,
                device,
                batch_prompts,
                args.ai_max_new_tokens,
                args.ai_min_new_tokens,
                args.num_candidates,
            )
            for (source_file, human_text), candidates in zip(batch_rows, generated_groups):
                generated = best_generation(candidates, args.min_ai_words)
                if not generated:
                    continue
                writer.writerow({"text": human_text, "label": 0, "source_file": source_file})
                writer.writerow({"text": generated, "label": 1, "source_file": source_file})

    print(f"Saved dataset to {args.output}")


if __name__ == "__main__":
    main()
