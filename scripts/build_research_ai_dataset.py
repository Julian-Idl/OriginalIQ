from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_texts(corpus_dir: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if len(text.split()) < 150:
            continue
        samples.append((path.name, text))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a research-paper AI vs human dataset.")
    parser.add_argument("--corpus-dir", default="data/raw/corpus/research_papers")
    parser.add_argument("--output", default="data/processed/research_ai_human.csv")
    parser.add_argument("--generator", default="google/flan-t5-small")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
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
            prompt = (
                "Rewrite the following research paper excerpt in a polished, original academic style "
                "while preserving the meaning and domain vocabulary:\n\n"
                f"{human_text[:1400]}"
            )
            batch_prompts.append(prompt)
            batch_rows.append((filename, human_text))

            if len(batch_prompts) < args.batch_size:
                continue

            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.92,
                    num_return_sequences=1,
                )

            for (source_file, human_text), output_ids in zip(batch_rows, outputs):
                generated = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                if len(generated.split()) < 20:
                    continue
                writer.writerow({"text": human_text, "label": 0, "source_file": source_file})
                writer.writerow({"text": generated, "label": 1, "source_file": source_file})
                written += 2

            handle.flush()
            batch_prompts.clear()
            batch_rows.clear()
            print(f"Wrote {written} rows so far")

        if batch_prompts:
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.92,
                    num_return_sequences=1,
                )
            for (source_file, human_text), output_ids in zip(batch_rows, outputs):
                generated = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                if len(generated.split()) < 20:
                    continue
                writer.writerow({"text": human_text, "label": 0, "source_file": source_file})
                writer.writerow({"text": generated, "label": 1, "source_file": source_file})

    print(f"Saved dataset to {args.output}")


if __name__ == "__main__":
    main()
