from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from transformers import pipeline


PROMPT_TEMPLATES = [
    "Write a concise academic explanation of the following topic:\n\n{topic}",
    "Summarize this passage in a polished student essay style:\n\n{topic}",
    "Create a short undergraduate-level response about:\n\n{topic}",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a starter human/AI text dataset.")
    parser.add_argument("--output", default="data/processed/ai_human.csv")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--generator", default="distilgpt2", help="Local HF text-generation model.")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    wiki = load_dataset("wikipedia", "20220301.en", split=f"train[:{args.samples}]")
    generator = pipeline(
        "text-generation",
        model=args.generator,
        device=0,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.85,
        top_p=0.95,
    )

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        for idx, row in enumerate(wiki):
            human_text = str(row["text"]).strip().replace("\n", " ")
            human_text = " ".join(human_text.split()[:240])
            if len(human_text.split()) < 80:
                continue
            writer.writerow({"text": human_text, "label": 0})

            prompt = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)].format(topic=human_text[:700])
            generated = generator(prompt)[0]["generated_text"]
            ai_text = generated.replace(prompt, "").strip()
            if len(ai_text.split()) >= 50:
                writer.writerow({"text": ai_text, "label": 1})


if __name__ == "__main__":
    main()

