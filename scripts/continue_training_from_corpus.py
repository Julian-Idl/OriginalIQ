from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue OriginalIQ training from the expanded research corpus.")
    parser.add_argument("--dataset-limit", type=int, default=22000)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--roberta-epochs", type=str, default="2")
    parser.add_argument("--roberta-batch-size", type=str, default="4")
    parser.add_argument("--gradient-accumulation-steps", type=str, default="4")
    parser.add_argument("--max-length", type=str, default="384")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-roberta", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    root = Path(__file__).resolve().parent.parent
    corpus_dir = root / "data" / "raw" / "corpus" / "research_papers"
    raw_dataset = root / "data" / "processed" / "research_ai_human.csv"
    balanced_dataset = root / "data" / "processed" / "research_ai_human_balanced.csv"

    if not args.skip_index:
        run([python, str(root / "ml_service" / "build_corpus_index.py"), "--input-dir", str(corpus_dir)])

    if not args.skip_generate:
        run([
            python,
            str(root / "scripts" / "build_research_ai_dataset.py"),
            "--limit",
            str(args.dataset_limit),
            "--batch-size",
            str(args.generation_batch_size),
        ])
        run([
            python,
            str(root / "scripts" / "prepare_balanced_ai_dataset.py"),
            "--input",
            str(raw_dataset),
            "--output",
            str(balanced_dataset),
        ])

    if not args.skip_roberta:
        cmd = [
            python,
            str(root / "ml_service" / "train_roberta.py"),
            "--data",
            str(balanced_dataset),
            "--output",
            str(root / "models" / "roberta-ai-detector"),
            "--base-model",
            str(root / "models" / "roberta-ai-detector"),
            "--epochs",
            args.roberta_epochs,
            "--batch-size",
            args.roberta_batch_size,
            "--gradient-accumulation-steps",
            args.gradient_accumulation_steps,
            "--max-length",
            args.max_length,
        ]
        if args.use_lora:
            cmd.append("--use-lora")
        if args.use_qlora:
            cmd.append("--use-qlora")
        if args.attn_implementation != "auto":
            cmd.extend(["--attn-implementation", args.attn_implementation])
        run(cmd)

    if not args.skip_ensemble:
        run([
            python,
            str(root / "ml_service" / "train_ensemble.py"),
            "--data",
            str(balanced_dataset),
            "--output",
            str(root / "models" / "ensemble.joblib"),
        ])


if __name__ == "__main__":
    main()
