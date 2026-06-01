from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download research papers and train the AI detector.")
    parser.add_argument("--corpus-arxiv", type=int, default=1200)
    parser.add_argument("--corpus-pubmed", type=int, default=1200)
    parser.add_argument("--dataset-limit", type=int, default=1000)
    args = parser.parse_args()

    python = sys.executable
    root = Path(__file__).resolve().parent.parent

    run([python, str(root / "scripts" / "download_research_corpus.py"), "--arxiv-samples", str(args.corpus_arxiv), "--pubmed-samples", str(args.corpus_pubmed)])
    run([python, str(root / "ml_service" / "build_corpus_index.py"), "--input-dir", str(root / "data" / "raw" / "corpus" / "research_papers")])
    run([python, str(root / "scripts" / "build_research_ai_dataset.py"), "--limit", str(args.dataset_limit)])
    run([python, str(root / "scripts" / "prepare_balanced_ai_dataset.py"), "--input", str(root / "data" / "processed" / "research_ai_human.csv"), "--output", str(root / "data" / "processed" / "research_ai_human_balanced.csv")])
    run([python, str(root / "ml_service" / "train_roberta.py"), "--data", str(root / "data" / "processed" / "research_ai_human_balanced.csv"), "--output", str(root / "models" / "roberta-ai-detector"), "--base-model", "roberta-base", "--epochs", "3", "--batch-size", "4", "--max-length", "384"])
    run([python, str(root / "ml_service" / "train_ensemble.py"), "--data", str(root / "data" / "processed" / "research_ai_human_balanced.csv"), "--output", str(root / "models" / "ensemble.joblib")])


if __name__ == "__main__":
    main()
