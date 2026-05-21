from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def load_csv_dataset(path: str) -> Dataset:
    dataset = load_dataset("csv", data_files=path)["train"]
    required = {"text", "label"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"Training CSV must include columns: {sorted(required)}")
    dataset = dataset.map(lambda row: {"label": int(row["label"])})
    return dataset.cast_column("label", ClassLabel(names=["human", "ai"]))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for AI vs human detection.")
    parser.add_argument("--data", default="data/processed/ai_human.csv", help="CSV with text,label columns.")
    parser.add_argument("--output", default="models/roberta-ai-detector")
    parser.add_argument("--base-model", default="roberta-base")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    dataset = load_csv_dataset(args.data).train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    remove_columns = [column for column in dataset["train"].column_names if column != "label"]
    tokenized = dataset.map(tokenize, batched=True, remove_columns=remove_columns)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        logging_steps=25,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["test"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
