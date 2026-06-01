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


def maybe_apply_lora(model, args):
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise ImportError("LoRA training requires peft. Install it with: pip install peft") from error

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--use-lora", action="store_true", help="Fine-tune only LoRA adapters plus classifier head.")
    parser.add_argument("--use-qlora", action="store_true", help="Load the model in 4-bit and train LoRA adapters.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        help="Attention backend. flash_attention_2 needs a compatible local install.",
    )
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    dataset = load_csv_dataset(args.data).train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    remove_columns = [column for column in dataset["train"].column_names if column != "label"]
    tokenized = dataset.map(tokenize, batched=True, remove_columns=remove_columns)
    model_kwargs = {"num_labels": 2}
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.use_qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires CUDA.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise ImportError("QLoRA requires a Transformers build with BitsAndBytesConfig.") from error
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model, **model_kwargs)
    except (ImportError, ValueError) as error:
        if args.attn_implementation == "flash_attention_2":
            print(f"flash_attention_2 unavailable for this environment/model: {error}")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForSequenceClassification.from_pretrained(args.base_model, **model_kwargs)
        else:
            raise

    model = maybe_apply_lora(model, args)

    fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
