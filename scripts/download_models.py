from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_service"))

from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"Using device: {settings.device}")

    print(f"Downloading Sentence-BERT: {settings.sentence_model}")
    SentenceTransformer(settings.sentence_model, device=settings.device)

    print(f"Downloading cross-encoder: {settings.cross_encoder_model}")
    CrossEncoder(settings.cross_encoder_model, device=settings.device)

    print("Downloading GPT-2 perplexity model")
    AutoTokenizer.from_pretrained("gpt2")
    AutoModelForCausalLM.from_pretrained("gpt2")

    print("Downloading RoBERTa base for fine-tuning")
    AutoTokenizer.from_pretrained("roberta-base")

    print("Done")


if __name__ == "__main__":
    main()

