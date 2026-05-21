from io import BytesIO
import re

import docx
import fitz
import spacy

from .config import get_settings

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        settings = get_settings()
        try:
            _nlp = spacy.load(settings.spacy_model)
        except OSError:
            _nlp = spacy.blank("en")
            _nlp.add_pipe("sentencizer")
    return _nlp


def clean_text(text: str, *, lowercase: bool = False, remove_references: bool = True) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if remove_references:
        text = re.split(r"\n\s*(references|bibliography|works cited)\s*\n", text, flags=re.I)[0]

    if lowercase:
        text = text.lower()
    return text


def extract_text_from_pdf(content: bytes) -> str:
    with fitz.open(stream=content, filetype="pdf") as pdf:
        pages = [page.get_text("text") for page in pdf]
    return clean_text("\n".join(pages))


def extract_text_from_docx(content: bytes) -> str:
    document = docx.Document(BytesIO(content))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return clean_text("\n".join(paragraphs))


def extract_text_from_upload(filename: str, content_type: str | None, content: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    content_type = content_type or ""

    if suffix == "pdf" or content_type == "application/pdf":
        return extract_text_from_pdf(content)
    if suffix == "docx" or "wordprocessingml.document" in content_type:
        return extract_text_from_docx(content)

    for encoding in ("utf-8", "utf-16", "cp1252"):
        try:
            return clean_text(content.decode(encoding))
        except UnicodeDecodeError:
            continue

    return clean_text(content.decode("utf-8", errors="ignore"))


def sentence_segment(text: str) -> list[str]:
    doc = get_nlp()(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

