from dataclasses import dataclass
import re


@dataclass(slots=True)
class TextChunk:
    id: int
    text: str
    start_char: int
    end_char: int
    word_count: int


_WORD_RE = re.compile(r"\S+")


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 50) -> list[TextChunk]:
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return []

    chunks: list[TextChunk] = []
    step = max(chunk_words - overlap_words, 1)
    chunk_id = 0

    for start_idx in range(0, len(matches), step):
        end_idx = min(start_idx + chunk_words, len(matches))
        selected = matches[start_idx:end_idx]
        if not selected:
            break

        start_char = selected[0].start()
        end_char = selected[-1].end()
        chunks.append(
            TextChunk(
                id=chunk_id,
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                word_count=len(selected),
            )
        )
        chunk_id += 1
        if end_idx == len(matches):
            break

    return chunks

