import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_service"))

from app.pipeline import analyze_document


SAMPLE_TEXT = """
Academic integrity depends on transparent authorship, careful citation, and a clear
distinction between original analysis and borrowed language. A plagiarism detection
system should compare submitted work with trusted local sources and web sources, but
it should also explain the matches it finds so that an instructor can review context.
AI detection should be treated as a probabilistic signal rather than an absolute
verdict, because writing style varies across authors, topics, and editing workflows.
"""


async def main() -> None:
    result = await analyze_document(SAMPLE_TEXT, filename="smoke-test.txt")
    print(
        {
            "plagiarism_score": result.plagiarism_score,
            "ai_score": result.ai_score,
            "chunks_analyzed": result.chunks_analyzed,
            "device": result.metadata.get("device"),
            "sources": len(result.source_urls),
            "highlights": len(result.highlighted_text_spans),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())

