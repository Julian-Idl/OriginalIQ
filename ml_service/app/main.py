from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile

from .config import get_settings
from .pipeline import analyze_document
from .preprocessing import extract_text_from_upload, upload_metadata
from .schemas import AnalyzeRequest, AnalyzeResponse
from .warmup import warmup_models, warmup_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("OriginalIQ ML warmup starting...")
    status = warmup_models()
    print(f"OriginalIQ ML warmup complete in {status['duration_seconds']}s on {status['device']}.")
    yield

app = FastAPI(
    title="OriginalIQ ML Service",
    version="1.0.0",
    description="FastAPI service for document extraction, plagiarism search, and AI detection.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    settings = get_settings()
    return {
        "ok": True,
        "device": settings.device,
        "sentence_model": settings.sentence_model,
        "web_search_enabled": bool(settings.serpapi_api_key),
        "warmup": warmup_status(),
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_upload(file.filename or "upload.txt", file.content_type, content)
    metadata = upload_metadata(file.filename or "upload.txt", file.content_type, content, text)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "character_count": len(text),
        **metadata,
        "text": text,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    return await analyze_document(
        payload.text,
        document_id=payload.document_id,
        filename=payload.filename,
    )
