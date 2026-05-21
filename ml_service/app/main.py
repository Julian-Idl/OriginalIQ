from fastapi import FastAPI, File, UploadFile

from .config import get_settings
from .pipeline import analyze_document
from .preprocessing import extract_text_from_upload
from .schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(
    title="Plagiarism and AI Detection ML Service",
    version="1.0.0",
    description="FastAPI service for document extraction, plagiarism search, and AI detection.",
)


@app.get("/health")
def health():
    settings = get_settings()
    return {
        "ok": True,
        "device": settings.device,
        "sentence_model": settings.sentence_model,
        "web_search_enabled": bool(settings.serpapi_api_key),
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_upload(file.filename or "upload.txt", file.content_type, content)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "character_count": len(text),
        "text": text,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    return await analyze_document(
        payload.text,
        document_id=payload.document_id,
        filename=payload.filename,
    )

