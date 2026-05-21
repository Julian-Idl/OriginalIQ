# OriginalIQ: Plagiarism and AI Detection

Full-stack NLP project based on `pipeline.txt`.

## Architecture

- `frontend/`: Next.js + React + Tailwind UI for upload, analysis, plagiarism/AI highlights, source evidence, and report download.
- `backend/`: Node.js Express API gateway with `/upload` and `/analyze`.
- `ml_service/`: FastAPI ML service with the requested pipeline modules.
- `infra/postgres/`: PostgreSQL schema for documents and analysis results.
- `storage/`: FAISS index, metadata, and embedding cache.
- `models/`: fine-tuned RoBERTa and ensemble artifacts.

## Runtime Flow

1. Upload PDF, DOCX, TXT, or pasted text through the frontend.
2. Express `/upload` forwards files to FastAPI `/extract`, stores extracted text in PostgreSQL, and returns a `document_id`.
3. Express `/analyze` sends text to FastAPI `/analyze`.
4. FastAPI chunks text into 250-word windows with 50-word overlap.
5. SBERT `all-mpnet-base-v2` embeddings search FAISS.
6. If local similarity is below `0.75`, SerpAPI web search is triggered and the top URLs are scraped.
7. Cross-encoder `ms-marco-MiniLM-L-6-v2` reranks matches.
8. Cosine similarity, n-gram overlap, and LCS highlighting produce plagiarism spans and sources.
9. RoBERTa, GPT-2 perplexity, stylometry, and logistic regression produce the AI score and explanation.

## Setup

Create `.env` from the template and fill values that apply to you:

```powershell
Copy-Item .env.example .env
```

Required for complete web plagiarism detection:

- `SERPAPI_API_KEY`
- `DATABASE_URL` if not using the bundled Docker PostgreSQL default

Install everything on Windows:

```powershell
.\scripts\setup_windows.ps1
```

Start PostgreSQL:

```powershell
docker compose up -d postgres
```

If you already have local PostgreSQL installed, update `DATABASE_URL` in `.env` with the correct username and password, then run:

```powershell
cd backend
npm run db:setup
```

Start the ML service:

```powershell
.\.venv\Scripts\activate
cd ml_service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Start the backend:

```powershell
cd backend
npm run dev
```

Start the frontend:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## GPU Notes

The Python requirements use PyTorch CUDA 12.1 wheels. The ML service automatically chooses `cuda` when PyTorch can see your RTX 4050. Verify with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Training

Generate a starter AI-vs-human dataset from Wikipedia plus local HF text generation:

```powershell
.\.venv\Scripts\python.exe .\ml_service\generate_ai_dataset.py --samples 1000
```

Fine-tune RoBERTa:

```powershell
.\.venv\Scripts\python.exe .\ml_service\train_roberta.py --data data/processed/ai_human.csv --epochs 3 --batch-size 16
```

Train the logistic-regression ensemble:

```powershell
.\.venv\Scripts\python.exe .\ml_service\train_ensemble.py --data data/processed/ai_human.csv
```

Build a local plagiarism corpus index from PDFs, DOCX, and TXT files:

```powershell
mkdir data\raw\corpus
.\.venv\Scripts\python.exe .\ml_service\build_corpus_index.py --input-dir data/raw/corpus
```

The FAISS corpus starts empty. Web search results and any local corpus you build are stored in `storage/faiss.index` and `storage/faiss_metadata.jsonl`; delete those two files when you want a clean plagiarism index.

## API

Backend:

- `POST /upload`: multipart `file`, or form text field `text`.
- `POST /analyze`: JSON `{ "document_id": "..." }` or `{ "text": "..." }`.

Analysis JSON includes:

```json
{
  "plagiarism_score": 0,
  "ai_score": 0,
  "highlighted_text_spans": [],
  "ai_highlighted_spans": [],
  "source_urls": [],
  "ai_explanation": "",
  "chunks_analyzed": 0
}
```

## Reports

OriginalIQ can generate a PDF report from `GET /report/:analysisId` only when the analyzed document was uploaded as PDF or DOCX and has at least 5 pages. PDF page counts are read directly; DOCX page count is estimated from extracted word count.
