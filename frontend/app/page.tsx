"use client";

import { useMemo, useState } from "react";
import {
  AlertCircle,
  Bot,
  FileText,
  Link as LinkIcon,
  Loader2,
  Search,
  ShieldCheck,
  Upload,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:4000";

type HighlightSpan = {
  start: number;
  end: number;
  text: string;
  source_url: string | null;
  score: number;
};

type SourceResult = {
  url: string | null;
  title: string | null;
  score: number;
  contribution: number;
  matched_text: string | null;
};

type AnalysisResult = {
  analysis_id: string;
  document_id: string | null;
  plagiarism_score: number;
  ai_score: number;
  highlighted_text_spans: HighlightSpan[];
  source_urls: SourceResult[];
  ai_explanation: string;
  chunks_analyzed: number;
  metadata: Record<string, unknown>;
};

type UploadResult = {
  document_id: string;
  filename: string;
  text_preview: string;
  text: string;
  character_count: number;
};

function scoreClass(score: number) {
  if (score >= 70) return "text-red-700";
  if (score >= 40) return "text-amber-700";
  return "text-emerald-700";
}

function renderHighlighted(text: string, spans: HighlightSpan[]) {
  if (!text || spans.length === 0) return text;
  const ordered = [...spans].sort((a, b) => a.start - b.start);
  const parts = [];
  let cursor = 0;
  for (const span of ordered) {
    if (span.start > cursor) parts.push(<span key={`plain-${cursor}`}>{text.slice(cursor, span.start)}</span>);
    parts.push(
      <mark
        key={`${span.start}-${span.end}`}
        title={`${span.score}% match${span.source_url ? ` - ${span.source_url}` : ""}`}
        className="rounded bg-amber-200 px-1 text-slate-900"
      >
        {text.slice(span.start, span.end)}
      </mark>
    );
    cursor = Math.max(cursor, span.end);
  }
  if (cursor < text.length) parts.push(<span key={`plain-${cursor}`}>{text.slice(cursor)}</span>);
  return parts;
}

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [uploaded, setUploaded] = useState<UploadResult | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const documentText = useMemo(() => text || uploaded?.text || "", [text, uploaded]);

  async function uploadDocument() {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      if (file) {
        form.append("file", file);
      } else {
        form.append("text", text);
        form.append("filename", "pasted-text.txt");
      }
      const response = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Upload failed");
      setUploaded(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function analyze() {
    setBusy(true);
    setError("");
    try {
      const body = uploaded?.document_id ? { document_id: uploaded.document_id } : { text };
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Analysis failed");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 text-ink">
      <section className="grid min-h-screen grid-cols-1 lg:grid-cols-[minmax(300px,380px)_1fr]">
        <aside className="flex flex-col gap-5 border-b border-line bg-white p-6 lg:border-b-0 lg:border-r">
          <div className="flex items-center gap-3">
            <ShieldCheck size={28} />
            <div>
              <h1 className="text-[22px] font-semibold leading-tight">Integrity Lab</h1>
              <p className="text-sm text-slate-500">Plagiarism and AI detection</p>
            </div>
          </div>

          <label className="grid min-h-28 cursor-pointer place-items-center gap-2 rounded-lg border border-dashed border-slate-400 bg-slate-50 p-5 text-center text-slate-700">
            <Upload size={24} />
            <span>{file ? file.name : "Upload PDF, DOCX, or TXT"}</span>
            <input
              className="hidden"
              type="file"
              accept=".pdf,.docx,.txt,text/plain,application/pdf"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </label>

          <textarea
            className="min-h-[230px] resize-y rounded-lg border border-slate-300 bg-slate-50 p-4 outline-none focus:border-brand focus:ring-4 focus:ring-sky-100"
            placeholder="Or paste text here..."
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setUploaded(null);
            }}
          />

          {error && (
            <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={uploadDocument}
              disabled={busy || (!file && text.trim().length < 20)}
              className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-brand text-white disabled:bg-slate-400"
            >
              {busy ? <Loader2 className="animate-spin" size={18} /> : <FileText size={18} />}
              Upload
            </button>
            <button
              type="button"
              onClick={analyze}
              disabled={busy || (!uploaded && text.trim().length < 20)}
              className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-brand text-white disabled:bg-slate-400"
            >
              {busy ? <Loader2 className="animate-spin" size={18} /> : <Search size={18} />}
              Analyze
            </button>
          </div>

          {uploaded && (
            <div className="grid gap-1 text-sm text-slate-600">
              <strong>{uploaded.filename}</strong>
              <span>{uploaded.character_count.toLocaleString()} characters indexed</span>
            </div>
          )}
        </aside>

        <section className="flex flex-col gap-5 p-6">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="grid gap-2 rounded-lg border border-line bg-white p-4">
              <span className="text-xs uppercase text-slate-500">Plagiarism</span>
              <strong className={`text-3xl ${result ? scoreClass(result.plagiarism_score) : ""}`}>
                {result ? `${result.plagiarism_score}%` : "--"}
              </strong>
            </div>
            <div className="grid gap-2 rounded-lg border border-line bg-white p-4">
              <span className="text-xs uppercase text-slate-500">AI probability</span>
              <strong className={`text-3xl ${result ? scoreClass(result.ai_score) : ""}`}>
                {result ? `${result.ai_score}%` : "--"}
              </strong>
            </div>
            <div className="grid gap-2 rounded-lg border border-line bg-white p-4">
              <span className="text-xs uppercase text-slate-500">Chunks</span>
              <strong className="text-3xl">{result ? result.chunks_analyzed : "--"}</strong>
            </div>
          </div>

          <div className="grid min-h-0 grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(280px,0.8fr)]">
            <section className="min-w-0 rounded-lg border border-line bg-white p-5">
              <div className="mb-3 flex items-center gap-2">
                <FileText size={20} />
                <h2 className="text-lg font-semibold">Document</h2>
              </div>
              <div className="max-h-[calc(100vh-190px)] min-h-[360px] overflow-auto whitespace-pre-wrap leading-7 text-slate-700 lg:min-h-[560px]">
                {result
                  ? renderHighlighted(text || uploaded?.text || "", result.highlighted_text_spans)
                  : documentText || "Upload or paste text to begin."}
              </div>
            </section>

            <section className="min-w-0 rounded-lg border border-line bg-white p-5">
              <div className="mb-3 flex items-center gap-2">
                <Bot size={20} />
                <h2 className="text-lg font-semibold">AI Detection</h2>
              </div>
              <p className="leading-7 text-slate-700">
                {result?.ai_explanation || "Analysis will combine RoBERTa, GPT-2 perplexity, and stylometric features."}
              </p>

              <div className="mb-3 mt-7 flex items-center gap-2">
                <LinkIcon size={20} />
                <h2 className="text-lg font-semibold">Sources</h2>
              </div>
              <div className="grid gap-3">
                {result?.source_urls.length ? (
                  result.source_urls.map((source, index) => (
                    <a
                      className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 rounded-lg border border-line bg-slate-50 p-3 text-slate-700 no-underline"
                      key={`${source.url}-${index}`}
                      href={source.url || undefined}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <span className="break-words">{source.title || source.url || "Local corpus"}</span>
                      <strong>{source.score}%</strong>
                      <small className="text-slate-500">{source.contribution}% contribution</small>
                    </a>
                  ))
                ) : (
                  <p className="leading-6 text-slate-500">No external matches have been returned yet.</p>
                )}
              </div>
            </section>
          </div>
        </section>
      </section>
    </main>
  );
}

