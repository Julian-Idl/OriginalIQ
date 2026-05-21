"use client";

import { useMemo, useState } from "react";
import {
  AlertCircle,
  Bot,
  Download,
  FileSearch,
  FileText,
  Globe2,
  Link as LinkIcon,
  Loader2,
  Search,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:4000";

type HighlightSpan = {
  start: number;
  end: number;
  text: string;
  source_url: string | null;
  score: number;
  kind?: string;
  explanation?: string | null;
};

type AISpan = {
  start: number;
  end: number;
  text: string;
  score: number;
  explanation: string;
};

type SourceResult = {
  url: string | null;
  title: string | null;
  score: number;
  contribution: number;
  matched_text: string | null;
  matched_spans: number;
  evidence: string[];
  source_type: string;
};

type AnalysisResult = {
  analysis_id: string;
  document_id: string | null;
  report_available: boolean;
  plagiarism_score: number;
  ai_score: number;
  highlighted_text_spans: HighlightSpan[];
  ai_highlighted_spans: AISpan[];
  source_urls: SourceResult[];
  ai_explanation: string;
  chunks_analyzed: number;
  metadata: {
    plagiarism?: {
      web_search_enabled?: boolean;
      web_search_triggered_for_chunks?: number;
      web_searched_chunks?: number;
      matches?: number;
    };
    document?: {
      page_count?: number;
      document_kind?: string;
      report_eligible?: boolean;
    };
    [key: string]: unknown;
  };
};

type UploadResult = {
  document_id: string;
  filename: string;
  text_preview: string;
  text: string;
  character_count: number;
  page_count: number;
  document_kind: string;
  report_eligible: boolean;
};

type RenderSpan = {
  start: number;
  end: number;
  score: number;
  kind: "plagiarism" | "ai";
  title: string;
};

function scoreClass(score: number) {
  if (score >= 70) return "text-red-700";
  if (score >= 40) return "text-amber-700";
  return "text-emerald-700";
}

function scoreBand(score: number) {
  if (score >= 70) return "High";
  if (score >= 40) return "Review";
  return "Low";
}

function renderHighlighted(text: string, plagiarism: HighlightSpan[], ai: AISpan[]) {
  if (!text) return "Upload or paste text to begin.";
  const spans: RenderSpan[] = [
    ...plagiarism.map((span) => ({
      start: span.start,
      end: span.end,
      score: span.score,
      kind: "plagiarism" as const,
      title: `${span.score}% match${span.source_url ? ` - ${span.source_url}` : ""}${span.explanation ? ` | ${span.explanation}` : ""}`,
    })),
    ...ai.map((span) => ({
      start: span.start,
      end: span.end,
      score: span.score,
      kind: "ai" as const,
      title: `${span.score}% AI-like | ${span.explanation}`,
    })),
  ].sort((a, b) => a.start - b.start || (a.kind === "plagiarism" ? -1 : 1));

  if (!spans.length) return text;

  const parts = [];
  let cursor = 0;
  spans.forEach((span, index) => {
    const start = Math.max(span.start, cursor);
    const end = Math.max(span.end, start);
    if (start > cursor) parts.push(<span key={`plain-${cursor}`}>{text.slice(cursor, start)}</span>);
    if (end > start) {
      const cls =
        span.kind === "plagiarism"
          ? "rounded bg-amber-200 px-1 text-slate-950"
          : "rounded bg-sky-200 px-1 text-slate-950";
      parts.push(
        <mark key={`${span.kind}-${index}-${start}-${end}`} title={span.title} className={cls}>
          {text.slice(start, end)}
        </mark>
      );
      cursor = end;
    }
  });
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
  const webStatus = result?.metadata?.plagiarism;
  const reportUrl = result?.report_available ? `${API_URL}/report/${result.analysis_id}` : "";

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
    <main className="min-h-screen bg-slate-100 text-ink">
      <section className="grid min-h-screen grid-cols-1 xl:grid-cols-[360px_1fr]">
        <aside className="border-b border-line bg-white p-5 xl:border-b-0 xl:border-r">
          <div className="mb-6 flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-lg bg-slate-900 text-white">
              <ShieldCheck size={24} />
            </div>
            <div>
              <h1 className="text-2xl font-semibold leading-tight">OriginalIQ</h1>
              <p className="text-sm text-slate-500">Academic originality workspace</p>
            </div>
          </div>

          <div className="grid gap-4">
            <label className="grid min-h-28 cursor-pointer place-items-center gap-2 rounded-lg border border-dashed border-slate-400 bg-slate-50 p-5 text-center text-slate-700">
              <Upload size={24} />
              <span className="font-medium">{file ? file.name : "Upload PDF, DOCX, or TXT"}</span>
              <small className="text-slate-500">PDF/DOCX reports unlock at 5+ pages</small>
              <input
                className="hidden"
                type="file"
                accept=".pdf,.docx,.txt,text/plain,application/pdf"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
            </label>

            <textarea
              className="min-h-[220px] resize-y rounded-lg border border-slate-300 bg-slate-50 p-4 outline-none focus:border-brand focus:ring-4 focus:ring-sky-100"
              placeholder="Or paste text here..."
              value={text}
              onChange={(event) => {
                setText(event.target.value);
                setUploaded(null);
                setResult(null);
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
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-slate-900 text-white disabled:bg-slate-400"
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

            {result?.report_available && (
              <a
                href={reportUrl}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white text-sm font-medium text-slate-800"
              >
                <Download size={18} />
                Download PDF report
              </a>
            )}

            {uploaded && (
              <div className="grid gap-2 rounded-lg border border-line bg-slate-50 p-4 text-sm text-slate-600">
                <strong className="break-words text-slate-900">{uploaded.filename}</strong>
                <span>{uploaded.character_count.toLocaleString()} characters</span>
                <span>
                  {uploaded.document_kind.toUpperCase()} · {uploaded.page_count || "unknown"} pages
                </span>
                <span className={uploaded.report_eligible ? "text-emerald-700" : "text-slate-500"}>
                  {uploaded.report_eligible ? "Report eligible" : "Report requires PDF/DOCX with 5+ pages"}
                </span>
              </div>
            )}
          </div>
        </aside>

        <section className="flex min-w-0 flex-col gap-5 p-5">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
            <div className="rounded-lg border border-line bg-white p-4">
              <span className="text-xs uppercase text-slate-500">Plagiarism</span>
              <div className="mt-2 flex items-end justify-between gap-3">
                <strong className={`text-3xl ${result ? scoreClass(result.plagiarism_score) : ""}`}>
                  {result ? `${result.plagiarism_score}%` : "--"}
                </strong>
                <small className="text-slate-500">{result ? scoreBand(result.plagiarism_score) : "Pending"}</small>
              </div>
            </div>
            <div className="rounded-lg border border-line bg-white p-4">
              <span className="text-xs uppercase text-slate-500">AI probability</span>
              <div className="mt-2 flex items-end justify-between gap-3">
                <strong className={`text-3xl ${result ? scoreClass(result.ai_score) : ""}`}>
                  {result ? `${result.ai_score}%` : "--"}
                </strong>
                <small className="text-slate-500">{result ? scoreBand(result.ai_score) : "Pending"}</small>
              </div>
            </div>
            <div className="rounded-lg border border-line bg-white p-4">
              <span className="text-xs uppercase text-slate-500">Web search</span>
              <div className="mt-2 flex items-center gap-2">
                <Globe2 size={22} className={webStatus?.web_search_enabled ? "text-emerald-700" : "text-slate-400"} />
                <strong>{result ? `${webStatus?.web_searched_chunks || 0} chunks` : "--"}</strong>
              </div>
            </div>
            <div className="rounded-lg border border-line bg-white p-4">
              <span className="text-xs uppercase text-slate-500">Evidence</span>
              <div className="mt-2 flex items-center gap-2">
                <FileSearch size={22} className="text-slate-700" />
                <strong>{result ? `${result.source_urls.length} sources` : "--"}</strong>
              </div>
            </div>
          </div>

          <div className="grid min-h-0 grid-cols-1 gap-5 2xl:grid-cols-[minmax(0,1.35fr)_minmax(360px,0.65fr)]">
            <section className="min-w-0 rounded-lg border border-line bg-white p-5">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <FileText size={20} />
                  <h2 className="text-lg font-semibold">Document Review</h2>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-600">
                  <span className="inline-flex items-center gap-1">
                    <span className="h-3 w-3 rounded bg-amber-200" /> plagiarism
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="h-3 w-3 rounded bg-sky-200" /> AI-like
                  </span>
                </div>
              </div>
              <div className="max-h-[calc(100vh-205px)] min-h-[520px] overflow-auto whitespace-pre-wrap rounded-lg border border-slate-200 bg-white p-4 leading-7 text-slate-700">
                {renderHighlighted(
                  text || uploaded?.text || "",
                  result?.highlighted_text_spans || [],
                  result?.ai_highlighted_spans || []
                )}
              </div>
            </section>

            <section className="grid min-w-0 gap-5">
              <div className="rounded-lg border border-line bg-white p-5">
                <div className="mb-3 flex items-center gap-2">
                  <Bot size={20} />
                  <h2 className="text-lg font-semibold">AI Detection</h2>
                </div>
                <p className="leading-7 text-slate-700">
                  {result?.ai_explanation || "Analysis combines fine-tuned RoBERTa, GPT-2 perplexity, stylometry, and an ensemble model."}
                </p>
                {result?.ai_highlighted_spans?.length ? (
                  <div className="mt-4 grid gap-2">
                    {result.ai_highlighted_spans.slice(0, 4).map((span, index) => (
                      <div key={`${span.start}-${index}`} className="rounded-lg bg-sky-50 p-3 text-sm text-slate-700">
                        <strong>{span.score}% AI-like</strong>
                        <p className="mt-1 text-slate-600">{span.explanation}</p>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>

              <div className="rounded-lg border border-line bg-white p-5">
                <div className="mb-3 flex items-center gap-2">
                  <LinkIcon size={20} />
                  <h2 className="text-lg font-semibold">Plagiarism Evidence</h2>
                </div>
                <div className="grid gap-3">
                  {result?.source_urls.length ? (
                    result.source_urls.map((source, index) => (
                      <a
                        className="grid gap-2 rounded-lg border border-line bg-slate-50 p-3 text-slate-700 no-underline"
                        key={`${source.url}-${index}`}
                        href={source.url || undefined}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <span className="break-words font-medium text-slate-900">{source.title || source.url || "Local corpus"}</span>
                          <strong className="shrink-0">{source.score}%</strong>
                        </div>
                        <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                          <span>{source.source_type}</span>
                          <span>{source.contribution}% contribution</span>
                          <span>{source.matched_spans} spans</span>
                        </div>
                        {source.evidence?.slice(0, 2).map((item, itemIndex) => (
                          <p key={itemIndex} className="line-clamp-3 text-sm leading-6 text-slate-600">
                            {item}
                          </p>
                        ))}
                      </a>
                    ))
                  ) : (
                    <p className="leading-6 text-slate-500">No plagiarized sources have been confirmed yet.</p>
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-line bg-white p-5">
                <div className="mb-3 flex items-center gap-2">
                  <Sparkles size={20} />
                  <h2 className="text-lg font-semibold">Run Notes</h2>
                </div>
                <ul className="grid gap-2 text-sm leading-6 text-slate-600">
                  <li>Web search enabled: {webStatus?.web_search_enabled ? "yes" : result ? "no" : "--"}</li>
                  <li>Chunks analyzed: {result?.chunks_analyzed ?? "--"}</li>
                  <li>Confirmed matches: {webStatus?.matches ?? "--"}</li>
                </ul>
              </div>
            </section>
          </div>
        </section>
      </section>
    </main>
  );
}

