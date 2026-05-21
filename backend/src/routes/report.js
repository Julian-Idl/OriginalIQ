const express = require("express");
const PDFDocument = require("pdfkit");
const { getAnalysis } = require("../services/documentService");

const router = express.Router();

const COLORS = {
  ink: "#111827",
  muted: "#4b5563",
  line: "#d1d5db",
  red: "#b91c1c",
  amber: "#b45309",
  green: "#047857",
  blue: "#1d4ed8",
  paleAmber: "#fff7ed",
  paleBlue: "#eff6ff",
  paleSlate: "#f8fafc"
};

function scoreLabel(score) {
  if (score >= 70) return "High";
  if (score >= 40) return "Review";
  return "Low";
}

function scoreColor(score) {
  if (score >= 70) return COLORS.red;
  if (score >= 40) return COLORS.amber;
  return COLORS.green;
}

function safeText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function addFooter(doc) {
  const range = doc.bufferedPageRange();
  for (let i = range.start; i < range.start + range.count; i += 1) {
    doc.switchToPage(i);
    doc.font("Helvetica").fontSize(8).fillColor(COLORS.muted);
    doc.text("OriginalIQ Similarity Report", 48, doc.page.height - 36, { continued: true });
    doc.text(`Page ${i + 1}`, { align: "right" });
    doc.fillColor(COLORS.ink);
  }
}

function section(doc, title, subtitle) {
  doc.moveDown(0.8);
  doc.font("Helvetica-Bold").fontSize(15).fillColor(COLORS.ink).text(title);
  if (subtitle) {
    doc.font("Helvetica").fontSize(9).fillColor(COLORS.muted).text(subtitle);
  }
  doc.moveDown(0.35);
  doc.strokeColor(COLORS.line).lineWidth(0.5).moveTo(48, doc.y).lineTo(doc.page.width - 48, doc.y).stroke();
  doc.moveDown(0.6);
}

function metricBox(doc, x, y, width, label, value, color, note) {
  doc.roundedRect(x, y, width, 82, 6).fillAndStroke("#ffffff", COLORS.line);
  doc.fillColor(COLORS.muted).font("Helvetica").fontSize(8).text(label.toUpperCase(), x + 12, y + 12, { width: width - 24 });
  doc.fillColor(color).font("Helvetica-Bold").fontSize(24).text(value, x + 12, y + 28, { width: width - 24 });
  doc.fillColor(COLORS.muted).font("Helvetica").fontSize(8).text(note, x + 12, y + 58, { width: width - 24 });
  doc.fillColor(COLORS.ink);
}

function keyValue(doc, key, value) {
  doc.font("Helvetica-Bold").fontSize(9).fillColor(COLORS.ink).text(`${key}: `, { continued: true });
  doc.font("Helvetica").fontSize(9).fillColor(COLORS.muted).text(String(value ?? "N/A"));
  doc.fillColor(COLORS.ink);
}

function addWrappedEvidence(doc, text, options = {}) {
  const color = options.color || COLORS.muted;
  doc.font("Helvetica").fontSize(options.size || 8.5).fillColor(color).text(safeText(text), {
    lineGap: 2,
    width: options.width || doc.page.width - 96,
  });
  doc.fillColor(COLORS.ink);
}

function ensureSpace(doc, minHeight = 120) {
  if (doc.y + minHeight > doc.page.height - 60) {
    doc.addPage();
  }
}

router.get("/:analysisId", async (req, res, next) => {
  try {
    const analysis = await getAnalysis(req.params.analysisId);
    if (!analysis) {
      return res.status(404).json({ error: "Analysis not found." });
    }
    if (!analysis.report_eligible) {
      return res.status(422).json({
        error: "Report generation is available only for uploaded PDF/DOCX documents with at least 5 pages."
      });
    }

    const result = analysis.result || {};
    const metadata = result.metadata || {};
    const plagiarismMeta = metadata.plagiarism || {};
    const sources = result.source_urls || [];
    const spans = result.highlighted_text_spans || [];
    const aiSpans = result.ai_highlighted_spans || [];
    const filename = (analysis.filename || "document").replace(/[^\w.-]+/g, "_");

    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", `attachment; filename="OriginalIQ-${filename}-report.pdf"`);

    const doc = new PDFDocument({ margin: 48, size: "A4", bufferPages: true });
    doc.pipe(res);

    doc.font("Helvetica-Bold").fontSize(26).fillColor(COLORS.ink).text("OriginalIQ Similarity Report");
    doc.font("Helvetica").fontSize(10).fillColor(COLORS.muted).text("Academic originality, source matching, and AI-writing indicators");
    doc.moveDown(1);

    const y = doc.y;
    metricBox(doc, 48, y, 150, "Similarity Index", `${Number(result.plagiarism_score || 0).toFixed(1)}%`, scoreColor(result.plagiarism_score || 0), scoreLabel(result.plagiarism_score || 0));
    metricBox(doc, 220, y, 150, "AI Probability", `${Number(result.ai_score || 0).toFixed(1)}%`, scoreColor(result.ai_score || 0), scoreLabel(result.ai_score || 0));
    metricBox(doc, 392, y, 150, "Sources", String(sources.length), COLORS.blue, `${spans.length} highlighted spans`);
    doc.y = y + 98;

    section(doc, "Submission Details");
    keyValue(doc, "File", analysis.filename);
    keyValue(doc, "Document type", String(analysis.document_kind || "").toUpperCase());
    keyValue(doc, "Pages", analysis.page_count);
    keyValue(doc, "Chunks analyzed", result.chunks_analyzed || 0);
    keyValue(doc, "Analysis ID", analysis.id);
    keyValue(doc, "Web search", plagiarismMeta.web_search_enabled ? `enabled, searched ${plagiarismMeta.web_searched_chunks || 0} chunks` : "disabled");

    section(doc, "Interpretation");
    addWrappedEvidence(
      doc,
      `The similarity score is a weighted aggregate of confirmed matched chunks. OriginalIQ requires semantic similarity plus lexical evidence and LCS-aligned spans before a source contributes to the final score. The AI score is an ensemble of the fine-tuned RoBERTa classifier, GPT-2 perplexity, and stylometric signals.`
    );
    doc.moveDown(0.5);
    addWrappedEvidence(doc, result.ai_explanation || "No AI explanation was generated.");

    section(doc, "Source Overview", "Ranked by contribution to confirmed similarity.");
    if (!sources.length) {
      doc.font("Helvetica").fontSize(10).fillColor(COLORS.muted).text("No confirmed plagiarized sources were detected.");
    }
    sources.slice(0, 20).forEach((source, index) => {
      ensureSpace(doc, 100);
      doc.roundedRect(48, doc.y, doc.page.width - 96, 86, 6).fillAndStroke(COLORS.paleSlate, COLORS.line);
      const boxY = doc.y + 10;
      doc.fillColor(COLORS.ink).font("Helvetica-Bold").fontSize(10).text(`${index + 1}. ${source.title || source.url || "Local corpus"}`, 60, boxY, { width: 360 });
      doc.fillColor(scoreColor(source.score || 0)).font("Helvetica-Bold").fontSize(13).text(`${source.score}%`, 480, boxY, { width: 50, align: "right" });
      doc.fillColor(COLORS.muted).font("Helvetica").fontSize(8).text(`${source.source_type || "source"} · ${source.contribution}% contribution · ${source.matched_spans || 0} spans`, 60, boxY + 18, { width: 460 });
      if (source.url) {
        doc.fillColor(COLORS.blue).font("Helvetica").fontSize(8).text(source.url, 60, boxY + 32, { width: 460, link: source.url, underline: true });
      }
      const evidence = (source.evidence || [source.matched_text]).filter(Boolean)[0];
      if (evidence) {
        doc.fillColor(COLORS.muted).font("Helvetica").fontSize(8).text(safeText(evidence).slice(0, 280), 60, boxY + 48, { width: 460, lineGap: 1 });
      }
      doc.y = boxY + 84;
      doc.fillColor(COLORS.ink);
    });

    doc.addPage();
    section(doc, "Matched Text Detail", "Highlighted passages that contributed to the similarity score.");
    if (!spans.length) {
      doc.font("Helvetica").fontSize(10).fillColor(COLORS.muted).text("No plagiarism spans were detected.");
    }
    spans.slice(0, 80).forEach((span, index) => {
      ensureSpace(doc, 105);
      doc.roundedRect(48, doc.y, doc.page.width - 96, 78, 6).fillAndStroke(COLORS.paleAmber, "#fed7aa");
      const boxY = doc.y + 9;
      doc.fillColor(COLORS.amber).font("Helvetica-Bold").fontSize(10).text(`Match ${index + 1}: ${span.score}%`, 60, boxY);
      doc.fillColor(COLORS.muted).font("Helvetica").fontSize(8).text(span.source_url || "Local corpus", 140, boxY, { width: 380 });
      doc.fillColor(COLORS.ink).font("Helvetica").fontSize(8.5).text(safeText(span.text).slice(0, 420), 60, boxY + 17, { width: 460, lineGap: 2 });
      if (span.explanation) {
        doc.fillColor(COLORS.muted).font("Helvetica").fontSize(7.5).text(span.explanation, 60, boxY + 54, { width: 460 });
      }
      doc.y = boxY + 78;
    });

    doc.addPage();
    section(doc, "AI Writing Indicators", "High-confidence chunks and the model signals that elevated them.");
    doc.font("Helvetica").fontSize(10).fillColor(COLORS.muted).text(result.ai_explanation || "No AI explanation was generated.", { lineGap: 3 });
    doc.moveDown();
    if (!aiSpans.length) {
      doc.font("Helvetica").fontSize(10).fillColor(COLORS.muted).text("No high-confidence AI-like spans were detected.");
    }
    aiSpans.slice(0, 60).forEach((span, index) => {
      ensureSpace(doc, 105);
      doc.roundedRect(48, doc.y, doc.page.width - 96, 82, 6).fillAndStroke(COLORS.paleBlue, "#bfdbfe");
      const boxY = doc.y + 9;
      doc.fillColor(COLORS.blue).font("Helvetica-Bold").fontSize(10).text(`AI-like span ${index + 1}: ${span.score}%`, 60, boxY);
      doc.fillColor(COLORS.muted).font("Helvetica").fontSize(8).text(span.explanation, 60, boxY + 16, { width: 460 });
      doc.fillColor(COLORS.ink).font("Helvetica").fontSize(8.5).text(safeText(span.text).slice(0, 430), 60, boxY + 32, { width: 460, lineGap: 2 });
      doc.y = boxY + 84;
    });

    doc.addPage();
    section(doc, "Method Appendix");
    [
      "Chunking: 250-word windows with 50-word overlap.",
      "Embedding retrieval: Sentence-BERT all-mpnet-base-v2 vectors searched with FAISS.",
      "Web search: SerpAPI exact-sentence queries trigger when local retrieval is weak or lacks confirmed text alignment.",
      "Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 scores query-source pairs for finer semantic ordering.",
      "Similarity: combined cosine similarity, cross-encoder score, n-gram overlap, and LCS coverage.",
      "Highlighting: LCS-aligned spans are reported only when lexical evidence is strong enough.",
      "AI detection: fine-tuned RoBERTa, GPT-2 perplexity, stylometry, and logistic-regression ensemble.",
    ].forEach((line) => {
      doc.font("Helvetica").fontSize(10).fillColor(COLORS.muted).text(`- ${line}`, { lineGap: 4 });
    });

    addFooter(doc);
    doc.end();
  } catch (error) {
    next(error);
  }
});

module.exports = router;

