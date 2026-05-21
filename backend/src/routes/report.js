const express = require("express");
const PDFDocument = require("pdfkit");
const { getAnalysis } = require("../services/documentService");

const router = express.Router();

function addKeyValue(doc, key, value) {
  doc.font("Helvetica-Bold").text(`${key}: `, { continued: true });
  doc.font("Helvetica").text(String(value ?? "N/A"));
}

function addScore(doc, label, value) {
  doc.font("Helvetica-Bold").fontSize(13).text(label);
  doc.font("Helvetica").fontSize(24).text(`${Number(value || 0).toFixed(2)}%`);
  doc.moveDown(0.5);
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
    const filename = (analysis.filename || "document").replace(/[^\w.-]+/g, "_");
    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", `attachment; filename="OriginalIQ-${filename}-report.pdf"`);

    const doc = new PDFDocument({ margin: 48, size: "A4" });
    doc.pipe(res);

    doc.font("Helvetica-Bold").fontSize(24).text("OriginalIQ Report");
    doc.font("Helvetica").fontSize(10).fillColor("#4b5563").text("Plagiarism and AI detection analysis");
    doc.moveDown();
    doc.fillColor("#111827");
    addKeyValue(doc, "File", analysis.filename);
    addKeyValue(doc, "Document type", String(analysis.document_kind || "").toUpperCase());
    addKeyValue(doc, "Pages", analysis.page_count);
    addKeyValue(doc, "Analysis ID", analysis.id);
    doc.moveDown();

    addScore(doc, "Plagiarism score", result.plagiarism_score);
    addScore(doc, "AI probability", result.ai_score);
    doc.font("Helvetica").fontSize(11).text(result.ai_explanation || "No AI explanation available.", {
      lineGap: 3
    });
    doc.moveDown();

    doc.font("Helvetica-Bold").fontSize(16).text("Matched Sources");
    const sources = result.source_urls || [];
    if (!sources.length) {
      doc.font("Helvetica").fontSize(11).text("No plagiarized sources were detected.");
    }
    for (const source of sources.slice(0, 10)) {
      doc.moveDown(0.6);
      doc.font("Helvetica-Bold").fontSize(12).text(source.title || source.url || "Local corpus");
      doc.font("Helvetica").fontSize(10);
      if (source.url) doc.fillColor("#2563eb").text(source.url, { link: source.url, underline: true });
      doc.fillColor("#111827");
      addKeyValue(doc, "Score", `${source.score}%`);
      addKeyValue(doc, "Contribution", `${source.contribution}%`);
      addKeyValue(doc, "Matched spans", source.matched_spans || 0);
      for (const evidence of (source.evidence || []).slice(0, 3)) {
        doc.font("Helvetica").fontSize(9).fillColor("#374151").text(`- ${evidence}`, { lineGap: 2 });
      }
      doc.fillColor("#111827");
    }

    doc.addPage();
    doc.font("Helvetica-Bold").fontSize(16).text("Highlighted Plagiarism Spans");
    const spans = result.highlighted_text_spans || [];
    if (!spans.length) {
      doc.font("Helvetica").fontSize(11).text("No highlighted plagiarism spans were detected.");
    }
    for (const span of spans.slice(0, 60)) {
      doc.moveDown(0.5);
      doc.font("Helvetica-Bold").fontSize(10).text(`${span.score}% match ${span.source_url ? `from ${span.source_url}` : ""}`);
      doc.font("Helvetica").fontSize(9).text(span.text, { lineGap: 2 });
    }

    doc.addPage();
    doc.font("Helvetica-Bold").fontSize(16).text("AI-Like Spans");
    const aiSpans = result.ai_highlighted_spans || [];
    if (!aiSpans.length) {
      doc.font("Helvetica").fontSize(11).text("No high-confidence AI-like spans were detected.");
    }
    for (const span of aiSpans.slice(0, 60)) {
      doc.moveDown(0.5);
      doc.font("Helvetica-Bold").fontSize(10).text(`${span.score}% AI probability: ${span.explanation}`);
      doc.font("Helvetica").fontSize(9).text(span.text.slice(0, 700), { lineGap: 2 });
    }

    doc.end();
  } catch (error) {
    next(error);
  }
});

module.exports = router;

