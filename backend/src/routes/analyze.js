const express = require("express");
const { z } = require("zod");
const { getDocument, saveAnalysis } = require("../services/documentService");
const mlClient = require("../services/mlClient");

const router = express.Router();

const analyzeSchema = z.object({
  document_id: z.string().uuid().optional(),
  text: z.string().min(20).optional(),
  filename: z.string().optional()
}).refine((data) => data.document_id || data.text, {
  message: "Provide either document_id or text."
});

router.post("/", async (req, res, next) => {
  try {
    const payload = analyzeSchema.parse(req.body);
    let text = payload.text;
    let filename = payload.filename;
    let document = null;

    if (payload.document_id) {
      document = await getDocument(payload.document_id);
      if (!document) {
        return res.status(404).json({ error: "Document not found." });
      }
      text = document.text_content;
      filename = document.filename;
    }

    const result = await mlClient.analyzeText({
      text,
      documentId: payload.document_id,
      filename
    });

    result.metadata = {
      ...(result.metadata || {}),
      document: document
        ? {
            filename: document.filename,
            page_count: document.page_count,
            document_kind: document.document_kind,
            report_eligible: document.report_eligible
          }
        : {
            filename: filename || "pasted-text.txt",
            page_count: 0,
            document_kind: "text",
            report_eligible: false
          }
    };

    const saved = await saveAnalysis({
      documentId: payload.document_id,
      result
    });

    res.json({
      analysis_id: saved.id,
      document_id: payload.document_id || null,
      report_available: Boolean(result.metadata.document.report_eligible),
      ...result
    });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
