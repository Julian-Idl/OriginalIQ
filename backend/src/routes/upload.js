const express = require("express");
const multer = require("multer");
const { z } = require("zod");
const { config } = require("../config");
const { createDocument } = require("../services/documentService");
const mlClient = require("../services/mlClient");

const router = express.Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.maxUploadMb * 1024 * 1024 }
});

const textUploadSchema = z.object({
  text: z.string().min(20, "text must contain at least 20 characters").optional(),
  filename: z.string().optional()
});

router.post("/", upload.single("file"), async (req, res, next) => {
  try {
    let extractedText = "";
    let metadata = {
      filename: req.body.filename || "pasted-text.txt",
      mimeType: "text/plain",
      sizeBytes: 0
    };

    if (req.file) {
      const extracted = await mlClient.extractText(req.file);
      extractedText = extracted.text || "";
      metadata = {
        filename: req.file.originalname,
        mimeType: req.file.mimetype,
        sizeBytes: req.file.size
      };
    } else {
      const parsed = textUploadSchema.parse(req.body);
      extractedText = parsed.text || "";
      metadata.filename = parsed.filename || metadata.filename;
      metadata.sizeBytes = Buffer.byteLength(extractedText, "utf8");
    }

    if (!extractedText.trim()) {
      return res.status(422).json({ error: "No readable text was found in the upload." });
    }

    const document = await createDocument({
      ...metadata,
      textContent: extractedText
    });

    res.status(201).json({
      document_id: document.id,
      filename: document.filename,
      created_at: document.created_at,
      text_preview: extractedText.slice(0, 800),
      text: extractedText,
      character_count: extractedText.length
    });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
