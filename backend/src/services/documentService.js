const { randomUUID } = require("crypto");
const { pool } = require("../db/pool");

async function createDocument({ filename, mimeType, sizeBytes, textContent }) {
  const id = randomUUID();
  const result = await pool.query(
    `INSERT INTO documents (id, filename, mime_type, size_bytes, text_content)
     VALUES ($1, $2, $3, $4, $5)
     RETURNING id, filename, mime_type, size_bytes, created_at`,
    [id, filename || null, mimeType || null, sizeBytes || null, textContent]
  );
  return result.rows[0];
}

async function getDocument(id) {
  const result = await pool.query("SELECT * FROM documents WHERE id = $1", [id]);
  return result.rows[0] || null;
}

async function saveAnalysis({ documentId, result }) {
  const id = randomUUID();
  const inserted = await pool.query(
    `INSERT INTO analyses (id, document_id, plagiarism_score, ai_score, result)
     VALUES ($1, $2, $3, $4, $5)
     RETURNING id, document_id, plagiarism_score, ai_score, created_at`,
    [
      id,
      documentId || null,
      result.plagiarism_score || 0,
      result.ai_score || 0,
      result
    ]
  );
  return inserted.rows[0];
}

module.exports = { createDocument, getDocument, saveAnalysis };

