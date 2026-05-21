const { Pool } = require("pg");
const { config } = require("../config");

const pool = new Pool({
  connectionString: config.databaseUrl
});

async function ensureSchema() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS documents (
      id UUID PRIMARY KEY,
      filename TEXT,
      mime_type TEXT,
      size_bytes BIGINT,
      page_count INTEGER NOT NULL DEFAULT 0,
      document_kind TEXT NOT NULL DEFAULT 'text',
      report_eligible BOOLEAN NOT NULL DEFAULT false,
      text_content TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_kind TEXT NOT NULL DEFAULT 'text';
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS report_eligible BOOLEAN NOT NULL DEFAULT false;

    CREATE TABLE IF NOT EXISTS analyses (
      id UUID PRIMARY KEY,
      document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
      plagiarism_score NUMERIC(5, 2) NOT NULL,
      ai_score NUMERIC(5, 2) NOT NULL,
      result JSONB NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
  `);
}

module.exports = { pool, ensureSchema };
