CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  filename TEXT,
  mime_type TEXT,
  size_bytes BIGINT,
  text_content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

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

