const path = require("path");
const dotenv = require("dotenv");

const envCandidates = [
  process.env.ENV_FILE,
  path.resolve(__dirname, "../../.env"),
  path.resolve(__dirname, "../.env"),
  path.resolve(process.cwd(), ".env")
].filter(Boolean);

for (const candidate of envCandidates) {
  const result = dotenv.config({ path: candidate });
  if (!result.error) {
    break;
  }
}

const config = {
  port: Number(process.env.BACKEND_PORT || process.env.PORT || 4000),
  mlServiceUrl: process.env.ML_SERVICE_URL || "http://localhost:8000",
  databaseUrl: process.env.DATABASE_URL || "postgresql://postgres:postgres@localhost:5432/plagiarism_detector",
  uploadDir: process.env.UPLOAD_DIR || "uploads",
  maxUploadMb: Number(process.env.MAX_UPLOAD_MB || 25),
  corsOrigin: process.env.CORS_ORIGIN || "*"
};

module.exports = { config };
