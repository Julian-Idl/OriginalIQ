const axios = require("axios");
const FormData = require("form-data");
const { config } = require("../config");

const client = axios.create({
  baseURL: config.mlServiceUrl,
  timeout: 1000 * 60 * 20
});

async function health() {
  const response = await client.get("/health");
  return response.data;
}

async function extractText(file) {
  const form = new FormData();
  form.append("file", file.buffer, {
    filename: file.originalname,
    contentType: file.mimetype,
    knownLength: file.size
  });

  const response = await client.post("/extract", form, {
    headers: form.getHeaders(),
    maxBodyLength: Infinity
  });

  return response.data;
}

async function analyzeText({ text, documentId, filename }) {
  const response = await client.post("/analyze", {
    text,
    document_id: documentId || null,
    filename: filename || null
  });
  return response.data;
}

module.exports = { health, extractText, analyzeText };

