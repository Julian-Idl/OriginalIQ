const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");
const uploadRoute = require("./routes/upload");
const analyzeRoute = require("./routes/analyze");
const mlClient = require("./services/mlClient");
const { config } = require("./config");

function createApp() {
  const app = express();

  app.use(helmet());
  app.use(cors({ origin: config.corsOrigin }));
  app.use(express.json({ limit: "2mb" }));
  app.use(morgan("dev"));

  app.get("/health", async (_req, res) => {
    let ml = { ok: false };
    try {
      ml = await mlClient.health();
    } catch (error) {
      ml = { ok: false, error: error.message };
    }

    res.json({ ok: true, ml });
  });

  app.use("/upload", uploadRoute);
  app.use("/analyze", analyzeRoute);

  app.use((req, res) => {
    res.status(404).json({ error: `Route not found: ${req.method} ${req.path}` });
  });

  app.use((error, _req, res, _next) => {
    const status = error.status || error.statusCode || 500;
    const issues = error.issues ? { issues: error.issues } : {};
    res.status(status).json({
      error: status >= 500 ? "Internal server error" : error.message,
      ...issues
    });
  });

  return app;
}

module.exports = { createApp };

