const { createApp } = require("./app");
const { ensureSchema, pool } = require("./db/pool");
const { config } = require("./config");

async function main() {
  await ensureSchema();
  const app = createApp();
  app.listen(config.port, () => {
    console.log(`Backend listening on http://localhost:${config.port}`);
  });
}

main().catch(async (error) => {
  console.error("Backend failed to start", error);
  await pool.end().catch(() => {});
  process.exit(1);
});

