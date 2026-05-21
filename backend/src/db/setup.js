const { Client } = require("pg");
const { config } = require("../config");
const { ensureSchema, pool } = require("./pool");

function databaseNameFromUrl(connectionString) {
  const parsed = new URL(connectionString);
  return parsed.pathname.replace(/^\//, "");
}

function maintenanceUrl(connectionString) {
  const parsed = new URL(connectionString);
  parsed.pathname = "/postgres";
  return parsed.toString();
}

function assertSafeDatabaseName(name) {
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
    throw new Error(`Unsafe database name: ${name}`);
  }
}

async function ensureDatabase() {
  const dbName = databaseNameFromUrl(config.databaseUrl);
  assertSafeDatabaseName(dbName);

  const client = new Client({ connectionString: maintenanceUrl(config.databaseUrl) });
  await client.connect();
  try {
    const existing = await client.query("SELECT 1 FROM pg_database WHERE datname = $1", [dbName]);
    if (existing.rowCount === 0) {
      await client.query(`CREATE DATABASE ${dbName}`);
      console.log(`Created database ${dbName}`);
    } else {
      console.log(`Database ${dbName} already exists`);
    }
  } finally {
    await client.end();
  }
}

async function main() {
  await ensureDatabase();
  await ensureSchema();
  await pool.end();
  console.log("Database schema is ready");
}

main().catch(async (error) => {
  console.error(error);
  await pool.end().catch(() => {});
  process.exit(1);
});

