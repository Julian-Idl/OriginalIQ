$ErrorActionPreference = "Stop"

if (!(Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
}

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\ml_service\requirements.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm

Push-Location backend
npm install
Pop-Location

Push-Location frontend
npm install
Pop-Location

Write-Host "Setup complete. Add SERPAPI_API_KEY and DATABASE_URL to .env if needed."

