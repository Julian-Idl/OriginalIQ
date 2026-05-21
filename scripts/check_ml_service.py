from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_service"))

from app.main import app

print(app.title)
