from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from database import init_db
from privacy.opacus_demo import run_opacus_demo


settings = load_settings()
init_db(settings.db_path)
print(json.dumps(run_opacus_demo(settings.db_path, settings.project_root / "artifacts" / "privacy"), indent=2))
