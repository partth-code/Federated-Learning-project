"""Bounded real Flower + Opacus experiment; no PPE training."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_settings
from database import init_db
from federated.dp_fl import run_dp_federated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    settings = load_settings()
    init_db(settings.db_path)
    print(json.dumps(run_dp_federated(settings.db_path,
        settings.project_root / "artifacts" / "federated", args.rounds), indent=2))
