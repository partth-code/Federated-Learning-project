"""Run one bounded, CPU-only Flower round for the PPE detection head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from database import init_db
from federated.ppe_federation import MAX_DEMO_FRACTION, run_ppe_head_federation
from federated.ppe_secagg import run_ppe_head_secagg


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one Flower FedAvg round with one local CPU epoch per simulated site. "
            "The per-site sample fraction is capped at 5%; no full training is started. "
            "Optional Flower SecAgg+ is available with --secure-aggregation and requires the configured WSL Flower environment."
        )
    )
    parser.add_argument(
        "--fraction-per-site",
        type=float,
        default=MAX_DEMO_FRACTION,
        help=f"fraction sampled from each simulated site (0, {MAX_DEMO_FRACTION}]",
    )
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--secure-aggregation",
        action="store_true",
        help="use the official Flower SecAgg+ workflow for this PPE head-only round (DP remains disabled)",
    )
    args = parser.parse_args()
    settings = load_settings()
    init_db(settings.db_path)
    run = run_ppe_head_secagg if args.secure_aggregation else run_ppe_head_federation
    result = run(
        settings,
        fraction_per_site=args.fraction_per_site,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
