from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from datasets.explorer import partition_train_manifest, read_yolo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a real YOLO dataset and write a portable manifest")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    settings = load_settings()
    config_path = settings.project_root / "datasets" / "construction-ppe.yaml"
    manifest = read_yolo_dataset(args.root, config_path=config_path)
    split_path = settings.project_root / "artifacts" / "dataset_partitions.json"
    partitions = partition_train_manifest(manifest, split_path)
    manifest["source"] = "Ultralytics Construction-PPE"
    # Keep generated evidence portable. The local dataset remains outside the
    # repository; never persist a machine-specific absolute path in artifacts.
    manifest["root"] = "<local dataset root>"
    manifest["license_note"] = "Upstream documentation states AGPL-3.0; verify terms before redistribution."
    manifest["simulated_federated_sites"] = {site: {"image_count": len(records), "instance_count": sum(len(record["class_ids"]) for record in records)} for site, records in partitions.items()}
    output = settings.project_root / "artifacts" / "dataset_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output.relative_to(settings.project_root)), "sites": manifest["simulated_federated_sites"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
