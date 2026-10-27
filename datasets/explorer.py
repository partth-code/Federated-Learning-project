"""Dataset inspection and deterministic simulated-client partitioning."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_yolo_dataset(root: Path, config_path: Path | None = None, splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, object]:
    root = Path(root)
    result: dict[str, object] = {"root": str(root), "splits": {}, "classes": {}}
    names_file = root / "classes.txt"
    names = [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()] if names_file.exists() else []
    if config_path and Path(config_path).exists():
        try:
            import yaml
            configured = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")).get("names", {})
            names = [configured[key] for key in sorted(configured, key=lambda value: int(value))] if isinstance(configured, dict) else list(configured)
        except Exception:
            pass
    result["classes"] = {str(index): name for index, name in enumerate(names)}
    for split in splits:
        image_dir = root / split / "images" if (root / split / "images").exists() else root / "images" / split
        label_dir = root / split / "labels" if (root / split / "labels").exists() else root / "labels" / split
        records: list[dict[str, object]] = []
        if image_dir.exists():
            for image_path in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}):
                label_path = label_dir / f"{image_path.stem}.txt"
                class_ids: list[int] = []
                if label_path.exists():
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if parts:
                            class_ids.append(int(parts[0]))
                records.append({"image": str(image_path.relative_to(root)), "label": str(label_path.relative_to(root)), "class_ids": class_ids})
        result["splits"][split] = {"images": records, "image_count": len(records), "instances": sum(len(r["class_ids"]) for r in records)}
    return result


def partition_train_manifest(
    explorer: dict[str, object],
    output_path: Path,
    sites: tuple[str, ...] = ("factory-a", "factory-b", "factory-c"),
    *,
    seed: int = 20261009,
    dirichlet_alpha: float = 0.3,
) -> dict[str, list[dict[str, object]]]:
    if len(sites) < 2 or len(set(sites)) != len(sites):
        raise ValueError("At least two distinct simulated sites are required")
    if not np.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be a positive finite value")
    train_records = explorer.get("splits", {}).get("train", {}).get("images", [])
    partitions = {site: [] for site in sites}
    class_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in train_records:
        class_ids = [int(class_id) for class_id in record.get("class_ids", [])]
        if class_ids:
            counts = Counter(class_ids)
            dominant_class = min(counts, key=lambda class_id: (-counts[class_id], class_id))
        else:
            dominant_class = -1
        class_groups[dominant_class].append(record)

    # Images with the same dominant label share a sampled Dirichlet allocation.
    # This preserves every image exactly once while making the client label
    # proportions intentionally heterogeneous and reproducible.
    rng = np.random.default_rng(seed)
    for dominant_class in sorted(class_groups):
        records = sorted(class_groups[dominant_class], key=lambda item: str(item["image"]))
        order = rng.permutation(len(records))
        shuffled = [records[int(index)] for index in order]
        proportions = rng.dirichlet(np.full(len(sites), dirichlet_alpha, dtype=float))
        if len(shuffled) >= len(sites):
            counts = rng.multinomial(len(shuffled) - len(sites), proportions) + 1
        else:
            counts = rng.multinomial(len(shuffled), proportions)
        offset = 0
        for site, count in zip(sites, counts, strict=True):
            partitions[site].extend(shuffled[offset : offset + int(count)])
            offset += int(count)

    for records in partitions.values():
        records.sort(key=lambda item: str(item["image"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "simulated": True,
                "strategy": "dominant-class Dirichlet allocation",
                "dirichlet_alpha": float(dirichlet_alpha),
                "seed": int(seed),
                "partitions": partitions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return partitions
