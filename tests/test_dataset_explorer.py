from __future__ import annotations

import json
from pathlib import Path

from datasets.explorer import partition_train_manifest, read_yolo_dataset


def test_yolo_manifest_and_partition_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    (root / "classes.txt").write_text("helmet\nPerson\n", encoding="utf-8")
    for index in range(6):
        image = root / "images" / "train" / f"image{index}.jpg"
        image.write_bytes(b"not decoded by the manifest parser")
        (root / "labels" / "train" / f"image{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest = read_yolo_dataset(root)
    output = tmp_path / "partitions.json"
    first = partition_train_manifest(manifest, output)
    second = partition_train_manifest(manifest, output)
    assert manifest["splits"]["train"]["image_count"] == 6
    assert first == second
    assert json.loads(output.read_text(encoding="utf-8"))["simulated"] is True


def test_partition_is_non_iid_reproducible_and_preserves_every_image(tmp_path: Path) -> None:
    records = [
        {"image": f"image-{class_id}-{index}.jpg", "class_ids": [class_id]}
        for class_id in range(3)
        for index in range(30)
    ]
    explorer = {"splits": {"train": {"images": records}}}
    output = tmp_path / "non_iid.json"
    partitions = partition_train_manifest(explorer, output)
    repeat = partition_train_manifest(explorer, output)
    assert partitions == repeat
    assert sum(map(len, partitions.values())) == len(records)
    assert len({record["image"] for rows in partitions.values() for record in rows}) == len(records)
    site_distributions = {
        site: tuple(sum(class_id in record["class_ids"] for record in rows) for class_id in range(3))
        for site, rows in partitions.items()
    }
    assert len(set(site_distributions.values())) == len(partitions)
    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["strategy"] == "dominant-class Dirichlet allocation"
    assert metadata["dirichlet_alpha"] == 0.3
