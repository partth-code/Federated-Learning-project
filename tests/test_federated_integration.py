from pathlib import Path

from database import init_db
from federated.lightweight_fl import run_federated


def test_real_flower_lightweight_round(tmp_path: Path) -> None:
    db_path = init_db(tmp_path / "fl.sqlite3")
    output = run_federated(db_path, tmp_path / "artifacts", rounds=1)
    assert output["workload"].startswith("lightweight PyTorch")
    assert len(output["simulated_sites"]) == 3
    assert output["round_metrics"]["1"] >= output["round_metrics"]["0"]
    assert (tmp_path / "artifacts" / output["model_artifact"]).exists()
    assert output["model_version_id"]
