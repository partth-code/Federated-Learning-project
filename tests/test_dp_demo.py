from pathlib import Path

from database import init_db
from privacy.opacus_demo import run_opacus_demo


def test_opacus_reports_computed_privacy_budget(tmp_path: Path) -> None:
    db = init_db(tmp_path / "dp.sqlite3")
    output = run_opacus_demo(db, tmp_path / "artifacts", epochs=1)
    assert output["mechanism"] == "Opacus DP-SGD"
    assert output["epsilon"] > 0
    assert output["delta"] == 1e-5
    assert (tmp_path / "artifacts" / output["model_artifact"]).exists()
