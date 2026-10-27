import json
import sqlite3

from database import init_db
from privacy.status import current_status


def test_privacy_runs_are_separate_and_deduplicated(tmp_path):
    db = init_db(tmp_path / "db.sqlite3")
    with sqlite3.connect(db) as conn:
        for run, round_number, metrics in [
            ("secure", 1, {"secure_aggregation": True, "differential_privacy": False}),
            ("dp", 1, {"secure_aggregation": False, "differential_privacy": True}),
            ("dp", 2, {"secure_aggregation": False, "differential_privacy": True}),
        ]:
            conn.execute("INSERT INTO federation_rounds(round_id,round_number,status,metrics_json) VALUES (?,?,?,?)",
                         (f"{run}-r{round_number}", round_number, "completed", json.dumps(metrics)))
    status = current_status(db, False)
    runs = {r["run_id"]: r for r in status["recorded_runs"]}
    assert len(runs) == 2
    assert runs["dp"]["latest_completed_round"] == 2
    assert runs["secure"]["differential_privacy"] is False
    assert runs["dp"]["secure_aggregation"] is False
    assert runs["dp"]["raw_update_bytes_per_direction"] is None
    assert status["federation_online"] is False


def test_malformed_or_partial_dp_record_does_not_enable_panel(tmp_path):
    db = init_db(tmp_path / "db.sqlite3")
    with sqlite3.connect(db) as conn:
        for i, raw in enumerate(['bad json', '[]', json.dumps({
            "workload": "Flower Opacus toy classifier; not PPE detector",
            "status": "completed", "differential_privacy": True, "accounting": [None]})]):
            conn.execute("INSERT INTO federation_rounds(round_id,round_number,status,metrics_json) VALUES (?,1,'completed',?)", (str(i), raw))
    assert current_status(db, True)["federated_dp"] is None


def test_empty_privacy_state_and_source_scope(tmp_path):
    db = init_db(tmp_path / "db.sqlite3")
    assert current_status(db, True)["recorded_runs"] == []
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO privacy_metrics(metric_id,mechanism,source) VALUES ('local','DP-SGD','standalone toy only')")
    assert current_status(db, True)["privacy_metrics"]["scope"] == "standalone toy only"
