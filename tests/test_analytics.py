import pytest
from analytics.queries import safety_analytics, incident_counts
from database import init_db, insert_incident
from incidents.schema import Incident
from feedback.service import record_feedback


def test_daily_filters_and_current_review(tmp_path):
    db = init_db(tmp_path / "db.sqlite3")
    insert_incident(db, Incident("a", "factory-a", "cam-a", "2026-10-09T01:00:00Z", "ppe_violation"))
    insert_incident(db, Incident("b", "factory-b", "cam-b", "2026-10-10T00:30:00+02:00", "zone_intrusion"))
    insert_incident(db, Incident("c", "factory-a", "cam-a", "2026-10-11T00:00:00Z", "ppe_violation"))
    record_feedback(db, "a", "confirmed")
    record_feedback(db, "a", "false_positive")
    report = safety_analytics(db, "2026-10-09", "2026-10-10")
    assert report["counts"]["total"] == 2
    assert report["counts"]["confirmed"] == 0
    assert report["counts"]["false_positive"] == 1
    assert report["daily"] == [{"date": "2026-10-09", "alerts": 2}, {"date": "2026-10-10", "alerts": 0}]
    assert len(report["most_common_alert_types"]) == 2
    assert incident_counts(db)["confirmed"] == 0
    assert safety_analytics(db, "2026-10-09", "2026-10-10", site_id="factory-a", camera_id="cam-a")["counts"]["total"] == 1
    assert safety_analytics(db, "2026-10-09", "2026-10-10", site_id="' OR 1=1 --")["counts"]["total"] == 0


def test_empty_analytics(tmp_path):
    db = init_db(tmp_path / "db.sqlite3")
    report = safety_analytics(db, "2026-10-09", "2026-10-09")
    assert not any(report["counts"].values())
    assert report["most_common_alert_types"] == []


@pytest.mark.parametrize("start,end", [("invalid", "2026-10-09"), ("2026-10-10", "2026-10-09"), ("2020-01-01", "2026-01-01")])
def test_invalid_range(tmp_path, start, end):
    with pytest.raises(ValueError):
        safety_analytics(tmp_path / "unused.sqlite3", start, end)
