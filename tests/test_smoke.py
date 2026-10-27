from pathlib import Path

from config import load_settings
from database import init_db, list_sites
from incidents.schema import Incident


def test_settings_are_project_relative(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)
    assert settings.project_root == tmp_path.resolve()
    assert settings.db_path == (tmp_path / "data/safesight.sqlite3").resolve()


def test_database_seeds_three_explicitly_simulated_sites(tmp_path: Path) -> None:
    db_path = init_db(tmp_path / "demo.sqlite3")
    sites = list_sites(db_path)
    assert [site["site_id"] for site in sites] == ["factory-a", "factory-b", "factory-c"]
    assert all(site["is_simulated"] == 1 for site in sites)


def test_incident_contract_contains_required_identity_fields() -> None:
    incident = Incident(
        event_id="evt-1",
        site_id="factory-a",
        camera_id="factory-a-camera-1",
        timestamp="2026-10-09T00:00:00+00:00",
        event_type="ppe_violation",
    )
    payload = incident.to_dict()
    assert payload["event_id"] == "evt-1"
    assert payload["status"] == "open"
    assert payload["created_at"]
