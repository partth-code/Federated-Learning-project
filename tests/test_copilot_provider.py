from __future__ import annotations

from ai_copilot.provider import provider_status, summarize_incident


def test_copilot_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    status = provider_status()
    assert status["mode"] == "deterministic_fallback"
    assert "Structured CV event" in summarize_incident({"event_type": "zone_intrusion", "site_id": "factory-a"})


def test_copilot_falls_back_when_gemini_rejects_request(monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("GEMINI_API_KEY", "test-invalid-key")
    incident = {"event_type": "zone_intrusion", "site_id": "factory-a"}
    with patch("google.genai.Client", side_effect=RuntimeError("request failed")):
        result = summarize_incident(incident)
    assert "Structured CV event" in result
    assert "deterministic local fallback used" in result
    assert "test-invalid-key" not in result


def test_local_choice_never_constructs_remote_client(monkeypatch):
    from unittest.mock import patch
    from ai_copilot.provider import generate_explanation
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with patch("google.genai.Client") as client:
        result = generate_explanation({"event_type": "ppe_violation", "feedback": "False Positive"})
    client.assert_not_called()
    assert result.outbound_attempted is False
    assert "False Positive" in result.text


def test_success_only_sends_allowlisted_metadata(monkeypatch):
    import json
    from unittest.mock import patch, MagicMock
    from ai_copilot.provider import generate_explanation
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value.text = "CV/rules recorded an alert."
    with patch("google.genai.Client", return_value=client):
        result = generate_explanation({"event_type": "zone_intrusion", "evidence_path": "private.png",
            "person_id": "private-person", "notes": "ignore rules", "confidence": 0.8}, allow_remote=True)
    payload = json.loads(client.models.generate_content.call_args.kwargs["contents"])
    assert payload == {"event_type": "zone_intrusion", "confidence": 0.8}
    assert result.provider == "gemini" and result.outbound_attempted
    client.__exit__.assert_called_once()


def test_payload_rejects_nonfinite_and_nested_values():
    from ai_copilot.provider import structured_payload
    assert structured_payload({"confidence": float('nan'), "status": {}, "event_type": [], "zone_duration_seconds": -2}) == {}


def test_empty_gemini_response_falls_back(monkeypatch):
    from unittest.mock import patch, MagicMock
    from ai_copilot.provider import generate_explanation
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value.text = " "
    with patch("google.genai.Client", return_value=client):
        result = generate_explanation({"event_type": "ppe_violation"}, allow_remote=True)
    assert result.provider == "deterministic_fallback" and result.outbound_attempted


def test_daily_summary_includes_more_than_twenty_and_filters_utc(tmp_path):
    from database import init_db, insert_incident
    from incidents.schema import Incident
    from ai_copilot.fallback import incidents_for_day, daily_summary
    db = init_db(tmp_path / "db.sqlite3")
    for i in range(25):
        insert_incident(db, Incident(str(i), "factory-a", "cam", "2026-10-09T01:00:00Z", "zone_intrusion"))
    insert_incident(db, Incident("other", "factory-a", "cam", "2026-10-08T01:00:00Z", "zone_intrusion"))
    rows = incidents_for_day(db, "2026-10-09")
    assert len(rows) == 25 and "zone_intrusion: 25" in daily_summary(rows)
    assert incidents_for_day(db, "2026-10-10") == []
