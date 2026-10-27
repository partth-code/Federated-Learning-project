from edge.zone_demo import run_simulated_zone_demo


def test_simulated_zone_demo_shows_entry_and_return_without_model_data():
    result = run_simulated_zone_demo()

    assert result.boundary == 0.5
    assert len(result.frames) == 7
    assert [frame.zone for frame in result.frames] == [
        "LEFT", "LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT", "LEFT"
    ]
    assert [
        (event.event_type, event.previous_zone, event.current_zone, event.elapsed_seconds)
        for event in result.events
    ] == [
        ("DANGER_ZONE_ENTRY", "LEFT", "RIGHT", 3.0),
        ("LEFT_ZONE_ENTRY", "RIGHT", "LEFT", 6.0),
    ]
    assert result.events[1].danger_duration_seconds == 3.0
    assert all(frame.annotated_image.size == (800, 450) for frame in result.frames)
