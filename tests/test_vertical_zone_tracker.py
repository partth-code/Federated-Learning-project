from detector.geometry import VerticalZoneTransitionTracker


def test_vertical_tracker_confirms_individual_transitions_and_ignores_boundary_jitter():
    tracker = VerticalZoneTransitionTracker(boundary=0.5, hysteresis=0.05, confirmations=2)
    tracker.begin_frame()
    assert tracker.update("a", 0.2) == {"zone": "LEFT", "transition": None}
    assert tracker.update("b", 0.8) == {"zone": "RIGHT", "transition": None}
    tracker.end_frame()

    tracker.begin_frame()
    assert tracker.update("a", 0.52)["transition"] is None
    assert tracker.update("b", 0.48)["transition"] is None
    tracker.end_frame()

    tracker.begin_frame()
    assert tracker.update("a", 0.8)["transition"] is None
    assert tracker.update("b", 0.2)["transition"] is None
    tracker.end_frame()

    tracker.begin_frame()
    assert tracker.update("a", 0.8)["transition"] == {
        "previous_zone": "LEFT", "current_zone": "RIGHT"
    }
    assert tracker.update("b", 0.2)["transition"] == {
        "previous_zone": "RIGHT", "current_zone": "LEFT"
    }
    tracker.end_frame()


def test_temporary_loss_does_not_create_transition_and_long_loss_resets_identity():
    tracker = VerticalZoneTransitionTracker(
        boundary=0.5,
        hysteresis=0.0,
        confirmations=2,
        missing_frame_tolerance=2,
    )
    tracker.begin_frame()
    tracker.update("a", 0.2)
    tracker.end_frame()

    tracker.begin_frame()
    tracker.update("a", 0.8)
    tracker.end_frame()
    tracker.begin_frame()
    tracker.end_frame()
    tracker.begin_frame()
    assert tracker.update("a", 0.8)["transition"] is None
    tracker.end_frame()
    tracker.begin_frame()
    assert tracker.update("a", 0.8)["transition"] == {
        "previous_zone": "LEFT", "current_zone": "RIGHT"
    }
    tracker.end_frame()

    tracker.begin_frame()
    tracker.end_frame()
    tracker.begin_frame()
    tracker.end_frame()
    tracker.begin_frame()
    tracker.end_frame()
    tracker.begin_frame()
    assert tracker.update("a", 0.2)["transition"] is None
    tracker.end_frame()
