from backend.app.services.vector_service import _distance_weighted_vote


def test_distance_weighted_vote_prefers_strong_neighbor() -> None:
    neighbors = [
        {"similarity": 0.9, "row": {"label_dept": "mechanical"}},
        {"similarity": 0.2, "row": {"label_dept": "electrical"}},
    ]
    label, confidence = _distance_weighted_vote(neighbors, "label_dept")
    assert label == "mechanical"
    assert confidence is not None
    assert confidence > 0.8


def test_distance_weighted_vote_confidence_increases_with_dominance() -> None:
    balanced = [
        {"similarity": 0.55, "row": {"label_level": "high"}},
        {"similarity": 0.45, "row": {"label_level": "low"}},
    ]
    dominant = [
        {"similarity": 0.9, "row": {"label_level": "high"}},
        {"similarity": 0.1, "row": {"label_level": "low"}},
    ]
    _, balanced_conf = _distance_weighted_vote(balanced, "label_level")
    _, dominant_conf = _distance_weighted_vote(dominant, "label_level")
    assert balanced_conf is not None
    assert dominant_conf is not None
    assert dominant_conf > balanced_conf
