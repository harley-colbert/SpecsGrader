from backend.app.services.aggregate_service import aggregate_outputs, DEFAULT_WEIGHTS


def test_vector_priority_over_llm_and_rules():
    outputs = {
        "vector": {"dept_pred": "mechanical", "dept_conf": 0.6, "level_pred": "high", "level_conf": 0.6},
        "llm": {"dept_pred": "electrical", "dept_conf": 0.9, "level_pred": "medium", "level_conf": 0.9},
        "rules": {"dept_pred": "controls", "dept_conf": 1.0, "level_pred": None, "level_conf": 0.0},
    }
    agg = aggregate_outputs(outputs, DEFAULT_WEIGHTS, mode="evaluate")
    assert agg["pred_dept"] == "mechanical"
    assert agg["pred_level"] == "high"


def test_abstain_when_no_outputs():
    agg = aggregate_outputs({})
    assert agg["pred_dept"] is None
    assert agg["pred_level"] is None
    assert agg["conf_dept"] == 0.0
    assert agg["conf_level"] == 0.0
