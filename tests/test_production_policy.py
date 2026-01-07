from backend.app.services.aggregate_service import aggregate_outputs


def test_hard_rules_win():
    outputs = {
        "rules": {"dept_pred": "mechanical", "dept_conf": 1.0, "level_pred": None, "level_conf": 0.0, "hard": True},
        "model": {"dept_pred": "controls", "dept_conf": 0.9, "level_pred": "high", "level_conf": 0.9},
    }
    result = aggregate_outputs(outputs, mode="production")
    assert result["pred_dept"] == "mechanical"
    assert result["trace"]["winner"] == "hard_rules"


def test_model_threshold_wins():
    outputs = {
        "model": {"dept_pred": "controls", "dept_conf": 0.9, "level_pred": "high", "level_conf": 0.9},
        "vector": {"dept_pred": "controls", "dept_conf": 0.2, "level_pred": "low", "level_conf": 0.2},
    }
    result = aggregate_outputs(outputs, mode="production", thresholds={"model_conf_threshold": 0.75})
    assert result["pred_dept"] == "controls"
    assert result["trace"]["winner"] == "model"


def test_consensus_wins_when_model_vector_agree():
    outputs = {
        "model": {"dept_pred": "controls", "dept_conf": 0.6, "level_pred": "high", "level_conf": 0.6},
        "vector": {
            "dept_pred": "controls",
            "dept_conf": 0.5,
            "level_pred": "high",
            "level_conf": 0.5,
            "top_similarity": 0.6,
            "margin": 0.1,
        },
    }
    result = aggregate_outputs(outputs, mode="production", thresholds={"model_conf_threshold": 0.75, "vector_similarity_threshold": 0.5})
    assert result["trace"]["winner"] == "consensus"


def test_vector_fallback_wins():
    outputs = {
        "model": {"dept_pred": "controls", "dept_conf": 0.3, "level_pred": "high", "level_conf": 0.3},
        "vector": {
            "dept_pred": "mechanical",
            "dept_conf": 0.8,
            "level_pred": "low",
            "level_conf": 0.8,
            "top_similarity": 0.8,
            "margin": 0.2,
        },
    }
    result = aggregate_outputs(
        outputs,
        mode="production",
        thresholds={"model_conf_threshold": 0.75, "vector_similarity_threshold": 0.5, "vector_margin_threshold": 0.1},
    )
    assert result["pred_dept"] == "mechanical"
    assert result["trace"]["winner"] == "vector"


def test_llm_last_resort():
    outputs = {
        "llm": {
            "dept_pred": "electrical",
            "dept_conf": 0.6,
            "level_pred": "medium",
            "level_conf": 0.6,
            "reason": "fallback",
        }
    }
    result = aggregate_outputs(outputs, mode="production", thresholds={"allow_llm": True})
    assert result["pred_dept"] == "electrical"
    assert result["trace"]["winner"] == "llm"


def test_abstain_when_ambiguous():
    outputs = {"model": {"dept_pred": None, "dept_conf": 0.0, "level_pred": None, "level_conf": 0.0}}
    result = aggregate_outputs(outputs, mode="production", thresholds={"allow_llm": False, "abstain_enabled": True})
    assert result["pred_dept"] is None
    assert result["trace"]["winner"] == "abstain"
