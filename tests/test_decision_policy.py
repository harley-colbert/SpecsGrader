from backend.app.decision_policy import default_decision_policy
from backend.app.services.aggregate_service import aggregate_outputs


def test_default_policy_matches_legacy_order():
    outputs = {
        "vector": {
            "dept_pred": "mechanical",
            "dept_conf": 0.8,
            "level_pred": "high",
            "level_conf": 0.8,
            "top_similarity": 0.9,
            "margin": 0.2,
        },
        "llm": {
            "dept_pred": "electrical",
            "dept_conf": 0.9,
            "level_pred": "medium",
            "level_conf": 0.9,
        },
    }

    result = aggregate_outputs(outputs, mode="production")
    assert result["pred_dept"] == "mechanical"
    assert result["trace"]["winner"] == "vector"


def test_custom_policy_reorders_layers():
    policy = default_decision_policy()
    policy["layers"] = [
        {"id": "llm", "type": "llm", "enabled": True},
        {"id": "vector", "type": "vector_confidence", "min_similarity": 0.5, "min_margin": 0.1},
        {"id": "abstain", "type": "abstain", "enabled": True},
    ]
    outputs = {
        "vector": {
            "dept_pred": "mechanical",
            "dept_conf": 0.8,
            "level_pred": "high",
            "level_conf": 0.8,
            "top_similarity": 0.9,
            "margin": 0.2,
        },
        "llm": {
            "dept_pred": "electrical",
            "dept_conf": 0.9,
            "level_pred": "medium",
            "level_conf": 0.9,
        },
    }

    result = aggregate_outputs(outputs, mode="production", policy=policy)
    assert result["pred_dept"] == "electrical"
    assert result["trace"]["winner"] == "llm"
