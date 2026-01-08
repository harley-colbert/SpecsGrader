import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.decision_policy import default_decision_policy
from backend.app.main import create_app


def _wait_for_classify(client: TestClient, timeout_s: float = 5.0) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        status = client.get("/api/classify/status").json()
        if status.get("status") in {"completed", "error", "canceled"}:
            return status
        time.sleep(0.05)
    return client.get("/api/classify/status").json()


def test_deep_layer_disabled_returns_unavailable() -> None:
    app = create_app()
    client = TestClient(app)

    classify_path = Path(__file__).resolve().parent / "fixtures" / "classify_sample.csv"
    load_classify = client.post("/api/data/load", data={"mode": "classify", "path": str(classify_path)})
    assert load_classify.status_code == 200

    policy = default_decision_policy()
    policy["layers"] = [
        {"id": "deep", "type": "deep_model", "min_confidence": 0.0, "enabled": True},
        {"id": "abstain", "type": "abstain", "enabled": True},
    ]

    classify_resp = client.post(
        "/api/classify/start",
        json={
            "mode": "production",
            "policy": policy,
            "enabled_methods": {"deep": True, "model": False, "rules": False, "vector": False, "llm": False},
        },
    )
    assert classify_resp.status_code == 200
    status = _wait_for_classify(client)
    assert status.get("status") == "completed"
    results = status.get("results", [])
    assert results
    methods_used = json.loads(results[0]["methods_used"])
    assert methods_used["deep"]["available"] is False
    assert results[0]["pred_level"] is None
    assert results[0]["pred_dept"] is None


def test_deep_layer_enabled_predicts() -> None:
    app = create_app()
    client = TestClient(app)

    training_path = Path(__file__).resolve().parent / "fixtures" / "training_sample.csv"
    classify_path = Path(__file__).resolve().parent / "fixtures" / "classify_sample.csv"

    load_train = client.post("/api/data/load", data={"mode": "train", "path": str(training_path)})
    assert load_train.status_code == 200
    load_classify = client.post("/api/data/load", data={"mode": "classify", "path": str(classify_path)})
    assert load_classify.status_code == 200

    policy = default_decision_policy()
    policy["layers"] = [
        {"id": "deep", "type": "deep_model", "min_confidence": 0.0, "enabled": True},
        {"id": "abstain", "type": "abstain", "enabled": True},
    ]

    classify_resp = client.post(
        "/api/classify/start",
        json={
            "mode": "production",
            "policy": policy,
            "enabled_methods": {"deep": True, "model": False, "rules": False, "vector": False, "llm": False},
        },
    )
    assert classify_resp.status_code == 200
    status = _wait_for_classify(client)
    assert status.get("status") == "completed"
    results = status.get("results", [])
    assert results
    methods_used = json.loads(results[0]["methods_used"])
    assert methods_used["deep"]["available"] is True
    assert results[0]["pred_level"] is not None
    assert results[0]["pred_dept"] is not None
