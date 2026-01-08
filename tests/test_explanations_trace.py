import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app


def _load_training_fixture(client: TestClient, name: str) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / name
    with fixture_path.open("rb") as handle:
        resp = client.post(
            "/api/data/load",
            data={"mode": "train"},
            files={"file": (fixture_path.name, handle, "text/csv")},
        )
    assert resp.status_code == 200


def _load_classify_rows(client: TestClient) -> None:
    content = """,,,,,,\n,,,,,,\n,,,,,,\n,,,,,,\n,,,,bearing issue,\n"""
    resp = client.post(
        "/api/data/load",
        data={"mode": "classify"},
        files={"file": ("classify_rule.csv", content, "text/csv")},
    )
    assert resp.status_code == 200


def _wait_for_training(client: TestClient, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status_resp = client.get("/api/train/status")
        status_resp.raise_for_status()
        status = status_resp.json()
        if status.get("status") != "running":
            return status
        time.sleep(0.1)
    raise AssertionError("Training did not finish before timeout")


def _wait_for_classify(client: TestClient, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status_resp = client.get("/api/classify/status")
        status_resp.raise_for_status()
        status = status_resp.json()
        if status.get("status") != "running":
            return status
        time.sleep(0.1)
    raise AssertionError("Classify did not finish before timeout")


def test_explanations_trace_contains_evidence() -> None:
    client = TestClient(create_app())
    _load_training_fixture(client, "training_balanced.csv")

    start_resp = client.post("/api/train/start", json={})
    assert start_resp.status_code == 200
    status = _wait_for_training(client)
    assert status.get("status") == "completed"

    vector_resp = client.post("/api/vector/build", json={"k": 3})
    assert vector_resp.status_code == 200

    _load_classify_rows(client)
    classify_resp = client.post(
        "/api/classify/start",
        json={
            "mode": "production",
            "enabled_methods": {"rules": True, "vector": True, "llm": False, "model": True},
            "k": 3,
        },
    )
    assert classify_resp.status_code == 200
    _wait_for_classify(client)

    rows_resp = client.get("/api/results/rows")
    assert rows_resp.status_code == 200
    rows = rows_resp.json()["rows"]
    assert rows

    for row in rows:
        trace = json.loads(row.get("trace") or "{}")
        assert trace.get("winner")
        assert trace.get("steps")
        evidence = trace.get("evidence", {})
        rules = evidence.get("rules", {})
        assert "matched" in rules
        vector = evidence.get("vector", {})
        assert isinstance(vector.get("neighbors"), list)
        model = evidence.get("model", {})
        assert "level_top_terms" in model
        assert "dept_top_terms" in model

    first_trace = json.loads(rows[0].get("trace") or "{}")
    matched = first_trace.get("evidence", {}).get("rules", {}).get("matched", {})
    assert "bearing" in (matched.get("mechanical") or [])
