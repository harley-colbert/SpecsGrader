import json
import time
from pathlib import Path

import joblib
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.main import create_app
from backend.app.services.model_inference_service import ModelInferenceService
from backend.app.state import AppState


def _build_pipeline() -> Pipeline:
    """Build a simple TF-IDF + LogisticRegression pipeline for tests.

    We intentionally avoid calibration here so tests remain stable even on
    very small synthetic datasets and across different scikit-learn versions.
    The production training pipeline still uses calibrated models with a
    fallback to an uncalibrated model when calibration is not feasible.
    """
    base = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1)
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
            ("clf", base),
        ]
    )


def _train_and_save_models(bundle_dir: Path) -> tuple[Path, Path]:
    texts = [
        "pump bearing failure",
        "bearing overheating",
        "plc panel fault",
        "controls logic error",
    ]
    levels = ["high", "high", "low", "low"]
    depts = ["mechanical", "mechanical", "controls", "controls"]

    level_model = _build_pipeline()
    dept_model = _build_pipeline()
    level_model.fit(texts, levels)
    dept_model.fit(texts, depts)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    level_path = bundle_dir / "level_model.joblib"
    dept_path = bundle_dir / "dept_model.joblib"
    joblib.dump(level_model, level_path)
    joblib.dump(dept_model, dept_path)
    return level_path, dept_path


def test_model_inference_service_returns_predictions(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "workspace_bundle"
    level_path, dept_path = _train_and_save_models(bundle_dir)
    app_state = AppState()
    app_state.training_job["level_model_path"] = str(level_path)
    app_state.training_job["dept_model_path"] = str(dept_path)

    service = ModelInferenceService(workspace=tmp_path, app_state=app_state)
    prediction = service.predict("bearing issue")

    assert prediction.available is True
    assert prediction.level_pred
    assert prediction.dept_pred
    assert prediction.level_conf >= 0.0
    assert prediction.dept_conf >= 0.0


def test_model_inference_service_handles_missing_models(tmp_path: Path) -> None:
    app_state = AppState()
    service = ModelInferenceService(workspace=tmp_path, app_state=app_state)
    prediction = service.predict("anything")

    assert prediction.available is False
    assert prediction.level_pred is None
    assert prediction.dept_pred is None


def test_classify_includes_model_method(tmp_path: Path) -> None:
    workspace_dir = Path(__file__).resolve().parents[1] / "workspace"
    bundle_dir = workspace_dir / "workspace_bundle"

    app = create_app()
    client = TestClient(app)

    training_path = Path(__file__).resolve().parent / "fixtures" / "training_sample.csv"
    classify_path = Path(__file__).resolve().parent / "fixtures" / "classify_sample.csv"

    try:
        load_train = client.post("/api/data/load", data={"mode": "train", "path": str(training_path)})
        assert load_train.status_code == 200
        load_classify = client.post("/api/data/load", data={"mode": "classify", "path": str(classify_path)})
        assert load_classify.status_code == 200

        train_resp = client.post("/api/train/start", json={})
        assert train_resp.status_code == 200
        for _ in range(100):
            status = client.get("/api/train/status").json()
            if status.get("status") in {"completed", "error", "canceled"}:
                break
            time.sleep(0.05)
        assert status.get("status") == "completed"

        classify_resp = client.post(
            "/api/classify/start",
            json={
                "mode": "production",
                "enabled_methods": {"model": True, "rules": False, "vector": False, "llm": False},
            },
        )
        assert classify_resp.status_code == 200
        for _ in range(100):
            status = client.get("/api/classify/status").json()
            if status.get("status") in {"completed", "error", "canceled"}:
                break
            time.sleep(0.05)

        assert status.get("status") == "completed"
        results = status.get("results", [])
        assert results
        methods_used = json.loads(results[0]["methods_used"])
        assert "model" in methods_used
    finally:
        for path in bundle_dir.glob("*.joblib"):
            path.unlink()
        meta_path = bundle_dir / "bundle_meta.json"
        if meta_path.exists():
            meta_path.unlink()
