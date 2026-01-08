import json
from pathlib import Path

from backend.app.state import AppState
from backend.app.services.training_service import TrainingParams, TrainingService


def _make_rows() -> list[dict[str, str]]:
    rows = []
    for idx in range(5):
        rows.append(
            {
                "risk_text": f"Mechanical risk {idx}",
                "label_level": "low",
                "label_dept": "mechanical",
            }
        )
        rows.append(
            {
                "risk_text": f"Electrical risk {idx}",
                "label_level": "high",
                "label_dept": "electrical",
            }
        )
    return rows


def test_cv_training_outputs_metrics_and_artifacts(tmp_path: Path) -> None:
    app_state = AppState()
    app_state.training_dataset = {"rows": _make_rows()}
    service = TrainingService(workspace=tmp_path / "workspace", app_state=app_state)
    params = TrainingParams(cv_folds=5, use_class_weight_balanced=True)

    service._train_task(params)

    cv_metrics = app_state.training_job.get("cv_metrics")
    assert cv_metrics is not None
    assert "level" in cv_metrics
    assert "dept" in cv_metrics
    assert isinstance(cv_metrics["level"]["averages"]["macro_f1"], float)
    assert isinstance(cv_metrics["dept"]["averages"]["macro_f1"], float)

    bundle_meta_path = Path(str(app_state.training_job.get("bundle_meta_path") or ""))
    assert bundle_meta_path.exists()
    bundle_meta = json.loads(bundle_meta_path.read_text(encoding="utf-8"))
    assert "cv_metrics" in bundle_meta
