import json
from pathlib import Path

from backend.app.state import AppState
from backend.app.services.modelset_service import ModelSetService
from backend.app.services.rule_service import RuleService
from backend.app.services.training_service import TrainingParams, TrainingService
from backend.app.services.vector_service import VectorService


def _make_rows() -> list[dict[str, str]]:
    rows = []
    for idx in range(4):
        rows.append(
            {
                "risk_text": f"Mechanical note {idx}",
                "label_level": "low",
                "label_dept": "mechanical",
            }
        )
        rows.append(
            {
                "risk_text": f"Electrical note {idx}",
                "label_level": "high",
                "label_dept": "electrical",
            }
        )
    return rows


def test_version_training_record_schema(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    app_state = AppState()
    app_state.training_dataset = {"rows": _make_rows()}
    training_service = TrainingService(workspace=workspace, app_state=app_state)
    training_service._train_task(TrainingParams(cv_folds=3))

    rule_service = RuleService(app_state.rules_config)
    vector_service = VectorService(workspace=workspace, app_state=app_state)
    modelset_service = ModelSetService(
        workspace=workspace,
        app_state=app_state,
        rule_service=rule_service,
        vector_service=vector_service,
    )
    modelset_service.create_modelset(modelset_id="meta", name="Meta", description="")
    version_meta = modelset_service.save_version(modelset_id="meta", note="v1")

    version_path = workspace / "modelsets" / "meta" / "versions" / version_meta["version_id"] / "version.json"
    version_payload = json.loads(version_path.read_text(encoding="utf-8"))
    record = version_payload.get("training_record") or {}
    assert "trained_at" in record
    assert "dataset_snapshot_hash" in record
    assert "training_params" in record
    assert "cv_summary" in record
    assert "artifacts" in record
