import asyncio
from pathlib import Path

from starlette.datastructures import UploadFile

from backend.app.state import AppState
from backend.app.services.modelset_service import ModelSetService
from backend.app.services.rule_service import RuleService
from backend.app.services.vector_service import VectorService


def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_modelset_save_export_import_load(tmp_path: Path):
    # Workspace #1: create fake artifacts
    ws1 = tmp_path / "ws1"
    ws1.mkdir(parents=True, exist_ok=True)

    # Fake trained model bundle artifacts
    bundle_dir = ws1 / "workspace_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_text(bundle_dir / "risk_level_model.joblib", "dummy")
    _write_text(bundle_dir / "department_model.joblib", "dummy")
    _write_text(bundle_dir / "bundle_meta.json", "{\"ok\": true}")

    # Fake vector store artifacts
    vector_dir = ws1 / "vector_store"
    vector_dir.mkdir(parents=True, exist_ok=True)
    _write_text(vector_dir / "rows.jsonl", "{\"risk_text\": \"x\"}\n")
    _write_text(vector_dir / "embeddings.npy", "not-a-real-npy")
    _write_text(vector_dir / "index.joblib", "dummy-index")
    _write_text(vector_dir / "embedder.json", "{\"kind\": \"dummy\"}")

    # App state with rules + training snapshot
    app_state1 = AppState()
    app_state1.rules_config = {
        "version": "1.0",
        "departments": {"mechanical": {"keywords": ["bearing"], "min_hits": 1}},
        "global": {"case_sensitive": False, "match_mode": "token_contains", "abstain_on_tie": True},
    }
    app_state1.training_job["params"] = {"oversample_enabled": True}
    app_state1.training_job["stats"] = {"n_rows": 1}
    app_state1.training_job["metrics"] = {"level": {"macro_f1": 0.5}, "dept": {"macro_f1": 0.6}}

    rule_service1 = RuleService(app_state1.rules_config)
    vector_service1 = VectorService(workspace=ws1, app_state=app_state1)
    ms1 = ModelSetService(workspace=ws1, app_state=app_state1, rule_service=rule_service1, vector_service=vector_service1)

    ms1.create_modelset(modelset_id="test", name="Test ModelSet", description="")
    version_meta = ms1.save_version(modelset_id="test", note="v1")
    assert version_meta["modelset_id"] == "test"
    assert version_meta["version_id"]

    # Export to .sgm
    sgm_path = ms1.export_sgm(modelset_id="test", version_id=version_meta["version_id"])
    assert sgm_path.exists()
    assert sgm_path.suffix == ".sgm"

    # Workspace #2: import the .sgm and load
    ws2 = tmp_path / "ws2"
    ws2.mkdir(parents=True, exist_ok=True)
    app_state2 = AppState()
    rule_service2 = RuleService(app_state2.rules_config)
    vector_service2 = VectorService(workspace=ws2, app_state=app_state2)
    ms2 = ModelSetService(workspace=ws2, app_state=app_state2, rule_service=rule_service2, vector_service=vector_service2)

    with sgm_path.open("rb") as f:
        upload = UploadFile(filename=sgm_path.name, file=f)
        result = asyncio.run(ms2.import_sgm(upload))

    assert result["imported"] is True
    assert result["modelset_id"] == "test"
    assert result["version_id"]

    load_result = ms2.load_version(modelset_id="test", version_id=result["version_id"])
    assert load_result["loaded"] is True
    assert app_state2.active_modelset_id == "test"
    assert app_state2.active_modelset_version_id == result["version_id"]
    assert app_state2.rules_config["departments"]["mechanical"]["keywords"] == ["bearing"]
    assert app_state2.vector_store["built"] is True
    assert app_state2.vector_store["path"]
