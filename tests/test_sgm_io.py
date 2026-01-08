import asyncio
import json
import zipfile
from pathlib import Path

from starlette.datastructures import UploadFile

from backend.app.state import AppState
from backend.app.services.modelset_service import ModelSetService
from backend.app.services.rule_service import RuleService
from backend.app.services.vector_service import VectorService
from backend.app.label_policy import save_label_policy, default_label_policy


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_service(tmp_path: Path) -> ModelSetService:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    bundle_dir = workspace / "workspace_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_text(bundle_dir / "level_model.joblib", "dummy")
    _write_text(bundle_dir / "dept_model.joblib", "dummy")
    _write_text(bundle_dir / "bundle_meta.json", "{\"ok\": true}")

    vector_dir = workspace / "vector_store"
    vector_dir.mkdir(parents=True, exist_ok=True)
    _write_text(vector_dir / "rows.jsonl", "{\"risk_text\": \"x\"}\n")
    _write_text(vector_dir / "embeddings.npy", "fake")
    _write_text(vector_dir / "index.joblib", "dummy-index")
    _write_text(vector_dir / "embedder.json", "{\"kind\": \"dummy\"}")

    app_state = AppState()
    rule_service = RuleService(app_state.rules_config)
    vector_service = VectorService(workspace=workspace, app_state=app_state)
    service = ModelSetService(
        workspace=workspace,
        app_state=app_state,
        rule_service=rule_service,
        vector_service=vector_service,
        app_version="test",
    )
    service.create_modelset(modelset_id="sgm", name="SGM", description="")
    policy = default_label_policy()
    policy["risk_levels"].append(
        {"id": "custom", "label": "CUSTOM", "description": "Custom level for test."}
    )
    save_label_policy(workspace, "sgm", policy)
    service.save_version(modelset_id="sgm", note="v1")
    return service


def test_export_contains_manifest_and_checksums(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    export_path = service.export_sgm(modelset_id="sgm")

    with zipfile.ZipFile(export_path, "r") as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "checksums.sha256" in names
        assert "rules.json" in names


def test_import_rejects_checksum_mismatch(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    export_path = service.export_sgm(modelset_id="sgm")

    tampered_path = tmp_path / "tampered.sgm"
    with zipfile.ZipFile(export_path, "r") as zf:
        with zipfile.ZipFile(tampered_path, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for name in zf.namelist():
                data = zf.read(name)
                if name == "rules.json":
                    data = data + b"\n"  # tamper
                out.writestr(name, data)

    with tampered_path.open("rb") as fh:
        upload = UploadFile(filename=tampered_path.name, file=fh)
        try:
            asyncio.run(service.import_sgm(upload))
        except ValueError as exc:
            assert "Checksum mismatch" in str(exc)
        else:
            raise AssertionError("Expected checksum mismatch")


def test_import_rejects_zip_slip(tmp_path: Path) -> None:
    zip_path = tmp_path / "slip.sgm"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../evil.txt", "bad")
        zf.writestr("manifest.json", "{}")

    app_state = AppState()
    rule_service = RuleService(app_state.rules_config)
    vector_service = VectorService(workspace=tmp_path, app_state=app_state)
    service = ModelSetService(
        workspace=tmp_path,
        app_state=app_state,
        rule_service=rule_service,
        vector_service=vector_service,
        app_version="test",
    )

    with zip_path.open("rb") as fh:
        upload = UploadFile(filename=zip_path.name, file=fh)
        try:
            asyncio.run(service.import_sgm(upload))
        except ValueError as exc:
            assert "path traversal" in str(exc) or "absolute paths" in str(exc)
        else:
            raise AssertionError("Expected zip-slip rejection")


def test_import_preserves_label_policy(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    export_path = service.export_sgm(modelset_id="sgm")

    workspace = tmp_path / "workspace_import"
    workspace.mkdir(parents=True, exist_ok=True)
    app_state = AppState()
    rule_service = RuleService(app_state.rules_config)
    vector_service = VectorService(workspace=workspace, app_state=app_state)
    service = ModelSetService(
        workspace=workspace,
        app_state=app_state,
        rule_service=rule_service,
        vector_service=vector_service,
        app_version="test",
    )

    with export_path.open("rb") as fh:
        upload = UploadFile(filename=export_path.name, file=fh)
        result = asyncio.run(service.import_sgm(upload))

    policy_path = workspace / "modelsets" / result["modelset_id"] / "label_policy.json"
    assert policy_path.exists()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    ids = {item["id"] for item in policy.get("risk_levels", [])}
    assert "custom" in ids
