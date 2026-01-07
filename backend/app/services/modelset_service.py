import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import UploadFile


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _zip_add_dir(zf: zipfile.ZipFile, root: Path, arc_prefix: str) -> None:
    root = root.resolve()
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        zf.write(p, f"{arc_prefix}/{rel.as_posix()}")


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract zip while preventing path traversal."""

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            # Guard against absolute paths or path traversal.
            if name.startswith("/") or name.startswith("\\"):
                raise ValueError("Invalid archive: absolute paths are not allowed")
            norm = Path(name)
            if any(part == ".." for part in norm.parts):
                raise ValueError("Invalid archive: path traversal detected")
        zf.extractall(dest_dir)


@dataclass
class ModelSetSummary:
    modelset_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    latest_version_id: Optional[str]
    versions: List[Dict[str, Any]]


class ModelSetService:
    """CRUD + import/export for "ModelSets".

    Storage layout (under workspace/modelsets):

      modelsets/
        <modelset_id>/
          modelset.json
          versions/
            <version_id>/
              version.json
              rules.json
              training_snapshot.json
              bundle/...
              vector_store/...

    Export format:
      .sgm (zip) containing manifest.json + payload/ directory
    """

    FORMAT_VERSION = "1.0"

    def __init__(self, workspace: Path, app_state: Any, rule_service: Any, vector_service: Any):
        self.workspace = workspace
        self.root = self.workspace / "modelsets"
        self.exports_dir = self.workspace / "exports"
        self.app_state = app_state
        self.rule_service = rule_service
        self.vector_service = vector_service
        _safe_mkdir(self.root)
        _safe_mkdir(self.exports_dir)

    # -----------------
    # Registry helpers
    # -----------------

    def _modelset_dir(self, modelset_id: str) -> Path:
        return self.root / modelset_id

    def _modelset_meta_path(self, modelset_id: str) -> Path:
        return self._modelset_dir(modelset_id) / "modelset.json"

    def _versions_dir(self, modelset_id: str) -> Path:
        return self._modelset_dir(modelset_id) / "versions"

    def _version_dir(self, modelset_id: str, version_id: str) -> Path:
        return self._versions_dir(modelset_id) / version_id

    def _version_meta_path(self, modelset_id: str, version_id: str) -> Path:
        return self._version_dir(modelset_id, version_id) / "version.json"

    @staticmethod
    def _normalize_id(value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("id is required")
        # Only allow [a-zA-Z0-9_-]
        safe = []
        for ch in value:
            if ch.isalnum() or ch in {"_", "-"}:
                safe.append(ch)
        result = "".join(safe)
        if not result:
            raise ValueError("id must contain alphanumeric characters")
        return result

    def list_modelsets(self) -> List[ModelSetSummary]:
        items: List[ModelSetSummary] = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "modelset.json"
            if not meta_path.exists():
                continue
            meta = _read_json(meta_path)
            modelset_id = str(meta.get("modelset_id") or d.name)
            name = str(meta.get("name") or modelset_id)
            description = str(meta.get("description") or "")
            created_at = str(meta.get("created_at") or "")
            updated_at = str(meta.get("updated_at") or created_at)
            versions = self.list_versions(modelset_id)
            latest = versions[0]["version_id"] if versions else None
            items.append(
                ModelSetSummary(
                    modelset_id=modelset_id,
                    name=name,
                    description=description,
                    created_at=created_at,
                    updated_at=updated_at,
                    latest_version_id=latest,
                    versions=versions,
                )
            )
        return items

    def get_modelset(self, modelset_id: str) -> ModelSetSummary:
        modelset_id = self._normalize_id(modelset_id)
        meta_path = self._modelset_meta_path(modelset_id)
        if not meta_path.exists():
            raise FileNotFoundError("ModelSet not found")
        meta = _read_json(meta_path)
        versions = self.list_versions(modelset_id)
        latest = versions[0]["version_id"] if versions else None
        return ModelSetSummary(
            modelset_id=modelset_id,
            name=str(meta.get("name") or modelset_id),
            description=str(meta.get("description") or ""),
            created_at=str(meta.get("created_at") or ""),
            updated_at=str(meta.get("updated_at") or ""),
            latest_version_id=latest,
            versions=versions,
        )

    def create_modelset(self, modelset_id: str, name: str, description: str = "") -> ModelSetSummary:
        modelset_id = self._normalize_id(modelset_id)
        name = (name or modelset_id).strip() or modelset_id
        description = (description or "").strip()

        d = self._modelset_dir(modelset_id)
        if d.exists():
            raise FileExistsError("ModelSet already exists")
        _safe_mkdir(d)
        _safe_mkdir(self._versions_dir(modelset_id))
        meta = {
            "modelset_id": modelset_id,
            "name": name,
            "description": description,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        _write_json(self._modelset_meta_path(modelset_id), meta)
        return self.get_modelset(modelset_id)

    def delete_modelset(self, modelset_id: str) -> None:
        modelset_id = self._normalize_id(modelset_id)
        d = self._modelset_dir(modelset_id)
        if not d.exists():
            raise FileNotFoundError("ModelSet not found")
        shutil.rmtree(d)
        if self.app_state.active_modelset_id == modelset_id:
            self.app_state.active_modelset_id = None
            self.app_state.active_modelset_version_id = None
            self.app_state.active_bundle_id = None

    def list_versions(self, modelset_id: str) -> List[Dict[str, Any]]:
        modelset_id = self._normalize_id(modelset_id)
        vdir = self._versions_dir(modelset_id)
        if not vdir.exists():
            return []
        versions: List[Dict[str, Any]] = []
        for d in sorted(vdir.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "version.json"
            if not meta_path.exists():
                continue
            meta = _read_json(meta_path)
            versions.append(meta)
        # newest first
        versions.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
        return versions

    # -----------------
    # CRUD operations
    # -----------------

    def save_version(
        self,
        modelset_id: str,
        note: str = "",
        include_bundle: bool = True,
        include_vector_store: bool = True,
        include_rules: bool = True,
        include_training_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """Create a new version in the ModelSet from current workspace artifacts."""

        modelset_id = self._normalize_id(modelset_id)
        if not self._modelset_meta_path(modelset_id).exists():
            raise FileNotFoundError("ModelSet not found")

        version_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        note = (note or "").strip()

        workspace_bundle = self.workspace / "workspace_bundle"
        vector_store = self.workspace / "vector_store"

        with tempfile.TemporaryDirectory(prefix="specsgrader_modelset_") as td:
            tmp = Path(td)
            version_dir = tmp / "version"
            _safe_mkdir(version_dir)

            included: Dict[str, bool] = {
                "bundle": False,
                "vector_store": False,
                "rules": False,
                "training_snapshot": False,
            }

            if include_bundle:
                included["bundle"] = _copy_if_exists(workspace_bundle, version_dir / "bundle")

            if include_vector_store:
                included["vector_store"] = _copy_if_exists(vector_store, version_dir / "vector_store")

            if include_rules:
                rules = self.app_state.rules_config or {}
                _write_json(version_dir / "rules.json", rules)
                included["rules"] = True

            if include_training_snapshot:
                snapshot = {
                    "captured_at": _now_iso(),
                    "training_job": dict(self.app_state.training_job or {}),
                    "training_dataset_summary": (self.app_state.training_dataset or {}).get("summary") if self.app_state.training_dataset else None,
                }
                _write_json(version_dir / "training_snapshot.json", snapshot)
                included["training_snapshot"] = True

            version_meta = {
                "modelset_id": modelset_id,
                "version_id": version_id,
                "created_at": _now_iso(),
                "note": note,
                "includes": included,
            }
            _write_json(version_dir / "version.json", version_meta)

            # commit
            target_dir = self._version_dir(modelset_id, version_id)
            if target_dir.exists():
                raise FileExistsError("Version id already exists")
            _safe_mkdir(self._versions_dir(modelset_id))
            shutil.move(str(version_dir), str(target_dir))

        # bump updated_at
        meta_path = self._modelset_meta_path(modelset_id)
        meta = _read_json(meta_path)
        meta["updated_at"] = _now_iso()
        _write_json(meta_path, meta)
        return _read_json(self._version_meta_path(modelset_id, version_id))

    def load_version(self, modelset_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Load a ModelSet version into the live app_state.

        This updates:
          - rules_config (and resets RuleService)
          - vector_store path and forces VectorService reload
          - training_job artifact paths (if present)
          - active_modelset_id/version
        """

        modelset_id = self._normalize_id(modelset_id)
        if not self._modelset_meta_path(modelset_id).exists():
            raise FileNotFoundError("ModelSet not found")

        versions = self.list_versions(modelset_id)
        if not versions:
            raise FileNotFoundError("ModelSet has no saved versions")

        if version_id is None:
            version_id = versions[0]["version_id"]
        version_id = self._normalize_id(version_id)

        vdir = self._version_dir(modelset_id, version_id)
        if not vdir.exists():
            raise FileNotFoundError("Version not found")

        # rules
        rules_path = vdir / "rules.json"
        if rules_path.exists():
            rules = _read_json(rules_path)
            self.app_state.rules_config = rules
            # Reset RuleService in-place
            self.rule_service.rules = rules

        # vector store
        vec_dir = vdir / "vector_store"
        if vec_dir.exists():
            self.app_state.vector_store = {"built": True, "path": str(vec_dir)}
            self.vector_service.store = None
        else:
            self.app_state.vector_store = {"built": False, "path": None}
            self.vector_service.store = None

        # bundle (trained models)
        bundle_dir = vdir / "bundle"
        if bundle_dir.exists():
            level_path = bundle_dir / "level_model.joblib"
            dept_path = bundle_dir / "dept_model.joblib"
            meta_path = bundle_dir / "bundle_meta.json"
            self.app_state.training_job["level_model_path"] = str(level_path) if level_path.exists() else None
            self.app_state.training_job["dept_model_path"] = str(dept_path) if dept_path.exists() else None
            self.app_state.training_job["bundle_meta_path"] = str(meta_path) if meta_path.exists() else None

        self.app_state.active_modelset_id = modelset_id
        self.app_state.active_modelset_version_id = version_id
        self.app_state.active_bundle_id = modelset_id

        return {
            "modelset_id": modelset_id,
            "version_id": version_id,
            "loaded": True,
            "rules_loaded": rules_path.exists(),
            "vector_loaded": vec_dir.exists(),
            "bundle_loaded": bundle_dir.exists(),
        }

    # -----------------
    # Export / Import (.sgm)
    # -----------------

    def export_sgm(self, modelset_id: str, version_id: Optional[str] = None) -> Path:
        modelset_id = self._normalize_id(modelset_id)
        modelset = self.get_modelset(modelset_id)
        if not modelset.versions:
            raise FileNotFoundError("ModelSet has no versions")

        if version_id is None:
            version_id = modelset.versions[0]["version_id"]
        version_id = self._normalize_id(version_id)

        vdir = self._version_dir(modelset_id, version_id)
        if not vdir.exists():
            raise FileNotFoundError("Version not found")

        manifest = {
            "format": "specsgrader-modelset",
            "format_version": self.FORMAT_VERSION,
            "exported_at": _now_iso(),
            "modelset": {
                "modelset_id": modelset.modelset_id,
                "name": modelset.name,
                "description": modelset.description,
                "created_at": modelset.created_at,
                "updated_at": modelset.updated_at,
            },
            "version": _read_json(vdir / "version.json") if (vdir / "version.json").exists() else {"version_id": version_id},
        }

        out_path = self.exports_dir / f"{modelset_id}__{version_id}.sgm"
        if out_path.exists():
            out_path.unlink()

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            # payload mirror
            _zip_add_dir(zf, vdir, "payload")
        return out_path

    async def import_sgm(self, upload: UploadFile) -> Dict[str, Any]:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix != ".sgm":
            raise ValueError("File must have .sgm extension")

        with tempfile.TemporaryDirectory(prefix="specsgrader_import_") as td:
            tmp = Path(td)
            zip_path = tmp / "import.sgm"
            zip_path.write_bytes(await upload.read())
            extract_dir = tmp / "extract"
            _safe_mkdir(extract_dir)
            _safe_extract_zip(zip_path, extract_dir)

            manifest_path = extract_dir / "manifest.json"
            payload_dir = extract_dir / "payload"
            if not manifest_path.exists():
                raise ValueError("Invalid .sgm: missing manifest.json")
            if not payload_dir.exists():
                raise ValueError("Invalid .sgm: missing payload/")

            manifest = _read_json(manifest_path)
            modelset_meta = manifest.get("modelset") or {}
            modelset_id = self._normalize_id(str(modelset_meta.get("modelset_id") or ""))
            name = str(modelset_meta.get("name") or modelset_id)
            description = str(modelset_meta.get("description") or "")
            version_meta = manifest.get("version") or {}
            source_version_id = str(version_meta.get("version_id") or "")
            if source_version_id:
                source_version_id = self._normalize_id(source_version_id)

            # Ensure modelset exists (create if missing)
            if not self._modelset_meta_path(modelset_id).exists():
                self.create_modelset(modelset_id=modelset_id, name=name, description=description)

            # Choose a new version id if collision
            target_version_id = source_version_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            if self._version_dir(modelset_id, target_version_id).exists():
                target_version_id = f"{target_version_id}_{datetime.now(timezone.utc).strftime('%f')}"

            # Copy payload into version directory
            target_dir = self._version_dir(modelset_id, target_version_id)
            _safe_mkdir(self._versions_dir(modelset_id))
            shutil.copytree(payload_dir, target_dir)

            # Ensure version meta exists and matches the target id.
            vmeta_path = target_dir / "version.json"
            vmeta = _read_json(vmeta_path) if vmeta_path.exists() else {}
            vmeta["version_id"] = target_version_id
            vmeta.setdefault("created_at", _now_iso())
            vmeta.setdefault("note", "Imported from .sgm")
            vmeta.setdefault("includes", {})
            _write_json(vmeta_path, vmeta)

            # bump updated_at
            meta_path = self._modelset_meta_path(modelset_id)
            meta = _read_json(meta_path)
            meta["updated_at"] = _now_iso()
            _write_json(meta_path, meta)

            return {
                "imported": True,
                "modelset_id": modelset_id,
                "version_id": target_version_id,
                "name": name,
            }


__all__ = ["ModelSetService", "ModelSetSummary"]
