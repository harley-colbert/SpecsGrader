import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import Body, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import confusion_matrix

from .settings import get_settings
from .state import AppState, get_state
from .services.ingest_service import load_classify_dataset, load_training_dataset
from .services.rule_service import RuleService
from .services.training_service import TrainingParams, TrainingService
from .services.vector_service import VectorService
from .services.llm_service import LLMService
from .services.modelset_service import ModelSetService
from .services.model_inference_service import ModelInferenceService
from .services.aggregate_service import aggregate_outputs
from .services.job_manager import JobManager

ALLOWED_PANES = {"train", "classify", "results"}

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()
    app = FastAPI(title="SpecsGrader", version="4.4.0")
    app_state: AppState = get_state()

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    workspace_dir = Path(__file__).resolve().parents[2] / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    index_path = frontend_dir / "index.html"
    styles_path = frontend_dir / "styles.css"
    src_dir = frontend_dir / "src"

    @app.get("/api/health", response_class=JSONResponse)
    async def health() -> Dict[str, Any]:
        return {"ok": True}

    rule_service = RuleService(app_state.rules_config)
    training_service = TrainingService(
        workspace=workspace_dir,
        app_state=app_state,
    )
    vector_service = VectorService(
        workspace=workspace_dir,
        app_state=app_state,
    )
    model_inference_service = ModelInferenceService(
        workspace=workspace_dir,
        app_state=app_state,
    )

    modelset_service = ModelSetService(
        workspace=workspace_dir,
        app_state=app_state,
        rule_service=rule_service,
        vector_service=vector_service,
        app_version=app.version,
    )
    llm_service = LLMService()
    classify_job = JobManager()

    @app.get("/api/state", response_class=JSONResponse)
    async def read_state() -> Dict[str, Any]:
        state = asdict(app_state)
        state["capabilities"] = {
            "model": model_inference_service.available(),
            "vector": bool(app_state.vector_store.get("built")),
            "rules": bool(app_state.rules_config),
            "llm": not app_state.never_send_externally,
        }
        state["default_mode"] = "production"
        return state

    @app.post("/api/ui/set_active_pane", response_class=JSONResponse)
    async def set_active_pane(payload: Dict[str, str] = Body(...)) -> Dict[str, Any]:
        requested_pane = payload.get("pane")
        if requested_pane not in ALLOWED_PANES:
            raise HTTPException(status_code=400, detail="Invalid pane requested")
        app_state.active_pane = requested_pane
        return asdict(app_state)

    @app.post("/api/data/load", response_class=JSONResponse)
    async def load_data(
        payload: Dict[str, str] | None = Body(None),
        mode: str | None = Form(None),
        path: str | None = Form(None),
        file: UploadFile | None = File(None),
    ) -> Dict[str, Any]:
        mode = mode or (payload.get("mode") if payload else None)
        path = path or (payload.get("path") if payload else None)

        # v4.5: be a bit more forgiving when the frontend forgets to send `mode`.
        # If a file is uploaded but no mode/path is provided, we assume a training
        # dataset load initiated from the Train pane.
        if mode not in {"train", "classify"}:
            inferred: str | None = None
            if mode is None and file is not None and path is None:
                inferred = "train"

            if inferred is not None:
                logger.info("Inferring load mode", extra={"inferred_mode": inferred})
                mode = inferred
            else:
                message = "Invalid load request; mode must be either 'train' or 'classify'"
                logger.warning(message)
                raise HTTPException(status_code=400, detail=message)

        if file is not None:
            uploads_dir = workspace_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(file.filename or "").suffix or ""
            target_path = uploads_dir / f"{mode}_{uuid4().hex}{suffix}"
            content = await file.read()
            target_path.write_bytes(content)
            path = str(target_path)

        if not path:
            message = "No file provided for load request"
            logger.warning(message)
            raise HTTPException(status_code=400, detail=message)

        normalized_path = Path(path).expanduser()
        if not normalized_path.exists():
            message = f"Load request failed: path does not exist -> {normalized_path}"
            logger.warning(message)
            raise HTTPException(status_code=400, detail=message)

        if normalized_path.is_dir():
            message = f"Load request failed: expected a file but found directory -> {normalized_path}"
            logger.warning(message)
            raise HTTPException(status_code=400, detail=message)

        safe_path = str(normalized_path)

        try:
            if mode == "train":
                dataset = load_training_dataset(safe_path)
                app_state.training_dataset = dataset
                app_state.data_loaded["train"] = True
            else:
                dataset = load_classify_dataset(safe_path)
                app_state.classify_dataset = dataset
                app_state.data_loaded["classify"] = True
        except Exception as exc:  # noqa: BLE001
            error_message = (
                "Failed to load dataset "
                f"(mode='{mode}', path='{safe_path}'): {exc}"
            )
            logger.exception(error_message)
            raise HTTPException(
                status_code=400,
                detail={
                    "message": error_message,
                    "mode": mode,
                    "path": safe_path,
                    "error": str(exc),
                    "exception": exc.__class__.__name__,
                },
            ) from exc

        return dataset.get("summary", {})

    @app.get("/api/data/preview", response_class=JSONResponse)
    async def preview_data(mode: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        if mode == "train":
            dataset = app_state.training_dataset
        elif mode == "classify":
            dataset = app_state.classify_dataset
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not loaded")

        rows = dataset.get("rows", [])
        sliced = rows[offset : offset + limit]
        return {"rows": sliced, "total": len(rows)}

    @app.get("/api/rules/get", response_class=JSONResponse)
    async def get_rules() -> Dict[str, Any]:
        return app_state.rules_config

    @app.post("/api/rules/set", response_class=JSONResponse)
    async def set_rules(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        rules = payload.get("rules")
        if not rules or not isinstance(rules, dict):
            raise HTTPException(status_code=400, detail="Invalid rules payload")
        if "version" not in rules:
            raise HTTPException(status_code=400, detail="Rules must include version")
        app_state.rules_config = rules
        nonlocal rule_service
        rule_service = RuleService(rules)
        return app_state.rules_config

    @app.post("/api/rules/test", response_class=JSONResponse)
    async def test_rules(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        text = payload.get("text", "")
        rules = payload.get("rules")
        svc = rule_service if not rules else RuleService(rules)
        prediction = svc.predict(text)
        return {
            "dept_pred": prediction.dept_pred,
            "dept_conf": prediction.dept_conf,
            "matched": prediction.matched,
        }

    @app.post("/api/train/start", response_class=JSONResponse)
    async def start_training(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        params = TrainingParams(
            oversample_enabled=bool(payload.get("oversample_enabled", False)),
            oversample_cap_ratio=float(payload.get("oversample_cap_ratio", 0.3)),
            min_recall_per_class=float(payload.get("min_recall_per_class", 0.5)),
            calibration_method=str(payload.get("calibration_method", "sigmoid")),
        )
        training_service.start_training(params)
        return training_service.status()

    @app.get("/api/train/status", response_class=JSONResponse)
    async def training_status() -> Dict[str, Any]:
        return training_service.status()

    @app.post("/api/train/cancel", response_class=JSONResponse)
    async def training_cancel() -> Dict[str, Any]:
        training_service.cancel()
        return training_service.status()

    @app.get("/api/train/metrics", response_class=JSONResponse)
    async def training_metrics() -> Dict[str, Any]:
        metrics = training_service.metrics()
        return {"available": metrics is not None, "metrics": metrics}

    @app.post("/api/train/sanity", response_class=JSONResponse)
    async def training_sanity() -> Dict[str, Any]:
        rows = training_service.labeled_rows()
        if not rows or not model_inference_service.available():
            report = {"available": False, "n_rows": len(rows)}
            app_state.sanity_report = report
            return report

        expected_levels: list[str] = []
        expected_depts: list[str] = []
        predicted_levels: list[str] = []
        predicted_depts: list[str] = []
        results: list[Dict[str, Any]] = []

        for idx, row in enumerate(rows):
            text = str(row.get("risk_text") or "")
            expected_level = str(row.get("label_level") or "")
            expected_dept = str(row.get("label_dept") or "")
            pred = model_inference_service.predict(text)
            pred_level = pred.level_pred
            pred_dept = pred.dept_pred

            expected_levels.append(expected_level)
            expected_depts.append(expected_dept)
            predicted_levels.append(str(pred_level) if pred_level else "None")
            predicted_depts.append(str(pred_dept) if pred_dept else "None")

            match_level = pred_level == expected_level
            match_dept = pred_dept == expected_dept
            results.append(
                {
                    "row_id": row.get("source_row") or idx,
                    "risk_text": text,
                    "expected_level": expected_level,
                    "pred_level": pred_level,
                    "conf_level": pred.level_conf,
                    "expected_dept": expected_dept,
                    "pred_dept": pred_dept,
                    "conf_dept": pred.dept_conf,
                    "match_level": match_level,
                    "match_dept": match_dept,
                }
            )

        n_rows = len(results)
        accuracy_level = sum(r["match_level"] for r in results) / n_rows if n_rows else 0.0
        accuracy_dept = sum(r["match_dept"] for r in results) / n_rows if n_rows else 0.0

        level_labels = sorted(set(expected_levels) | set(predicted_levels))
        dept_labels = sorted(set(expected_depts) | set(predicted_depts))
        confusion_level = confusion_matrix(expected_levels, predicted_levels, labels=level_labels).tolist()
        confusion_dept = confusion_matrix(expected_depts, predicted_depts, labels=dept_labels).tolist()

        report = {
            "available": True,
            "n_rows": n_rows,
            "accuracy_level": accuracy_level,
            "accuracy_dept": accuracy_dept,
            "rows": results,
            "confusion": {
                "level": {"labels": level_labels, "matrix": confusion_level},
                "dept": {"labels": dept_labels, "matrix": confusion_dept},
            },
        }
        app_state.sanity_report = report
        return report

    @app.post("/api/train/evaluate", response_class=JSONResponse)
    async def training_evaluate(payload: Dict[str, Any] = Body(None)) -> Dict[str, Any]:
        payload = payload or {}
        vector_k_values = payload.get("vector_k_values") or [1]
        test_size = float(payload.get("test_size", 0.2))
        random_state = int(payload.get("random_state", 42))

        report = training_service.evaluate(
            rules=rule_service,
            vector_k_values=vector_k_values,
            test_size=test_size,
            random_state=random_state,
        )
        app_state.evaluation_report = report
        return report

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        """Avoid a noisy 404 in the browser dev console.

        SpecsGrader is a local tool; we don't currently ship a favicon.
        Returning 204 keeps the browser happy.
        """

        return Response(status_code=204)

    @app.get("/api/settings", response_class=JSONResponse)
    async def get_settings_state() -> Dict[str, Any]:
        return {"never_send_externally": app_state.never_send_externally}

    @app.post("/api/settings", response_class=JSONResponse)
    async def set_settings(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        app_state.never_send_externally = bool(payload.get("never_send_externally", False))
        return {"never_send_externally": app_state.never_send_externally}

    @app.post("/api/vector/build", response_class=JSONResponse)
    async def vector_build(payload: Dict[str, Any] = Body(None)) -> Dict[str, Any]:
        k = int(payload.get("k", 5)) if payload else 5
        if not app_state.training_dataset:
            raise HTTPException(status_code=400, detail="No training dataset loaded")
        rows = app_state.training_dataset.get("rows", [])
        if not rows:
            raise HTTPException(status_code=400, detail="Training dataset empty")
        vector_service.build(rows)
        return {"built": True, "k": k, "path": app_state.vector_store.get("path")}

    @app.get("/api/vector/status", response_class=JSONResponse)
    async def vector_status() -> Dict[str, Any]:
        built = bool(app_state.vector_store.get("built"))
        return {"available": built, "path": app_state.vector_store.get("path") if built else None}

    @app.post("/api/vector/test", response_class=JSONResponse)
    async def vector_test(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        text = payload.get("text", "")
        k = int(payload.get("k", 5))
        result = vector_service.predict(text, k=k)
        return {
            "dept_pred": result.dept_pred,
            "dept_conf": result.dept_conf,
            "level_pred": result.level_pred,
            "level_conf": result.level_conf,
            "top_similarity": result.top_similarity,
            "second_similarity": result.second_similarity,
            "margin": result.margin,
            "top_neighbors": result.top_neighbors,
            "neighbors": result.neighbors,
        }

    # -----------------
    # ModelSet CRUD + .sgm import/export
    # -----------------

    @app.get("/api/modelsets", response_class=JSONResponse)
    async def modelsets_list() -> Dict[str, Any]:
        items = modelset_service.list_modelsets()
        return {
            "modelsets": [
                {
                    "modelset_id": ms.modelset_id,
                    "name": ms.name,
                    "description": ms.description,
                    "tags": ms.tags,
                    "created_at": ms.created_at,
                    "updated_at": ms.updated_at,
                    "latest_version_id": ms.latest_version_id,
                    "versions": ms.versions,
                }
                for ms in items
            ],
            "active_modelset_id": app_state.active_modelset_id,
            "active_modelset_version_id": app_state.active_modelset_version_id,
        }

    @app.post("/api/modelsets", response_class=JSONResponse)
    async def modelsets_create(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        modelset_id = str(payload.get("modelset_id") or "").strip()
        description = str(payload.get("description") or "").strip()
        tags = payload.get("tags")
        if not name and not modelset_id:
            raise HTTPException(status_code=400, detail="name or modelset_id is required")
        if not modelset_id:
            # derive a stable-ish id from name
            modelset_id = "".join([c for c in name.lower().replace(" ", "-") if c.isalnum() or c in {"-", "_"}])
        try:
            ms = modelset_service.create_modelset(
                modelset_id=modelset_id,
                name=name or modelset_id,
                description=description,
                tags=tags,
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "created": True,
            "modelset": {
                "modelset_id": ms.modelset_id,
                "name": ms.name,
                "description": ms.description,
                "tags": ms.tags,
                "created_at": ms.created_at,
                "updated_at": ms.updated_at,
                "latest_version_id": ms.latest_version_id,
                "versions": ms.versions,
            },
        }

    @app.patch("/api/modelsets/{modelset_id}", response_class=JSONResponse)
    async def modelsets_update(modelset_id: str, payload: Dict[str, Any] = Body(None)) -> Dict[str, Any]:
        payload = payload or {}
        name = payload.get("name")
        description = payload.get("description")
        tags = payload.get("tags")
        try:
            ms = modelset_service.update_modelset(
                modelset_id=modelset_id,
                name=str(name) if name is not None else None,
                description=str(description) if description is not None else None,
                tags=tags,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "updated": True,
            "modelset": {
                "modelset_id": ms.modelset_id,
                "name": ms.name,
                "description": ms.description,
                "tags": ms.tags,
                "created_at": ms.created_at,
                "updated_at": ms.updated_at,
                "latest_version_id": ms.latest_version_id,
                "versions": ms.versions,
            },
        }

    @app.delete("/api/modelsets/{modelset_id}", response_class=JSONResponse)
    async def modelsets_delete(modelset_id: str) -> Dict[str, Any]:
        try:
            modelset_service.delete_modelset(modelset_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"deleted": True, "modelset_id": modelset_id}

    @app.post("/api/modelsets/{modelset_id}/versions", response_class=JSONResponse)
    async def modelsets_save_version(modelset_id: str, payload: Dict[str, Any] = Body(None)) -> Dict[str, Any]:
        payload = payload or {}
        note = payload.get("notes")
        if note is None:
            note = payload.get("note")
        parent_version_id = payload.get("parent_version_id")
        try:
            meta = modelset_service.save_version(
                modelset_id=modelset_id,
                note=str(note or ""),
                notes=str(note or ""),
                parent_version_id=str(parent_version_id) if parent_version_id else None,
                include_bundle=bool(payload.get("include_bundle", True)),
                include_vector_store=bool(payload.get("include_vector_store", True)),
                include_rules=bool(payload.get("include_rules", True)),
                include_training_snapshot=bool(payload.get("include_training_snapshot", True)),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"saved": True, "version": meta}

    @app.delete("/api/modelsets/{modelset_id}/versions/{version_id}", response_class=JSONResponse)
    async def modelsets_delete_version(modelset_id: str, version_id: str, force: bool = False) -> Dict[str, Any]:
        try:
            modelset_service.delete_version(modelset_id=modelset_id, version_id=version_id, force=force)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": True, "modelset_id": modelset_id, "version_id": version_id, "forced": force}

    @app.post("/api/modelsets/{modelset_id}/load", response_class=JSONResponse)
    async def modelsets_load(modelset_id: str, payload: Dict[str, Any] = Body(None)) -> Dict[str, Any]:
        payload = payload or {}
        version_id = payload.get("version_id")
        try:
            result = modelset_service.load_version(modelset_id=modelset_id, version_id=version_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result

    @app.get("/api/modelsets/{modelset_id}/export")
    async def modelsets_export(modelset_id: str, version_id: str | None = None):
        try:
            path = modelset_service.export_sgm(modelset_id=modelset_id, version_id=version_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=path.name,
        )

    @app.post("/api/modelsets/import", response_class=JSONResponse)
    async def modelsets_import(file: UploadFile = File(...)) -> Dict[str, Any]:
        try:
            return await modelset_service.import_sgm(file)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/llm/test", response_class=JSONResponse)
    async def llm_test(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        text = payload.get("text", "")
        model_name = payload.get("model", "openrouter/auto")
        try:
            pred = llm_service.predict(text, model_name, app_state.never_send_externally)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "risk_level": pred.risk_level,
            "department": pred.department,
            "confidence": pred.confidence,
            "reason": pred.reason,
        }

    @app.post("/api/classify/start", response_class=JSONResponse)
    async def classify_start(payload: Dict[str, Any] = Body(None)) -> Dict[str, Any]:
        thresholds = payload.get("thresholds", {}) if payload else {}
        mode = (payload.get("mode") if payload else None) or "production"
        if mode not in {"sanity", "evaluate", "production"}:
            raise HTTPException(status_code=400, detail="Invalid mode")

        policy = payload.get("policy") if payload else None
        if not isinstance(policy, dict):
            policy = dict(app_state.production_policy)

        default_enabled = {
            "sanity": {"model": True, "rules": False, "vector": False, "llm": False},
            "evaluate": {"model": True, "rules": True, "vector": True, "llm": True},
            "production": {"model": True, "rules": True, "vector": True, "llm": True},
        }
        enabled = (
            payload.get("enabled_methods", default_enabled[mode])
            if payload
            else default_enabled[mode]
        )

        if app_state.classify_dataset is None:
            raise HTTPException(status_code=400, detail="No classify dataset loaded")
        rows = app_state.classify_dataset.get("rows", [])
        if not rows:
            raise HTTPException(status_code=400, detail="Classify dataset empty")

        def worker(row: Dict[str, Any]) -> Dict[str, Any]:
            text = row.get("risk_text", "")
            method_outputs: Dict[str, Dict[str, Any]] = {}
            if enabled.get("rules"):
                rules_pred = rule_service.predict(text)
                method_outputs["rules"] = {
                    "dept_pred": rules_pred.dept_pred,
                    "dept_conf": rules_pred.dept_conf,
                    "level_pred": None,
                    "level_conf": 0.0,
                    "matched": rules_pred.matched,
                    "hard_hits": rules_pred.hard_hits,
                    "hard": rules_pred.is_hard,
                }
            if enabled.get("vector") and app_state.vector_store.get("built"):
                try:
                    vector_pred = vector_service.predict(text, k=int(payload.get("k", 5)) if payload else 5)
                    method_outputs["vector"] = {
                        "dept_pred": vector_pred.dept_pred,
                        "dept_conf": vector_pred.dept_conf,
                        "level_pred": vector_pred.level_pred,
                        "level_conf": vector_pred.level_conf,
                        "top_similarity": vector_pred.top_similarity,
                        "second_similarity": vector_pred.second_similarity,
                        "margin": vector_pred.margin,
                        "neighbors": vector_pred.top_neighbors,
                    }
                except Exception:
                    method_outputs["vector"] = None
            if enabled.get("llm") and not app_state.never_send_externally:
                try:
                    llm_pred = llm_service.predict(
                        text,
                        payload.get("llm_model", "openrouter/auto") if payload else "openrouter/auto",
                        app_state.never_send_externally,
                    )
                    method_outputs["llm"] = {
                        "dept_pred": llm_pred.department,
                        "dept_conf": float(llm_pred.confidence or 0.5),
                        "level_pred": llm_pred.risk_level,
                        "level_conf": float(llm_pred.confidence or 0.5),
                        "reason": llm_pred.reason,
                    }
                except Exception:
                    method_outputs["llm"] = None

            if enabled.get("model"):
                model_pred = model_inference_service.predict(text)
                if model_pred.available:
                    method_outputs["model"] = {
                        "dept_pred": model_pred.dept_pred,
                        "dept_conf": model_pred.dept_conf,
                        "level_pred": model_pred.level_pred,
                        "level_conf": model_pred.level_conf,
                    }

            aggregated = aggregate_outputs(method_outputs, mode=mode, thresholds=policy)
            level_threshold = float(thresholds.get("level", 0.0))
            dept_threshold = float(thresholds.get("dept", 0.0))
            below_threshold = aggregated["conf_level"] < level_threshold or aggregated["conf_dept"] < dept_threshold

            return {
                "risk_text": text,
                "pred_level": aggregated["pred_level"],
                "pred_dept": aggregated["pred_dept"],
                "conf_level": aggregated["conf_level"],
                "conf_dept": aggregated["conf_dept"],
                "methods_used": json.dumps(method_outputs),
                "trace": json.dumps(aggregated.get("trace", {})),
                "below_threshold": below_threshold,
                "user_override_level": None,
                "user_override_dept": None,
            }

        classify_job.start(rows, worker)
        app_state.results_rows = classify_job.status.get("results", [])
        return classify_job.current_status()

    @app.get("/api/classify/status", response_class=JSONResponse)
    async def classify_status() -> Dict[str, Any]:
        app_state.results_rows = classify_job.status.get("results", [])
        return classify_job.current_status()

    @app.post("/api/classify/cancel", response_class=JSONResponse)
    async def classify_cancel() -> Dict[str, Any]:
        classify_job.cancel()
        return classify_job.current_status()

    @app.get("/api/results/rows", response_class=JSONResponse)
    async def results_rows(limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        rows = getattr(app_state, "results_rows", []) or []
        return {"rows": rows[offset : offset + limit], "total": len(rows)}

    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> Any:
        return index_path.read_text(encoding="utf-8")

    @app.get("/styles.css")
    async def serve_styles() -> FileResponse:
        return FileResponse(styles_path, media_type="text/css")

    app.mount("/src", StaticFiles(directory=src_dir), name="src")

    app.state.settings = settings
    return app


__all__ = ["create_app"]
