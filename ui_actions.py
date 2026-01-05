from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

import logic


def list_model_sets() -> Tuple[bool, list[str], str]:
    try:
        return True, logic.list_model_sets(), ""
    except Exception as exc:  # pragma: no cover - UI helper
        return False, [], str(exc)


def load_model_set(name: str) -> Tuple[bool, Optional[Dict], str]:
    try:
        files_dict = logic.load_model_set(name)
        models = logic.load_all_models(files_dict=files_dict)
        return True, models, ""
    except Exception as exc:  # pragma: no cover - UI helper
        return False, None, str(exc)


def load_last_used_model_set() -> Tuple[Optional[str], Optional[Dict]]:
    last_used = logic.get_last_used_model_set()
    if not last_used:
        return None, None
    ok, models, _ = load_model_set(last_used)
    if not ok:
        return None, None
    return last_used, models


def train_models(train_csv: str) -> Tuple[bool, Optional[str], Optional[Dict], Optional[Dict], str]:
    try:
        report, files_dict = logic.train_all_models(train_csv, return_file_dict=True)
        models = logic.load_all_models(files_dict=files_dict)
        return True, report, models, files_dict, ""
    except Exception as exc:  # pragma: no cover - UI helper
        return False, None, None, None, str(exc)


def save_model_set(set_name: str, files_dict: Dict) -> Tuple[bool, str]:
    try:
        logic.save_model_set(set_name, files_dict)
        return True, ""
    except Exception as exc:  # pragma: no cover - UI helper
        return False, str(exc)


def classify_from_path(
    path: str,
    models: Dict,
    enable_similarity: bool = True,
    top_k: int = 5,
    similarity_threshold: float = 0.55,
) -> Tuple[bool, Optional[pd.DataFrame], str]:
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        result_df = logic.multipass_classify(
            df,
            models,
            enable_similarity,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        result_df = result_df.copy()
        if "Needs Review" in result_df.columns:
            result_df["Needs Review"] = result_df["Needs Review"].astype(bool)
        if "Similarity Evidence" in result_df.columns:
            try:
                result_df["Top Similarity"] = result_df["Similarity Evidence"].apply(_top_similarity_from_payload)
            except Exception:
                pass
        return True, result_df, ""
    except Exception as exc:  # pragma: no cover - UI helper
        return False, None, str(exc)


def _top_similarity_from_payload(payload: str) -> float:
    if not payload:
        return 0.0
    try:
        parsed = json.loads(payload)
    except Exception:
        return 0.0
    if not parsed:
        return 0.0
    top = parsed[0]
    if isinstance(top, dict):
        return float(top.get("similarity", 0.0))
    return 0.0


def perform_export(
    df: pd.DataFrame,
    path: str,
    fmt: str = "csv",
    include_conf: bool = True,
    include_source: bool = True,
    include_context: bool = False,
    preset: Optional[dict] = None,
) -> Tuple[bool, str]:
    preset = preset or {}
    data = df.copy()
    if not include_conf:
        for col in ["Top Similarity", "Similarity Trust", "Rule Trust", "Classic Trust"]:
            if col in data.columns:
                data.drop(columns=[col], inplace=True)
    if not include_source and "Source File" in data.columns:
        data.drop(columns=["Source File"], inplace=True)
    if not include_context:
        for col in ["Top Match (Preview)", "Similarity Evidence"]:
            if col in data.columns:
                data.drop(columns=[col], inplace=True)
    if not preset.get("include_trust"):
        for col in ["Rule Trust", "Classic Trust", "Similarity Trust", "Semantic Risk Proba", "Semantic Dept Proba"]:
            if col in data.columns:
                data.drop(columns=[col], inplace=True)
    try:
        if fmt == "json":
            data.to_json(path, orient="records", indent=2)
        else:
            data.to_csv(path, index=False)
        return True, ""
    except Exception as exc:  # pragma: no cover - UI helper
        return False, str(exc)


def get_current_model_paths(models: Dict) -> Dict:
    return logic.get_current_model_file_paths(models)
