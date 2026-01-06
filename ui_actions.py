from __future__ import annotations

import json
from datetime import datetime
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
    """
    Thin UI wrapper around logic.train_all_models.

    `train_csv` may actually be a CSV or an Excel file; the core logic
    handles format detection and treats the source as read-only.
    """
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


def _load_table_for_path(path: str) -> pd.DataFrame:
    """
    Load a table from a training/classification file.

    - If CSV: simple pd.read_csv(path)
    - If Excel: choose the FIRST worksheet whose header row contains a
      column with 'Quote #' (case-insensitive). If none match, fall
      back to the first sheet.
    """
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        print(f"[DEBUG] _load_table_for_path: inspecting Excel file {path}")
        xls = pd.ExcelFile(path)
        target_sheet: Optional[str] = None

        for sheet_name in xls.sheet_names:
            try:
                # Read just the header row for inspection
                head_df = pd.read_excel(xls, sheet_name=sheet_name, nrows=1)
                cols = [str(c) for c in head_df.columns]
                print(f"[DEBUG]  Sheet '{sheet_name}' columns: {cols}")
                for col in cols:
                    col_norm = col.replace(" ", "").lower()
                    if "quote#" in col_norm:
                        target_sheet = sheet_name
                        print(f"[DEBUG]  -> Selected sheet '{sheet_name}' (found 'Quote #' in '{col}')")
                        break
                if target_sheet is not None:
                    break
            except Exception as e:
                print(f"[DEBUG]  Could not inspect sheet '{sheet_name}': {e}")

        if target_sheet is None:
            # Fallback: first sheet
            target_sheet = xls.sheet_names[0]
            print(f"[DEBUG]  No 'Quote #' column found; falling back to first sheet '{target_sheet}'")

        df = pd.read_excel(xls, sheet_name=target_sheet)
        print(f"[DEBUG]  Loaded sheet '{target_sheet}' with shape {df.shape}")
        return df

    # CSV path
    print(f"[DEBUG] _load_table_for_path: loading CSV file {path}")
    df = pd.read_csv(path)
    print(f"[DEBUG]  Loaded CSV with shape {df.shape}")
    return df


def classify_from_path(
    path: str,
    models: Dict,
    enable_similarity: bool = True,
    top_k: int = 5,
    similarity_threshold: float = 0.55,
) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """
    Load the source file, selecting the correct worksheet if Excel,
    run multipass classification, and return a DataFrame that includes:

    - ALL original columns from the chosen sheet
    - PLUS all classification columns (Final Risk Level, etc.)

    This ensures the UI table on the Classify tab shows the full
    worksheet columns along with the model outputs.
    """
    try:
        # Load the appropriate sheet / table
        df_source = _load_table_for_path(path)

        # Run classification using the source data
        result_df = logic.multipass_classify(
            df_source,
            models,
            enable_similarity,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # Post-process result_df
        result_df = result_df.copy()
        if "Needs Review" in result_df.columns:
            result_df["Needs Review"] = result_df["Needs Review"].astype(bool)
        if "Similarity Evidence" in result_df.columns:
            try:
                result_df["Top Similarity"] = result_df[
                    "Similarity Evidence"
                ].apply(_top_similarity_from_payload)
            except Exception:
                pass

        # Merge original sheet columns with classification output
        # We reset index to avoid any mismatch.
        full_df = pd.concat(
            [df_source.reset_index(drop=True), result_df.reset_index(drop=True)],
            axis=1,
        )
        print(
            "[DEBUG] classify_from_path: merged source and results; final shape:",
            full_df.shape,
        )
        return True, full_df, ""
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
        for col in [
            "Rule Trust",
            "Classic Trust",
            "Similarity Trust",
            "Semantic Risk Proba",
            "Semantic Dept Proba",
        ]:
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


def summarize_training_file(
    path: str,
) -> Tuple[bool, Optional[int], Optional[int], Optional[str], str]:
    """
    Read-only summary of the training file.

    - If Excel: use the *same* sheet-selection logic as classify_from_path,
      i.e., pick the first sheet whose header contains 'Quote #', or fall
      back to the first sheet.
    - If CSV: just read the file normally.

    Returns:
    - ok flag
    - total rows
    - labeled rows (based on a best-guess label column)
    - last_modified (human-readable string)
    - error message
    """
    try:
        df = _load_table_for_path(path)

        rows = len(df)
        # Try to find a reasonable label column
        label_cols = [
            c
            for c in df.columns
            if c.lower() in {"label", "risk level", "final risk level"}
        ]
        if label_cols:
            labeled = int(df[label_cols[0]].notna().sum())
        else:
            labeled = rows

        last_modified_ts = Path(path).stat().st_mtime
        last_modified = datetime.fromtimestamp(last_modified_ts).strftime(
            "%Y-%m-%d %H:%M"
        )
        return True, rows, labeled, last_modified, ""
    except Exception as exc:  # pragma: no cover - UI helper
        return False, None, None, None, str(exc)
