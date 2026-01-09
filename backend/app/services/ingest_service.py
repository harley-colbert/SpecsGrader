import pathlib
from typing import Dict, List, Optional

import pandas as pd

from backend.app.config.xlsx_contract import (
    DEPT_COL,
    RISK_LEVEL_COL,
    RISK_LEVELS,
    SPEC_TEXT_COL,
    SPECIFIC_RISK_COL,
    column_index,
    normalize_risk_level,
)

DEPARTMENTS = {"mechanical", "electrical", "controls", "project_management"}


def _select_sheet(excel_file: pd.ExcelFile) -> str:
    for sheet in excel_file.sheet_names:
        if "standards risk matrix" in sheet.lower():
            return sheet
    raise ValueError("No sheet containing 'Standards Risk Matrix' found")


def _read_dataframe(path: pathlib.Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, header=None, skiprows=4)
    elif suffix in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(path)
        sheet = _select_sheet(excel)
        df = excel.parse(sheet_name=sheet, header=None)
        df = df.iloc[4:]
    else:
        raise ValueError("Unsupported file type; expected CSV or XLSX")
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _to_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_rows(df: pd.DataFrame, training: bool) -> List[Dict[str, Optional[str]]]:
    rows: List[Dict[str, Optional[str]]] = []
    spec_text_index = column_index(SPEC_TEXT_COL)
    risk_text_index = column_index(SPECIFIC_RISK_COL)
    risk_level_index = column_index(RISK_LEVEL_COL)
    dept_index = column_index(DEPT_COL)
    for idx, row in df.iterrows():
        spec_text = _to_text(row[spec_text_index]) if len(row) > spec_text_index else ""
        if not spec_text.strip():
            continue
        specific_risk = _to_text(row[risk_text_index]) if len(row) > risk_text_index else ""
        risk_level_raw = _to_text(row[risk_level_index]) if len(row) > risk_level_index else ""
        risk_level = normalize_risk_level(risk_level_raw) if risk_level_raw else None
        dept = _to_text(row[dept_index]) if len(row) > dept_index else ""
        canonical: Dict[str, Optional[str]] = {
            "source_row": idx + 5,  # Excel 1-indexed row number
            "id": None,
            "spec_text": spec_text,
            "specific_risk_existing": specific_risk,
            "risk_level_existing": risk_level or "",
            "dept_existing": dept.lower(),
            "risk_text": specific_risk,
        }
        if training:
            canonical["label_level"] = risk_level or ""
            canonical["label_dept"] = dept.lower()
        rows.append(canonical)
    return rows


def load_training_dataset(path: str) -> Dict[str, object]:
    file_path = pathlib.Path(path)
    df = _read_dataframe(file_path)
    rows = _canonical_rows(df, training=True)

    summary = {
        "total_rows": len(rows),
        "missing_risk_text": 0,
        "missing_labels": 0,
        "invalid_levels": 0,
        "invalid_departments": 0,
    }

    for row in rows:
        spec_text = row.get("spec_text", "") or ""
        if not spec_text.strip():
            summary["missing_risk_text"] += 1
        level = normalize_risk_level(row.get("label_level") or "")
        dept = _to_text(row.get("label_dept") or "").lower()
        if not level or not dept:
            summary["missing_labels"] += 1
        if level and level not in RISK_LEVELS:
            summary["invalid_levels"] += 1
        if dept and dept not in DEPARTMENTS:
            summary["invalid_departments"] += 1

    return {"rows": rows, "summary": summary}


def load_classify_dataset(path: str) -> Dict[str, object]:
    file_path = pathlib.Path(path)
    df = _read_dataframe(file_path)
    rows = _canonical_rows(df, training=False)

    summary = {
        "total_rows": len(rows),
        "missing_risk_text": 0,
    }

    for row in rows:
        spec_text = row.get("spec_text", "") or ""
        if not spec_text.strip():
            summary["missing_risk_text"] += 1

    return {"rows": rows, "summary": summary}


__all__ = ["load_training_dataset", "load_classify_dataset"]
