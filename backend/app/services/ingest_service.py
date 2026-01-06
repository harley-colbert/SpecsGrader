import pathlib
from typing import Dict, List, Optional

import pandas as pd

RISK_LEVELS = {"none", "low", "medium", "high", "extreme"}
DEPARTMENTS = {"mechanical", "electrical", "controls", "project_management"}


def _select_sheet(excel_file: pd.ExcelFile) -> str:
    for sheet in excel_file.sheet_names:
        if "quote #" in sheet.lower():
            return sheet
    raise ValueError("No sheet containing 'Quote #' found")


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
    for idx, row in df.iterrows():
        canonical: Dict[str, Optional[str]] = {
            "source_row": idx + 5,  # Excel 1-indexed row number
            "id": None,
            "risk_text": _to_text(row[4]) if len(row) > 4 else "",
        }
        if training:
            canonical["label_level"] = _to_text(row[5]) if len(row) > 5 else ""
            canonical["label_dept"] = _to_text(row[6]) if len(row) > 6 else ""
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
        risk_text = row.get("risk_text", "") or ""
        if not risk_text.strip():
            summary["missing_risk_text"] += 1
        level = (row.get("label_level") or "").lower()
        dept = (row.get("label_dept") or "").lower()
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
        risk_text = row.get("risk_text", "") or ""
        if not risk_text.strip():
            summary["missing_risk_text"] += 1

    return {"rows": rows, "summary": summary}


__all__ = ["load_training_dataset", "load_classify_dataset"]
