from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from backend.app.config.xlsx_contract import DEPT_COL, RISK_LEVEL_COL
from backend.app.services.xlsx_output_service import write_classify_predictions_to_xlsx


def _read_cell(path: Path, cell: str) -> str:
    wb = load_workbook(path)
    sheet = wb.active
    value = sheet[cell].value
    return "" if value is None else str(value)


def _write_fixture(path: Path) -> None:
    rows = [["", "", "", "", "", "", ""] for _ in range(4)]
    rows += [
        ["", "", "", "Spec A", "Risk A", "", ""],
        ["", "", "", "Spec B", "Risk B", "low", "mechanical"],
    ]
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Standards Risk Matrix Sample", index=False, header=False)


def test_classify_writes_fg_with_overwrite(tmp_path: Path):
    target = tmp_path / "contract_v410_classify_in.xlsx"
    _write_fixture(target)

    results = [
        {"source_row": 5, "pred_level": "medium", "pred_dept": "electrical"},
        {"source_row": 6, "pred_level": "high", "pred_dept": "controls"},
    ]

    output_path = write_classify_predictions_to_xlsx(str(target), results, overwrite_predictions=True)

    assert _read_cell(output_path, f"{RISK_LEVEL_COL}5") == "medium"
    assert _read_cell(output_path, f"{DEPT_COL}5") == "electrical"
    assert _read_cell(output_path, f"{RISK_LEVEL_COL}6") == "high"
    assert _read_cell(output_path, f"{DEPT_COL}6") == "controls"


def test_classify_respects_existing_predictions_when_disabled(tmp_path: Path):
    target = tmp_path / "contract_v410_classify_in.xlsx"
    _write_fixture(target)

    results = [
        {"source_row": 5, "pred_level": "medium", "pred_dept": "electrical"},
        {"source_row": 6, "pred_level": "high", "pred_dept": "controls"},
    ]

    output_path = write_classify_predictions_to_xlsx(str(target), results, overwrite_predictions=False)

    assert _read_cell(output_path, f"{RISK_LEVEL_COL}5") == "medium"
    assert _read_cell(output_path, f"{DEPT_COL}5") == "electrical"
    assert _read_cell(output_path, f"{RISK_LEVEL_COL}6") == "low"
    assert _read_cell(output_path, f"{DEPT_COL}6") == "mechanical"
