from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from backend.app.config.xlsx_contract import SPECIFIC_RISK_COL
from backend.app.services.xlsx_output_service import write_classify_predictions_to_xlsx


def _read_cell(path: Path, cell: str) -> str:
    wb = load_workbook(path)
    sheet = wb.active
    value = sheet[cell].value
    return "" if value is None else str(value)


def _write_fixture(path: Path) -> None:
    rows = [["", "", "", "", "", "", ""] for _ in range(4)]
    rows += [
        ["", "", "", "Spec Medium", "", "", ""],
        ["", "", "", "Spec High", "Existing specific risk", "", ""],
        ["", "", "", "Spec Low", "", "", ""],
    ]
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Standards Risk Matrix Sample", index=False, header=False)


def test_specific_risk_written_only_for_medium_plus(tmp_path: Path):
    target = tmp_path / "contract_v410_specific_risk_in.xlsx"
    _write_fixture(target)

    results = [
        {"source_row": 5, "pred_level": "medium", "pred_dept": "mechanical"},
        {"source_row": 6, "pred_level": "high", "pred_dept": "electrical"},
        {"source_row": 7, "pred_level": "low", "pred_dept": "controls"},
    ]

    output_path = write_classify_predictions_to_xlsx(str(target), results, overwrite_specific_risk=True)

    assert _read_cell(output_path, f"{SPECIFIC_RISK_COL}5") != ""
    assert _read_cell(output_path, f"{SPECIFIC_RISK_COL}6") != ""
    assert _read_cell(output_path, f"{SPECIFIC_RISK_COL}7") == ""


def test_specific_risk_respects_overwrite_policy(tmp_path: Path):
    target = tmp_path / "contract_v410_specific_risk_in.xlsx"
    _write_fixture(target)

    results = [
        {"source_row": 5, "pred_level": "medium", "pred_dept": "mechanical"},
        {"source_row": 6, "pred_level": "high", "pred_dept": "electrical"},
    ]

    output_path = write_classify_predictions_to_xlsx(str(target), results, overwrite_specific_risk=False)

    assert _read_cell(output_path, f"{SPECIFIC_RISK_COL}5") != ""
    assert _read_cell(output_path, f"{SPECIFIC_RISK_COL}6") == "Existing specific risk"
