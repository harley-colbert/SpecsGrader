import pathlib

import pandas as pd
import pytest

from backend.app.services.ingest_service import (
    load_classify_dataset,
    load_training_dataset,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_training_xlsx_selects_quote_sheet_and_row5():
    dataset = load_training_dataset(str(FIXTURES / "training_sample.xlsx"))

    assert dataset["summary"]["total_rows"] == 1
    row = dataset["rows"][0]
    assert row["source_row"] == 5
    assert row["risk_text"] == "Risk text XLSX"
    assert row["label_level"] == "medium"
    assert row["label_dept"] == "electrical"


def test_training_csv_reads_columns_and_validates_labels(tmp_path):
    csv_path = tmp_path / "bad_training.csv"
    csv_path.write_text(
        "\n\n\n\n,,,,,,\n,,,,,BAD,dept\n,,,,,medium,controls\n",
        encoding="utf-8",
    )

    dataset = load_training_dataset(str(csv_path))

    assert dataset["summary"]["total_rows"] == 2
    assert dataset["summary"]["missing_risk_text"] == 2
    assert dataset["summary"]["invalid_levels"] == 1
    assert dataset["summary"]["invalid_departments"] == 1


def test_classify_loader_handles_csv_and_xlsx():
    csv_dataset = load_classify_dataset(str(FIXTURES / "classify_sample.csv"))
    xlsx_dataset = load_classify_dataset(str(FIXTURES / "classify_sample.xlsx"))

    assert csv_dataset["rows"][0]["risk_text"] == "Classify text A"
    assert xlsx_dataset["rows"][0]["risk_text"] == "Classify XLSX Text"
    assert csv_dataset["summary"]["total_rows"] == 2


def test_missing_quote_sheet_raises(tmp_path):
    path = tmp_path / "no_quote.xlsx"
    df = pd.DataFrame({"E": ["risk"], "F": ["low"], "G": ["mechanical"]})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Other", index=False, header=False)

    with pytest.raises(ValueError):
        load_training_dataset(str(path))
