from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class PandasTableModel(QAbstractTableModel):
    def __init__(self, dataframe: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self._df = dataframe if dataframe is not None else pd.DataFrame()

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = dataframe if dataframe is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # type: ignore
        if parent.isValid():
            return 0
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()) -> int:  # type: ignore
        if parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # type: ignore
        if not index.isValid() or role not in {Qt.DisplayRole, Qt.EditRole}:
            return None
        value = self._df.iat[index.row(), index.column()]
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # type: ignore
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            try:
                return str(self._df.columns[section])
            except IndexError:
                return ""
        return str(section + 1)

    def get_row(self, row: int) -> dict:
        if row < 0 or row >= len(self._df.index):
            return {}
        return self._df.iloc[row].to_dict()


def summarize_results(df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {"total": 0, "risks": 0, "uncertain": 0}
    total = len(df)
    uncertain = int(df.get("Needs Review", pd.Series(dtype=bool)).astype(bool).sum()) if "Needs Review" in df else 0
    risks = 0
    if "Risk Level" in df.columns:
        risks = df["Risk Level"].astype(str).str.lower().str.contains("risk|high|medium").sum()
    elif "Final Risk Level" in df.columns:
        risks = df["Final Risk Level"].astype(str).str.lower().str.contains("risk|high|medium").sum()
    return {"total": int(total), "risks": int(risks), "uncertain": int(uncertain)}
