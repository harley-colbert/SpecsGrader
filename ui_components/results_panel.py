from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget, QTableView)

from ui_models import PandasTableModel, summarize_results


class ResultsPanel(QFrame):
    rowSelected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.summary_row = QHBoxLayout()
        self.summary_row.setSpacing(8)
        self.summary_label = QLabel("No results yet.")
        self.summary_label.setObjectName("status")
        self.summary_row.addWidget(self.summary_label)
        self.summary_row.addStretch(1)
        layout.addLayout(self.summary_row)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Table tab
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        self.table_view = QTableView()
        self.model = PandasTableModel()
        self.table_view.setModel(self.model)
        self.table_view.selectionModel().selectionChanged.connect(self.on_selection_changed)
        table_layout.addWidget(self.table_view)
        self.tabs.addTab(table_tab, "Table")

        # Log tab
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setAcceptRichText(False)
        self.tabs.addTab(self.log_box, "Logs")

        # Stats tab
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.tabs.addTab(self.stats_box, "Stats")

    def set_results(self, df: Optional[pd.DataFrame]) -> None:
        self.model.set_dataframe(df if df is not None else pd.DataFrame())
        counts = summarize_results(df)
        self.summary_label.setText(
            f"Specs: {counts['total']} • Risks: {counts['risks']} • Uncertain: {counts['uncertain']}"
        )

    def append_log(self, message: str) -> None:
        if message:
            self.log_box.append(message)

    def set_stats(self, text: str) -> None:
        self.stats_box.setPlainText(text)

    def on_selection_changed(self):
        indexes = self.table_view.selectionModel().selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        data = self.model.get_row(row)
        self.rowSelected.emit(data)
