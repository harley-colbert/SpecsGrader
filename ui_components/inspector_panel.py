from __future__ import annotations

from typing import Dict

from PySide6.QtWidgets import QFrame, QFormLayout, QLabel, QTextEdit, QVBoxLayout


class InspectorPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title = QLabel("Inspector")
        self.title.setObjectName("panelTitle")
        layout.addWidget(self.title)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.label_type = QLabel("—")
        self.label_confidence = QLabel("—")
        self.label_source = QLabel("—")

        form.addRow("Type", self.label_type)
        form.addRow("Confidence", self.label_confidence)
        form.addRow("Source", self.label_source)

        layout.addLayout(form)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Select a row to inspect details.")
        layout.addWidget(self.text)

    def set_data(self, row: Dict) -> None:
        if not row:
            self.label_type.setText("—")
            self.label_confidence.setText("—")
            self.label_source.setText("—")
            self.text.clear()
            return
        self.label_type.setText(str(row.get("Final Risk Level") or row.get("Risk Level") or "—"))
        confidence = row.get("Top Similarity") or row.get("Similarity Trust") or row.get("Semantic Risk Proba")
        if isinstance(confidence, float):
            confidence = f"{confidence:.2f}"
        self.label_confidence.setText(str(confidence or "—"))
        self.label_source.setText(str(row.get("Label Source", "—")))
        snippet = row.get("Risk Description") or row.get("text") or ""
        self.text.setPlainText(str(snippet))
