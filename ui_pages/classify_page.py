from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ui_components.cards import CardFrame
from ui_components.file_picker_row import FilePickerRow
from ui_strings import BUTTONS


class ClassifyPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.doc_card = CardFrame("Classify Document")
        self.picker = FilePickerRow("Document")
        self.doc_card.body_layout.addWidget(self.picker)
        self.summary = QLabel("No document selected")
        self.doc_card.body_layout.addWidget(self.summary)
        self.run_btn = QPushButton(BUTTONS["classify"])
        self.run_btn.setProperty("variant", "primary")
        self.run_btn.setEnabled(False)
        self.doc_card.body_layout.addWidget(self.run_btn)
        layout.addWidget(self.doc_card)
        layout.addStretch(1)

    def set_document(self, path: str) -> None:
        self.picker.set_path(path)
        self.summary.setText(path or "No document selected")
        self.run_btn.setEnabled(bool(path))
