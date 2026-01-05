from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton

from ui_components.cards import CardFrame
from ui_components.file_picker_row import FilePickerRow
from ui_strings import BUTTONS, WARNINGS


class ImportPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.model_card = CardFrame("Load Model Set")
        self.model_card.body_layout.addWidget(QLabel("Model set dropdown placeholder"))
        layout.addWidget(self.model_card)

        self.train_card = CardFrame("Select Labeled Training Data (CSV)")
        self.file_picker = FilePickerRow("Training file")
        self.train_card.body_layout.addWidget(self.file_picker)
        self.dataset_summary = QLabel("No file selected")
        self.train_card.body_layout.addWidget(self.dataset_summary)
        self.warning = QLabel(WARNINGS["confirm_labels"])
        self.train_card.body_layout.addWidget(self.warning)
        self.continue_btn = QPushButton(BUTTONS["continue_review"])
        self.continue_btn.setProperty("variant", "primary")
        self.continue_btn.setEnabled(False)
        self.train_card.body_layout.addWidget(self.continue_btn)
        layout.addWidget(self.train_card)
        layout.addStretch(1)

    def set_training_path(self, path: Optional[str]) -> None:
        self.file_picker.set_path(path or "")
        self.continue_btn.setEnabled(bool(path))
        if path:
            self.dataset_summary.setText(path)
        else:
            self.dataset_summary.setText("No file selected")
