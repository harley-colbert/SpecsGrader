from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout, QComboBox

from ui_components.cards import CardFrame
from ui_components.file_picker_row import FilePickerRow
from ui_strings import BUTTONS, WARNINGS


class ImportPage(QWidget):
    model_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.model_card = CardFrame("Load Model Set")
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        self.model_dropdown = QComboBox()
        self.model_dropdown.currentTextChanged.connect(self.model_changed)
        model_row.addWidget(self.model_dropdown, stretch=1)

        self.open_folder_btn = QPushButton("Open model folder")
        model_row.addWidget(self.open_folder_btn)

        self.import_btn = QPushButton("Import model set…")
        model_row.addWidget(self.import_btn)

        self.model_card.body_layout.addLayout(model_row)
        self.model_status = QLabel("None currently loaded.")
        self.model_status.setObjectName("status")
        self.model_card.body_layout.addWidget(self.model_status)
        layout.addWidget(self.model_card)

        self.train_card = CardFrame("Select Labeled Training Data (CSV)")
        self.file_picker = FilePickerRow("Training file")
        self.train_card.body_layout.addWidget(self.file_picker)
        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)
        self.rows_label = QLabel("Rows: —")
        self.labeled_label = QLabel("Labeled: —")
        self.modified_label = QLabel("Last Modified: —")
        summary_row.addWidget(self.rows_label)
        summary_row.addWidget(self.labeled_label)
        summary_row.addWidget(self.modified_label)
        summary_row.addStretch(1)
        self.train_card.body_layout.addLayout(summary_row)

        self.dataset_summary = QLabel("No file selected")
        self.dataset_summary.setObjectName("status")
        self.train_card.body_layout.addWidget(self.dataset_summary)
        self.warning = QLabel(WARNINGS["confirm_labels"])
        self.train_card.body_layout.addWidget(self.warning)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.continue_btn = QPushButton(BUTTONS["continue_review"])
        self.continue_btn.setProperty("variant", "primary")
        self.continue_btn.setEnabled(False)
        button_row.addWidget(self.continue_btn)

        self.more_btn = QPushButton(f"{BUTTONS['more']} ▾")
        button_row.addWidget(self.more_btn)
        button_row.addStretch(1)
        self.train_card.body_layout.addLayout(button_row)
        layout.addWidget(self.train_card)
        layout.addStretch(1)

    def set_training_path(self, path: Optional[str]) -> None:
        self.file_picker.set_path(path or "")
        self.continue_btn.setEnabled(bool(path))
        if path:
            self.dataset_summary.setText(path)
        else:
            self.dataset_summary.setText("No file selected")
            self.set_dataset_summary(None, None, None)

    def set_model_sets(self, model_names: list[str], active: Optional[str]) -> None:
        self.model_dropdown.blockSignals(True)
        self.model_dropdown.clear()
        self.model_dropdown.addItems(model_names)
        self.model_dropdown.setCurrentIndex(0 if model_names else -1)
        if active and active in model_names:
            self.model_dropdown.setCurrentText(active)
            self.model_status.setText(active)
        else:
            self.model_status.setText("None currently loaded.")
        self.model_dropdown.blockSignals(False)

    def set_active_model(self, model_name: Optional[str]) -> None:
        if model_name:
            self.model_dropdown.setCurrentText(model_name)
            self.model_status.setText(model_name)
        else:
            if self.model_dropdown.count():
                self.model_dropdown.setCurrentIndex(0)
            self.model_status.setText("None currently loaded.")

    def set_dataset_summary(self, rows: Optional[int], labeled: Optional[int], last_modified: Optional[str]) -> None:
        self.rows_label.setText(f"Rows: {rows if rows is not None else '—'}")
        self.labeled_label.setText(f"Labeled: {labeled if labeled is not None else '—'}")
        self.modified_label.setText(f"Last Modified: {last_modified or '—'}")
