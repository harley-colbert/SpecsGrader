from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel,
                               QPushButton)

from ui_strings import APP_TITLE, BUTTONS, STATUS_LABELS


class TopBar(QFrame):
    exportClicked = Signal()
    modelSetChanged = Signal(str)
    projectChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        self.title_label = QLabel(APP_TITLE)
        self.title_label.setObjectName("panelTitle")
        layout.addWidget(self.title_label)

        self.project_dropdown = QComboBox()
        self.project_dropdown.setEditable(True)
        self.project_dropdown.lineEdit().setPlaceholderText("Project name…")
        self.project_dropdown.currentTextChanged.connect(self.on_project_changed)
        layout.addWidget(self.project_dropdown, stretch=1)

        self.model_dropdown = QComboBox()
        self.model_dropdown.currentTextChanged.connect(self.modelSetChanged)
        layout.addWidget(self.model_dropdown, stretch=1)

        self.trained_label = QLabel(STATUS_LABELS["trained"].format(trained="—"))
        self.trained_label.setObjectName("status")
        layout.addWidget(self.trained_label)

        self.export_btn = QPushButton(BUTTONS["export"])
        self.export_btn.setProperty("variant", "primary")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.exportClicked.emit)
        layout.addWidget(self.export_btn)

    # --- Public API ---
    def set_models(self, model_names: list[str], active: Optional[str]) -> None:
        self.model_dropdown.blockSignals(True)
        self.model_dropdown.clear()
        self.model_dropdown.addItems(model_names)
        if active and active in model_names:
            self.model_dropdown.setCurrentText(active)
        self.model_dropdown.blockSignals(False)

    def set_project(self, project: Optional[str]) -> None:
        self.project_dropdown.blockSignals(True)
        self.project_dropdown.clear()
        if project:
            self.project_dropdown.addItem(project)
            self.project_dropdown.setCurrentIndex(0)
        self.project_dropdown.blockSignals(False)

    def set_trained_label(self, trained_text: str) -> None:
        self.trained_label.setText(STATUS_LABELS["trained"].format(trained=trained_text or "—"))

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_btn.setEnabled(enabled)

    # --- Slots ---
    def on_project_changed(self, text: str) -> None:
        if text:
            self.projectChanged.emit(text)
