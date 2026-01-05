from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from ui_strings import BUTTONS, STATUS_LABELS, WORKFLOW_STEPS


class TopBar(QFrame):
    exportClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        self.title_label = QLabel(WORKFLOW_STEPS[0])
        self.title_label.setObjectName("topBarTitle")
        layout.addWidget(self.title_label)

        self.status_line = QLabel()
        self.status_line.setObjectName("status")
        layout.addWidget(self.status_line, stretch=1)

        self.export_btn = QPushButton(BUTTONS["export"])
        self.export_btn.setProperty("variant", "primary")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.exportClicked.emit)
        layout.addWidget(self.export_btn)

        self.project_name: Optional[str] = None
        self.model_set: Optional[str] = None
        self.trained_text: str = "—"
        self._update_status_line()

    # --- Public API ---
    def set_title(self, step_name: str) -> None:
        self.title_label.setText(step_name)

    def set_project(self, project: Optional[str]) -> None:
        self.project_name = project or "(not project selected)"
        self._update_status_line()

    def set_model_status(self, model_name: Optional[str]) -> None:
        self.model_set = model_name or "None"
        self._update_status_line()

    def set_trained_label(self, trained_text: Optional[str]) -> None:
        self.trained_text = trained_text or "—"
        self._update_status_line()

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_btn.setEnabled(enabled)

    # --- Helpers ---
    def _update_status_line(self) -> None:
        project_text = STATUS_LABELS["project"].format(project=self.project_name or "(not project selected)")
        model_text = STATUS_LABELS["model_set"].format(model=self.model_set or "None")
        trained_text = STATUS_LABELS["trained"].format(trained=self.trained_text or "—")
        self.status_line.setText(f"{project_text} • {model_text} • {trained_text}")
