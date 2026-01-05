from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui_strings import WORKFLOW_STEPS


class WorkflowSidebar(QFrame):
    stepSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel("Workflow")
        header.setObjectName("sectionTitle")
        layout.addWidget(header)

        self.list = QListWidget()
        self.list.setObjectName("workflowList")
        self.list.setSelectionMode(QListWidget.SingleSelection)
        for step in WORKFLOW_STEPS:
            item = QListWidgetItem(step)
            self.list.addItem(item)
        self.list.currentRowChanged.connect(self.stepSelected)
        layout.addWidget(self.list)

        # Advanced section placeholder
        self.advanced_label = QLabel("Advanced ▾")
        self.advanced_label.setObjectName("sectionTitle")
        self.similarity_checkbox = QCheckBox("Similarity Settings")
        self.similarity_checkbox.setChecked(True)
        self.similarity_checkbox.setTristate(False)
        self.similarity_checkbox.setAutoExclusive(False)
        advanced = QWidget()
        adv_layout = QVBoxLayout(advanced)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(6)
        adv_layout.addWidget(self.advanced_label)
        adv_layout.addWidget(self.similarity_checkbox)
        layout.addWidget(advanced)
        layout.addStretch(1)

    def set_active_step(self, index: int) -> None:
        if 0 <= index < self.list.count():
            self.list.setCurrentRow(index)

    def set_similarity_checked(self, enabled: bool) -> None:
        self.similarity_checkbox.setChecked(enabled)

    def active_step(self) -> int:
        return self.list.currentRow()
