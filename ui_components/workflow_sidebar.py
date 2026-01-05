from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QFrame, QVBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton, QWidget, QCheckBox, QLabel)

from ui_strings import WORKFLOW_STEPS


class WorkflowSidebar(QFrame):
    stepSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.list = QListWidget()
        for step in WORKFLOW_STEPS:
            item = QListWidgetItem(step)
            self.list.addItem(item)
        self.list.currentRowChanged.connect(self.stepSelected)
        layout.addWidget(self.list)

        # Advanced section placeholder
        self.advanced_label = QLabel("Advanced")
        self.similarity_checkbox = QCheckBox("Enable similarity")
        self.similarity_checkbox.setChecked(True)
        advanced = QWidget()
        adv_layout = QVBoxLayout(advanced)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.addWidget(self.advanced_label)
        adv_layout.addWidget(self.similarity_checkbox)
        layout.addWidget(advanced)
        layout.addStretch(1)

    def set_active_step(self, index: int) -> None:
        if 0 <= index < self.list.count():
            self.list.setCurrentRow(index)

    def set_similarity_checked(self, enabled: bool) -> None:
        self.similarity_checkbox.setChecked(enabled)
