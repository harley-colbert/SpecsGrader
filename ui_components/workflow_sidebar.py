from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
        layout.setSpacing(16)

        # Header
        header = QLabel("Workflow")
        header.setObjectName("sidebarSectionTitle")
        layout.addWidget(header)

        # Step list
        self.list = QListWidget()
        self.list.setObjectName("workflowList")
        self.list.setFrameShape(QFrame.NoFrame)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setUniformItemSizes(True)

        for index, step in enumerate(WORKFLOW_STEPS):
            item = QListWidgetItem(step)
            item.setData(Qt.UserRole, index)
            self.list.addItem(item)

        self.list.currentRowChanged.connect(self.stepSelected)
        layout.addWidget(self.list)

        # Divider
        divider = QFrame()
        divider.setObjectName("sidebarDivider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        # Advanced section
        self.advanced_label = QLabel("Advanced")
        self.advanced_label.setObjectName("sidebarSectionTitle")

        self.similarity_checkbox = QCheckBox("Similarity Settings")
        self.similarity_checkbox.setObjectName("similarityToggle")
        self.similarity_checkbox.setChecked(True)

        advanced = QWidget()
        adv_layout = QVBoxLayout(advanced)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(4)
        adv_layout.addWidget(self.advanced_label)
        adv_layout.addWidget(self.similarity_checkbox)

        layout.addWidget(advanced)
        layout.addStretch(1)

    def set_active_step(self, index: int) -> None:
        if 0 <= index < self.list.count():
            self.list.setCurrentRow(index)

    def set_similarity_checked(self, enabled: bool) -> None:
        self.similarity_checkbox.setChecked(enabled)
