from __future__ import annotations

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel


class CardFrame(QFrame):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        if title:
            label = QLabel(title)
            label.setObjectName("panelTitle")
            layout.addWidget(label)
        self.body_layout = layout
