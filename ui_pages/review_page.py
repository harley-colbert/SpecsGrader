from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QSplitter, QVBoxLayout, QWidget

from ui_components.cards import CardFrame
from ui_components.inspector_panel import InspectorPanel


class ReviewPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Filter"))
        toolbar.addWidget(QLineEdit())
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("0 remaining"))
        layout.addLayout(toolbar)

        splitter = QSplitter()
        self.table_card = CardFrame("Review Queue")
        splitter.addWidget(self.table_card)

        self.inspector = InspectorPanel()
        splitter.addWidget(self.inspector)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.save_next = QLabel("Save & Next")
        footer.addWidget(self.save_next)
        layout.addLayout(footer)
