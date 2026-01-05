from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from ui_strings import BUTTONS


class FilePickerRow(QFrame):
    browseRequested = Signal()
    pathChanged = Signal(str)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.label = QLabel(label)
        layout.addWidget(self.label)

        self.path_display = QLabel(BUTTONS["browse"])
        layout.addWidget(self.path_display, stretch=1)

        self.browse_btn = QPushButton(BUTTONS["browse"])
        self.browse_btn.clicked.connect(self.browseRequested.emit)
        layout.addWidget(self.browse_btn)

    def set_path(self, path: str) -> None:
        self.path_display.setText(path or BUTTONS["browse"])
        self.pathChanged.emit(path)
