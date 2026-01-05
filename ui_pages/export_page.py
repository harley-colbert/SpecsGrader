from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ui_components.cards import CardFrame
from ui_strings import BUTTONS


class ExportPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.preset_card = CardFrame("Export Options")
        self.preset_card.body_layout.addWidget(QLabel("Preset dropdown placeholder"))
        self.export_btn = QPushButton(BUTTONS["export"])
        self.export_btn.setProperty("variant", "primary")
        self.preset_card.body_layout.addWidget(self.export_btn)
        layout.addWidget(self.preset_card)
        layout.addStretch(1)
