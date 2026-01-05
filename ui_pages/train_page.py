from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ui_components.cards import CardFrame
from ui_strings import BUTTONS


class TrainPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.readiness_card = CardFrame("Training Readiness")
        self.readiness_label = QLabel("Select training data to begin.")
        self.readiness_card.body_layout.addWidget(self.readiness_label)
        self.train_btn = QPushButton(BUTTONS["train"])
        self.train_btn.setProperty("variant", "primary")
        self.train_card_footer = QVBoxLayout()
        self.readiness_card.body_layout.addWidget(self.train_btn)
        layout.addWidget(self.readiness_card)

        self.output_card = CardFrame("Training Output")
        self.metrics_label = QLabel("No training run yet.")
        self.save_btn = QPushButton("Save Model Set")
        self.save_btn.setEnabled(False)
        self.output_card.body_layout.addWidget(self.metrics_label)
        self.output_card.body_layout.addWidget(self.save_btn)
        layout.addWidget(self.output_card)
        layout.addStretch(1)

    def set_ready(self, ready: bool) -> None:
        self.train_btn.setEnabled(ready)

    def set_metrics(self, text: str) -> None:
        self.metrics_label.setText(text)

    def set_save_enabled(self, enabled: bool) -> None:
        self.save_btn.setEnabled(enabled)
