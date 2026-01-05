from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QFileDialog, QTextEdit, QMessageBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QSplitter, QTabWidget, QToolButton, QSizePolicy, QFrame, QTableWidget, QTableWidgetItem, QListWidget, QDialog, QDialogButtonBox, QCheckBox as QDialogCheckBox, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from datetime import datetime
import json
import os
import uuid
from pathlib import Path
import pandas as pd
import logic  # Your business logic module
from ui_theme import LIGHT_THEME, build_stylesheet
from ux_state import AppState, derive_ux_state, summarize_stepper, STEPS
from ui_strings import LABELS, TOOLTIPS, EMPTY_STATES, MIN_LABELS, EXPORT_PRESETS

class DualSpecClassifierApp(QMainWindow):
    def __init__(self, preloaded=None):
        super().__init__()
        self.setWindowTitle("Risk Level & Review Department Classifier - Multipass Ensemble")
        self.resize(1300, 900)
        self.setStyleSheet(build_stylesheet(LIGHT_THEME))

        # --- State ---
        self.train_csv_path = None
        self.classify_file_path = None
        self.models = None
        self.last_pred_df = None
        self.last_model_set_name = logic.get_last_used_model_set() or ""
        self.last_trained_display = "—"
        self.table_filter = "all"
        self.training_in_progress = False
        self.classify_in_progress = False
        self.last_error_stage = ""
        self.base_train_count = 0
        self.train_label_count = 0
        self.persisted_review_labels = []
        self.review_items = []
        self.review_filter = "uncertain"
        self.review_search = ""
        self.review_selected_id = None
        self.model_history = []
        self.active_model_id = None
        self.selected_result_index = None
        self.result_row_index_map = []

        # Use preloaded models if provided (from splash)
        if preloaded is not None:
            self.models = preloaded

        # --- Layout ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 14, 18, 14)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        left_panel = QWidget()
        left_panel.setMinimumWidth(360)
        left_panel.setMaximumWidth(480)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(14)
        left_layout.setContentsMargins(8, 8, 8, 8)

        self.left_header = QLabel(LABELS["left_header"])
        self.left_header.setObjectName("panelTitle")
        left_layout.addWidget(self.left_header)

        # --- Model Set Dropdown ---
        model_group = QGroupBox(LABELS["project_group"])
        model_group.setToolTip(TOOLTIPS["project_group"])
        model_layout = QFormLayout(model_group)
        model_layout.setSpacing(8)
        model_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        model_layout.setContentsMargins(16, 12, 16, 12)

        self.model_label = QLabel("Loaded: 0 classifiers • Last trained: —")
        model_layout.addRow("Status", self.model_label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.currentIndexChanged.connect(self.on_model_select)

        self.refresh_model_dropdown_btn = QPushButton("Refresh")
        self.refresh_model_dropdown_btn.clicked.connect(self.refresh_model_dropdown)

        model_controls = QHBoxLayout()
        model_controls.setSpacing(8)
        model_controls.addWidget(self.model_dropdown, stretch=1)
        model_controls.addWidget(self.refresh_model_dropdown_btn)
        model_layout.addRow("Model set", model_controls)
        left_layout.addWidget(model_group)

        # --- Training file selection ---
        training_group = QGroupBox(LABELS["train_group"])
        training_group.setToolTip(TOOLTIPS["train_group"])
        training_layout = QFormLayout(training_group)
        training_layout.setSpacing(8)
        training_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        training_layout.setContentsMargins(16, 12, 16, 12)

        self.train_file_label = QLabel("No labeled training file selected")
        self.train_browse_btn = QPushButton("Browse")
        self.train_browse_btn.clicked.connect(self.browse_train_file)

        train_file_controls = QHBoxLayout()
        train_file_controls.setSpacing(8)
        train_file_controls.addWidget(self.train_file_label, stretch=1)
        train_file_controls.addWidget(self.train_browse_btn)
        training_layout.addRow("Training file", train_file_controls)

        self.train_btn = QPushButton(LABELS["train_cta"])
        self.train_btn.setEnabled(False)
        self.train_btn.setProperty("variant", "primary")
        self.train_btn.clicked.connect(self.train_model)

        self.train_override_btn = QToolButton()
        self.train_override_btn.setText(LABELS["train_override"])
        self.train_override_btn.setCheckable(False)
        self.train_override_btn.setToolTip(f"Override the recommended minimum of {MIN_LABELS} labeled rows.")
        self.train_override_btn.clicked.connect(lambda: self.train_model(allow_override=True))

        self.save_model_set_btn = QPushButton("Save Model Set")
        self.save_model_set_btn.clicked.connect(self.save_model_set_dialog)

        train_action_controls = QHBoxLayout()
        train_action_controls.setSpacing(8)
        train_action_controls.addWidget(self.train_btn)
        train_action_controls.addWidget(self.train_override_btn)
        train_action_controls.addWidget(self.save_model_set_btn)
        train_action_controls.addStretch(1)
        training_layout.addRow("Actions", train_action_controls)

        self.train_status_label = QLabel("")
        training_layout.addRow("Status", self.train_status_label)
        self.train_summary_label = QLabel("No training run yet.")
        self.train_summary_label.setWordWrap(True)
        training_layout.addRow("Summary", self.train_summary_label)
        left_layout.addWidget(training_group)

        # --- Classify file selection ---
        classify_group = QGroupBox(LABELS["classify_group"])
        classify_group.setToolTip(TOOLTIPS["classify_group"])
        classify_layout = QFormLayout(classify_group)
        classify_layout.setSpacing(8)
        classify_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        classify_layout.setContentsMargins(16, 12, 16, 12)

        self.classify_file_label = QLabel("No file selected for classification")
        self.classify_browse_btn = QPushButton("Browse")
        self.classify_browse_btn.clicked.connect(self.browse_classify_file)

        classify_file_controls = QHBoxLayout()
        classify_file_controls.setSpacing(8)
        classify_file_controls.addWidget(self.classify_file_label, stretch=1)
        classify_file_controls.addWidget(self.classify_browse_btn)
        classify_layout.addRow("Input file", classify_file_controls)

        self.classify_btn = QPushButton(LABELS["classify_cta"])
        self.classify_btn.setEnabled(False)
        self.classify_btn.setProperty("variant", "primary")
        self.classify_btn.clicked.connect(self.classify_items)
        classify_layout.addRow("Run", self.classify_btn)

        self.pred_status_label = QLabel("")
        classify_layout.addRow("Status", self.pred_status_label)
        left_layout.addWidget(classify_group)

        # --- Advanced / Similarity Controls ---
        advanced_group = QGroupBox()
        advanced_group.setTitle("")
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(8)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.clicked.connect(self.toggle_advanced_section)
        advanced_layout.addWidget(self.advanced_toggle)

        self.advanced_contents = QWidget()
        similarity_layout = QFormLayout(self.advanced_contents)
        similarity_layout.setSpacing(8)
        similarity_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        similarity_layout.setContentsMargins(16, 6, 16, 12)

        self.sim_checkbox = QCheckBox("Enable Vector DB Similarity")
        self.sim_checkbox.setChecked(True)
        self.sim_checkbox.stateChanged.connect(self.update_similarity_controls)
        similarity_layout.addRow("Use similarity", self.sim_checkbox)

        self.top_k_label = QLabel("Top K")
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 20)
        self.top_k_spin.setValue(5)
        similarity_layout.addRow(self.top_k_label, self.top_k_spin)

        self.sim_threshold_label = QLabel("Similarity Threshold")
        self.sim_threshold_spin = QDoubleSpinBox()
        self.sim_threshold_spin.setDecimals(2)
        self.sim_threshold_spin.setSingleStep(0.05)
        self.sim_threshold_spin.setRange(0.0, 1.0)
        self.sim_threshold_spin.setValue(0.55)
        similarity_layout.addRow(self.sim_threshold_label, self.sim_threshold_spin)

        self.advanced_contents.setVisible(False)
        advanced_layout.addWidget(self.advanced_contents)
        left_layout.addWidget(advanced_group)
        left_layout.addStretch(1)

        splitter.addWidget(left_panel)

        # --- Results area ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(12, 8, 12, 8)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        self.right_title = QLabel(LABELS["results_header"])
        self.right_title.setObjectName("panelTitle")
        header_row.addWidget(self.right_title)

        self.state_visibility = QLabel("Project: (not set) • Model: None • Trained: —")
        self.state_visibility.setObjectName("status")
        header_row.addWidget(self.state_visibility, stretch=1)

        self.specs_chip = QPushButton("Specs: 0")
        self.risks_chip = QPushButton("Risks: 0")
        self.uncertain_chip = QPushButton("Uncertain: 0")
        for chip in (self.specs_chip, self.risks_chip, self.uncertain_chip):
            chip.setObjectName("summaryChip")
            chip.setCheckable(True)
        self.specs_chip.setToolTip(TOOLTIPS["specs_chip"])
        self.risks_chip.setToolTip(TOOLTIPS["risks_chip"])
        self.uncertain_chip.setToolTip(TOOLTIPS["uncertain_chip_hint"])
        header_row.addWidget(self.specs_chip)
        header_row.addWidget(self.risks_chip)
        header_row.addWidget(self.uncertain_chip)

        self.save_btn = QPushButton(LABELS["export"])
        self.save_btn.setEnabled(False)
        self.save_btn.setProperty("variant", "primary")
        self.save_btn.clicked.connect(self.export_results_dialog)
        header_row.addWidget(self.save_btn)
        right_layout.addLayout(header_row)

        # --- Stepper ---
        self.stepper_frame = QFrame()
        self.stepper_frame.setObjectName("stepperFrame")
        stepper_layout = QVBoxLayout(self.stepper_frame)
        stepper_layout.setContentsMargins(4, 0, 4, 4)
        self.stepper_row = QHBoxLayout()
        self.stepper_row.setSpacing(12)
        self.step_labels = []
        for step in STEPS:
            container = QVBoxLayout()
            icon_label = QLabel("○")
            icon_label.setAlignment(Qt.AlignHCenter)
            title_label = QLabel(step)
            title_label.setAlignment(Qt.AlignHCenter)
            container.addWidget(icon_label)
            container.addWidget(title_label)
            self.step_labels.append((icon_label, title_label))
            wrapper = QVBoxLayout()
            wrapper.addLayout(container)
            stepper_cell = QFrame()
            stepper_cell.setLayout(wrapper)
            self.stepper_row.addWidget(stepper_cell)
        stepper_layout.addLayout(self.stepper_row)
        self.stepper_helper = QLabel("")
        self.stepper_helper.setObjectName("status")
        stepper_layout.addWidget(self.stepper_helper)
        right_layout.addWidget(self.stepper_frame)

        # --- Uncertain Banner ---
        self.uncertain_banner = QFrame()
        self.uncertain_banner.setVisible(False)
        banner_layout = QHBoxLayout(self.uncertain_banner)
        self.uncertain_label = QLabel("")
        self.open_review_btn = QPushButton(LABELS["review_cta"])
        self.open_review_btn.setProperty("variant", "primary")
        self.open_review_btn.clicked.connect(self.open_review_tab)
        banner_layout.addWidget(self.uncertain_label)
        banner_layout.addStretch(1)
        banner_layout.addWidget(self.open_review_btn)
        right_layout.addWidget(self.uncertain_banner)

        # --- Empty state panel ---
        self.empty_state_frame = QFrame()
        self.empty_state_frame.setObjectName("emptyStateFrame")
        self.empty_state_frame.setVisible(False)
        empty_layout = QVBoxLayout(self.empty_state_frame)
        empty_layout.setSpacing(6)
        empty_layout.setContentsMargins(12, 8, 12, 8)
        self.empty_headline = QLabel("")
        self.empty_headline.setObjectName("panelTitle")
        self.empty_body = QLabel("")
        self.empty_body.setWordWrap(True)
        self.empty_cta = QPushButton("")
        self.empty_cta.setProperty("variant", "primary")
        self.empty_cta.clicked.connect(self.handle_empty_state_cta)
        empty_layout.addWidget(self.empty_headline)
        empty_layout.addWidget(self.empty_body)
        empty_layout.addWidget(self.empty_cta, alignment=Qt.AlignLeft)
        right_layout.addWidget(self.empty_state_frame)

        self.results_tabs = QTabWidget()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["Type", "Confidence", "Snippet", "Source", "Status", "Action"])
        self.results_table.setSelectionBehavior(self.results_table.SelectRows)
        self.results_table.setEditTriggers(self.results_table.NoEditTriggers)
        self.results_table.cellClicked.connect(self.on_result_select)

        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setPlaceholderText("Select a result row to see details and evidence.")
        self.details_decision_row = QHBoxLayout()
        self.mark_spec_btn = QPushButton("Mark as Spec")
        self.mark_spec_btn.clicked.connect(lambda: self.details_mark_decision("spec"))
        self.mark_risk_btn = QPushButton("Mark as Risk")
        self.mark_risk_btn.clicked.connect(lambda: self.details_mark_decision("risk"))
        self.mark_not_relevant_btn = QPushButton("Not Relevant")
        self.mark_not_relevant_btn.clicked.connect(lambda: self.details_mark_decision("ignore"))
        for btn in (self.mark_spec_btn, self.mark_risk_btn, self.mark_not_relevant_btn):
            btn.setProperty("variant", "primary")
        self.details_decision_row.addWidget(self.mark_spec_btn)
        self.details_decision_row.addWidget(self.mark_risk_btn)
        self.details_decision_row.addWidget(self.mark_not_relevant_btn)
        self.details_decision_row.addStretch(1)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Run logs will appear here.")
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setPlaceholderText("Summary statistics will appear here.")

        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        details_layout.addWidget(self.details_box)
        details_layout.addLayout(self.details_decision_row)

        self.results_tabs.addTab(self.results_table, "Table")
        self.results_tabs.addTab(details_tab, "Details")
        self.review_tab = QWidget()
        review_layout = QVBoxLayout(self.review_tab)
        review_layout.setContentsMargins(12, 12, 12, 12)
        controls_row = QHBoxLayout()
        self.review_filter_dropdown = QComboBox()
        self.review_filter_dropdown.addItems(["Uncertain", "All", "Accepted", "Corrected", "Skipped"])
        self.review_filter_dropdown.currentIndexChanged.connect(self.refresh_review_table)
        self.review_search_box = QTextEdit()
        self.review_search_box.setFixedHeight(32)
        self.review_search_box.setPlaceholderText("Search text or source…")
        self.review_search_box.textChanged.connect(self.refresh_review_table)
        self.review_remaining_label = QLabel("0 remaining")
        controls_row.addWidget(self.review_filter_dropdown)
        controls_row.addWidget(self.review_search_box, stretch=1)
        controls_row.addWidget(self.review_remaining_label)
        review_layout.addLayout(controls_row)

        self.review_table = QTableWidget()
        self.review_table.setColumnCount(5)
        self.review_table.setHorizontalHeaderLabels(["Snippet", "Predicted", "Confidence", "Source", "Status"])
        self.review_table.setSelectionBehavior(self.review_table.SelectRows)
        self.review_table.setEditTriggers(self.review_table.NoEditTriggers)
        self.review_table.cellClicked.connect(self.on_review_select)
        review_layout.addWidget(self.review_table, stretch=1)

        actions_row = QHBoxLayout()
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.clicked.connect(self.review_accept)
        self.change_label_btn = QPushButton("Change Label")
        self.change_label_btn.clicked.connect(self.review_change_label)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self.review_skip)
        self.save_next_btn = QPushButton("Save & Next")
        self.save_next_btn.clicked.connect(self.review_save_and_next)
        for btn in (self.accept_btn, self.change_label_btn, self.skip_btn, self.save_next_btn):
            btn.setProperty("variant", "primary")
        actions_row.addWidget(self.accept_btn)
        actions_row.addWidget(self.change_label_btn)
        actions_row.addWidget(self.skip_btn)
        actions_row.addWidget(self.save_next_btn)
        actions_row.addStretch(1)
        review_layout.addLayout(actions_row)

        history_row = QHBoxLayout()
        history_col = QVBoxLayout()
        history_label = QLabel("Model History")
        self.model_history_list = QListWidget()
        self.set_active_btn = QPushButton("Set Active")
        self.set_active_btn.clicked.connect(self.set_active_model_from_history)
        history_col.addWidget(history_label)
        history_col.addWidget(self.model_history_list)
        history_col.addWidget(self.set_active_btn)
        history_row.addLayout(history_col)
        history_row.addStretch(1)
        review_layout.addLayout(history_row)
        self.results_tabs.addTab(self.review_tab, "Review")
        self.results_tabs.addTab(self.log_box, "Log")
        self.results_tabs.addTab(self.stats_box, "Stats")
        right_layout.addWidget(self.results_tabs, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.statusBar().showMessage("Ready")

        self.apply_typography_and_spacing(
            section_headers=[
                model_group,
                training_group,
                classify_group,
            ],
            labels=[
                self.train_file_label,
                self.classify_file_label,
                self.top_k_label,
                self.sim_threshold_label,
                self.state_visibility,
                self.uncertain_label,
                self.stepper_helper,
            ],
            status_labels=[
                self.model_label,
                self.train_status_label,
                self.pred_status_label,
            ],
            buttons=[
                self.refresh_model_dropdown_btn,
                self.train_browse_btn,
                self.train_btn,
                self.train_override_btn,
                self.save_model_set_btn,
                self.classify_browse_btn,
                self.classify_btn,
                self.save_btn,
                self.advanced_toggle,
                self.specs_chip,
                self.risks_chip,
                self.uncertain_chip,
                self.open_review_btn,
                self.accept_btn,
                self.change_label_btn,
                self.skip_btn,
                self.save_next_btn,
                self.set_active_btn,
                self.mark_spec_btn,
                self.mark_risk_btn,
                self.mark_not_relevant_btn,
            ],
            inputs=[
                self.model_dropdown,
                self.top_k_spin,
                self.sim_threshold_spin,
            ],
        )
        self.update_similarity_controls()
        self.update_results_summary()
        self.connect_counter_actions()
        self.load_persisted_labels()
        self.refresh_label_counts()
        self.load_model_history()
        self.refresh_model_dropdown()
        self.refresh_ux_state()

    # --- UI Logic Functions ---
    def apply_typography_and_spacing(
        self,
        section_headers,
        labels,
        status_labels,
        buttons,
        inputs,
    ):
        header_font = QFont("Segoe UI", 15, QFont.DemiBold)
        panel_header_font = QFont("Segoe UI", 18, QFont.DemiBold)
        label_font = QFont("Segoe UI", 13)
        status_font = QFont("Segoe UI", 12)
        chip_font = QFont("Segoe UI", 11, QFont.DemiBold)

        for header in section_headers:
            header.setFont(header_font)

        for title_label in (self.left_header, self.right_title):
            title_label.setFont(panel_header_font)

        for label in labels:
            label.setFont(label_font)

        self.sim_checkbox.setFont(label_font)

        for status_label in status_labels:
            status_label.setFont(status_font)
            status_label.setObjectName("status")

        for chip in (self.specs_chip, self.risks_chip, self.uncertain_chip):
            chip.setFont(chip_font)

        control_min_height = 38
        button_min_width = 160
        button_padding = "padding: 6px 14px;"
        input_padding = "padding: 6px 10px;"

        for button in buttons:
            if button in {self.specs_chip, self.risks_chip, self.uncertain_chip}:
                button.setMinimumHeight(30)
                button.setMinimumWidth(110)
                button.setStyleSheet("padding: 4px 10px;")
                button.setCheckable(True)
                button.setFlat(True)
            else:
                button.setMinimumHeight(control_min_height)
                button.setMinimumWidth(button_min_width)
                button.setStyleSheet(button_padding)

        for input_widget in inputs:
            input_widget.setMinimumHeight(control_min_height)
            input_widget.setStyleSheet(input_padding)

        self.advanced_toggle.setMinimumHeight(32)
        self.advanced_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # --- Review Queue Helpers ---
    def load_persisted_labels(self):
        path = Path("models/review_labels.jsonl")
        if not path.exists():
            return []
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        self.persisted_review_labels = items
        return items

    def persisted_label_count(self):
        if not self.persisted_review_labels:
            self.load_persisted_labels()
        return len(self.persisted_review_labels)

    def write_persisted_label(self, entry):
        path = Path("models/review_labels.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        # remove duplicate by text+source
        existing = [e for e in self.persisted_review_labels if not (e.get("text") == entry.get("text") and e.get("source_file") == entry.get("source_file"))]
        existing.append(entry)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in existing:
                f.write(json.dumps(row) + "\n")
        tmp_path.replace(path)
        self.persisted_review_labels = existing
        self.refresh_label_counts()

    def refresh_label_counts(self):
        self.train_label_count = self.base_train_count + self.persisted_label_count()
        self.apply_guardrails(self.gather_app_state())

    # --- Model history helpers ---
    def model_history_path(self):
        return Path("models/model_history.json")

    def load_model_history(self):
        path = self.model_history_path()
        if not path.exists():
            self.model_history = []
            return
        try:
            self.model_history = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.model_history = []
        if self.model_history and not self.active_model_id:
            self.active_model_id = self.model_history[-1].get("id")
        self.render_model_history()

    def save_model_history(self):
        path = self.model_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_history, indent=2), encoding="utf-8")

    def render_model_history(self):
        if not hasattr(self, "model_history_list"):
            return
        self.model_history_list.clear()
        for entry in reversed(self.model_history):
            label = f"{entry.get('name','')} — {entry.get('trained_at','')} — {entry.get('metrics','N/A')}"
            if entry.get("id") == self.active_model_id:
                label = f"* {label}"
            self.model_history_list.addItem(label)

    def add_model_history_entry(self, name, metrics, dataset_size):
        entry = {
            "id": str(uuid.uuid4()),
            "name": name,
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics or "N/A",
            "dataset_size": dataset_size,
        }
        self.model_history.append(entry)
        self.active_model_id = entry["id"]
        if not self.last_model_set_name:
            self.last_model_set_name = name
        self.save_model_history()
        self.render_model_history()

    def set_active_model_from_history(self):
        row = self.model_history_list.currentRow()
        if row < 0:
            return
        index = len(self.model_history) - 1 - row
        if index < 0 or index >= len(self.model_history):
            return
        entry = self.model_history[index]
        self.active_model_id = entry.get("id")
        if entry.get("name"):
            try:
                files_dict = logic.load_model_set(entry["name"])
                self.models = logic.load_all_models(files_dict=files_dict)
                self.last_model_set_name = entry["name"]
                self.classify_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load selected model:\n{e}")
        self.render_model_history()
        self.refresh_ux_state()
    def connect_counter_actions(self):
        self.specs_chip.clicked.connect(lambda: self.set_table_filter("all", focus_tab=True))
        self.risks_chip.clicked.connect(lambda: self.set_table_filter("risks", focus_tab=True))
        self.uncertain_chip.clicked.connect(lambda: self.set_table_filter("uncertain", focus_tab=True, open_review=True))

    def toggle_advanced_section(self):
        is_open = self.advanced_toggle.isChecked()
        self.advanced_contents.setVisible(is_open)
        self.advanced_toggle.setArrowType(Qt.DownArrow if is_open else Qt.RightArrow)

    def set_table_filter(self, filter_key, focus_tab=False, open_review=False):
        self.table_filter = filter_key
        for chip, key in (
            (self.specs_chip, "all"),
            (self.risks_chip, "risks"),
            (self.uncertain_chip, "uncertain"),
        ):
            chip.setChecked(key == filter_key)
        self.render_results_table()
        if focus_tab:
            self.results_tabs.setCurrentWidget(self.results_table)
        if open_review:
            self.open_review_tab()

    def open_review_tab(self):
        index = self.results_tabs.indexOf(self.review_tab)
        if index != -1:
            self.results_tabs.setCurrentIndex(index)

    def get_filtered_results(self):
        if self.last_pred_df is None:
            return None
        df = self.last_pred_df
        if self.table_filter == "risks":
            if "Final Risk Level" in df.columns:
                return df[df["Final Risk Level"].astype(str).str.len() > 0]
            return df
        if self.table_filter == "uncertain":
            if "Needs Review" in df.columns:
                return df[df["Needs Review"].astype(bool)]
            return df.iloc[0:0]
        return df

    def render_results_table(self):
        filtered = self.get_filtered_results()
        if filtered is None:
            self.results_table.clear()
            self.results_table.setPlaceholderText(
                "Select an input file and click Classify. Results will appear here."
            )
            return
        self.populate_results_table(filtered)

    def build_review_queue(self):
        self.review_items = []
        if self.last_pred_df is None:
            return
        df = self.last_pred_df.copy()
        if "Needs Review" in df.columns:
            df = df[df["Needs Review"].astype(bool)]
        for idx, row in df.iterrows():
            confidence = 0.0
            if "Top Similarity" in row:
                confidence = float(row.get("Top Similarity") or 0.0)
            elif "Similarity Score" in row:
                confidence = float(row.get("Similarity Score") or 0.0)
            item = {
                "id": str(uuid.uuid4()),
                "text": row.get("Risk Description", ""),
                "pred_risk": row.get("Final Risk Level", row.get("Risk Level", "")),
                "pred_dept": row.get("Final Review Dept", row.get("Review Department", "")),
                "confidence": confidence,
                "source": self.classify_file_path or "",
                "status": "uncertain",
                "selected_risk": "",
                "selected_dept": "",
            }
            self.review_items.append(item)
        self.refresh_review_table()

    def refresh_review_table(self):
        from PySide6.QtWidgets import QTableWidgetItem
        filter_text = self.review_filter_dropdown.currentText().lower() if hasattr(self, "review_filter_dropdown") else "uncertain"
        search = ""
        if hasattr(self, "review_search_box"):
            search = self.review_search_box.toPlainText().strip().lower()
        items = self.review_items or []
        if filter_text == "uncertain":
            filtered = [i for i in items if i["status"] in {"uncertain", "pending"}]
        elif filter_text == "accepted":
            filtered = [i for i in items if i["status"] == "accepted"]
        elif filter_text == "corrected":
            filtered = [i for i in items if i["status"] == "corrected"]
        elif filter_text == "skipped":
            filtered = [i for i in items if i["status"] == "skipped"]
        else:
            filtered = list(items)
        if search:
            filtered = [i for i in filtered if search in i.get("text", "").lower() or search in i.get("source", "").lower()]
        filtered = sorted(filtered, key=lambda i: i.get("confidence", 0.0))
        self.review_table.setRowCount(len(filtered))
        for r, item in enumerate(filtered):
            self.review_table.setItem(r, 0, QTableWidgetItem((item.get("text") or "")[:80]))
            self.review_table.setItem(r, 1, QTableWidgetItem(f"{item.get('pred_risk','')} / {item.get('pred_dept','')}"))
            self.review_table.setItem(r, 2, QTableWidgetItem(f"{item.get('confidence',0.0):.2f}"))
            self.review_table.setItem(r, 3, QTableWidgetItem(os.path.basename(item.get("source",""))))
            self.review_table.setItem(r, 4, QTableWidgetItem(item.get("status","")))
        remaining = len([i for i in items if i["status"] in {"uncertain", "pending"}])
        self.review_remaining_label.setText(f"{remaining} remaining")
        if filtered:
            self.review_table.selectRow(0)
            self.review_selected_id = filtered[0]["id"]
            self.update_details_from_review(filtered[0])
        else:
            self.review_selected_id = None

    def update_details_from_review(self, item):
        self.details_box.setPlainText(
            "\n".join(
                [
                    f"Text: {item.get('text','')}",
                    f"Predicted Risk: {item.get('pred_risk','')}",
                    f"Predicted Dept: {item.get('pred_dept','')}",
                    f"Confidence: {item.get('confidence',0.0):.2f}",
                    f"Status: {item.get('status','')}",
                ]
            )
        )

    def get_review_item_by_row(self, row):
        if row < 0 or row >= self.review_table.rowCount():
            return None
        snippet = self.review_table.item(row, 0).text()
        for item in self.review_items:
            if item.get("text", "").startswith(snippet):
                return item
        return None

    def on_result_select(self, row, col):
        if not self.result_row_index_map:
            return
        if row < 0 or row >= len(self.result_row_index_map):
            return
        orig_idx = self.result_row_index_map[row]
        self.selected_result_index = orig_idx
        self.update_details_from_result(orig_idx)

    def update_details_from_result(self, orig_idx):
        if self.last_pred_df is None or orig_idx not in self.last_pred_df.index:
            return
        row = self.last_pred_df.loc[orig_idx]
        confidence = row.get("Top Similarity", row.get("Similarity Trust", 0.0))
        text = str(row.get("Risk Description", ""))
        risk = row.get("Final Risk Level", row.get("Risk Level", ""))
        dept = row.get("Final Review Dept", row.get("Review Department", ""))
        status = "Needs Review" if row.get("Needs Review") else "Predicted"
        self.details_box.setPlainText(
            "\n".join(
                [
                    f"Status: {status}",
                    f"Predicted Risk: {risk}",
                    f"Predicted Dept: {dept}",
                    f"Confidence: {confidence}",
                    "",
                    text,
                ]
            )
        )

    def result_accept(self, table_row):
        idx = self.result_row_index_map[table_row]
        self.update_result_decision(idx, status="accepted")

    def result_correct(self, table_row):
        from PySide6.QtWidgets import QInputDialog
        idx = self.result_row_index_map[table_row]
        risk, ok = QInputDialog.getText(self, "Correct Risk Level", "Risk Level:")
        if not ok:
            return
        dept, ok = QInputDialog.getText(self, "Correct Review Department", "Review Department:")
        if not ok:
            return
        self.update_result_decision(idx, status="corrected", selected_risk=risk, selected_dept=dept)

    def result_send_to_review(self, table_row):
        idx = self.result_row_index_map[table_row]
        if self.last_pred_df is not None and "Needs Review" in self.last_pred_df.columns:
            self.last_pred_df.loc[idx, "Needs Review"] = True
        self.build_review_queue()
        self.refresh_results_summary_and_tables()

    def result_ignore(self, table_row):
        idx = self.result_row_index_map[table_row]
        self.update_result_decision(idx, status="ignored", mark_review=False)

    def details_mark_decision(self, decision):
        if self.selected_result_index is None:
            return
        if decision == "spec":
            self.update_result_decision(self.selected_result_index, status="accepted", selected_risk="spec", selected_dept="")
        elif decision == "risk":
            self.update_result_decision(self.selected_result_index, status="accepted", selected_risk="risk", selected_dept="")
        elif decision == "ignore":
            self.update_result_decision(self.selected_result_index, status="ignored", mark_review=False)

    def update_result_decision(self, idx, status, selected_risk=None, selected_dept=None, mark_review=True):
        if self.last_pred_df is None or idx not in self.last_pred_df.index:
            return
        if "Needs Review" in self.last_pred_df.columns:
            self.last_pred_df.loc[idx, "Needs Review"] = status == "uncertain"
        self.last_pred_df.loc[idx, "Status"] = status
        if selected_risk:
            self.last_pred_df.loc[idx, "Final Risk Level"] = selected_risk
        if selected_dept:
            self.last_pred_df.loc[idx, "Final Review Dept"] = selected_dept
        item = {
            "id": str(uuid.uuid4()),
            "text": self.last_pred_df.loc[idx].get("Risk Description", ""),
            "pred_risk": self.last_pred_df.loc[idx].get("Final Risk Level", ""),
            "pred_dept": self.last_pred_df.loc[idx].get("Final Review Dept", ""),
            "confidence": self.last_pred_df.loc[idx].get("Top Similarity", 0.0),
            "source": self.classify_file_path or "",
            "status": status,
            "selected_risk": selected_risk or self.last_pred_df.loc[idx].get("Final Risk Level", ""),
            "selected_dept": selected_dept or self.last_pred_df.loc[idx].get("Final Review Dept", ""),
        }
        if status in {"accepted", "corrected"}:
            self.persist_review_decision(item)
        if status == "ignored":
            # ensure it won't be counted as uncertain
            if "Needs Review" in self.last_pred_df.columns:
                self.last_pred_df.loc[idx, "Needs Review"] = False
        if mark_review and status == "accepted":
            if "Needs Review" in self.last_pred_df.columns:
                self.last_pred_df.loc[idx, "Needs Review"] = False
        self.refresh_results_summary_and_tables()

    def on_review_select(self, row, col):
        item = self.get_review_item_by_row(row)
        if item:
            self.review_selected_id = item["id"]
            self.update_details_from_review(item)

    def review_accept(self):
        item = self.get_selected_review_item()
        if not item:
            return
        item["status"] = "accepted"
        item["selected_risk"] = item.get("pred_risk", "")
        item["selected_dept"] = item.get("pred_dept", "")
        self.persist_review_decision(item)
        self.refresh_review_table()
        self.refresh_results_summary_and_tables()

    def review_change_label(self):
        from PySide6.QtWidgets import QInputDialog
        item = self.get_selected_review_item()
        if not item:
            return
        risk, ok = QInputDialog.getText(self, "Change Risk Level", "Risk Level:", text=item.get("pred_risk",""))
        if not ok:
            return
        dept, ok = QInputDialog.getText(self, "Change Review Department", "Review Department:", text=item.get("pred_dept",""))
        if not ok:
            return
        item["status"] = "corrected"
        item["selected_risk"] = risk
        item["selected_dept"] = dept
        self.persist_review_decision(item)
        self.refresh_review_table()
        self.refresh_results_summary_and_tables()

    def review_skip(self):
        item = self.get_selected_review_item()
        if not item:
            return
        item["status"] = "skipped"
        self.persist_review_decision(item)
        self.refresh_review_table()
        self.refresh_results_summary_and_tables()

    def review_save_and_next(self):
        self.review_accept()
        current_row = self.review_table.currentRow()
        next_row = current_row + 1
        if next_row < self.review_table.rowCount():
            self.review_table.selectRow(next_row)
            self.on_review_select(next_row, 0)

    def get_selected_review_item(self):
        if self.review_selected_id:
            for item in self.review_items:
                if item["id"] == self.review_selected_id:
                    return item
        if self.review_table.currentRow() >= 0:
            return self.get_review_item_by_row(self.review_table.currentRow())
        return None

    def persist_review_decision(self, item):
        entry = {
            "id": item["id"],
            "text": item.get("text", ""),
            "source_file": os.path.basename(item.get("source", "")),
            "predicted_risk": item.get("pred_risk", ""),
            "predicted_dept": item.get("pred_dept", ""),
            "selected_risk": item.get("selected_risk", ""),
            "selected_dept": item.get("selected_dept", ""),
            "confidence": item.get("confidence", 0.0),
            "label_source": "Review Queue",
            "status": item.get("status", ""),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "user_note": "",
        }
        self.write_persisted_label(entry)
        self.refresh_label_counts()
        if self.last_pred_df is not None and "Risk Description" in self.last_pred_df.columns:
            mask = self.last_pred_df["Risk Description"] == item.get("text", "")
            if mask.any():
                self.last_pred_df.loc[mask, "Status"] = item.get("status", "")
                if "Needs Review" in self.last_pred_df.columns:
                    self.last_pred_df.loc[mask, "Needs Review"] = False
            self.refresh_results_summary_and_tables()

    def set_training_stage(self, stage_text):
        self.train_status_label.setText(stage_text)
        self.log_message(f"[Training] {stage_text}")

    def refresh_results_summary_and_tables(self):
        self.update_results_summary(self.last_pred_df)
        filtered = self.get_filtered_results()
        if filtered is not None:
            self.populate_results_table(filtered)
        self.build_review_queue()

    # --- Export dialog and presets ---
    def export_results_dialog(self):
        if self.last_pred_df is None:
            QMessageBox.information(self, "No Results", "Run classification before exporting.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(LABELS["export_dialog_title"])
        layout = QFormLayout(dialog)
        format_combo = QComboBox()
        format_combo.addItems(["CSV", "JSON"])
        preset_combo = QComboBox()
        preset_combo.addItems(list(EXPORT_PRESETS.keys()))
        conf_toggle = QDialogCheckBox("Include confidence scores")
        source_toggle = QDialogCheckBox("Include source info")
        context_toggle = QDialogCheckBox("Include context")
        conf_toggle.setChecked(True)
        source_toggle.setChecked(True)
        context_toggle.setChecked(False)
        layout.addRow("Format", format_combo)
        layout.addRow("Preset", preset_combo)
        layout.addRow(conf_toggle)
        layout.addRow(source_toggle)
        layout.addRow(context_toggle)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow(buttons)

        def on_preset_change(idx):
            name = preset_combo.currentText()
            preset = EXPORT_PRESETS.get(name, {})
            conf_toggle.setChecked(preset.get("include_confidence", True))
            source_toggle.setChecked(preset.get("include_source", True))
            context_toggle.setChecked(preset.get("include_context", False))
        preset_combo.currentIndexChanged.connect(on_preset_change)
        on_preset_change(0)

        def accept():
            fmt = format_combo.currentText().lower()
            preset_name = preset_combo.currentText()
            include_conf = conf_toggle.isChecked()
            include_source = source_toggle.isChecked()
            include_context = context_toggle.isChecked()
            fname_filter = "CSV Files (*.csv)" if fmt == "csv" else "JSON Files (*.json)"
            fname, _ = QFileDialog.getSaveFileName(self, LABELS["export_dialog_title"], "", fname_filter)
            if fname:
                self.perform_export(fname, preset_name, format_override=fmt, include_conf=include_conf, include_source=include_source, include_context=include_context)
            dialog.accept()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

    def handle_empty_state_cta(self):
        state = self.gather_app_state()
        ux_state = derive_ux_state(state)
        config = EMPTY_STATES.get(ux_state, {})
        action = config.get("action")
        if action == "browse_train":
            self.browse_train_file()
        elif action == "train":
            self.train_model()
        elif action == "refresh_models":
            self.refresh_model_dropdown()
        elif action == "browse_classify":
            self.browse_classify_file()
        elif action == "open_review":
            self.open_review_tab()
        elif action == "export":
            self.export_results_dialog()

    def update_empty_state(self, ux_state):
        should_show = not self.last_pred_df
        config = EMPTY_STATES.get(ux_state)
        if not should_show or not config:
            self.empty_state_frame.setVisible(False)
            return
        self.empty_headline.setText(config.get("headline", ""))
        self.empty_body.setText(config.get("body", ""))
        self.empty_cta.setText(config.get("cta", ""))
        self.empty_state_frame.setVisible(True)

    def update_state_visibility(self):
        project_label = "(not set)"
        if self.classify_file_path:
            project_label = os.path.basename(self.classify_file_path)
        model_label = self.last_model_set_name or "None"
        if self.active_model_id:
            model_label = f"{model_label} ({self.active_model_id[:6]})"
        trained = self.last_trained_display or "—"
        self.state_visibility.setText(
            LABELS["state_visibility"].format(project=project_label, model=model_label, trained=trained)
        )

    def update_stepper(self, statuses, helper_text):
        icon_map = {
            "not_started": ("○", "#9aa1a9"),
            "in_progress": ("●", "#2b7de9"),
            "done": ("✓", "#1f8b4c"),
            "needs_attention": ("!", "#e67e22"),
        }
        for (icon_label, title_label), status in zip(self.step_labels, statuses):
            icon, color = icon_map.get(status, ("○", "#9aa1a9"))
            icon_label.setText(icon)
            icon_label.setStyleSheet(f"color:{color}; font-size:18px;")
            title_label.setStyleSheet(f"color:{color};")
        self.stepper_helper.setText(helper_text)

    def update_review_banner(self, uncertain_count):
        has_uncertain = uncertain_count > 0
        self.uncertain_banner.setVisible(has_uncertain)
        if has_uncertain:
            self.uncertain_label.setText(LABELS["uncertain_banner"].format(count=uncertain_count))

    def gather_app_state(self):
        uncertain_count = 0
        if self.last_pred_df is not None and "Needs Review" in self.last_pred_df.columns:
            uncertain_count = int(self.last_pred_df["Needs Review"].astype(bool).sum())
        try:
            saved_sets = logic.list_model_sets()
            has_saved = len(saved_sets) > 0
        except Exception:
            has_saved = False
        return AppState(
            has_model_set=bool(self.models),
            has_train_file=bool(self.train_csv_path),
            has_classify_file=bool(self.classify_file_path),
            has_results=self.last_pred_df is not None,
            uncertain_count=uncertain_count,
            training_in_progress=self.training_in_progress,
            classify_in_progress=self.classify_in_progress,
            label_count=self.train_label_count,
            has_saved_model_sets=has_saved,
            last_error_stage=self.last_error_stage,
        )

    def apply_guardrails(self, state: AppState):
        # Train gating
        can_train = state.has_train_file and not state.training_in_progress and state.label_count >= MIN_LABELS
        if not state.has_train_file:
            self.train_btn.setToolTip(TOOLTIPS["train_disabled_no_file"])
        elif state.training_in_progress:
            self.train_btn.setToolTip(TOOLTIPS["train_disabled_running"])
        elif state.label_count < MIN_LABELS:
            self.train_btn.setToolTip(TOOLTIPS["train_disabled_threshold"])
        else:
            self.train_btn.setToolTip("")
        self.train_btn.setEnabled(can_train)
        # Train override visibility
        show_override = state.has_train_file and state.label_count < MIN_LABELS
        self.train_override_btn.setVisible(show_override)

        # Classify gating
        can_classify = state.has_model_set and state.has_classify_file and not state.classify_in_progress
        if not state.has_model_set:
            self.classify_btn.setToolTip(TOOLTIPS["classify_disabled_no_model"])
        elif not state.has_classify_file:
            self.classify_btn.setToolTip(TOOLTIPS["classify_disabled_no_file"])
        elif state.classify_in_progress:
            self.classify_btn.setToolTip(TOOLTIPS["classify_disabled_running"])
        else:
            self.classify_btn.setToolTip("")
        self.classify_btn.setEnabled(can_classify)

        # Export gating
        can_export = state.has_results and not state.classify_in_progress and not state.training_in_progress
        self.save_btn.setEnabled(bool(can_export))
        if not can_export:
            self.save_btn.setToolTip(TOOLTIPS["export_gating"])
        else:
            self.save_btn.setToolTip("")

    def refresh_ux_state(self):
        state = self.gather_app_state()
        ux_state = derive_ux_state(state)
        statuses, helper_text = summarize_stepper(ux_state, state)
        self.update_stepper(statuses, helper_text)
        self.update_state_visibility()
        self.update_review_banner(state.uncertain_count)
        self.update_empty_state(ux_state)
        self.apply_guardrails(state)

    def refresh_model_dropdown(self):
        sets = ["None (Unload)"] + logic.list_model_sets()
        self.model_dropdown.clear()
        self.model_dropdown.addItems(sets)
        # Select last used, or first
        ix = sets.index(self.last_model_set_name) if self.last_model_set_name in sets else 0
        self.model_dropdown.setCurrentIndex(ix)
        self.refresh_ux_state()

    def on_model_select(self, idx):
        selection = self.model_dropdown.currentText()
        if selection == "None (Unload)":
            logic.unload_all_models(self.models)
            self.models = None
            self.model_label.setText(f"Loaded: 0 classifiers • Last trained: {self.last_trained_display}")
            self.classify_btn.setEnabled(False)
            self.last_model_set_name = ""
        else:
            try:
                files_dict = logic.load_model_set(selection)
                self.models = logic.load_all_models(files_dict=files_dict)
                self.model_label.setText(
                    f"Loaded: {len(files_dict)} classifiers • Last trained: {self.last_trained_display}"
                )
                self.classify_btn.setEnabled(True)
                self.last_model_set_name = selection
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load model set:\n{e}")
                self.models = None
                self.model_label.setText(f"Loaded: 0 classifiers • Last trained: {self.last_trained_display}")
                self.classify_btn.setEnabled(False)
        if selection and self.model_history:
            match = next((m for m in self.model_history if m.get("name") == selection), None)
            if match:
                self.active_model_id = match.get("id")
        self.render_model_history()
        self.update_similarity_controls()
        self.refresh_ux_state()

    def browse_train_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Training CSV", "", "CSV Files (*.csv)")
        if fname:
            self.train_csv_path = fname
            self.train_file_label.setText(fname.split("/")[-1])
            try:
                df = pd.read_csv(fname)
                self.base_train_count = len(df)
            except Exception:
                self.base_train_count = 0
        else:
            self.train_csv_path = None
            self.train_file_label.setText("No file selected")
            self.base_train_count = 0
        self.refresh_label_counts()
        self.refresh_ux_state()

    def train_model(self, allow_override=False):
        if self.train_label_count < MIN_LABELS and not allow_override:
            QMessageBox.information(
                self,
                "Training blocked",
                f"A minimum of {MIN_LABELS} labeled rows is recommended before training. Use 'Train anyway' to override.",
            )
            return
        if self.train_label_count < MIN_LABELS and allow_override:
            proceed = QMessageBox.question(
                self,
                "Train anyway?",
                f"Only {self.train_label_count} labeled rows detected (recommended: {MIN_LABELS}). Proceed anyway?",
            )
            if proceed != QMessageBox.Yes:
                return
        self.train_status_label.setText("Training models (this may take a moment)...")
        self.log_message("Training started.")
        self.statusBar().showMessage("Training in progress…")
        self.repaint()
        self.training_in_progress = True
        self.last_error_stage = ""
        self.refresh_ux_state()
        try:
            self.set_training_stage("Preparing data")
            report, files_dict = logic.train_all_models(self.train_csv_path, return_file_dict=True)
            self.set_training_stage("Training and evaluating")
            self.results_table.clear()
            self.results_table.append(self.format_report_output("Training Report", report))
            self.set_training_stage("Loading models")
            self.models = logic.load_all_models(files_dict=files_dict)
            self.last_trained_display = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.train_status_label.setText("Training complete. All models and embeddings saved.")
            self.log_message("Training complete.")
            self.statusBar().showMessage("Training complete.")
            dataset_size = self.base_train_count + self.persisted_label_count()
            summary_metrics = report.splitlines()[0] if report else "Metrics: N/A"
            self.train_summary_label.setText(
                f"{summary_metrics}\nTrained at: {self.last_trained_display}\nDataset size: {dataset_size}"
            )
            model_name = self.last_model_set_name or files_dict.get("vector_collection", "default")
            self.add_model_history_entry(model_name, summary_metrics, dataset_size)
            self.set_training_stage("Saving model")

            # ====== Prompt to Save Model Set immediately after training completes ======
            from PySide6.QtWidgets import QInputDialog
            set_name, ok = QInputDialog.getText(self, "Save Model Set", "Enter a name for this model set:")
            if ok and set_name:
                try:
                    logic.save_model_set(set_name, files_dict)
                    self.model_label.setText(
                        f"Loaded: {len(files_dict)} classifiers • Last trained: {self.last_trained_display}"
                    )
                    self.last_model_set_name = set_name
                    self.refresh_model_dropdown()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not save model set:\n{e}")
            self.set_training_stage("Done")
            # ====== END NEW CODE ======

            self.save_model_set_btn.setEnabled(True)
            self.refresh_model_dropdown()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during training: {e}")
            self.train_status_label.setText("Error during training.")
            self.log_message(f"Training error: {e}")
            self.statusBar().showMessage("Training failed.")
            self.last_error_stage = "train"
        finally:
            self.training_in_progress = False
            self.refresh_ux_state()

    def save_model_set_dialog(self):
        if not self.models:
            QMessageBox.information(self, "No Models", "No models to save. Train or load first.")
            return
        from PySide6.QtWidgets import QInputDialog
        set_name, ok = QInputDialog.getText(self, "Save Model Set", "Enter a name for this model set:")
        if ok and set_name:
            try:
                files_dict = logic.get_current_model_file_paths(self.models)
                logic.save_model_set(set_name, files_dict)
                self.model_label.setText(
                    f"Loaded: {len(files_dict)} classifiers • Last trained: {self.last_trained_display}"
                )
                self.last_model_set_name = set_name
                self.refresh_model_dropdown()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save model set:\n{e}")

    def browse_classify_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select File to Classify", "",
                                               "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)")
        if fname:
            self.classify_file_path = fname
            self.classify_file_label.setText(fname.split("/")[-1])
            if self.models:
                self.classify_btn.setEnabled(True)
        else:
            self.classify_file_path = None
            self.classify_file_label.setText("No file selected")
            self.classify_btn.setEnabled(False)
        self.refresh_ux_state()

    def update_similarity_controls(self):
        has_vector_db = bool(self.models and self.models.get("vector_db"))
        enable_controls = self.sim_checkbox.isChecked() and has_vector_db
        self.top_k_spin.setEnabled(enable_controls)
        self.top_k_label.setEnabled(enable_controls)
        self.sim_threshold_spin.setEnabled(enable_controls)
        self.sim_threshold_label.setEnabled(enable_controls)
        if not has_vector_db:
            self.sim_checkbox.setToolTip("Vector DB not loaded; similarity will fall back to embeddings.")
        else:
            self.sim_checkbox.setToolTip("")

    def classify_items(self):
        self.pred_status_label.setText("Classifying (Multipass)...")
        self.log_message("Classification started.")
        self.statusBar().showMessage("Classification in progress…")
        self.repaint()
        self.classify_in_progress = True
        self.last_error_stage = ""
        self.refresh_ux_state()
        if self.active_model_id:
            self.log_message(f"Using active model id: {self.active_model_id}")
        if not self.models or not self.classify_file_path:
            QMessageBox.critical(self, "Error", "Models or file missing.")
            self.pred_status_label.setText("Missing input.")
            self.last_error_stage = "classify"
            self.classify_in_progress = False
            self.refresh_ux_state()
            return
        try:
            if self.classify_file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(self.classify_file_path)
            else:
                df = pd.read_csv(self.classify_file_path)
            result_df = logic.multipass_classify(
                df,
                self.models,
                self.sim_checkbox.isChecked(),
                top_k=self.top_k_spin.value(),
                similarity_threshold=self.sim_threshold_spin.value()
            )
            result_df = result_df.copy()
            if "Needs Review" in result_df.columns:
                result_df["Needs Review"] = result_df["Needs Review"].astype(bool)
            result_df["Status"] = result_df.get("Status", "pending")

            evidence_json = result_df.get("Similarity Evidence")
            if evidence_json is not None:
                def build_top_evidence(payload):
                    if not payload:
                        return 0.0, ""
                    try:
                        data = json.loads(payload)
                    except (TypeError, json.JSONDecodeError):
                        return 0.0, ""
                    if not data:
                        return 0.0, ""
                    top = data[0]
                    similarity = float(top.get("similarity", 0.0)) if isinstance(top, dict) else 0.0
                    text = top.get("text", "") if isinstance(top, dict) else ""
                    return similarity, text

                top_values = evidence_json.apply(build_top_evidence)
                result_df["Top Similarity"] = top_values.apply(lambda val: val[0])
                result_df["Top Match (Preview)"] = top_values.apply(
                    lambda val: (val[1][:120] + "…") if val[1] and len(val[1]) > 120 else val[1]
                )
                result_df["Top-K Evidence (JSON)"] = evidence_json
            else:
                if "Similarity Score" in result_df.columns:
                    result_df["Top Similarity"] = result_df["Similarity Score"]
                if "Similarity Match" in result_df.columns:
                    result_df["Top Match (Preview)"] = result_df["Similarity Match"].apply(
                        lambda text: (text[:120] + "…") if isinstance(text, str) and len(text) > 120 else text
                    )

            self.last_pred_df = result_df
            self.set_table_filter("all")
            self.save_btn.setEnabled(True)
            self.pred_status_label.setText("Classification (Multipass) done.")
            self.update_results_summary(result_df)
            self.build_review_queue()
            self.log_message("Classification complete.")
            self.statusBar().showMessage("Classification complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during classification: {e}")
            self.pred_status_label.setText("Classification error.")
            self.log_message(f"Classification error: {e}")
            self.statusBar().showMessage("Classification failed.")
            self.last_error_stage = "classify"
        finally:
            self.classify_in_progress = False
            self.refresh_ux_state()

    def save_results(self):
        if self.last_pred_df is None:
            QMessageBox.information(self, "No Results", "No results to save yet.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, LABELS["export_dialog_title"], "", "CSV Files (*.csv);;JSON Files (*.json)")
        if not fname:
            return
        preset = "SpecGrader Standard"
        self.perform_export(fname, preset, format_override=None, include_conf=True, include_source=True, include_context=False)

    def perform_export(self, fname, preset_name, format_override=None, include_conf=True, include_source=True, include_context=False):
        fmt = format_override or ("json" if fname.lower().endswith(".json") else "csv")
        preset = EXPORT_PRESETS.get(preset_name, {})
        df = self.last_pred_df.copy() if self.last_pred_df is not None else None
        if df is None:
            QMessageBox.information(self, "No Results", "No results to export.")
            return
        # Build metadata columns
        df = df.copy()
        df["active_model_name"] = self.last_model_set_name or "unknown"
        df["active_model_version"] = self.active_model_id or ""
        df["trained_at"] = self.last_trained_display or ""
        df["project_name"] = os.path.basename(self.classify_file_path) if self.classify_file_path else ""
        if not include_conf:
            for col in ["Top Similarity", "Similarity Trust", "Rule Trust", "Classic Trust"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
        if not include_source:
            if "Source File" in df.columns:
                df.drop(columns=["Source File"], inplace=True)
        if not include_context:
            for col in ["Top Match (Preview)", "Similarity Evidence"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
        if preset.get("include_trust"):
            pass  # trust columns already present
        else:
            for col in ["Rule Trust", "Classic Trust", "Similarity Trust", "Semantic Risk Proba", "Semantic Dept Proba"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
        try:
            if fmt == "json":
                df.to_json(fname, orient="records", indent=2)
            else:
                df.to_csv(fname, index=False)
            self.log_message(f"Exported results to {fname} with preset '{preset_name}' (format={fmt})")
            QMessageBox.information(self, "Exported", f"Results exported to {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not export file:\n{str(e)}")

    def format_report_output(self, title, body):
        if not body:
            return f"{title}\n{'=' * len(title)}\nNo details available."
        return f"{title}\n{'=' * len(title)}\n\n{body}"

    def format_results_table(self, result_df, filter_key="all"):
        title = "Classification Results"
        if filter_key == "risks":
            title += " — Risks"
        elif filter_key == "uncertain":
            title += " — Needs Review"
        columns = ", ".join(result_df.columns)
        lines = [
            title,
            "=" * len(title),
            f"Rows: {len(result_df)}",
            f"Columns: {columns}",
            "",
            result_df.to_string(index=False)
        ]
        return "\n".join(lines)

    def populate_results_table(self, df):
        self.result_row_index_map = []
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(6)
        for row_idx, (orig_idx, row) in enumerate(df.iterrows()):
            self.result_row_index_map.append(orig_idx)
            confidence = 0.0
            if "Top Similarity" in row:
                confidence = float(row.get("Top Similarity") or 0.0)
            elif "Similarity Trust" in row:
                confidence = float(row.get("Similarity Trust") or 0.0)
            type_label = "Uncertain" if bool(row.get("Needs Review")) else "Risk"
            snippet = str(row.get("Risk Description", ""))[:120]
            source = os.path.basename(self.classify_file_path) if self.classify_file_path else ""
            status = row.get("Status", "pending")
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(type_label))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(f"{confidence:.2f}"))
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(snippet))
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(source))
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(status))
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(0, 0, 0, 0)
            btn_accept = QPushButton("Accept")
            btn_accept.clicked.connect(lambda _, r=row_idx: self.result_accept(r))
            btn_correct = QPushButton("Correct…")
            btn_correct.clicked.connect(lambda _, r=row_idx: self.result_correct(r))
            btn_review = QPushButton("Review")
            btn_review.clicked.connect(lambda _, r=row_idx: self.result_send_to_review(r))
            btn_ignore = QPushButton("Ignore")
            btn_ignore.clicked.connect(lambda _, r=row_idx: self.result_ignore(r))
            for b in (btn_accept, btn_correct, btn_review, btn_ignore):
                b.setProperty("variant", "primary")
                action_layout.addWidget(b)
            action_widget.setLayout(action_layout)
            self.results_table.setCellWidget(row_idx, 5, action_widget)
            self.results_table.resizeColumnsToContents()

    def update_results_summary(self, result_df=None):
        if result_df is None:
            self.specs_chip.setText("Specs: 0")
            self.risks_chip.setText("Risks: 0")
            self.uncertain_chip.setText("Uncertain: 0")
            self.stats_box.setPlainText("")
            self.refresh_ux_state()
            return
        total = len(result_df)
        risks = 0
        if "Risk Level" in result_df.columns:
            risks = result_df["Risk Level"].astype(str).str.lower().str.contains("high|medium|risk").sum()
        elif "Final Risk Level" in result_df.columns:
            risks = result_df["Final Risk Level"].astype(str).str.lower().str.contains("high|medium|risk").sum()
        uncertain = 0
        if "Needs Review" in result_df.columns:
            uncertain = result_df["Needs Review"].astype(bool).sum()
        self.specs_chip.setText(f"Specs: {total}")
        self.risks_chip.setText(f"Risks: {int(risks)}")
        self.uncertain_chip.setText(f"Uncertain: {int(uncertain)}")
        self.stats_box.setPlainText(
            "\n".join(
                [
                    "Summary Stats",
                    "-------------",
                    f"Specs: {total}",
                    f"Risks: {int(risks)}",
                    f"Uncertain: {int(uncertain)}",
                ]
            )
        )
        self.update_review_banner(int(uncertain))
        self.refresh_ux_state()

    def log_message(self, message):
        if not message:
            return
        self.log_box.append(message)
