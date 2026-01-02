from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QFileDialog, QTextEdit, QMessageBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QSplitter, QTabWidget, QToolButton, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from datetime import datetime
import json
import pandas as pd
import logic  # Your business logic module
from ui_theme import LIGHT_THEME, build_stylesheet

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

        self.left_header = QLabel("Control Panel")
        self.left_header.setObjectName("panelTitle")
        left_layout.addWidget(self.left_header)

        # --- Model Set Dropdown ---
        model_group = QGroupBox("Project / Model Set")
        model_layout = QFormLayout(model_group)
        model_layout.setSpacing(8)
        model_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        model_layout.setContentsMargins(16, 12, 16, 12)

        self.model_label = QLabel("Loaded: 0 classifiers • Last trained: —")
        model_layout.addRow("Status", self.model_label)

        self.model_dropdown = QComboBox()
        self.refresh_model_dropdown()
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
        training_group = QGroupBox("Train")
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

        self.train_btn = QPushButton("Train")
        self.train_btn.setEnabled(False)
        self.train_btn.setProperty("variant", "primary")
        self.train_btn.clicked.connect(self.train_model)

        self.save_model_set_btn = QPushButton("Save Model Set")
        self.save_model_set_btn.clicked.connect(self.save_model_set_dialog)

        train_action_controls = QHBoxLayout()
        train_action_controls.setSpacing(8)
        train_action_controls.addWidget(self.train_btn)
        train_action_controls.addWidget(self.save_model_set_btn)
        train_action_controls.addStretch(1)
        training_layout.addRow("Actions", train_action_controls)

        self.train_status_label = QLabel("")
        training_layout.addRow("Status", self.train_status_label)
        left_layout.addWidget(training_group)

        # --- Classify file selection ---
        classify_group = QGroupBox("Classify")
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

        self.classify_btn = QPushButton("Classify (Multipass)")
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
        self.right_title = QLabel("Results")
        self.right_title.setObjectName("panelTitle")
        header_row.addWidget(self.right_title)

        self.specs_chip = QLabel("Specs: 0")
        self.risks_chip = QLabel("Risks: 0")
        self.uncertain_chip = QLabel("Uncertain: 0")
        for chip in (self.specs_chip, self.risks_chip, self.uncertain_chip):
            chip.setObjectName("summaryChip")
            header_row.addWidget(chip)

        header_row.addStretch(1)

        self.save_btn = QPushButton("Save Results to CSV")
        self.save_btn.setEnabled(False)
        self.save_btn.setProperty("variant", "primary")
        self.save_btn.clicked.connect(self.save_results)
        header_row.addWidget(self.save_btn)
        right_layout.addLayout(header_row)

        self.results_tabs = QTabWidget()
        self.results_table = QTextEdit()
        self.results_table.setReadOnly(True)
        self.results_table.setPlaceholderText(
            "Select an input file and click Classify. Results will appear here."
        )
        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setPlaceholderText("Select a result row to see details and evidence.")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Run logs will appear here.")
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setPlaceholderText("Summary statistics will appear here.")

        self.results_tabs.addTab(self.results_table, "Table")
        self.results_tabs.addTab(self.details_box, "Details")
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
                self.save_model_set_btn,
                self.classify_browse_btn,
                self.classify_btn,
                self.save_btn,
                self.advanced_toggle,
            ],
            inputs=[
                self.model_dropdown,
                self.top_k_spin,
                self.sim_threshold_spin,
            ],
        )
        self.update_similarity_controls()
        self.update_results_summary()

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
            button.setMinimumHeight(control_min_height)
            button.setMinimumWidth(button_min_width)
            button.setStyleSheet(button_padding)

        for input_widget in inputs:
            input_widget.setMinimumHeight(control_min_height)
            input_widget.setStyleSheet(input_padding)

        self.advanced_toggle.setMinimumHeight(32)
        self.advanced_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def toggle_advanced_section(self):
        is_open = self.advanced_toggle.isChecked()
        self.advanced_contents.setVisible(is_open)
        self.advanced_toggle.setArrowType(Qt.DownArrow if is_open else Qt.RightArrow)

    def refresh_model_dropdown(self):
        sets = ["None (Unload)"] + logic.list_model_sets()
        self.model_dropdown.clear()
        self.model_dropdown.addItems(sets)
        # Select last used, or first
        ix = sets.index(self.last_model_set_name) if self.last_model_set_name in sets else 0
        self.model_dropdown.setCurrentIndex(ix)

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
        self.update_similarity_controls()

    def browse_train_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Training CSV", "", "CSV Files (*.csv)")
        if fname:
            self.train_csv_path = fname
            self.train_file_label.setText(fname.split("/")[-1])
            self.train_btn.setEnabled(True)
        else:
            self.train_csv_path = None
            self.train_file_label.setText("No file selected")
            self.train_btn.setEnabled(False)

    def train_model(self):
        self.train_status_label.setText("Training models (this may take a moment)...")
        self.log_message("Training started.")
        self.statusBar().showMessage("Training in progress…")
        self.repaint()
        try:
            report, files_dict = logic.train_all_models(self.train_csv_path, return_file_dict=True)
            self.results_table.clear()
            self.results_table.append(self.format_report_output("Training Report", report))
            self.models = logic.load_all_models(files_dict=files_dict)
            self.last_trained_display = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.train_status_label.setText("Training complete. All models and embeddings saved.")
            self.log_message("Training complete.")
            self.statusBar().showMessage("Training complete.")

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
            # ====== END NEW CODE ======

            self.save_model_set_btn.setEnabled(True)
            self.refresh_model_dropdown()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during training: {e}")
            self.train_status_label.setText("Error during training.")
            self.log_message(f"Training error: {e}")
            self.statusBar().showMessage("Training failed.")

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
        if not self.models or not self.classify_file_path:
            QMessageBox.critical(self, "Error", "Models or file missing.")
            self.pred_status_label.setText("Missing input.")
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
            output = self.format_results_table(result_df)
            self.results_table.clear()
            self.results_table.append(output)
            self.save_btn.setEnabled(True)
            self.pred_status_label.setText("Classification (Multipass) done.")
            self.update_results_summary(result_df)
            self.log_message("Classification complete.")
            self.statusBar().showMessage("Classification complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during classification: {e}")
            self.pred_status_label.setText("Classification error.")
            self.log_message(f"Classification error: {e}")
            self.statusBar().showMessage("Classification failed.")

    def save_results(self):
        if self.last_pred_df is None:
            QMessageBox.information(self, "No Results", "No results to save yet.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv);;All Files (*)")
        if fname:
            try:
                self.last_pred_df.to_csv(fname, index=False)
                QMessageBox.information(self, "Saved", f"Results saved to {fname}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file:\n{str(e)}")

    def format_report_output(self, title, body):
        if not body:
            return f"{title}\n{'=' * len(title)}\nNo details available."
        return f"{title}\n{'=' * len(title)}\n\n{body}"

    def format_results_table(self, result_df):
        title = "Classification Results"
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

    def update_results_summary(self, result_df=None):
        if result_df is None:
            self.specs_chip.setText("Specs: 0")
            self.risks_chip.setText("Risks: 0")
            self.uncertain_chip.setText("Uncertain: 0")
            self.stats_box.setPlainText("")
            return
        total = len(result_df)
        risks = 0
        if "Risk Level" in result_df.columns:
            risks = result_df["Risk Level"].astype(str).str.lower().str.contains("high|medium|risk").sum()
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

    def log_message(self, message):
        if not message:
            return
        self.log_box.append(message)
