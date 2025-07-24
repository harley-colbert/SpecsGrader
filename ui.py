from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QTextEdit, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt
import pandas as pd
import logic  # Your business logic module

class DualSpecClassifierApp(QMainWindow):
    def __init__(self, preloaded=None):
        super().__init__()
        self.setWindowTitle("Risk Level & Review Department Classifier - Multipass Ensemble")
        self.resize(1300, 900)
        self.setStyleSheet("""
            QMainWindow { background: #222326; }
            QLabel, QCheckBox { color: #eaf6fb; font-family: 'Segoe UI', Arial, sans-serif; }
            QPushButton, QComboBox { font-family: 'Segoe UI', Arial, sans-serif; font-size:14px; }
            QComboBox, QPushButton { height: 36px; border-radius: 8px; }
            QPushButton { background: #44c0fa; color: #1e2227; border: none; }
            QPushButton:disabled { background: #444d56; color: #949ba7; }
            QComboBox { background: #232629; color: #eaf6fb; }
            QTextEdit { background: #232629; color: #fff; font-family: Consolas, monospace; font-size:13px; border-radius: 8px; }
            QCheckBox::indicator { width: 20px; height: 20px; }
        """)

        # --- State ---
        self.train_csv_path = None
        self.classify_file_path = None
        self.models = None
        self.last_pred_df = None
        self.last_model_set_name = logic.get_last_used_model_set() or ""

        # Use preloaded models if provided (from splash)
        if preloaded is not None:
            self.models = preloaded

        # --- Layout ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(14)
        layout.setContentsMargins(28, 20, 28, 20)

        grid = QGridLayout()
        grid.setSpacing(10)
        layout.addLayout(grid)

        # --- Model Set Dropdown ---
        self.model_label = QLabel("No classifiers loaded")
        grid.addWidget(self.model_label, 0, 0, 1, 1)

        self.model_dropdown = QComboBox()
        self.refresh_model_dropdown()
        self.model_dropdown.currentIndexChanged.connect(self.on_model_select)
        grid.addWidget(self.model_dropdown, 0, 1, 1, 2)

        self.refresh_model_dropdown_btn = QPushButton("Refresh Model Sets")
        self.refresh_model_dropdown_btn.clicked.connect(self.refresh_model_dropdown)
        grid.addWidget(self.refresh_model_dropdown_btn, 0, 3, 1, 1)

        # --- Training file selection ---
        self.train_file_label = QLabel("No labeled training file selected")
        grid.addWidget(self.train_file_label, 1, 0, 1, 1)

        self.train_browse_btn = QPushButton("Select Training CSV")
        self.train_browse_btn.clicked.connect(self.browse_train_file)
        grid.addWidget(self.train_browse_btn, 1, 1, 1, 1)

        self.train_btn = QPushButton("Train Models")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.train_model)
        grid.addWidget(self.train_btn, 1, 2, 1, 1)

        self.train_status_label = QLabel("")
        grid.addWidget(self.train_status_label, 2, 0, 1, 4)

        self.save_model_set_btn = QPushButton("Save Model Set")
        self.save_model_set_btn.clicked.connect(self.save_model_set_dialog)
        grid.addWidget(self.save_model_set_btn, 3, 0, 1, 1)

        # --- Classify file selection ---
        self.classify_file_label = QLabel("No file selected for classification")
        grid.addWidget(self.classify_file_label, 4, 0, 1, 1)

        self.classify_browse_btn = QPushButton("Select File to Classify")
        self.classify_browse_btn.clicked.connect(self.browse_classify_file)
        grid.addWidget(self.classify_browse_btn, 4, 1, 1, 1)

        self.classify_btn = QPushButton("Classify (Multipass)")
        self.classify_btn.setEnabled(False)
        self.classify_btn.clicked.connect(self.classify_items)
        grid.addWidget(self.classify_btn, 4, 2, 1, 1)

        self.pred_status_label = QLabel("")
        grid.addWidget(self.pred_status_label, 5, 0, 1, 4)

        # --- Checkbox ---
        self.sim_checkbox = QCheckBox("Show Most Similar Training Spec for Each")
        self.sim_checkbox.setChecked(True)
        grid.addWidget(self.sim_checkbox, 6, 0, 1, 2)

        # --- Results area ---
        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)
        layout.addWidget(self.results_box, stretch=1)

        self.save_btn = QPushButton("Save Results to CSV")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        layout.addWidget(self.save_btn)

    # --- UI Logic Functions ---
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
            self.model_label.setText("No classifiers loaded")
            self.classify_btn.setEnabled(False)
            self.last_model_set_name = ""
        else:
            try:
                files_dict = logic.load_model_set(selection)
                self.models = logic.load_all_models(files_dict=files_dict)
                self.model_label.setText(f"Model set '{selection}' loaded.")
                self.classify_btn.setEnabled(True)
                self.last_model_set_name = selection
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load model set:\n{e}")
                self.models = None
                self.model_label.setText("No classifiers loaded")
                self.classify_btn.setEnabled(False)

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
        self.repaint()
        try:
            report, files_dict = logic.train_all_models(self.train_csv_path, return_file_dict=True)
            self.results_box.clear()
            self.results_box.append(report)
            self.models = logic.load_all_models(files_dict=files_dict)
            self.train_status_label.setText("Training complete. All models and embeddings saved.")

            # ====== Prompt to Save Model Set immediately after training completes ======
            from PySide6.QtWidgets import QInputDialog
            set_name, ok = QInputDialog.getText(self, "Save Model Set", "Enter a name for this model set:")
            if ok and set_name:
                try:
                    logic.save_model_set(set_name, files_dict)
                    self.model_label.setText(f"Model set '{set_name}' saved.")
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
                self.model_label.setText(f"Model set '{set_name}' saved.")
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

    def classify_items(self):
        self.pred_status_label.setText("Classifying (Multipass)...")
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
            result_df = logic.multipass_classify(df, self.models, self.sim_checkbox.isChecked())
            self.last_pred_df = result_df
            output = result_df.to_string(index=False)
            self.results_box.clear()
            self.results_box.append(output)
            self.save_btn.setEnabled(True)
            self.pred_status_label.setText("Classification (Multipass) done.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during classification: {e}")
            self.pred_status_label.setText("Classification error.")

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
