from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui_actions import (
    classify_from_path,
    list_model_sets,
    load_last_used_model_set,
    load_model_set,
    summarize_training_file,
    get_current_model_paths,
    perform_export,
    save_model_set,
    train_models,
)
from ui_components import InspectorPanel, ResultsPanel, TopBar, WorkflowSidebar
from ui_pages import ClassifyPage, ExportPage, ImportPage, ReviewPage, TrainPage
from ui_state import UIState
from ui_strings import APP_TITLE, WORKFLOW_STEPS
from ui_theme import apply_theme


class SpecsGraderMainWindow(QMainWindow):
    def __init__(self, preloaded=None, debug: bool = False):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.debug = debug
        self.resize(1300, 900)
        apply_theme(self)

        self.state = UIState()
        if preloaded:
            self.state.models = preloaded

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # Sidebar
        self.sidebar = WorkflowSidebar()
        self.sidebar.stepSelected.connect(self.on_step_selected)
        self.sidebar.similarity_checkbox.stateChanged.connect(
            self.on_similarity_toggled
        )
        splitter.addWidget(self.sidebar)

        # Right content
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        self.top_bar = TopBar()
        self.top_bar.exportClicked.connect(self.on_export_clicked)
        right_layout.addWidget(self.top_bar)

        self.pages = QStackedWidget()
        self.import_page = ImportPage()
        self.train_page = TrainPage()
        self.classify_page = ClassifyPage()
        self.review_page = ReviewPage()
        self.export_page = ExportPage()

        self.pages.addWidget(self.import_page)
        self.pages.addWidget(self.review_page)
        self.pages.addWidget(self.train_page)
        self.pages.addWidget(self.classify_page)
        self.pages.addWidget(self.export_page)
        right_layout.addWidget(self.pages, stretch=1)

        self.import_page.file_picker.browseRequested.connect(self.pick_training_file)
        self.import_page.model_changed.connect(self.on_model_selected)
        self.classify_page.picker.browseRequested.connect(self.pick_classify_file)
        self.import_page.continue_btn.clicked.connect(
            lambda: self.sidebar.set_active_step(1)
        )
        self.train_page.train_btn.clicked.connect(self.run_training)
        self.train_page.save_btn.clicked.connect(self.prompt_save_model_set)
        self.classify_page.run_btn.clicked.connect(self.run_classification_from_page)

        # Results + inspector area
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        bottom.setContentsMargins(0, 0, 0, 0)
        self.results_panel = ResultsPanel()
        self.results_panel.rowSelected.connect(self.update_inspector)
        bottom.addWidget(self.results_panel, stretch=2)
        self.inspector = InspectorPanel()
        self.inspector.setVisible(False)
        bottom.addWidget(self.inspector, stretch=1)
        right_layout.addLayout(bottom)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.refresh_models()
        self.sidebar.set_active_step(0)
        self.top_bar.set_title(WORKFLOW_STEPS[0])
        self.top_bar.set_project(None)
        self.top_bar.set_model_status(None)
        self.top_bar.set_trained_label("—")

    # --- Sidebar / navigation ---
    def on_step_selected(self, idx: int) -> None:
        if idx < 0 or idx >= self.pages.count():
            return
        self.pages.setCurrentIndex(idx)
        self.top_bar.set_title(WORKFLOW_STEPS[idx])
        self.update_inspector_visibility(idx)

    def on_similarity_toggled(self, state: int) -> None:
        self.state.similarity_enabled = bool(state)

    # --- Models ---
    def refresh_models(self):
        last_name, last_models = load_last_used_model_set()
        active = last_name
        if last_models:
            self.state.models = last_models
            self.state.active_model_set = last_name
        success, names, _ = list_model_sets()
        names = names if success else []
        self.import_page.set_model_sets(["None (Unload)"] + names, active)
        self.top_bar.set_model_status(active)
        self.classify_page.run_btn.setEnabled(
            bool(self.state.models and self.state.classify_file_path)
        )

    def on_model_selected(self, name: str) -> None:
        if name == "None (Unload)":
            self.state.models = None
            self.state.active_model_set = None
            self.top_bar.set_model_status(None)
            self.import_page.set_active_model(None)
            self.update_export_enabled()
            self.classify_page.run_btn.setEnabled(False)
            return
        ok, models, err = load_model_set(name)
        if not ok:
            QMessageBox.critical(self, "Error", f"Could not load model set: {err}")
            return
        self.state.models = models
        self.state.active_model_set = name
        self.top_bar.set_model_status(name)
        self.import_page.set_active_model(name)
        self.classify_page.run_btn.setEnabled(bool(self.state.classify_file_path))
        self.update_export_enabled()

    # --- Import page ---
    def set_training_file(self, path: str) -> None:
        self.state.train_csv_path = path
        self.state.update_project_name_from_path(path)
        self.import_page.set_training_path(path)
        self.train_page.set_ready(bool(path))
        self.top_bar.set_project(self.state.project_name)
        self.update_training_summary()

    def pick_training_file(self):
        """
        Let the user choose a training file (CSV or Excel).
        The underlying training logic handles both formats read-only.
        """
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training File",
            "",
            "Training Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
        )
        if fname:
            self.set_training_file(fname)

    def update_training_summary(self) -> None:
        if not self.state.train_csv_path:
            self.import_page.set_dataset_summary(None, None, None)
            return
        ok, rows, labeled, last_modified, err = summarize_training_file(
            self.state.train_csv_path
        )
        if ok:
            self.import_page.set_dataset_summary(rows, labeled, last_modified)
        else:
            self.import_page.set_dataset_summary(None, None, None)

    # --- Train ---
    def run_training(self):
        if not self.state.train_csv_path:
            return
        ok, report, models, files_dict, err = train_models(self.state.train_csv_path)
        if not ok:
            QMessageBox.critical(self, "Error", f"Training failed: {err}")
            return
        self.state.trained_timestamp = "just now"
        self.state.models = models or self.state.models
        self.train_page.set_metrics(report or "Training complete")
        self.train_page.set_save_enabled(True)
        self.top_bar.set_trained_label(self.state.trained_timestamp or "—")
        self.update_export_enabled()

    def prompt_save_model_set(self) -> None:
        if not self.state.models:
            return
        name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model Set",
            str(Path.home() / "model_set.json"),
            "JSON (*.json)",
        )
        if not name:
            return
        files_dict = get_current_model_paths(self.state.models)
        ok, err = save_model_set(Path(name).stem, files_dict)
        if not ok:
            QMessageBox.critical(self, "Error", f"Could not save model set: {err}")

    # --- Classification ---
    def pick_classify_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select File to Classify",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)",
        )
        if fname:
            self.classify_page.set_document(fname)
            self.state.classify_file_path = fname
            self.state.update_project_name_from_path(fname)
            self.sidebar.set_active_step(3)
            self.top_bar.set_project(self.state.project_name)
            self.classify_page.run_btn.setEnabled(bool(self.state.models))
            self.update_inspector_visibility(3)

    def run_classification_from_page(self):
        if not self.state.classify_file_path:
            return
        self.run_classification(self.state.classify_file_path)

    def run_classification(self, path: str) -> None:
        if not path or not self.state.models:
            return
        ok, df, err = classify_from_path(
            path,
            self.state.models,
            enable_similarity=self.sidebar.similarity_checkbox.isChecked(),
        )
        if not ok or df is None:
            QMessageBox.critical(self, "Error", f"Classification failed: {err}")
            return
        self.state.last_pred_df = df
        self.state.update_project_name_from_path(path)
        self.state.update_review_count()
        self.results_panel.set_results(df)
        self.top_bar.set_project(self.state.project_name)
        self.top_bar.set_model_status(self.state.active_model_set)
        self.update_export_enabled()
        self.update_inspector_visibility()

    # --- Export ---
    def on_export_clicked(self):
        if self.state.last_pred_df is None:
            QMessageBox.information(self, "No results", "Run classification first")
            return
        path = str(Path.home() / "specs_results.csv")
        fmt = "json" if path.lower().endswith(".json") else "csv"
        ok, err = perform_export(self.state.last_pred_df, path, fmt=fmt)
        if ok:
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        else:
            QMessageBox.critical(self, "Error", f"Export failed: {err}")
        self.update_export_enabled()

    def update_export_enabled(self):
        self.top_bar.set_export_enabled(self.state.has_results())

    # --- Inspector ---
    def update_inspector(self, row: dict):
        self.inspector.set_data(row)

    def update_inspector_visibility(self, step_idx: Optional[int] = None) -> None:
        step = step_idx if step_idx is not None else self.sidebar.active_step()
        show = (step == 1) or (self.state.has_results() and step != 0)
        self.inspector.setVisible(show)
