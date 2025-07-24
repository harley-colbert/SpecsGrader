import sys
import time
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

# --------- Loader Thread ---------
class LoaderThread(QThread):
    progress_update = Signal(int, str)
    finished_loading = Signal(dict)
    error = Signal(str)

    def run(self):
        try:
            import joblib
            from embeddings import EmbeddingManager
            from similarity_engine import load_training_embeddings

            def try_load(label, fn, *args, **kwargs):
                try:
                    self.progress_update.emit(None, f"Loading {label}...")
                    return fn(*args, **kwargs)
                except Exception as e:
                    self.progress_update.emit(None, f"Warning: Could not load {label}: {e}")
                    return None

            self.progress_update.emit(10, "Loading Risk Level Classifier...")
            clf_rl = try_load("Risk Level Classifier", joblib.load, "models/risklevel_classifier.joblib")

            self.progress_update.emit(20, "Loading Review Dept Classifier...")
            clf_rd = try_load("Review Dept Classifier", joblib.load, "models/reviewdept_classifier.joblib")

            self.progress_update.emit(30, "Loading Embedding Model...")
            embedder = try_load("Embedding Model", EmbeddingManager.load, "models/embedder.joblib")

            self.progress_update.emit(40, "Loading Training Embeddings...")
            training_embeddings = try_load("Training Embeddings", load_training_embeddings, "models/training_data_embeddings.pkl")

            from classic_ml import load_classic_ml
            classic_ml_objs = None
            try:
                self.progress_update.emit(50, "Loading Classic ML Models...")
                classic_ml_objs = load_classic_ml(
                    tfidf_path="models/classic_tfidf.joblib",
                    risklevel_model_path="models/classic_risklevel_clf.joblib",
                    reviewdept_model_path="models/classic_reviewdept_clf.joblib",
                    risklevel_le_path="models/classic_risklevel_le.joblib",
                    reviewdept_le_path="models/classic_reviewdept_le.joblib"
                )
            except Exception as e:
                self.progress_update.emit(60, f"Warning: Classic ML models not found ({e})")
                classic_ml_objs = None

            self.progress_update.emit(100, "Startup complete!")
            time.sleep(0.3)
            preloaded = {
                "clf_rl": clf_rl,
                "clf_rd": clf_rd,
                "embedder": embedder,
                "training_embeddings": training_embeddings,
                "classic_ml_objs": classic_ml_objs
            }
            self.finished_loading.emit(preloaded)
        except Exception as e:
            self.error.emit(str(e))

# --------- Splash Dialog ---------
class SplashDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setFixedSize(440, 200)
        self.setStyleSheet("""
            QDialog { background: #212326; }
            QLabel, QProgressBar { color: #eaf6fb; font-family: 'Segoe UI', Arial, sans-serif; }
            QProgressBar {
                border: 1px solid #2a3642;
                border-radius: 7px;
                background: #232629;
                color: #212326;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background: #44c0fa;
            }
        """)

        self.setWindowTitle("Loading Risk Classifier...")
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(32, 24, 32, 24)

        self.title_label = QLabel("Risk & Department Classifier")
        self.title_label.setStyleSheet("font-size:22px; font-weight:bold; color:#44c0fa;")
        layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)

        self.status_label = QLabel("Starting up...")
        self.status_label.setStyleSheet("font-size:13px;")
        layout.addWidget(self.status_label, alignment=Qt.AlignHCenter)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress, alignment=Qt.AlignHCenter)

        self.setLayout(layout)

        self.loader_thread = LoaderThread()
        self.loader_thread.progress_update.connect(self.update_progress)
        self.loader_thread.finished_loading.connect(self.finish_and_launch)
        self.loader_thread.error.connect(self.on_error)
        self.loader_thread.start()

        self.main_window = None

    def update_progress(self, pct, msg):
        if pct is not None:
            self.progress.setValue(pct)
        self.status_label.setText(msg)

    def finish_and_launch(self, preloaded):
        # Import main window class only when needed (avoids double QApplication)
        from ui import DualSpecClassifierApp
        self.accept()  # Closes the splash dialog
        self.main_window = DualSpecClassifierApp(preloaded=preloaded)  # Or without preloaded=preloaded if not needed
        self.main_window.show()

    def on_error(self, err):
        QMessageBox.critical(self, "Startup Error", str(err))
        sys.exit(1)

# --------- Run Splash (and Main) ---------
def show_splash_and_load():
    app = QApplication(sys.argv)
    splash = SplashDialog()
    splash.exec()
    # Execution resumes here after splash is closed and main window is shown
    sys.exit(app.exec())

if __name__ == "__main__":
    show_splash_and_load()
