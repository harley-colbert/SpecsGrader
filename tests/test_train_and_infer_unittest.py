import os
import unittest
from unittest import mock
import zlib

import joblib
import numpy as np
import pandas as pd

import logic


class FakeEmbeddingManager:
    def __init__(self, model_name="fake-embedder"):
        self.model_name = model_name

    def encode(self, texts):
        vectors = []
        for text in texts:
            seed = zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=5)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vectors.append(vec)
        return np.vstack(vectors)

    def save(self, path):
        joblib.dump({"model_name": self.model_name}, path)

    @staticmethod
    def load(path):
        meta = joblib.load(path)
        return FakeEmbeddingManager(model_name=meta["model_name"])


class FakeLogisticRegression:
    def __init__(self, *args, **kwargs):
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if self.classes_ is None:
            raise ValueError("Model has not been fit.")
        return np.array([self.classes_[0]] * X.shape[0])

    def predict_proba(self, X):
        if self.classes_ is None:
            raise ValueError("Model has not been fit.")
        n_classes = len(self.classes_)
        return np.full((X.shape[0], n_classes), 1.0 / n_classes)


class TestTrainAndInfer(unittest.TestCase):
    def test_train_and_infer_smoke(self):
        train_csv = os.path.join("tests", "data", "train_min.csv")
        infer_csv = os.path.join("tests", "data", "infer_min.csv")

        with (
            mock.patch("train_classifier.EmbeddingManager", FakeEmbeddingManager),
            mock.patch("logic.EmbeddingManager", FakeEmbeddingManager),
            mock.patch("train_classifier.LogisticRegression", FakeLogisticRegression),
            mock.patch("classic_ml.LogisticRegression", FakeLogisticRegression),
        ):
            report, files_dict = logic.train_all_models(train_csv, return_file_dict=True)
            self.assertIn("Training complete.", report)

            models = logic.load_all_models(files_dict=files_dict)
            infer_df = pd.read_csv(infer_csv)
            result_df = logic.multipass_classify(infer_df, models, sim_checkbox=False)

        self.assertFalse(result_df.empty)
        self.assertIn("Final Risk Level", result_df.columns)
        self.assertIn("Final Review Dept", result_df.columns)


if __name__ == "__main__":
    unittest.main()
