import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """
    Handles loading, encoding, and saving of sentence embeddings using SentenceTransformer.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Encodes a list of strings to L2-normalized embedding vectors.
        Args:
            texts (list[str]): Sentences to embed.
        Returns:
            np.ndarray: 2D array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # L2 normalization (unit vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings

    def save(self, path):
        """
        Save only the model name (weights can always be re-loaded).
        """
        joblib.dump({"model_name": self.model_name}, path)

    @staticmethod
    def load(path):
        """
        Loads the EmbeddingManager from a saved model name.
        """
        meta = joblib.load(path)
        return EmbeddingManager(model_name=meta["model_name"])
