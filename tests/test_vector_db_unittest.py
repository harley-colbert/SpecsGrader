import shutil
import tempfile
import unittest

from vector_db import VectorDB


class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="vector_db_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_open_upsert_query(self):
        db = VectorDB.open(self.temp_dir, "unit-test")
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        ids = ["a", "b"]
        metadatas = [
            {"text": "alpha", "risk_level": "low", "review_dept": "safety"},
            {"text": "beta", "risk_level": "high", "review_dept": "controls"},
        ]
        db.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

        self.assertEqual(db.count(), 2)

        results = db.query([1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "alpha")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)


if __name__ == "__main__":
    unittest.main()
