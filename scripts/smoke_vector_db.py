import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vector_db import VectorDB


def main() -> None:
    db = VectorDB.open("models/vector_db/_smoke/chroma", "_smoke")
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.0, 0.2, 0.1],
    ]
    metadatas = [
        {
            "text": "Spec one",
            "risk_level": "Low",
            "review_dept": "QA",
            "model_set": "_smoke",
            "row_index": 0,
        },
        {
            "text": "Spec two",
            "risk_level": "High",
            "review_dept": "Security",
            "model_set": "_smoke",
            "row_index": 1,
        },
    ]
    ids = ["spec-1", "spec-2"]
    db.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    results = db.query(embedding=[0.1, 0.2, 0.25], top_k=1)
    if not results:
        raise SystemExit("No results returned from vector DB")
    result = results[0]
    required_keys = {"text", "risk_level", "review_dept", "similarity"}
    if not required_keys.issubset(result.keys()):
        raise SystemExit(f"Missing keys in result: {result}")
    print("Top match:", result)


if __name__ == "__main__":
    main()
