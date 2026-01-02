import argparse
import json
import os

from logic import build_vector_index, get_safe_model_set_name
from vector_db import VectorDB

MODEL_SET_DB = "models/model_sets.json"


def load_model_sets():
    if not os.path.exists(MODEL_SET_DB):
        raise FileNotFoundError(f"Model set database not found at {MODEL_SET_DB}")
    with open(MODEL_SET_DB, "r") as handle:
        return json.load(handle)


def save_model_sets(data):
    os.makedirs(os.path.dirname(MODEL_SET_DB), exist_ok=True)
    with open(MODEL_SET_DB, "w") as handle:
        json.dump(data, handle, indent=2)


def rebuild(model_set_name, confirm):
    if not confirm:
        raise ValueError("Refusing to rebuild without --confirm")
    model_sets = load_model_sets()
    if model_set_name not in model_sets:
        raise ValueError(f"Model set '{model_set_name}' not found in {MODEL_SET_DB}")

    entry = model_sets[model_set_name]
    embeddings_path = entry.get("embeddings")
    if not embeddings_path:
        raise ValueError("Model set is missing 'embeddings' path")

    safe_name = get_safe_model_set_name(model_set_name)
    vector_db_dir = entry.get("vector_db_dir") or os.path.join("models", "vector_db", safe_name, "chroma")
    vector_collection = entry.get("vector_collection") or safe_name

    db = VectorDB.open(vector_db_dir, vector_collection)
    db.reset_collection(confirm=True)
    inserted = build_vector_index(
        embeddings_path=embeddings_path,
        model_set_name=model_set_name,
        vector_db_dir=vector_db_dir,
        vector_collection=vector_collection,
    )

    if entry.get("vector_db_dir") is None or entry.get("vector_collection") is None:
        entry.update(
            {
                "vector_db_dir": vector_db_dir,
                "vector_collection": vector_collection,
                "vector_db_backend": entry.get("vector_db_backend", "chromadb"),
                "vector_db_metric": entry.get("vector_db_metric", "cosine"),
            }
        )
        model_sets[model_set_name] = entry
        save_model_sets(model_sets)

    db_count = VectorDB.open(vector_db_dir, vector_collection).count()
    print("Rebuild complete.")
    print(f"Rows inserted: {inserted}")
    print(f"DB count: {db_count}")
    print(f"Vector DB dir: {vector_db_dir}")
    print(f"Vector collection: {vector_collection}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild Vector DB from embeddings pickle.")
    parser.add_argument("model_set_name", help="Model set name to rebuild")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion and rebuild of the vector DB collection",
    )
    args = parser.parse_args()
    rebuild(args.model_set_name, args.confirm)


if __name__ == "__main__":
    main()
