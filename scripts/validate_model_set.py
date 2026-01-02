import argparse
import json
import os
import sys

MODEL_SET_DB = "models/model_sets.json"

REQUIRED_KEYS = [
    "risklevel_classifier",
    "reviewdept_classifier",
    "embedder",
    "embeddings",
    "classic_tfidf",
    "classic_risklevel_clf",
    "classic_reviewdept_clf",
    "classic_risklevel_le",
    "classic_reviewdept_le",
]


def load_model_sets():
    if not os.path.exists(MODEL_SET_DB):
        raise FileNotFoundError(f"Model set database not found at {MODEL_SET_DB}")
    with open(MODEL_SET_DB, "r") as handle:
        return json.load(handle)


def validate(model_set_name):
    model_sets = load_model_sets()
    if model_set_name not in model_sets:
        raise ValueError(f"Model set '{model_set_name}' not found in {MODEL_SET_DB}")

    entry = model_sets[model_set_name]
    errors = []

    for key in REQUIRED_KEYS:
        if key not in entry:
            errors.append(f"Missing key: {key}")

    for key in REQUIRED_KEYS:
        path = entry.get(key)
        if path and not os.path.exists(path):
            errors.append(f"Missing file for {key}: {path}")

    vector_db_dir = entry.get("vector_db_dir")
    vector_collection = entry.get("vector_collection")
    if vector_db_dir and not os.path.isdir(vector_db_dir):
        errors.append(f"Vector DB dir not found: {vector_db_dir}")
    if vector_db_dir and not vector_collection:
        errors.append("Vector DB dir provided without vector_collection")

    if errors:
        print("Model set validation failed:")
        for error in errors:
            print(f"- {error}")
        return False

    print(f"Model set '{model_set_name}' is valid.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate a model set entry and files.")
    parser.add_argument("model_set_name", help="Model set name to validate")
    args = parser.parse_args()

    ok = validate(args.model_set_name)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
