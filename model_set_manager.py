import os
import json

MODEL_SET_DB = "models/model_sets.json"

def ensure_models_dir():
    os.makedirs(os.path.dirname(MODEL_SET_DB), exist_ok=True)

def save_model_set(model_set_name, files_dict):
    """
    files_dict = {
        "risklevel_classifier": path,
        "reviewdept_classifier": path,
        ...
    }
    """
    ensure_models_dir()
    if os.path.exists(MODEL_SET_DB):
        with open(MODEL_SET_DB, "r") as f:
            db = json.load(f)
    else:
        db = {}

    db[model_set_name] = files_dict
    with open(MODEL_SET_DB, "w") as f:
        json.dump(db, f, indent=2)
    print(f"Saved model set '{model_set_name}' to {MODEL_SET_DB}")

def get_model_set(model_set_name):
    ensure_models_dir()
    if not os.path.exists(MODEL_SET_DB):
        raise FileNotFoundError("No model set database found.")
    with open(MODEL_SET_DB, "r") as f:
        db = json.load(f)
    if model_set_name not in db:
        raise ValueError(f"Model set '{model_set_name}' not found.")
    return db[model_set_name]
