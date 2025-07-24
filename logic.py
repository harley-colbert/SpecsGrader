import pandas as pd
import joblib
import os
from embeddings import EmbeddingManager
from train_classifier import train_dual_classifiers
from similarity_engine import load_training_embeddings, find_most_similar, is_outlier
from classic_ml import train_classic_ml, load_classic_ml, batch_predict as classic_batch_predict
from rules_engine import apply_rules
import numpy as np
import json

MODEL_SET_DB = "models/model_sets.json"
LAST_USED_PATH = "models/last_model_set.txt"

def train_all_models(train_csv, return_file_dict=False, model_set_name=None):
    df = pd.read_csv(train_csv)
    df['Risk Level'] = df['Risk Level'].astype(str).str.lower().str.strip()
    df['Review Department'] = df['Review Department'].astype(str).str.lower().str.strip()
    df.to_csv(train_csv, index=False)
    report_sem = train_dual_classifiers(
        train_csv,
        text_col='Risk Description',
        risklevel_col='Risk Level',
        reviewdept_col='Review Department',
        embedder_out='models/embedder.joblib',
        risklevel_clf_out='models/risklevel_classifier.joblib',
        reviewdept_clf_out='models/reviewdept_classifier.joblib',
        embeddings_out='models/training_data_embeddings.pkl'
    )
    report_cl = train_classic_ml(
        train_csv,
        text_col='Risk Description',
        risklevel_col='Risk Level',
        reviewdept_col='Review Department',
        risklevel_model_out='models/classic_risklevel_clf.joblib',
        reviewdept_model_out='models/classic_reviewdept_clf.joblib',
        tfidf_out='models/classic_tfidf.joblib',
        risklevel_le_out='models/classic_risklevel_le.joblib',
        reviewdept_le_out='models/classic_reviewdept_le.joblib'
    )
    file_dict = {
        "risklevel_classifier": "models/risklevel_classifier.joblib",
        "reviewdept_classifier": "models/reviewdept_classifier.joblib",
        "embedder": "models/embedder.joblib",
        "embeddings": "models/training_data_embeddings.pkl",
        "classic_tfidf": "models/classic_tfidf.joblib",
        "classic_risklevel_clf": "models/classic_risklevel_clf.joblib",
        "classic_reviewdept_clf": "models/classic_reviewdept_clf.joblib",
        "classic_risklevel_le": "models/classic_risklevel_le.joblib",
        "classic_reviewdept_le": "models/classic_reviewdept_le.joblib"
    }
    if model_set_name:
        save_model_set(model_set_name, file_dict)
    if return_file_dict:
        return report_sem + "\n\n" + report_cl, file_dict
    else:
        return report_sem + "\n\n" + report_cl

def load_all_models(files_dict=None):
    file_dict = files_dict
    if file_dict is None:
        file_dict = {
            "risklevel_classifier": "models/risklevel_classifier.joblib",
            "reviewdept_classifier": "models/reviewdept_classifier.joblib",
            "embedder": "models/embedder.joblib",
            "embeddings": "models/training_data_embeddings.pkl",
            "classic_tfidf": "models/classic_tfidf.joblib",
            "classic_risklevel_clf": "models/classic_risklevel_clf.joblib",
            "classic_reviewdept_clf": "models/classic_reviewdept_clf.joblib",
            "classic_risklevel_le": "models/classic_risklevel_le.joblib",
            "classic_reviewdept_le": "models/classic_reviewdept_le.joblib"
        }
    models = {}
    models['clf_rl'] = joblib.load(file_dict["risklevel_classifier"])
    models['clf_rd'] = joblib.load(file_dict["reviewdept_classifier"])
    models['embedder'] = EmbeddingManager.load(file_dict["embedder"])
    models['training_embeddings'] = load_training_embeddings(file_dict["embeddings"])
    models['classic_ml_objs'] = load_classic_ml(
        tfidf_path=file_dict["classic_tfidf"],
        risklevel_model_path=file_dict["classic_risklevel_clf"],
        reviewdept_model_path=file_dict["classic_reviewdept_clf"],
        risklevel_le_path=file_dict["classic_risklevel_le"],
        reviewdept_le_path=file_dict["classic_reviewdept_le"]
    )
    models['sem_risklevel_le'] = joblib.load("models/sem_risklevel_le.joblib")
    models['sem_reviewdept_le'] = joblib.load("models/sem_reviewdept_le.joblib")
    print("[DEBUG] Loaded all models and encoders:")
    for key in models:
        print(f"  [DEBUG] {key}: {type(models[key])}")
    return models

def multipass_classify(df, models, sim_checkbox=True):
    print("\n[DEBUG] multipass_classify called with df shape:", df.shape)
    text_col = None
    for col in df.columns:
        if "risk" in col.lower() and "description" in col.lower():
            text_col = col
    if not text_col:
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                break
    if not text_col:
        raise ValueError("No 'Risk Description' column found!")
    texts = df[text_col].astype(str).fillna("")
    print(f"[DEBUG] Using text column: '{text_col}'")
    print("[DEBUG] First 3 input texts:", texts.head(3).tolist())

    rules_results = [apply_rules(t) for t in texts]
    print("[DEBUG] rules_results (first 3):", rules_results[:3])

    classic_preds_df = classic_batch_predict(texts, models['classic_ml_objs'], threshold=0.9)
    print("[DEBUG] classic_preds_df head:\n", classic_preds_df.head(3))

    X_vec = models['embedder'].encode(texts.tolist())
    print("[DEBUG] Encoded embeddings shape:", getattr(X_vec, 'shape', 'N/A'))

    preds_rl = [x for x in models['clf_rl'].predict(X_vec)]
    preds_rd = [x for x in models['clf_rd'].predict(X_vec)]
    print("[DEBUG] Semantic preds_rl:", preds_rl[:3])
    print("[DEBUG] Semantic preds_rd:", preds_rd[:3])

    sim_results = []
    if sim_checkbox and models['training_embeddings'] is not None:
        print("[DEBUG] Running similarity engine for each embedding...")
        for idx, vec in enumerate(X_vec):
            print(f"\n[DEBUG] >>> find_most_similar for input {idx}:")
            print("[DEBUG]  - Embedding (shape):", vec.shape if hasattr(vec, 'shape') else type(vec))
            sim = find_most_similar(vec, models['training_embeddings'])
            print("[DEBUG]  - Similarity engine result:", sim)
            sim_results.append(sim)
    else:
        print("[DEBUG] Skipping similarity search (sim_checkbox is False or missing embeddings).")
        sim_results = [None] * len(texts)

    outlier_flags = [is_outlier(vec, models['training_embeddings']) for vec in X_vec]
    print("[DEBUG] Outlier flags (first 10):", outlier_flags[:10])

    risklevel_le = models['classic_ml_objs'].get('le_risk', None)
    reviewdept_le = models['classic_ml_objs'].get('le_dept', None)
    sem_risklevel_le = models.get('sem_risklevel_le', None)
    sem_reviewdept_le = models.get('sem_reviewdept_le', None)

    output_rows = []
    for i in range(len(df)):
        print(f"\n[DEBUG] --- Processing row {i} ---")
        rule = rules_results[i]
        c = classic_preds_df.iloc[i]
        sim = sim_results[i] if sim_results else None

        print("[DEBUG] Rule result:", rule)
        print("[DEBUG] Classic ML result:", dict(c))
        print("[DEBUG] Similarity result:", sim)

        # Decode helpers
        def decode(le, v):
            if le is None:
                return str(v)
            try:
                if isinstance(v, (list, tuple, np.ndarray)):
                    v = v[0]
                decoded = le.inverse_transform([int(v)])[0]
                return str(decoded)
            except Exception as e:
                print(f"[DEBUG] Could not decode '{v}' with {le}: {e}")
                return str(v)

        # Compute trust for each
        # -- Rule trust: 1 if triggered, else 0
        rule_trust = 1.0 if rule['triggered'] else 0.0
        # -- Classic trust: min(proba, proba)
        try:
            classic_trust = float(min(
                c.get('risk_level_proba', 0.0) if c.get('risk_level_proba', 0.0) is not None else 0.0,
                c.get('review_dept_proba', 0.0) if c.get('review_dept_proba', 0.0) is not None else 0.0
            ))
        except Exception as e:
            print(f"[DEBUG] Could not compute classic_trust: {e}")
            classic_trust = 0.0
        # -- Similarity trust: sim['similarity'] or 0
        sim_trust = float(sim.get('similarity', 0.0)) if sim else 0.0

        trusts = [("Rule", rule_trust), ("Classic", classic_trust), ("Similarity", sim_trust)]
        print("[DEBUG] Trust scores (Rule, Classic, Sim):", trusts)
        winner = max(trusts, key=lambda x: x[1])[0]
        print("[DEBUG] Trust winner:", winner)

        if winner == "Rule":
            final_risk = rule['forced_label'].get('Risk Level', '').lower().strip()
            final_dept = rule['forced_label'].get('Review Department', '').lower().strip()
            src = "Rule"
        elif winner == "Classic":
            final_risk = decode(risklevel_le, c["risk_level"]).lower().strip()
            final_dept = decode(reviewdept_le, c["review_dept"]).lower().strip()
            src = f"Classic (proba={classic_trust:.3f})"
        else:
            final_risk = sim.get("risk_level", "") if sim else ""
            final_dept = sim.get("review_dept", "") if sim else ""
            src = f"Similarity (score={sim_trust:.3f})"

        row = {
            "Risk Description": texts.iloc[i],
            "Final Risk Level": final_risk,
            "Final Review Dept": final_dept,
            "Label Source": src,
            "Rule Trust": rule_trust,
            "Classic Trust": classic_trust,
            "Similarity Trust": sim_trust,
        }
        if sim:
            row["Similarity Match"] = sim.get("text", "")
            row["Similarity Score"] = sim_trust
        print("[DEBUG] OUTPUT ROW:", row)
        output_rows.append(row)

    print("\n[DEBUG] Finished multipass_classify, returning DataFrame with shape:", len(output_rows))
    return pd.DataFrame(output_rows)

def ensure_model_db():
    if not os.path.exists(MODEL_SET_DB):
        with open(MODEL_SET_DB, "w") as f:
            json.dump({}, f)

def save_model_set(set_name, files_dict):
    ensure_model_db()
    with open(MODEL_SET_DB, "r") as f:
        db = json.load(f)
    db[set_name] = files_dict
    with open(MODEL_SET_DB, "w") as f:
        json.dump(db, f, indent=2)
    with open(LAST_USED_PATH, "w") as f:
        f.write(set_name)

def load_model_set(set_name):
    ensure_model_db()
    with open(MODEL_SET_DB, "r") as f:
        db = json.load(f)
    if set_name not in db:
        raise ValueError(f"Model set '{set_name}' not found.")
    return db[set_name]

def list_model_sets():
    ensure_model_db()
    with open(MODEL_SET_DB, "r") as f:
        db = json.load(f)
    print("[DEBUG] Available model sets:", list(db.keys()))
    return list(db.keys())

def get_last_used_model_set():
    if os.path.exists(LAST_USED_PATH):
        with open(LAST_USED_PATH, "r") as f:
            name = f.read().strip()
            if name:
                return name
    return None

def get_current_model_file_paths(models):
    return {
        "risklevel_classifier": "models/risklevel_classifier.joblib",
        "reviewdept_classifier": "models/reviewdept_classifier.joblib",
        "embedder": "models/embedder.joblib",
        "embeddings": "models/training_data_embeddings.pkl",
        "classic_tfidf": "models/classic_tfidf.joblib",
        "classic_risklevel_clf": "models/classic_risklevel_clf.joblib",
        "classic_reviewdept_clf": "models/classic_reviewdept_clf.joblib",
        "classic_risklevel_le": "models/classic_risklevel_le.joblib",
        "classic_reviewdept_le": "models/classic_reviewdept_le.joblib"
    }

def get_model_set_names_for_dropdown():
    names = list_model_sets()
    names = ['None (Unload)'] + names
    print("[DEBUG] Dropdown model set options:", names)
    return names

def unload_all_models(models_dict):
    print("[DEBUG] Unloading all models...")
    if models_dict is not None:
        models_dict.clear()
    print("[DEBUG] Models dict after unload:", models_dict)
