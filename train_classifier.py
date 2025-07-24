import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from embeddings import EmbeddingManager
import numpy as np

def train_dual_classifiers(
    train_csv,
    text_col,
    risklevel_col,
    reviewdept_col,
    embedder_out,
    risklevel_clf_out,
    reviewdept_clf_out,
    embeddings_out=None
):
    # Load data
    df = pd.read_csv(train_csv)
    if df[text_col].isnull().any():
        df[text_col] = df[text_col].fillna("")
    if df[risklevel_col].isnull().any():
        df[risklevel_col] = df[risklevel_col].fillna("Unknown")
    if df[reviewdept_col].isnull().any():
        df[reviewdept_col] = df[reviewdept_col].fillna("Unknown")

    # --- SANITIZE LABELS ---
    df[risklevel_col] = df[risklevel_col].astype(str).str.lower().str.strip()
    df[reviewdept_col] = df[reviewdept_col].astype(str).str.lower().str.strip()

    # Embedding model
    embedder = EmbeddingManager()
    X = embedder.encode(df[text_col].astype(str).tolist())

    # Risk Level classifier
    le_risk = LabelEncoder()
    y_risk = le_risk.fit_transform(df[risklevel_col])
    clf_risk = LogisticRegression(max_iter=1000, multi_class="auto", solver="lbfgs")
    clf_risk.fit(X, y_risk)

    # Review Department classifier
    le_dept = LabelEncoder()
    y_dept = le_dept.fit_transform(df[reviewdept_col])
    clf_dept = LogisticRegression(max_iter=1000, multi_class="auto", solver="lbfgs")
    clf_dept.fit(X, y_dept)

    # Save models and embedder
    joblib.dump(clf_risk, risklevel_clf_out)
    joblib.dump(clf_dept, reviewdept_clf_out)
    embedder.save(embedder_out)

    # >>>>>> ADD THESE LINES <<<<<<
    joblib.dump(le_risk, "models/sem_risklevel_le.joblib")
    joblib.dump(le_dept, "models/sem_reviewdept_le.joblib")
    # >>>>>> END ADDED LINES <<<<<<

    # Save embeddings, text, and labels for semantic similarity
    if embeddings_out is not None:
        embed_df = pd.DataFrame({
            "text": df[text_col],
            "risk_level": df[risklevel_col],
            "review_dept": df[reviewdept_col],
            "embedding": list(X)
        })
        embed_df.to_pickle(embeddings_out)

    report = f"Training complete. Files saved:\n"
    report += f"  Risk Level classifier: {risklevel_clf_out}\n"
    report += f"  Review Department classifier: {reviewdept_clf_out}\n"
    report += f"  Embedding model: {embedder_out}\n"
    report += f"  Semantic Risk Level label encoder: models/sem_risklevel_le.joblib\n"
    report += f"  Semantic Review Dept label encoder: models/sem_reviewdept_le.joblib\n"
    if embeddings_out is not None:
        report += f"  Training data embeddings: {embeddings_out}\n"
    report += f"\nLabels (Risk Level): {list(le_risk.classes_)}"
    report += f"\nLabels (Review Department): {list(le_dept.classes_)}"
    return report
