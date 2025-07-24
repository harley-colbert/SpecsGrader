import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# ============ TRAINING ============

def train_classic_ml(
    train_csv,
    text_col,
    risklevel_col,
    reviewdept_col,
    risklevel_model_out,
    reviewdept_model_out,
    tfidf_out,
    risklevel_le_out,
    reviewdept_le_out
):
    """
    Trains two classifiers (risk level & review department) using TF-IDF + Logistic Regression.

    Args:
        train_csv: path to CSV file with training data
        text_col: column with specification descriptions
        risklevel_col: column with risk level labels
        reviewdept_col: column with department labels
        risklevel_model_out: path to save risk level classifier
        reviewdept_model_out: path to save department classifier
        tfidf_out: path to save TF-IDF vectorizer
        risklevel_le_out: path to save risk level label encoder
        reviewdept_le_out: path to save department label encoder

    Returns:
        str: Training summary report
    """
    df = pd.read_csv(train_csv)
    texts = df[text_col].astype(str).fillna("")

    # --- SANITIZE LABELS ---
    df[risklevel_col] = df[risklevel_col].astype(str).str.lower().str.strip()
    df[reviewdept_col] = df[reviewdept_col].astype(str).str.lower().str.strip()

    # Fit TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=8000)
    X = tfidf.fit_transform(texts)

    # Risk Level classifier
    le_risk = LabelEncoder()
    y_risk = le_risk.fit_transform(df[risklevel_col].fillna("unknown"))
    clf_risk = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf_risk.fit(X, y_risk)

    # Department classifier
    le_dept = LabelEncoder()
    y_dept = le_dept.fit_transform(df[reviewdept_col].fillna("unknown"))
    clf_dept = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf_dept.fit(X, y_dept)

    # Save models and encoders
    joblib.dump(tfidf, tfidf_out)
    joblib.dump(clf_risk, risklevel_model_out)
    joblib.dump(clf_dept, reviewdept_model_out)
    joblib.dump(le_risk, risklevel_le_out)
    joblib.dump(le_dept, reviewdept_le_out)

    report = f"Classic ML training complete. Files saved:\n"
    report += f"  TF-IDF: {tfidf_out}\n"
    report += f"  Risk Level classifier: {risklevel_model_out}\n"
    report += f"  Review Dept classifier: {reviewdept_model_out}\n"
    report += f"  Risk Level label encoder: {risklevel_le_out}\n"
    report += f"  Review Dept label encoder: {reviewdept_le_out}\n"
    report += f"\nRisk Level labels: {list(le_risk.classes_)}"
    report += f"\nDept labels: {list(le_dept.classes_)}"
    return report

# ============ INFERENCE ============

def load_classic_ml(
    tfidf_path,
    risklevel_model_path,
    reviewdept_model_path,
    risklevel_le_path,
    reviewdept_le_path
):
    """
    Loads all classic ML models/components.
    """
    tfidf = joblib.load(tfidf_path)
    clf_risk = joblib.load(risklevel_model_path)
    clf_dept = joblib.load(reviewdept_model_path)
    le_risk = joblib.load(risklevel_le_path)
    le_dept = joblib.load(reviewdept_le_path)
    return {
        "tfidf": tfidf,
        "clf_risk": clf_risk,
        "clf_dept": clf_dept,
        "le_risk": le_risk,
        "le_dept": le_dept
    }

def predict_classic_ml(texts, classic_ml_objs, threshold=0.9):
    """
    Predicts classes and probabilities for a list of texts using classic ML models.

    Args:
        texts: list/Series of strings
        classic_ml_objs: dict as returned from load_classic_ml
        threshold: if set, only results with max proba >= threshold are "high confidence"

    Returns:
        List of dict per item:
            {
                "risk_level": predicted class label,
                "risk_level_proba": float,
                "risk_level_high_conf": bool,
                "review_dept": predicted dept,
                "review_dept_proba": float,
                "review_dept_high_conf": bool
            }
    """
    tfidf = classic_ml_objs["tfidf"]
    clf_risk = classic_ml_objs["clf_risk"]
    clf_dept = classic_ml_objs["clf_dept"]
    le_risk = classic_ml_objs["le_risk"]
    le_dept = classic_ml_objs["le_dept"]

    X = tfidf.transform([str(t) for t in texts])

    # Risk Level predictions
    proba_risk = clf_risk.predict_proba(X)
    idx_risk = np.argmax(proba_risk, axis=1)
    maxprob_risk = np.max(proba_risk, axis=1)
    pred_risk = le_risk.inverse_transform(idx_risk)
    high_conf_risk = maxprob_risk >= threshold

    # Review Dept predictions
    proba_dept = clf_dept.predict_proba(X)
    idx_dept = np.argmax(proba_dept, axis=1)
    maxprob_dept = np.max(proba_dept, axis=1)
    pred_dept = le_dept.inverse_transform(idx_dept)
    high_conf_dept = maxprob_dept >= threshold

    # Assemble output
    results = []
    for i in range(X.shape[0]):
        results.append({
            "risk_level": pred_risk[i],
            "risk_level_proba": float(maxprob_risk[i]),
            "risk_level_high_conf": bool(high_conf_risk[i]),
            "review_dept": pred_dept[i],
            "review_dept_proba": float(maxprob_dept[i]),
            "review_dept_high_conf": bool(high_conf_dept[i]),
        })
    return results

# ============ UTILITIES & BATCH ============

def batch_predict(texts, classic_ml_objs, threshold=0.9):
    """
    Predicts for a list or Series of texts. Returns DataFrame with predictions and confidences.
    """
    preds = predict_classic_ml(texts, classic_ml_objs, threshold=threshold)
    return pd.DataFrame(preds)

# ============ Example Training/Testing ============

if __name__ == "__main__":
    # Training example (run once)
    if not os.path.exists("models/classic_tfidf.joblib"):
        print(train_classic_ml(
            train_csv="data/example_training.csv",
            text_col="Risk Description",
            risklevel_col="Risk Level",
            reviewdept_col="Review Department",
            risklevel_model_out="models/classic_risklevel_clf.joblib",
            reviewdept_model_out="models/classic_reviewdept_clf.joblib",
            tfidf_out="models/classic_tfidf.joblib",
            risklevel_le_out="models/classic_risklevel_le.joblib",
            reviewdept_le_out="models/classic_reviewdept_le.joblib"
        ))

    # Load models and predict on some text
    classic_objs = load_classic_ml(
        tfidf_path="models/classic_tfidf.joblib",
        risklevel_model_path="models/classic_risklevel_clf.joblib",
        reviewdept_model_path="models/classic_reviewdept_clf.joblib",
        risklevel_le_path="models/classic_risklevel_le.joblib",
        reviewdept_le_path="models/classic_reviewdept_le.joblib"
    )
    sample_texts = [
        "System may overheat if ventilation is blocked.",
        "PLCs must be isolated by safety relay.",
        "Hydraulic pressure above 4000psi."
    ]
    df = batch_predict(sample_texts, classic_objs, threshold=0.85)
    print(df)
