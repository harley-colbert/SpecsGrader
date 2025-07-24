import numpy as np
import pandas as pd

def load_training_embeddings(path):
    """
    Loads the pickled DataFrame of training specs and their embeddings.

    Returns:
        dict with:
          - df: DataFrame (columns: text, risk_level, review_dept, embedding)
          - embeddings: numpy array of shape (n_examples, dim)
    """
    print(f"[DEBUG] Loading training embeddings from: {path}")
    df = pd.read_pickle(path)
    print("[DEBUG] Embedding DataFrame columns:", df.columns.tolist())
    print("[DEBUG] First 3 rows:\n", df.head(3))
    embeddings = np.stack(df["embedding"].values)
    print("[DEBUG] Embedding array shape:", embeddings.shape)
    return {
        "df": df,
        "embeddings": embeddings
    }

def cosine_similarity(vec1, mat):
    """
    Computes cosine similarity between a single vector and a matrix of vectors.

    Args:
        vec1: numpy 1D vector (d,)
        mat: numpy 2D array (n, d)
    Returns:
        similarities: numpy array (n,)
    """
    print("[DEBUG] cosine_similarity - vec1 shape:", vec1.shape)
    print("[DEBUG] cosine_similarity - mat shape:", mat.shape)
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(mat_norm, vec1_norm)
    print("[DEBUG] cosine_similarity - min/max/sum:", np.min(sim), np.max(sim), np.sum(sim))
    return sim

def find_most_similar(vec, train_embs_obj, filter_fn=None, top_k=1):
    """
    Finds the most similar training specs to the provided vector.

    Args:
        vec: numpy 1D vector for query
        train_embs_obj: dict from load_training_embeddings
        filter_fn: optional function(df) -> boolean mask for subsetting
        top_k: how many results to return (int)

    Returns:
        list of dicts (each: text, risk_level, review_dept, similarity)
        or just one dict if top_k==1
    """
    df = train_embs_obj["df"]
    embeddings = train_embs_obj["embeddings"]

    print("[DEBUG] find_most_similar - vec shape:", vec.shape if hasattr(vec, 'shape') else type(vec))
    print("[DEBUG] find_most_similar - df shape:", df.shape)
    print("[DEBUG] find_most_similar - embeddings shape:", embeddings.shape)

    # Optional filtering (by department, label, etc.)
    if filter_fn is not None:
        print("[DEBUG] Applying filter function...")
        mask = filter_fn(df)
        print("[DEBUG] Filter mask sum:", np.sum(mask))
        if not np.any(mask):
            print("[DEBUG] Filter mask empty, returning []")
            return []
        df = df[mask]
        embeddings = embeddings[mask]
        print("[DEBUG] After filtering - df shape:", df.shape, "embeddings shape:", embeddings.shape)

    sims = cosine_similarity(vec, embeddings)
    print("[DEBUG] find_most_similar - cosine similarities:", sims[:10], "...")

    if top_k == 1:
        idx = np.argmax(sims)
        row = df.iloc[idx]
        print("[DEBUG] Selected row index:", idx)
        print("[DEBUG] Selected row values:", row.to_dict())
        result = {
            "text": row["text"],
            "risk_level": row["risk_level"],
            "review_dept": row["review_dept"],
            "similarity": float(sims[idx])
        }
        print("[DEBUG] Returning single most similar:", result)
        return result
    else:
        # Return top-k (sorted descending)
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        sorted_idx = top_indices[np.argsort(sims[top_indices])[::-1]]
        results = []
        print("[DEBUG] Top-k indices:", sorted_idx)
        for i in sorted_idx:
            row = df.iloc[i]
            row_result = {
                "text": row["text"],
                "risk_level": row["risk_level"],
                "review_dept": row["review_dept"],
                "similarity": float(sims[i])
            }
            print("[DEBUG] Top-k row:", row_result)
            results.append(row_result)
        print("[DEBUG] Returning top-k results:", results)
        return results

def find_batch_topk(queries, train_embs_obj, top_k=1, filter_fn=None):
    """
    For a batch of query vectors, returns top-k most similar for each.

    Args:
        queries: numpy array shape (m, d)
        train_embs_obj: as above
        top_k: int
        filter_fn: optional function(df) -> boolean mask

    Returns:
        list (len=m) of list-of-dict results (see find_most_similar)
    """
    print(f"[DEBUG] find_batch_topk - queries shape: {queries.shape}, top_k: {top_k}")
    results = []
    for idx, qvec in enumerate(queries):
        print(f"[DEBUG] Batch {idx}:")
        res = find_most_similar(qvec, train_embs_obj, filter_fn=filter_fn, top_k=top_k)
        print(f"[DEBUG] find_most_similar result (batch {idx}):", res)
        results.append(res)
    return results

def is_outlier(vec, train_embs_obj, similarity_threshold=0.50):
    """
    Returns True if the max similarity to training set is below threshold.
    """
    print("[DEBUG] is_outlier - Checking similarity for vector...")
    sims = cosine_similarity(vec, train_embs_obj["embeddings"])
    max_sim = np.max(sims)
    print("[DEBUG] Max similarity:", max_sim)
    return max_sim < similarity_threshold

def batch_outlier_flags(queries, train_embs_obj, similarity_threshold=0.50):
    """
    Batch version of is_outlier. Returns list of bools.
    """
    print(f"[DEBUG] batch_outlier_flags - queries shape: {queries.shape}")
    flags = []
    for idx, vec in enumerate(queries):
        print(f"[DEBUG] Outlier check for {idx}:")
        flag = is_outlier(vec, train_embs_obj, similarity_threshold)
        print(f"[DEBUG] Outlier flag: {flag}")
        flags.append(flag)
    return flags

def pretty_similarity_result(sim, max_len=100):
    """
    Formats a single similarity result for printing or display.
    """
    txt = sim['text']
    if len(txt) > max_len:
        txt = txt[:max_len-3] + "..."
    msg = (
        f"Text: \"{txt}\"\n"
        f"Risk Level: {sim['risk_level']}   |   Review Dept: {sim['review_dept']}\n"
        f"Cosine Similarity: {sim['similarity']:.3f}"
    )
    print("[DEBUG] pretty_similarity_result output:", msg)
    return msg

def filter_by_dept(dept):
    """
    Returns a filter function for restricting search to a department.
    """
    print(f"[DEBUG] filter_by_dept - restricting to dept: {dept}")
    return lambda df: df["review_dept"].astype(str).str.lower() == str(dept).lower()

def filter_by_risk_level(level):
    """
    Returns a filter function for restricting search to a risk level.
    """
    print(f"[DEBUG] filter_by_risk_level - restricting to level: {level}")
    return lambda df: df["risk_level"].astype(str).str.lower() == str(level).lower()

def get_topk_stats(batch_results, k=3):
    """
    For diagnostics: returns frequency counts for top-k most similar labels.
    Useful for ensemble voting, analytics.
    """
    print("[DEBUG] get_topk_stats called...")
    from collections import Counter
    risk_ct = Counter()
    dept_ct = Counter()
    for topk in batch_results:
        if isinstance(topk, dict):
            items = [topk]
        else:
            items = topk
        for sim in items[:k]:
            risk_ct[sim["risk_level"]] += 1
            dept_ct[sim["review_dept"]] += 1
    print("[DEBUG] Top-k risk counts:", risk_ct)
    print("[DEBUG] Top-k dept counts:", dept_ct)
    return {"risk_level": risk_ct, "review_dept": dept_ct}

# --- Example usage/testing below (not run in prod):
if __name__ == "__main__":
    import os
    # Assume models/training_data_embeddings.pkl exists and query_embs available
    path = "models/training_data_embeddings.pkl"
    if os.path.exists(path):
        print("[DEBUG] Running main test on similarity engine...")
        train_embs_obj = load_training_embeddings(path)
        # Example: Find top-3 similar for a dummy vector
        dummy_vec = train_embs_obj["embeddings"][0]
        print("[DEBUG] Dummy vec shape:", dummy_vec.shape)
        top3 = find_most_similar(dummy_vec, train_embs_obj, top_k=3)
        print("Top-3 similar:", top3)
        # Outlier test
        print("Is outlier?", is_outlier(dummy_vec, train_embs_obj))
