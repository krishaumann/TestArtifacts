"""
Test Case Analysis & Functional Area Modeling
=============================================

This script ingests test cases (CSV or Gherkin), validates input,
clusters test cases semantically, identifies functional areas,
builds representative path models, and exports enriched data.

Steps:
------
1. Load and validate test cases (CSV or Gherkin).
2. Normalize text and compute embeddings (SentenceTransformer).
3. Cluster test cases (HDBSCAN with fallback to KMeans).
4. Auto-label clusters with top keywords.
5. Normalize steps into canonical actions.
6. Build functional area step graphs and extract representative paths.
7. Save enhanced analysis results to CSV.

Expected Input (CSV mode):
--------------------------
Columns required: ["test_case_id", "step_no", "test_step", "expected_result"]

Output:
-------
Exports/exported_test_analysis.csv
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan


# ============================================================
# Configuration
# ============================================================
REQUIRED_COLUMNS = ["test_case_id", "step_no", "test_step", "expected_result"]

CSV_FOLDER = "CSV"
FEATURES_FOLDER = "features"
EXPORT_FOLDER = "Exports"
ERROR_FOLDER = "Error"
EXCEPTION_FILE = os.path.join(ERROR_FOLDER, "import_exceptions.log")

os.makedirs(EXPORT_FOLDER, exist_ok=True)
os.makedirs(ERROR_FOLDER, exist_ok=True)


# ============================================================
# Logging & Error Handling
# ============================================================
def log(msg: str):
    """Lightweight logger with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def log_exception(line_num, col_name, value, reason):
    """Log validation errors to exception file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EXCEPTION_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] Line {line_num}, Column '{col_name}', Value '{value}': {reason}\n")


# ============================================================
# Data Ingestion & Validation
# ============================================================
def validate_and_import_csv(csv_path: str) -> pd.DataFrame:
    """
    Validate and load a CSV file into a DataFrame.
    - Ensures required columns exist.
    - Logs missing/blank values to exception file.
    """
    valid_rows = []
    seen_values = {col: set() for col in REQUIRED_COLUMNS}

    try:
        df_raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as e:
        log_exception(0, "FILE", csv_path, f"File read error: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Check for missing columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        for col in missing_cols:
            log_exception(0, col, "", "Missing required column")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Validate rows
    for idx, row in df_raw.iterrows():
        row_valid = True
        for col in REQUIRED_COLUMNS:
            value = row[col]
            if value is None or str(value).strip() == "":
                log_exception(idx + 2, col, value, "Blank value")
                row_valid = False
            else:
                seen_values[col].add(value)
        if row_valid:
            valid_rows.append(row)

    return pd.DataFrame(valid_rows, columns=df_raw.columns)


def ingest_gherkin_features(features_folder="features") -> pd.DataFrame:
    """Placeholder: ingest .feature files. Currently returns empty DataFrame."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def detect_source_type(csv_folder=CSV_FOLDER, features_folder=FEATURES_FOLDER) -> str:
    """Determine whether to process CSV or Gherkin inputs."""
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    gherkin_exists = os.path.isdir(features_folder) and glob.glob(os.path.join(features_folder, "*.feature"))

    if csv_files and not gherkin_exists:
        return "csv"
    if gherkin_exists and not csv_files:
        return "gherkin"
    if csv_files and gherkin_exists:
        log("Both CSV and Gherkin found. Defaulting to CSV.")
        return "csv"
    raise FileNotFoundError("No valid CSV or Gherkin feature files found.")


# ============================================================
# Preprocessing & Embeddings
# ============================================================
def preprocess_text(text: str) -> str:
    """Lowercase and remove punctuation for embeddings."""
    return re.sub(r"[^\w\s]", "", str(text)).lower()


def compute_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Encode texts into embeddings."""
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()


# ============================================================
# Clustering
# ============================================================
def cluster_embeddings(X: np.ndarray) -> np.ndarray:
    """Cluster embeddings using HDBSCAN with KMeans fallback."""
    if X.shape[0] < 2:
        log("Only one sample found. Skipping clustering.")
        return np.zeros(X.shape[0], dtype=int)

    if X.shape[0] < 5:
        log("Too few samples for HDBSCAN. Falling back to KMeans.")
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=min(2, X.shape[0])).fit_predict(X)

    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
        return clusterer.fit_predict(X)
    except ValueError as e:
        log(f"HDBSCAN failed: {e}. Falling back to KMeans.")
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=min(2, X.shape[0])).fit_predict(X)


def reassign_noise(grouped, embeddings, labels):
    """Reassign noise cluster (-1) to nearest valid clusters."""
    if set(labels) == {-1}:
        log("All samples in noise cluster. Forcing KMeans fallback.")
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=min(2, len(grouped))).fit_predict(embeddings)

    misc_mask = grouped["cluster"] == -1
    if misc_mask.any():
        misc_embeddings = embeddings[misc_mask]
        valid_mask = grouped["cluster"] != -1
        sims = cosine_similarity(misc_embeddings, embeddings[valid_mask])
        nearest = sims.argmax(axis=1)
        grouped.loc[misc_mask, "cluster"] = grouped.loc[valid_mask, "cluster"].values[nearest]
    return grouped["cluster"].values


# ============================================================
# Cluster Labeling
# ============================================================
def top_keywords(texts, n=5):
    """Extract top keywords for cluster labeling."""
    stopwords = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
        'did', 'she', 'use', 'way', 'will', 'step', 'expected'
    }
    words = []
    for t in texts:
        words.extend([w for w in re.findall(r"\w+", str(t).lower()) if len(w) > 2 and w not in stopwords])
    return [w for w, _ in Counter(words).most_common(n)]


# ============================================================
# Step Normalization
# ============================================================
def canonicalize_action(action: str) -> str:
    """Map raw action descriptions to normalized canonical forms."""
    if pd.isna(action) or not str(action).strip():
        return ""
    action = str(action).lower().strip()
    if "username" in action or "user id" in action:
        return "enter_username"
    if "password" in action:
        return "enter_password"
    if "login" in action and "click" in action:
        return "click_login"
    if "logout" in action:
        return "click_logout"
    if "submit" in action:
        return "click_submit"
    if "navigate" in action or "go to" in action:
        return "navigate"
    if "verify" in action or "check" in action:
        return "verify"
    return re.sub(r"\s+", "_", action)


# ============================================================
# Graph Modeling & Path Extraction
# ============================================================
def build_step_graph(df: pd.DataFrame):
    """Build a step transition graph per cluster."""
    models = {}
    for cluster_id, group in df.groupby("cluster"):
        if cluster_id == -1:
            continue
        G = nx.DiGraph()
        for _, steps in group.groupby("test_case_id"):
            ordered = steps.sort_values("step_no")["normalized_step"].tolist()
            ordered = [s for s in ordered if s]
            for u, v in zip(ordered, ordered[1:]):
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=0, test_ids=set())
                G[u][v]["weight"] += 1
        if G.number_of_nodes() > 0:
            models[cluster_id] = G
    return models


def get_representative_paths(graph: nx.DiGraph, max_paths=5):
    """Extract representative paths from graph (simple source->sink paths)."""
    if graph.number_of_nodes() == 0:
        return []
    sources = [n for n in graph.nodes if graph.in_degree(n) == 0] or list(graph.nodes)[:2]
    sinks = [n for n in graph.nodes if graph.out_degree(n) == 0] or list(graph.nodes)[-2:]
    paths = []
    for s in sources[:3]:
        for t in sinks[:3]:
            try:
                simple_paths = list(nx.all_simple_paths(graph, s, t, cutoff=10))
                paths.extend(simple_paths[:2])
                if len(paths) >= max_paths:
                    return paths[:max_paths]
            except (nx.NetworkXNoPath, nx.NetworkXError):
                continue
    return paths[:max_paths]


# ============================================================
# Main Pipeline
# ============================================================
def main():
    # Clean exception log
    if os.path.exists(EXCEPTION_FILE):
        os.remove(EXCEPTION_FILE)

    # Detect input source
    source_type = detect_source_type()
    if source_type == "csv":
        dfs = [validate_and_import_csv(f) for f in glob.glob(os.path.join(CSV_FOLDER, "*.csv"))]
        df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    else:
        df = ingest_gherkin_features(FEATURES_FOLDER)

    if df.empty or len(df) < 2:
        log(f"Not enough valid rows. See {EXCEPTION_FILE}.")
        return

    # Preprocess text
    df["test_step"] = df["test_step"].fillna("")
    df["expected_result"] = df["expected_result"].fillna("")
    grouped = df.groupby("test_case_id", group_keys=False).apply(
        lambda g: " ".join([f"Step {r.step_no}: {r.test_step}. Expected: {r.expected_result}." for r in g.itertuples()])
    ).reset_index(name="text")
    grouped["text"] = grouped["text"].apply(preprocess_text)

    # Embeddings + Clustering
    log("Computing embeddings...")
    X = compute_embeddings(grouped["text"].tolist())
    labels = cluster_embeddings(X)
    grouped["cluster"] = labels
    labels = reassign_noise(grouped, X, labels)

    # Map back to steps
    df = df.merge(grouped[["test_case_id", "cluster"]], on="test_case_id", how="left")

    # Auto-label clusters
    functional_areas = {c: top_keywords(grouped[grouped["cluster"] == c]["text"]) or ["misc"] for c in set(labels)}
    df["functional_area"] = df["cluster"].map(functional_areas)

    # Normalize steps
    df["normalized_step"] = df["test_step"].apply(canonicalize_action)

    # Build step graphs & paths
    log("Building step graphs...")
    models = build_step_graph(df)
    consolidated = {}
    cluster_to_area = {}
    for cid, G in models.items():
        paths = get_representative_paths(G)
        if paths:
            keywords = functional_areas.get(cid, ["unknown"])
            area_name = "_".join(keywords[:2])
            consolidated[area_name] = paths
            cluster_to_area[cid] = area_name

    # Prepare output
    cluster_to_paths = {
        cid: " | ".join([" -> ".join(p) for p in consolidated.get(cluster_to_area[cid], [])])
        for cid in cluster_to_area
    }
    df["area_name"] = df["cluster"].map(cluster_to_area)
    df["consolidated_paths"] = df["cluster"].map(cluster_to_paths)
    df["functional_area_keywords"] = df["functional_area"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

    # Save enhanced analysis
    export_path = os.path.join(EXPORT_FOLDER, "exported_test_analysis.csv")
    cols = [
        "test_case_id", "step_no", "test_step", "expected_result",
        "normalized_step", "cluster", "functional_area_keywords",
        "area_name", "consolidated_paths"
    ]
    df[cols].to_csv(export_path, index=False)
    log(f"Enhanced analysis saved to '{export_path}' ({len(df)} rows).")

if __name__ == "__main__":
    main()