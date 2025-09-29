"""
Test Ambiguity Analysis Tool
----------------------------
This script loads CSV files containing test cases, validates their structure,
and analyzes them for ambiguity using NLP and clustering techniques.

Outputs:
    - exported_test_analysis.csv  (full enhanced dataset)
    - ambiguity_analysis.csv      (ambiguity scores only)

Expected CSV columns (Include CSV files in the CSV folder):
    - test_case_id (str): Unique identifier for each test case
    - step_no (int/str): Step number within the test case
    - test_step (str): Action to perform
    - expected_result (str): Expected outcome
"""

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan


# ============================================================
# Core Analyzer Class
# ============================================================
class TestAmbiguityDetector:
    """
    Detects ambiguous test cases by combining NLP features,
    clustering, and statistical heuristics.
    """

    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words="english")

    def compute_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute heuristic-based quality scores for each test case row.
        Returns DataFrame with columns:
            - clarity_score, completeness_score, coherence_score, has_specifics
        """
        quality_scores = []
        for _, row in df.iterrows():
            test_text = str(row.get("test_step", ""))
            expected_text = str(row.get("expected_result", ""))

            scores = {
                "text_length": len(test_text) + len(expected_text),
                "has_specifics": self._has_specific_details(test_text + " " + expected_text),
                "clarity_score": self._compute_clarity_score(test_text, expected_text),
                "completeness_score": self._compute_completeness_score(test_text, expected_text),
                "coherence_score": self._compute_coherence(test_text, expected_text),
            }
            quality_scores.append(scores)

        return pd.DataFrame(quality_scores)

    # -------------------- Helper scoring methods --------------------
    def _has_specific_details(self, text: str) -> float:
        """Detect presence of concrete testable details (UI elements, actions, etc.)."""
        specific_patterns = [
            r"\d+",
            r"button|link|field|form|page",
            r"click|enter|select|verify|check",
            r"should|must|will|expected",
        ]
        return sum(bool(re.search(p, text.lower())) for p in specific_patterns) / len(specific_patterns)

    def _compute_clarity_score(self, test_step: str, expected_result: str) -> float:
        """Score reduced for vague words, increased for specific language."""
        combined_text = f"{test_step} {expected_result}".lower()
        vague_words = ["something", "somehow", "maybe", "perhaps", "might", "could", "probably"]
        specific_words = ["verify", "validate", "confirm", "ensure", "exactly", "precisely"]

        penalty = sum(1 for w in vague_words if w in combined_text) * 0.2
        bonus = sum(1 for w in specific_words if w in combined_text) * 0.1

        return max(0, min(1, 0.5 + bonus - penalty))

    def _compute_completeness_score(self, test_step: str, expected_result: str) -> float:
        """Score based on presence of preconditions and expected outcomes."""
        step_score = 0.5 if test_step.strip() else 0
        result_score = 0.5 if expected_result.strip() else 0
        if "given" in test_step.lower() or "when" in test_step.lower():
            step_score += 0.2
        if "then" in expected_result.lower() or "should" in expected_result.lower():
            result_score += 0.2
        return step_score + result_score

    def _compute_coherence(self, test_step: str, expected_result: str) -> float:
        """Semantic similarity between step and expected result."""
        if not test_step.strip() or not expected_result.strip():
            return 0.0
        embeddings = self.embed_model.encode([test_step, expected_result])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


# ============================================================
# Validation Utilities
# ============================================================
def validate_csv(df: pd.DataFrame) -> bool:
    """
    Validate that the CSV contains useful test cases.
    Checks:
        - Required columns exist
        - At least one non-empty test_step and expected_result
    Returns True if valid, False otherwise.
    """
    required_cols = {"test_case_id", "step_no", "test_step", "expected_result"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return False

    # Filter out rows without meaningful content
    useful = df[df["test_step"].str.strip().astype(bool) & df["expected_result"].str.strip().astype(bool)]
    if useful.empty:
        print("❌ No useful test cases found (empty steps/results).")
        return False

    print(f"✅ CSV validation passed: {len(useful)} useful test cases found")
    return True


# ============================================================
# Main Analysis Pipeline
# ============================================================
def analyze_tests_with_ml(df: pd.DataFrame):
    """Run ML-based ambiguity analysis end-to-end."""
    df = df.copy()
    df["test_step"] = df["test_step"].fillna("")
    df["expected_result"] = df["expected_result"].fillna("")

    # Validate before processing
    if not validate_csv(df):
        raise ValueError("CSV file did not pass validation checks")

    # Group steps per test case
    grouped = df.groupby("test_case_id").apply(
        lambda g: " ".join(
            [f"Step {row['step_no']}: {row['test_step']}. Expected: {row['expected_result']}." for _, row in g.iterrows()]
        )
    ).reset_index(name="text")

    # Embeddings
    print("Computing embeddings...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(grouped["text"].tolist(), convert_to_tensor=True)

    # Clustering
    print("Clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    grouped["cluster"] = clusterer.fit_predict(embeddings.cpu().numpy())
    df = df.merge(grouped[["test_case_id", "cluster"]], on="test_case_id", how="left")

    # Ambiguity detection
    detector = TestAmbiguityDetector()
    ambiguity_results = detector.compute_quality_scores(df)
    df_enhanced = pd.concat([df, ambiguity_results], axis=1)

    return df_enhanced, ambiguity_results


# ============================================================
# Script Entry Point
# ============================================================
if __name__ == "__main__":
    csv_folder = "CSV"
    exports_folder = "Exports"
    os.makedirs(exports_folder, exist_ok=True)

    # Load CSVs
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found in folder:", csv_folder)
        exit(1)

    df_list = []
    for filename in csv_files:
        try:
            filepath = os.path.join(csv_folder, filename)
            df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
            if "cluster" in df.columns:
                df = df.rename(columns={"cluster": "pre_cluster"})
            print(f"Loaded {filename} with {len(df)} rows")
            df_list.append(df)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    if not df_list:
        print("❌ No valid CSV files loaded.")
        exit(1)

    # Merge all
    df = pd.concat(df_list, ignore_index=True)

    # Run analysis
    try:
        enhanced_df, ambiguity_results = analyze_tests_with_ml(df)

        # Save outputs
        enhanced_df.to_csv(os.path.join(exports_folder, "exported_test_analysis.csv"), index=False)
        ambiguity_results.to_csv(os.path.join(exports_folder, "ambiguity_analysis.csv"), index=False)
        print("✅ Analysis complete! Results saved in", exports_folder)

    except Exception as e:
        print("❌ Error during analysis:", str(e))
