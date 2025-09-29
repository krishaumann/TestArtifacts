#!/usr/bin/env python3
"""
Active Learning Test Case Classifier
------------------------------------------------

This script implements an active-learning workflow for classifying software
test cases into functional areas. It is a self-contained, well-documented,
and maintainable rewrite of the provided implementation.

Main features:
- Robust CSV loading and validation (auto-handles pre-existing `cluster` column)
- Clean, modular feature extraction (text stats, keywords, embeddings)
- Bootstrap labeling (from cluster metadata or keyword heuristics)
- Initialize/train a classifier, predict with uncertainty
- Several sample selection strategies for active learning (uncertainty, entropy,
  margin, diverse)
- Model saving/loading and simple CLI for running the pipeline

Usage (CLI):
    python active_learning_classifier.py <input_folder> <output_folder>

Requirements:
    pandas, numpy, scikit-learn, sentence-transformers, joblib

"""
from __future__ import annotations

import os
import re
import sys
import json
import joblib
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def safe_read_csv(filepath: str) -> pd.DataFrame:
    """Read CSV with safe defaults and return DataFrame."""
    try:
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
        logger.info("Loaded CSV %s (%d rows, %d cols)", os.path.basename(filepath), *df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to read CSV %s: %s", filepath, e)
        raise


def validate_input_df(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains required columns.
    Raises ValueError with a clear message if validation fails.
    """
    required = {"test_case_id", "step_no", "test_step", "expected_result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure at least one useful row exists
    has_content = df["test_step"].astype(str).str.strip().astype(bool) & df["expected_result"].astype(str).str.strip().astype(bool)
    if not has_content.any():
        raise ValueError("No rows with both non-empty 'test_step' and 'expected_result' found.")


def rename_conflicting_columns(df: pd.DataFrame, reserved_names: List[str]) -> pd.DataFrame:
    """
    If df contains any columns that clash with reserved_names (e.g. 'cluster'),
    rename them to '<name>_pre' to avoid collisions later in the pipeline.
    """
    df = df.copy()
    for name in reserved_names:
        if name in df.columns:
            new_name = f"{name}_pre"
            logger.info("Renaming existing column '%s' to '%s' to avoid collision.", name, new_name)
            df = df.rename(columns={name: new_name})
    return df


# ---------------------------------------------------------------------
# ActiveLearningTestClassifier - well-documented, modular implementation
# ---------------------------------------------------------------------
class ActiveLearningTestClassifier:
    """
    Active learning classifier for test cases.

    Responsibilities:
    - Extract features for classifier (text stats + embeddings)
    - Bootstrap labels using clustering metadata or keyword heuristics
    - Initialize with labeled data, perform predictions with uncertainty
    - Select informative samples for manual labeling
    - Update / retrain with newly labeled samples and persist model
    """

    def __init__(self, embedding_model: str = DEFAULT_EMBED_MODEL):
        self.embed_model_name = embedding_model
        self.embed_model = SentenceTransformer(self.embed_model_name)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.training_history: List[Dict[str, Any]] = []
        self.uncertainty_threshold: float = 0.3

        # In-memory training stash
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    # -------------------------
    # Feature extraction
    # -------------------------
    def extract_features(self, df: pd.DataFrame, max_embed_dims: int = 50) -> np.ndarray:
        """
        Extract features from DataFrame rows and return a numeric matrix.

        Features include:
        - text_length, word_count, step_length, result_length
        - counts of UI/action/data keywords
        - boolean indicators (has_numbers, has_actions)
        - quality scores if present (clarity_score, completeness_score, coherence_score, ambiguity_score)
        - top embedding dimensions (up to max_embed_dims)
        """
        rows = []
        # keywords
        ui_keywords = ['button', 'link', 'field', 'form', 'page', 'menu', 'dialog']
        action_keywords = ['click', 'enter', 'select', 'verify', 'check', 'navigate', 'submit']
        data_keywords = ['username', 'password', 'email', 'name', 'id', 'number']

        for _, r in df.iterrows():
            step_no = str(r.get('step_no', ''))
            test_step = str(r.get('test_step', ''))
            expected_result = str(r.get('expected_result', ''))
            combined = f"{step_no} {test_step} {expected_result}"

            d = {
                'text_length': len(combined),
                'word_count': len(combined.split()),
                'step_length': len(test_step),
                'result_length': len(expected_result),
                'ui_keyword_count': sum(1 for kw in ui_keywords if kw in combined.lower()),
                'action_keyword_count': sum(1 for kw in action_keywords if kw in combined.lower()),
                'data_keyword_count': sum(1 for kw in data_keywords if kw in combined.lower()),
                'has_numbers': int(bool(re.search(r'\d', combined))),
                'has_specific_elements': int(any(kw in combined.lower() for kw in ui_keywords)),
                'has_actions': int(any(kw in combined.lower() for kw in action_keywords))
            }

            # quality scores fallback to neutral if not present
            for kval in ['clarity_score', 'completeness_score', 'coherence_score', 'ambiguity_score']:
                if kval in r and pd.notna(r[kval]):
                    try:
                        d[kval] = float(r[kval])
                    except Exception:
                        d[kval] = 0.5
                else:
                    d[kval] = 0.5
            rows.append(d)

        feat_df = pd.DataFrame(rows).fillna(0)

        # embeddings (semantic features)
        texts = [f"{r.get('test_step', '')} {r.get('expected_result', '')}" for _, r in df.iterrows()]
        # SentenceTransformer encode: request numpy output
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)

        n_dims = min(max_embed_dims, embeddings.shape[1])
        for i in range(n_dims):
            feat_df[f'embed_{i}'] = embeddings[:, i]

        self.feature_names = feat_df.columns.tolist()
        return feat_df.values

    # -------------------------
    # Bootstrap labeling
    # -------------------------
    def create_bootstrap_labels(self, df: pd.DataFrame, n_per_cluster: int = 10, sample_limit: int = 200) -> pd.DataFrame:
        """
        Create bootstrap labels using precomputed cluster and functional area keywords
        if available. Otherwise fall back to keyword-based heuristics.

        Returns a DataFrame with a 'bootstrap_label' column and sampled rows.
        """
        df = df.copy()
        # Prefer cluster + functional_area_keywords if available
        if 'cluster_pre' in df.columns and 'functional_area_keywords' in df.columns:
            logger.info("Using precomputed 'cluster_pre' + 'functional_area_keywords' for bootstrap labels.")
            # ignore noise clusters
            bootstrap_df = df[df['cluster_pre'].astype(str) != '-1'].copy()
            if bootstrap_df.empty:
                logger.warning("No usable clusters found in precomputed cluster column; falling back to heuristics.")
                return self._keyword_bootstrap(df, sample_limit)
            # prepare label from functional_area_keywords (first keyword)
            bootstrap_df['bootstrap_label'] = bootstrap_df['functional_area_keywords'].fillna('general') \
                .apply(lambda s: str(s).split(',')[0].strip())
            # sample up to n_per_cluster from each cluster
            samples = []
            for cid, group in bootstrap_df.groupby('cluster_pre'):
                n = min(n_per_cluster, len(group))
                samples.append(group.sample(n))
            sampled = pd.concat(samples, ignore_index=True)
            return sampled.reset_index(drop=True)
        else:
            logger.info("No precomputed clustering metadata found; using keyword heuristics for bootstrap labeling.")
            return self._keyword_bootstrap(df, sample_limit)

    def _keyword_bootstrap(self, df: pd.DataFrame, sample_limit: int = 200) -> pd.DataFrame:
        """Fallback bootstrap labeling based on keyword heuristics."""
        df = df.copy()
        def assign_basic_category(text: str) -> str:
            t = str(text).lower()
            if any(k in t for k in ['login', 'username', 'password', 'authenticate']):
                return 'authentication'
            if any(k in t for k in ['navigate', 'page', 'menu', 'link']):
                return 'navigation'
            if any(k in t for k in ['form', 'submit', 'save', 'create']):
                return 'data_entry'
            if any(k in t for k in ['verify', 'check', 'validate', 'confirm']):
                return 'validation'
            return 'general'
        combined = df['test_step'].fillna('') + ' ' + df['expected_result'].fillna('')
        sample_df = df.sample(min(sample_limit, len(df))).copy()
        sample_df['bootstrap_label'] = combined.loc[sample_df.index].apply(assign_basic_category)
        return sample_df.reset_index(drop=True)

    # -------------------------
    # Initialization & training
    # -------------------------
    def initialize_with_labeled_data(self, df: pd.DataFrame, label_column: str = 'functional_area') -> bool:
        """
        Initialize model using an existing labeled column.
        If none found, returns False (caller may choose to bootstrap).
        """
        if label_column not in df.columns:
            logger.warning("Label column '%s' not found", label_column)
            return False

        # Filter out empty/misc labels
        labeled_mask = df[label_column].notna() & (df[label_column].astype(str).str.strip() != '') & (df[label_column].astype(str).str.strip().str.lower() != 'misc')
        if 'cluster_pre' in df.columns:
            labeled_mask &= (df['cluster_pre'].astype(str) != '-1')

        labeled = df[labeled_mask].copy()
        if labeled.empty:
            logger.warning("No usable labeled rows in column '%s'", label_column)
            return False

        X = self.extract_features(labeled)
        y = labeled[label_column].astype(str).values

        # store class names and train
        self.class_names = sorted(list(set(y)))
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)

        self._X_train = X_scaled
        self._y_train = y

        if len(set(y)) > 1:
            cv = cross_val_score(self.classifier, X_scaled, y, cv=min(3, len(y)))
            logger.info("Initial CV accuracy: %.3f (+/- %.3f)", cv.mean(), cv.std() * 2)

        logger.info("Initialized with %d labeled examples and classes: %s", len(y), self.class_names)
        return True

    # -------------------------
    # Prediction & uncertainty
    # -------------------------
    def predict_with_uncertainty(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict classes and return:
        - predictions (labels)
        - uncertainties (1 - max probability)
        - entropies (per-row entropy)
        - probabilities (2D array)
        """
        if not hasattr(self.classifier, "predict_proba"):
            raise RuntimeError("Classifier does not support probability estimates (predict_proba).")

        X = self.extract_features(df)
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)
        preds = self.classifier.predict(X_scaled)

        max_probs = np.max(probs, axis=1)
        uncertainties = 1.0 - max_probs
        entropies = -np.sum(probs * np.log(probs + 1e-12), axis=1)

        return preds, uncertainties, entropies, probs

    # -------------------------
    # Sample selection strategies
    # -------------------------
    def select_samples_for_labeling(self, unlabeled_df: pd.DataFrame, strategy: str = 'uncertainty', n_samples: int = 10) -> pd.DataFrame:
        """
        Select informative samples from unlabeled_df for manual labeling.
        Strategies:
            - 'uncertainty': highest uncertainty
            - 'entropy': highest entropy
            - 'margin': smallest margin between top two probs
            - 'diverse': uncertainty + diversity via embeddings
            - 'random': random selection
        Returns DataFrame with extra columns: predicted_class, uncertainty, entropy, and class probabilities.
        """
        preds, uncertainties, entropies, probs = self.predict_with_uncertainty(unlabeled_df)
        n = min(n_samples, len(unlabeled_df))
        indices = np.arange(len(unlabeled_df))

        if strategy == 'uncertainty':
            selected_idx = indices[np.argsort(uncertainties)[-n:]]
        elif strategy == 'entropy':
            selected_idx = indices[np.argsort(entropies)[-n:]]
        elif strategy == 'margin':
            sorted_probs = np.sort(probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            selected_idx = indices[np.argsort(margins)[:n]]
        elif strategy == 'diverse':
            selected_idx = self._select_diverse_uncertain_samples(unlabeled_df, uncertainties, n)
        else:
            selected_idx = np.random.choice(indices, n, replace=False)

        selected = unlabeled_df.iloc[selected_idx].copy().reset_index(drop=True)
        selected['predicted_class'] = preds[selected_idx]
        selected['uncertainty'] = uncertainties[selected_idx]
        selected['entropy'] = entropies[selected_idx]

        # attach probabilities for each class (aligned to self.class_names)
        if len(self.class_names) > 0 and probs.shape[1] == len(self.class_names):
            for i, cname in enumerate(self.class_names):
                selected[f'prob_{cname}'] = probs[selected_idx, i]
        else:
            # fallback: store top-k probs
            for i in range(min(3, probs.shape[1])):
                selected[f'prob_{i}'] = probs[selected_idx, i]

        return selected

    def _select_diverse_uncertain_samples(self, df: pd.DataFrame, uncertainties: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Greedy selection combining uncertainty and diversity (embedding distance).
        Return indices into df.
        """
        texts = [f"{r.get('test_step', '')} {r.get('expected_result', '')}" for _, r in df.iterrows()]
        emb = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb.reshape(-1, 1)

        remaining = list(range(len(df)))
        selected = []
        # pick the most uncertain first
        first = int(np.argmax(uncertainties))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < min(n_samples, len(df)):
            best_score = -np.inf
            best_idx = None
            for idx in remaining:
                uncertainty_score = uncertainties[idx]
                # diversity: distance to nearest selected
                distances = [np.linalg.norm(emb[idx] - emb[s]) for s in selected]
                min_dist = min(distances) if distances else 0.0
                score = 0.7 * uncertainty_score + 0.3 * min_dist
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        return np.array(selected)

    # -------------------------
    # Model update / retrain
    # -------------------------
    def update_model_with_labels(self, newly_labeled_df: pd.DataFrame, label_column: str = 'manual_label') -> float:
        """
        Update the classifier with newly labeled rows (contains label_column).
        Returns updated cross-validated accuracy (or 1.0 if single class).
        """
        if label_column not in newly_labeled_df.columns:
            raise ValueError(f"Label column '{label_column}' not found in newly labeled data")

        X_new = self.extract_features(newly_labeled_df)
        y_new = newly_labeled_df[label_column].astype(str).values

        X_new_scaled = self.scaler.transform(X_new) if (self._X_train is not None) else self.scaler.fit_transform(X_new)

        if self._X_train is not None and self._y_train is not None:
            X_combined = np.vstack([self._X_train, X_new_scaled])
            y_combined = np.concatenate([self._y_train, y_new])
        else:
            X_combined = X_new_scaled
            y_combined = y_new

        # update class names
        new_classes = set(y_new) - set(self.class_names)
        if new_classes:
            self.class_names = sorted(list(set(self.class_names) | new_classes))
            logger.info("New classes added: %s", new_classes)

        # retrain classifier
        self.classifier.fit(X_combined, y_combined)
        self._X_train, self._y_train = X_combined, y_combined

        if len(set(y_combined)) > 1:
            cv_scores = cross_val_score(self.classifier, X_combined, y_combined, cv=min(3, len(y_combined)))
            accuracy = float(cv_scores.mean())
        else:
            accuracy = 1.0

        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'n_total': len(y_combined),
            'n_new': len(y_new),
            'accuracy': accuracy,
            'n_classes': len(set(y_combined))
        })
        logger.info("Model updated: total_samples=%d, accuracy=%.3f", len(y_combined), accuracy)
        return accuracy

    # -------------------------
    # Persistence & evaluation
    # -------------------------
    def save_model(self, filepath: str) -> None:
        """Save model and metadata to disk using joblib."""
        data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'training_history': self.training_history,
            'embed_model_name': self.embed_model_name,
            'uncertainty_threshold': self.uncertainty_threshold
        }
        joblib.dump(data, filepath)
        logger.info("Saved model to %s", filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        data = joblib.load(filepath)
        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.feature_names = data.get('feature_names', [])
        self.class_names = data.get('class_names', [])
        self.training_history = data.get('training_history', [])
        self.embed_model_name = data.get('embed_model_name', DEFAULT_EMBED_MODEL)
        self.uncertainty_threshold = data.get('uncertainty_threshold', 0.3)
        # Reload embed_model using saved name to ensure reproducibility
        self.embed_model = SentenceTransformer(self.embed_model_name)
        logger.info("Loaded model from %s", filepath)

    def get_model_performance(self, test_df: pd.DataFrame, label_column: str = 'true_label') -> Dict[str, Any]:
        """Evaluate the model on labeled test data and return metrics."""
        if label_column not in test_df.columns:
            raise ValueError(f"Test label column '{label_column}' not found.")
        X_test = self.extract_features(test_df)
        X_test_scaled = self.scaler.transform(X_test)
        y_test = test_df[label_column].astype(str).values

        preds = self.classifier.predict(X_test_scaled)
        probs = self.classifier.predict_proba(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        return {
            'accuracy': acc,
            'report': report,
            'predictions': preds,
            'probabilities': probs
        }


# ---------------------------------------------------------------------
# Orchestration helpers (run full active learning loop)
# ---------------------------------------------------------------------
def run_active_learning_cycle(
    df: pd.DataFrame,
    classifier: ActiveLearningTestClassifier,
    n_iterations: int = 5,
    samples_per_iteration: int = 10,
    label_columns: List[str] = ('functional_area', 'functional_area_keywords', 'area_name')
) -> Tuple[ActiveLearningTestClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Run an active learning process:
    - Initialize classifier (from label columns or bootstrap)
    - Iteratively select samples for labeling and update the model
    - Return trained classifier, labeled_df, unlabeled_df
    """

    # Validate & prepare
    validate_input_df(df)
    df = rename_conflicting_columns(df, reserved_names=['cluster'])  # rename 'cluster' -> 'cluster_pre' if present

    labeled_df = pd.DataFrame()
    unlabeled_df = df.copy()

    # try to find existing labels
    for col in label_columns:
        if col in df.columns:
            mask = df[col].notna() & (df[col].astype(str).str.strip() != '') & (df[col].astype(str).str.lower() != 'misc')
            if 'cluster_pre' in df.columns:
                mask &= (df['cluster_pre'].astype(str) != '-1')
            if mask.sum() > 0:
                labeled_df = df[mask].copy()
                unlabeled_df = df[~mask].copy()
                logger.info("Found %d labeled rows in column '%s'", len(labeled_df), col)
                break

    # Initialize model
    if len(labeled_df) > 0:
        # choose first available label column
        chosen_col = None
        for c in label_columns:
            if c in labeled_df.columns and labeled_df[c].notna().sum() > 0:
                chosen_col = c
                break
        if chosen_col:
            ok = classifier.initialize_with_labeled_data(labeled_df, label_column=chosen_col)
            if not ok:
                logger.info("Initialization with existing labels failed; bootstrapping instead.")
                labeled_df = pd.DataFrame()
                unlabeled_df = df.copy()
        else:
            logger.info("No proper labeled column found despite earlier check; will bootstrap.")
            labeled_df = pd.DataFrame()
            unlabeled_df = df.copy()
    else:
        logger.info("No existing labeled data found; will bootstrap.")

    # If not initialized, create bootstrap labels and initialize with that sample
    if classifier._X_train is None:
        bootstrap = classifier.create_bootstrap_labels(df)
        if bootstrap is None or bootstrap.empty:
            raise RuntimeError("Bootstrap sampling failed; cannot initialize classifier.")
        # train on bootstrap sample
        success = classifier.initialize_with_labeled_data(bootstrap, label_column='bootstrap_label')
        if not success:
            # fallback: directly train classifier using bootstrap labels (extract features and train)
            Xb = classifier.extract_features(bootstrap)
            yb = bootstrap['bootstrap_label'].astype(str).values
            classifier.class_names = sorted(list(set(yb)))
            Xb_scaled = classifier.scaler.fit_transform(Xb)
            classifier.classifier.fit(Xb_scaled, yb)
            classifier._X_train, classifier._y_train = Xb_scaled, yb
            logger.info("Trained on bootstrap sample: %d rows", len(yb))

    # Active learning iterations (manual labeling would be performed externally)
    labeled_accumulator = labeled_df.copy()
    for it in range(n_iterations):
        logger.info("Active learning iteration %d/%d", it + 1, n_iterations)
        if len(unlabeled_df) == 0:
            logger.info("No unlabeled rows remaining. Stopping early.")
            break

        # Select samples for labeling
        samples = classifier.select_samples_for_labeling(unlabeled_df, strategy='diverse', n_samples=min(samples_per_iteration, len(unlabeled_df)))
        logger.info("Selected %d samples for labeling", len(samples))

        # For this script, we'll simulate labeling by asking the user (CLI) OR expect external labels to be provided.
        # To keep this module non-blocking, we'll store selected samples to disk and stop here so a human can label them.
        # Caller may reload labeled samples and call update_model_with_labels().
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging_file = f"label_stage_{timestamp}.csv"
        samples.to_csv(staging_file, index=False)
        logger.info("Wrote %d samples to %s for manual labeling (please label column 'manual_label')", len(samples), staging_file)
        # Stop the iterative loop here; assumes manual labeling step outside script
        break

    # Final predictions for remaining unlabeled
    if len(unlabeled_df) > 0:
        preds, uncerts, _, probs = classifier.predict_with_uncertainty(unlabeled_df)
        unlabeled_df = unlabeled_df.copy()
        unlabeled_df['predicted_class'] = preds
        unlabeled_df['prediction_uncertainty'] = uncerts

    return classifier, labeled_accumulator, unlabeled_df


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------
def main_cli(input_folder: str, output_folder: str) -> None:
    ensure_dir(output_folder)
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    if not csv_files:
        logger.error("No CSV files found in '%s'", input_folder)
        sys.exit(1)

    # Load and concatenate
    frames = []
    for fp in csv_files:
        frames.append(safe_read_csv(fp))
    df = pd.concat(frames, ignore_index=True)

    # Run pipeline
    classifier = ActiveLearningTestClassifier()
    classifier, labeled_df, unlabeled_df = run_active_learning_cycle(df, classifier, n_iterations=1, samples_per_iteration=10)

    # Save outputs
    model_path = os.path.join(output_folder, "active_learning_model.pkl")
    ensure_dir(os.path.dirname(model_path) or ".")
    classifier.save_model(model_path)

    labeled_out = os.path.join(output_folder, "final_labeled.csv")
    unlabeled_out = os.path.join(output_folder, "final_unlabeled_predictions.csv")
    if not labeled_df.empty:
        labeled_df.to_csv(labeled_out, index=False)
        logger.info("Saved labeled dataframe to %s", labeled_out)
    if unlabeled_df is not None and len(unlabeled_df) > 0:
        unlabeled_df.to_csv(unlabeled_out, index=False)
        logger.info("Saved unlabeled predictions to %s", unlabeled_out)

    logger.info("Active learning pipeline complete. Model saved to %s", model_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python active_learning_classifier.py <input_csv_folder> <output_folder>")
        sys.exit(1)
    in_folder = sys.argv[1]
    out_folder = sys.argv[2]
    main_cli(in_folder, out_folder)
