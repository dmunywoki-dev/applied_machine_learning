"""Multimodal preprocessing and anomaly-ranking model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    text = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("text", text, TEXT_FEATURE),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )


def ensure_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Create absent model columns for API requests and old processed snapshots."""
    frame = frame.copy()
    for col in NUMERIC_FEATURES:
        if col not in frame:
            frame[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in frame:
            frame[col] = "unknown"
    if TEXT_FEATURE not in frame:
        title = frame.get("title", pd.Series([""] * len(frame), index=frame.index)).fillna("").astype(str)
        description = frame.get("description", pd.Series([""] * len(frame), index=frame.index)).fillna("").astype(str)
        frame[TEXT_FEATURE] = (title + " " + description).str.strip().replace("", "untitled procurement")
    frame[TEXT_FEATURE] = frame[TEXT_FEATURE].fillna("untitled procurement").astype(str)
    return frame


def fit_isolation_forest(frame: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    if len(frame) < 10:
        raise ValueError("At least 10 procurement process observations are required for anomaly training.")
    frame = ensure_model_columns(frame)
    preprocessor = make_preprocessor()
    matrix = preprocessor.fit_transform(frame)
    detector = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    detector.fit(matrix)
    train_raw_scores = -detector.score_samples(matrix)
    return {
        "model_type": "isolation_forest",
        "preprocessor": preprocessor,
        "model": detector,
        "reference_scores": np.sort(np.asarray(train_raw_scores, dtype=float)),
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE],
        "metadata": {
            "algorithm": "IsolationForest",
            "text_encoder": "TF-IDF (word unigrams/bigrams)",
            "structured_encoder": "median imputation + training-fitted standard scaling + one-hot categories",
            "random_state": random_state,
            "training_rows": int(len(frame)),
            "score_semantics": "empirical anomaly percentile; not a probability of fraud or corruption",
        },
    }


def raw_anomaly_scores(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    frame = ensure_model_columns(frame)
    matrix = bundle["preprocessor"].transform(frame)
    if bundle.get("model_type") == "nnpu_logistic":
        return np.asarray(bundle["model"].decision_function(matrix), dtype=float)
    return -bundle["model"].score_samples(matrix)


def percentile_scores(raw_scores: np.ndarray, reference_scores: np.ndarray) -> np.ndarray:
    reference_scores = np.asarray(reference_scores, dtype=float)
    if reference_scores.size == 0:
        return np.full(len(raw_scores), 0.5, dtype=float)
    return np.searchsorted(reference_scores, raw_scores, side="right") / reference_scores.size


def predict_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    raw = raw_anomaly_scores(bundle, frame)
    priority = percentile_scores(raw, bundle["reference_scores"])
    result = pd.DataFrame({
        "model_raw_score": raw,
        "risk_priority_score": priority,
        "risk_band": np.where(priority >= 0.95, "priority_review",
                       np.where(priority >= 0.80, "review", "routine")),
    }, index=frame.index)
    return result


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
