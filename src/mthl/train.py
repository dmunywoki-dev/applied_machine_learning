"""Train the reproducible MTHL risk-prioritisation pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import fit_isolation_forest, make_preprocessor, predict_frame, save_bundle
from .pu import NnPULogistic


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15):
    """Split by unique prediction timestamps so ties never cross partitions."""
    ordered = frame.copy()
    ordered["prediction_time"] = pd.to_datetime(ordered["prediction_time"], utc=True, errors="coerce")
    ordered = ordered.dropna(subset=["prediction_time"]).sort_values(["prediction_time", "ocid"]).reset_index(drop=True)
    unique_times = ordered["prediction_time"].drop_duplicates().sort_values().to_list()
    if len(unique_times) < 3:
        raise ValueError("At least three distinct prediction timestamps are required for chronological train/validation/test splits.")
    train_end = max(1, int(len(unique_times) * train_fraction))
    validation_end = max(train_end + 1, int(len(unique_times) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(unique_times) - 1)
    train_cutoff = unique_times[train_end - 1]
    validation_cutoff = unique_times[validation_end - 1]
    train = ordered[ordered["prediction_time"] <= train_cutoff].copy()
    validation = ordered[(ordered["prediction_time"] > train_cutoff) & (ordered["prediction_time"] <= validation_cutoff)].copy()
    test = ordered[ordered["prediction_time"] > validation_cutoff].copy()
    return train, validation, test


def _write_mlflow(run_name: str, params: dict, metrics: dict) -> str | None:
    try:
        import mlflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mlflow.log_metric(key, float(value))
            return mlflow.active_run().info.run_id
    except ImportError:
        return None


def train(processed_path: str, artifact_path: str, report_path: str, seed: int = 42, use_mlflow: bool = False, positive_labels_path: str | None = None, positive_prior: float | None = None):
    frame = pd.read_csv(processed_path)
    if len(frame) < 10:
        raise ValueError("Processed procurement dataset contains fewer than 10 observations.")
    if "prediction_time" not in frame:
        raise ValueError("Processed dataset must include prediction_time for chronological validation.")
    train_df, validation_df, test_df = chronological_split(frame)
    holdout = pd.concat([validation_df, test_df], ignore_index=True)
    synthetic_values = frame.get("is_synthetic", pd.Series([0] * len(frame))).fillna(0)
    synthetic_warning = bool(pd.to_numeric(synthetic_values, errors="coerce").fillna(0).astype(int).eq(1).any())

    if positive_labels_path:
        if positive_prior is None:
            raise ValueError("--positive-prior is required when --positive-labels is supplied; it must be justified and sensitivity-tested.")
        positive_labels = pd.read_csv(positive_labels_path)
        required_label_columns = {"ocid", "published_at"}
        missing_label_columns = required_label_columns - set(positive_labels.columns)
        if missing_label_columns:
            raise ValueError(
                "Positive-label file must contain 'ocid' and 'published_at'. "
                "The latter is required to prevent a finding published in the future "
                "from entering an earlier training run."
            )
        positive_labels = positive_labels.copy()
        positive_labels["ocid"] = positive_labels["ocid"].astype(str).str.strip()
        positive_labels["published_at"] = pd.to_datetime(positive_labels["published_at"], utc=True, errors="coerce")
        invalid_label_dates = int(positive_labels["published_at"].isna().sum())
        positive_labels = positive_labels.dropna(subset=["published_at"])
        positive_labels = positive_labels[positive_labels["ocid"].ne("")].sort_values("published_at").drop_duplicates("ocid", keep="first")
        train_cutoff = pd.to_datetime(train_df["prediction_time"], utc=True).max()
        # Only findings already public by the end of the training window can
        # supervise that window. Later publications remain holdout outcomes.
        train_label_rows = positive_labels[positive_labels["published_at"] <= train_cutoff]
        known_positive_train_ocids = set(train_label_rows["ocid"])
        known_positive_ocids = set(positive_labels["ocid"])
        train_y = train_df["ocid"].astype(str).isin(known_positive_train_ocids).astype(int).to_numpy()
        if train_y.sum() < 2:
            raise ValueError("At least two verified positive OCIDs must appear in the chronological training period for nnPU.")
        if len(train_y) - int(train_y.sum()) < 5:
            raise ValueError("At least five unlabelled training records are required for nnPU.")
        pre = make_preprocessor()
        train_matrix = pre.fit_transform(train_df)
        pu_model = NnPULogistic(positive_prior=positive_prior, max_iter=500).fit(train_matrix, train_y)
        eval_bundle = {
            "model_type": "nnpu_logistic", "preprocessor": pre, "model": pu_model,
            "reference_scores": np.sort(pu_model.decision_function(train_matrix)),
            "feature_columns": list(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else [],
            "metadata": {"algorithm": "non-negative PU logistic risk estimator", "training_rows": len(train_df), "positive_prior": positive_prior,
                         "label_publication_cutoff": str(train_cutoff), "invalid_label_publication_dates_dropped": invalid_label_dates}
        }
        holdout_scores = predict_frame(eval_bundle, holdout) if len(holdout) else pd.DataFrame()
        holdout_positive = holdout["ocid"].astype(str).isin(known_positive_ocids).to_numpy() if len(holdout) else np.array([])
        if len(holdout_scores) and holdout_positive.any():
            order = np.argsort(-holdout_scores["risk_priority_score"].to_numpy())
            k = min(20, len(order))
            observed_recall_at_k = float(holdout_positive[order[:k]].sum() / holdout_positive.sum())
            observed_precision_at_k = float(holdout_positive[order[:k]].mean())
        else:
            observed_recall_at_k = None
            observed_precision_at_k = None
        unsup_metrics = {
            "train_rows": int(len(train_df)), "validation_rows": int(len(validation_df)), "test_rows": int(len(test_df)),
            "holdout_rows_scored": int(len(holdout)), "known_positive_train_count": int(train_y.sum()),
            "known_positive_holdout_count": int(holdout_positive.sum()) if len(holdout_positive) else 0,
            "observed_positive_recall_at_k20": observed_recall_at_k,
            "observed_positive_precision_at_k20": observed_precision_at_k,
            "positive_prior": float(positive_prior), "mode": "positive_unlabelled_learning",
            "label_publication_cutoff": str(train_cutoff),
            "positive_labels_excluded_as_not_yet_published_at_training_cutoff": int(len(positive_labels) - len(train_label_rows)),
            "invalid_label_publication_dates_dropped": invalid_label_dates,
            "interpretation": "Retrieval of documented positives only; unlabelled records may also be positive, so these are not full precision/recall estimates."
        }
        final_cutoff = pd.to_datetime(frame["prediction_time"], utc=True).max()
        final_label_ocids = set(positive_labels.loc[positive_labels["published_at"] <= final_cutoff, "ocid"])
        full_y = frame["ocid"].astype(str).isin(final_label_ocids).astype(int).to_numpy()
        final_pre = make_preprocessor()
        full_matrix = final_pre.fit_transform(frame)
        final_model = NnPULogistic(positive_prior=positive_prior, max_iter=500).fit(full_matrix, full_y)
        final_bundle = {
            "model_type": "nnpu_logistic", "preprocessor": final_pre, "model": final_model,
            "reference_scores": np.sort(final_model.decision_function(full_matrix)),
            "feature_columns": list(final_pre.feature_names_in_) if hasattr(final_pre, "feature_names_in_") else [],
            "metadata": {"algorithm": "non-negative PU logistic risk estimator", "training_rows": len(frame), "positive_prior": positive_prior,
                         "training_mode": "positive_unlabelled_learning", "score_semantics": "empirical PU ranking percentile; not a calibrated probability",
                         "label_publication_cutoff": str(final_cutoff), "invalid_label_publication_dates_dropped": invalid_label_dates}
        }
    else:
        # No label file: validate out-of-time anomaly scoring without treating unlabelled
        # processes as confirmed negatives.
        evaluation_bundle = fit_isolation_forest(train_df, random_state=seed)
        holdout_scores = predict_frame(evaluation_bundle, holdout) if len(holdout) else pd.DataFrame()
        unsup_metrics = {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "holdout_rows_scored": int(len(holdout)),
            "holdout_score_mean": float(holdout_scores["risk_priority_score"].mean()) if len(holdout_scores) else None,
            "holdout_score_p95": float(holdout_scores["risk_priority_score"].quantile(0.95)) if len(holdout_scores) else None,
            "confirmed_outcome_labels_available": False,
            "supervised_metrics_reported": False,
            "mode": "unsupervised_anomaly_prioritisation",
            "reason": "No independently verified audit/sanction outcome labels were supplied. Unlabelled records are not treated as legitimate negatives.",
        }
        final_bundle = fit_isolation_forest(frame, random_state=seed)
        final_bundle["metadata"].update({"training_mode": "unsupervised_anomaly_prioritisation"})

    # The service model is fitted on the full, frozen snapshot after the out-of-time check.
    final_bundle["metadata"].update({
        "training_snapshot": str(Path(processed_path).name),
        "training_time_min": str(frame["prediction_time"].min()),
        "training_time_max": str(frame["prediction_time"].max()),
        "synthetic_data_warning": synthetic_warning,
    })
    save_bundle(final_bundle, artifact_path)
    report = {
        "project": "MTHL - Explainable Procurement Integrity Risk Prioritisation",
        "model_type": final_bundle.get("metadata", {}).get("algorithm", "IsolationForest with structured, text, temporal and historical relational features"),
        "evaluation_design": "chronological 70/15/15 split; the validation/test periods were excluded from the evaluation fit",
        "evaluation": unsup_metrics,
        "final_service_model": final_bundle["metadata"],
        "interpretation": [
            "The anomaly percentile is a ranking signal, not a probability of fraud, corruption or legal wrongdoing.",
            "High anomaly scores identify records that differ from the training distribution and need contextual human review.",
            "No supervised fraud metric is reported without traceable outcome labels and defensible negative examples.",
            "If the source data are synthetic, the artifact is for integration testing only and must not be used for real procurement decisions."
        ],
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if use_mlflow:
        run_id = _write_mlflow("mthl-unsupervised-baseline", final_bundle["metadata"], unsup_metrics)
        report["mlflow_run_id"] = run_id
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train MTHL anomaly ranking model")
    parser.add_argument("--input", default="data/processed/procurements.csv")
    parser.add_argument("--artifact", default="artifacts/model.joblib")
    parser.add_argument("--report", default="reports/evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow", action="store_true", help="Log this run to MLflow if installed")
    parser.add_argument("--positive-labels", default=None, help="CSV with verified positive OCIDs only (PU mode)")
    parser.add_argument("--positive-prior", type=float, default=None, help="Estimated positive class prior for PU mode; justify and sensitivity-test")
    args = parser.parse_args()
    report = train(args.input, args.artifact, args.report, args.seed, args.mlflow, args.positive_labels, args.positive_prior)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
