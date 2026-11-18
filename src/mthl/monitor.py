"""Lightweight input drift monitor using KS tests and population stability index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NUMERIC = [
    "estimated_value_log1p", "duration_days", "num_tenderers",
    "buyer_prior_process_count", "buyer_prior_supplier_hhi",
    "buyer_prior_awarded_process_count", "days_since_buyer_previous_tender",
]
CATEGORICAL = ["procurement_method", "procurement_category", "tender_status"]


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def monitor(reference_path: str, current_path: str, output_path: str):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    rows = []
    for col in NUMERIC:
        if col not in reference or col not in current:
            continue
        a = pd.to_numeric(reference[col], errors="coerce").dropna()
        b = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(a) < 10 or len(b) < 10:
            rows.append({"feature": col, "type": "numeric", "status": "insufficient_data", "n_reference": len(a), "n_current": len(b)})
            continue
        ks = ks_2samp(a, b)
        psi = population_stability_index(a, b)
        alert = bool((psi > 0.20) or (ks.pvalue < 0.01))
        rows.append({
            "feature": col, "type": "numeric", "ks_statistic": float(ks.statistic),
            "ks_p_value": float(ks.pvalue), "psi": float(psi),
            "n_reference": len(a), "n_current": len(b),
            "status": "drift_alert" if alert else "no_alert",
        })
    for col in CATEGORICAL:
        if col not in reference or col not in current:
            continue
        a = reference[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)
        b = current[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)
        levels = sorted(set(a.index) | set(b.index))
        pa = np.array([a.get(x, 0.0) for x in levels], dtype=float)
        pb = np.array([b.get(x, 0.0) for x in levels], dtype=float)
        pa = np.clip(pa, 1e-6, None); pb = np.clip(pb, 1e-6, None)
        psi = float(np.sum((pb - pa) * np.log(pb / pa)))
        rows.append({"feature": col, "type": "categorical", "psi": psi, "n_reference": len(reference), "n_current": len(current), "status": "drift_alert" if psi > 0.20 else "no_alert"})
    report = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "feature_results": rows,
        "alerts": int(sum(1 for r in rows if r["status"] == "drift_alert")),
        "thresholds": {"psi_alert": 0.20, "ks_p_value_alert": 0.01},
        "limitation": "This detects covariate/data drift only. Concept drift and predictive performance require later, valid outcome labels.",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default="reports/drift_report.json")
    args = parser.parse_args()
    print(json.dumps(monitor(args.reference, args.current, args.output), indent=2))


if __name__ == "__main__":
    main()
