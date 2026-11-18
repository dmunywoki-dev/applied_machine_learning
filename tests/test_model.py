import numpy as np
import pandas as pd

from mthl.model import fit_isolation_forest, predict_frame


def test_model_fits_and_returns_bounded_priority_scores():
    n = 40
    frame = pd.DataFrame({
        "estimated_value_log1p": np.linspace(10, 18, n),
        "duration_days": [14] * n,
        "num_tenderers": [3] * n,
        "tenderer_data_available": [1] * n,
        "buyer_prior_process_count": np.arange(n),
        "buyer_prior_direct_share": [0.1] * n,
        "buyer_prior_awarded_process_count": np.arange(n),
        "buyer_prior_award_value_log1p": np.linspace(0, 12, n),
        "buyer_prior_unique_suppliers": np.arange(n) % 10,
        "buyer_prior_supplier_hhi": [0.2] * n,
        "buyer_prior_top_supplier_share": [0.3] * n,
        "days_since_buyer_previous_tender": [30] * n,
        "pub_month_sin": [0.5] * n,
        "pub_month_cos": [0.8] * n,
        "pub_weekday": [1] * n,
        "flag_short_tender": [0] * n,
        "flag_non_open_method": [0] * n,
        "flag_single_disclosed_tenderer": [0] * n,
        "flag_high_prior_supplier_concentration": [0] * n,
        "text_missing": [0] * n,
        "buyer_missing": [0] * n,
        "duration_missing": [0] * n,
        "estimated_value_missing": [0] * n,
        "hyperedge_size": [3] * n,
        "hyperedge_prior_recurrence": [0] * n,
        "hyperedge_prior_pair_recurrence_mean": [0] * n,
        "procurement_method": ["open"] * n,
        "procurement_category": ["goods"] * n,
        "tender_status": ["active"] * n,
        "text": ["supply office stationery goods procurement"] * n,
    })
    bundle = fit_isolation_forest(frame)
    scores = predict_frame(bundle, frame)
    assert len(scores) == n
    assert scores["risk_priority_score"].between(0, 1).all()
