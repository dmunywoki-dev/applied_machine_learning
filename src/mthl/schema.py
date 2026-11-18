"""Feature schema shared by ingestion, training and serving."""

NUMERIC_FEATURES = [
    "estimated_value_log1p",
    "duration_days",
    "num_tenderers",
    "tenderer_data_available",
    "buyer_prior_process_count",
    "buyer_prior_direct_share",
    "buyer_prior_awarded_process_count",
    "buyer_prior_award_value_log1p",
    "buyer_prior_unique_suppliers",
    "buyer_prior_supplier_hhi",
    "buyer_prior_top_supplier_share",
    "days_since_buyer_previous_tender",
    "pub_month_sin",
    "pub_month_cos",
    "pub_weekday",
    "flag_short_tender",
    "flag_non_open_method",
    "flag_single_disclosed_tenderer",
    "flag_high_prior_supplier_concentration",
    "text_missing",
    "buyer_missing",
    "duration_missing",
    "estimated_value_missing",
    "hyperedge_size",
    "hyperedge_prior_recurrence",
    "hyperedge_prior_pair_recurrence_mean",
]

CATEGORICAL_FEATURES = ["procurement_method", "procurement_category", "tender_status"]
TEXT_FEATURE = "text"
IDENTIFIER_COLUMNS = ["ocid", "prediction_time", "buyer_id", "buyer_name", "county"]


def model_feature_columns():
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]
