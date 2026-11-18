"""Rule-based evidence trace accompanying model scores (not causal explanations)."""
from __future__ import annotations

import math
from typing import Any


def _number(record: dict, key: str):
    try:
        value = float(record.get(key))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def evidence_trace(record: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, str]] = []
    duration = _number(record, "duration_days")
    if duration is not None and 0 < duration < 7:
        flags.append({"code": "SHORT_TENDER_PERIOD", "description": f"Recorded tender period is {duration:.1f} days (<7 days). Context and applicable rules require human review."})

    method = str(record.get("procurement_method") or "unknown").lower()
    if method not in {"open", "unknown", ""}:
        flags.append({"code": "NON_OPEN_METHOD", "description": f"Published procurement method is '{method}'. This may be lawful and is not evidence of wrongdoing by itself."})

    tenderer_count = _number(record, "num_tenderers")
    if tenderer_count == 1 and bool(record.get("tenderer_data_available", 0)):
        flags.append({"code": "ONE_DISCLOSED_TENDERER", "description": "One tenderer is recorded in the available source. Tenderer disclosure may be incomplete; this is not proof of restricted competition."})

    prior_hhi = _number(record, "buyer_prior_supplier_hhi")
    if prior_hhi is not None and prior_hhi >= 0.50 and _number(record, "buyer_prior_awarded_process_count"):
        flags.append({"code": "HISTORICAL_BUYER_SUPPLIER_CONCENTRATION", "description": f"The buyer's prior supplier-award concentration HHI is {prior_hhi:.2f}; market structure and purchasing needs should be checked."})

    quality_fields = ["buyer_missing", "duration_missing", "estimated_value_missing", "text_missing"]
    missing = sum(int(bool(record.get(k, 0))) for k in quality_fields)
    confidence = max(0.0, 1.0 - missing / len(quality_fields))
    if missing:
        flags.append({"code": "INCOMPLETE_SOURCE_FIELDS", "description": f"{missing} of {len(quality_fields)} key feature groups are missing. Missingness lowers confidence; it is not itself a risk indicator."})

    if not bool(record.get("tenderer_data_available", 0)):
        flags.append({"code": "TENDERER_DATA_NOT_DISCLOSED", "description": "No explicit tenderer-role identities were available at the prediction time. No bidder-count or bidder-group inference is made."})

    return {
        "data_confidence": round(confidence, 3),
        "evidence_flags": flags,
        "caveat": "These are source-traceable screening indicators, not findings of fraud, corruption or legal liability.",
    }
