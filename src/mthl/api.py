"""FastAPI scoring service for the MTHL procurement risk-prioritisation model."""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .explain import evidence_trace
from .model import load_bundle, predict_frame

MODEL_PATH = Path(os.getenv("MTHL_MODEL_PATH", "artifacts/demo_model.joblib"))


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model_bundle = load_bundle(MODEL_PATH) if MODEL_PATH.exists() else None
    yield


app = FastAPI(
    title="MTHL Procurement Integrity Screening API",
    version="0.1.0",
    description=(
        "Explainable anomaly-priority scoring for procurement review. Scores are not fraud probabilities "
        "and do not make legal findings."
    ),
    lifespan=lifespan,
)


class ProcurementInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    procurement_method: str = "unknown"
    procurement_category: str = "unknown"
    tender_status: str = "unknown"
    title: str = ""
    description: str = ""
    text: str | None = None
    estimated_value_log1p: float | None = None
    duration_days: float | None = None
    num_tenderers: int | None = None
    tenderer_data_available: int = 0
    buyer_prior_process_count: int | None = None
    buyer_prior_direct_share: float | None = None
    buyer_prior_awarded_process_count: int | None = None
    buyer_prior_award_value_log1p: float | None = None
    buyer_prior_unique_suppliers: int | None = None
    buyer_prior_supplier_hhi: float | None = None
    buyer_prior_top_supplier_share: float | None = None
    days_since_buyer_previous_tender: float | None = None
    pub_month_sin: float | None = None
    pub_month_cos: float | None = None
    pub_weekday: int | None = None
    flag_short_tender: int | None = None
    flag_non_open_method: int | None = None
    flag_single_disclosed_tenderer: int | None = None
    flag_high_prior_supplier_concentration: int | None = None
    text_missing: int | None = None
    buyer_missing: int | None = None
    duration_missing: int | None = None
    estimated_value_missing: int | None = None
    hyperedge_size: int | None = None
    hyperedge_prior_recurrence: int | None = None
    hyperedge_prior_pair_recurrence_mean: float | None = None


@app.get("/health")
def health():
    bundle = getattr(app.state, "model_bundle", None)
    return {
        "status": "ok" if bundle is not None else "model_not_loaded",
        "model_path": str(MODEL_PATH),
        "model_type": bundle.get("model_type") if bundle else None,
        "training_mode": bundle.get("metadata", {}).get("training_mode", "demo") if bundle else None,
        "synthetic_data_warning": bundle.get("metadata", {}).get("synthetic_data_warning", False) if bundle else None,
        "score_semantics": "risk-priority percentile; not a probability of wrongdoing",
    }


@app.get("/")
def root():
    return {"service": "MTHL Procurement Integrity Screening API", "docs": "/docs", "health": "/health"}


@app.post("/score")
def score(payload: ProcurementInput):
    bundle = getattr(app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail=f"No model loaded. Train a model and set MTHL_MODEL_PATH (current: {MODEL_PATH}).")
    record: dict[str, Any] = payload.model_dump()
    # Keep the request text as a single text modality while allowing the service to accept title/description.
    if not record.get("text"):
        record["text"] = (str(record.get("title") or "") + " " + str(record.get("description") or "")).strip()
    if not record.get("text"):
        record["text"] = "untitled procurement"
    if record.get("duration_days") is not None and record["duration_days"] < 0:
        raise HTTPException(status_code=422, detail="duration_days cannot be negative")
    frame = pd.DataFrame([record])
    try:
        prediction = predict_frame(bundle, frame).iloc[0].to_dict()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not score request: {exc}") from exc
    trace = evidence_trace(record)
    return {
        "request_id": str(uuid.uuid4()),
        "risk_priority_score": round(float(prediction["risk_priority_score"]), 4),
        "risk_band": str(prediction["risk_band"]),
        "model_raw_score": round(float(prediction["model_raw_score"]), 6),
        "score_kind": "empirical anomaly percentile; not a fraud probability",
        "evidence_trace": trace,
        "model_version": bundle.get("metadata", {}).get("model_version", "0.1.0"),
        "synthetic_data_warning": bool(bundle.get("metadata", {}).get("synthetic_data_warning", False)),
        "decision": "human_review_required_for_any_follow_up",
    }
