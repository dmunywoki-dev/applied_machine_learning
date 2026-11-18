from fastapi.testclient import TestClient
from mthl.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "score_semantics" in response.json()
        assert "synthetic_data_warning" in response.json()


def test_score_endpoint_returns_review_priority_not_probability():
    payload = {
        "procurement_method": "open",
        "procurement_category": "goods",
        "tender_status": "active",
        "title": "Supply of ICT equipment",
        "description": "Supply and delivery of computers and printers",
        "duration_days": 5,
        "estimated_value_log1p": 15.2,
        "buyer_prior_process_count": 8,
        "buyer_prior_supplier_hhi": 0.62,
    }
    with TestClient(app) as client:
        response = client.post("/score", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_priority_score"] <= 1.0
    assert body["score_kind"] == "empirical anomaly percentile; not a fraud probability"
    assert body["decision"] == "human_review_required_for_any_follow_up"
    assert "evidence_trace" in body
