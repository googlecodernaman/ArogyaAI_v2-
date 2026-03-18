"""
Integration tests — HTTP route verification (in-process ASGI, no real server needed).

Tests cover:
  /api/hybrid/assess    — full hybrid prediction, all sources present
  /api/fnn/predict      — structured FNN prediction with vitals
  /api/federated/update — correct and incorrect gradient dimensions
  /api/federated/kb/update — knowledge base update and RAG re-index
  /api/triage           — deterministic red-flag triage, emergency and normal paths
  /api/classify         — local ML text classifier
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pytest
import pytest_asyncio
import httpx

from main import app
from ml.federated_nn import NN_WEIGHT_DIM

ASGI = httpx.ASGITransport(app=app)
VITALS = {
    "age": 65,
    "systolic_bp": 160,
    "diastolic_bp": 96,
    "heart_rate": 92,
    "spo2": 97,
    "bmi": 28.0,
    "symptom_severity": 2,
    "symptom_duration_days": 3,
    "has_diabetes": True,
    "has_hypertension": True,
    "has_family_history": False,
    "is_smoker": False,
}
EMERGENCY_VITALS = {**VITALS, "spo2": 78, "systolic_bp": 210, "heart_rate": 155}


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(
        transport=ASGI, base_url="http://test", timeout=60.0
    ) as c:
        yield c


# ── /api/triage ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_triage_normal(client):
    r = await client.post("/api/triage", json={
        "symptom_text": "mild headache after waking up, no chest pain",
    })
    assert r.status_code == 200
    d = r.json()
    assert "is_emergency" in d
    assert d["is_emergency"] is False


@pytest.mark.asyncio
async def test_triage_emergency_vitals(client):
    r = await client.post("/api/triage", json={
        "symptom_text": "sudden chest pain and dizziness",
        "vitals": EMERGENCY_VITALS,
    })
    assert r.status_code == 200
    # low SpO2 (78) should trigger emergency
    assert r.json()["is_emergency"] is True


@pytest.mark.asyncio
async def test_triage_empty_prompt_rejected(client):
    r = await client.post("/api/triage", json={"symptom_text": "   "})
    assert r.status_code == 400


# ── /api/classify ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_returns_category(client):
    r = await client.post("/api/classify", json={
        "symptom_text": "crushing chest pressure radiating to left arm with sweating",
    })
    assert r.status_code == 200
    d = r.json()
    assert d["category"] in [
        "cardiac_emergency", "cardiac_chronic", "cardiac_arrhythmia",
        "cardiac_risk", "non_cardiac",
    ]
    assert 0.0 <= d["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_classify_non_cardiac_text(client):
    r = await client.post("/api/classify", json={
        "symptom_text": "stomach pain after eating spicy food, no fever",
    })
    assert r.status_code == 200
    d = r.json()
    assert d["category"] in [
        "cardiac_emergency", "cardiac_chronic", "cardiac_arrhythmia",
        "cardiac_risk", "non_cardiac",
    ]


# ── /api/fnn/predict ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fnn_predict_with_vitals(client):
    r = await client.post("/api/fnn/predict", json={"vitals": VITALS})
    assert r.status_code == 200
    d = r.json()
    pred = d["prediction"]
    assert pred["category"] in [
        "cardiac_emergency", "cardiac_chronic", "cardiac_arrhythmia",
        "cardiac_risk", "non_cardiac",
    ]
    assert 0.0 <= pred["confidence"] <= 1.0
    assert d["feature_dim"] == 12
    assert "vital_interpretation" in d


@pytest.mark.asyncio
async def test_fnn_predict_missing_vitals_rejected(client):
    r = await client.post("/api/fnn/predict", json={"symptom_text": "chest pain"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_fnn_predicts_all_five_probability_keys(client):
    r = await client.post("/api/fnn/predict", json={"vitals": VITALS})
    assert r.status_code == 200
    probs = r.json()["prediction"]["probabilities"]
    expected = {
        "cardiac_emergency", "cardiac_chronic", "cardiac_arrhythmia",
        "cardiac_risk", "non_cardiac",
    }
    assert set(probs.keys()) == expected


# ── /api/hybrid/assess ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hybrid_assess_returns_full_structure(client):
    r = await client.post("/api/hybrid/assess", json={
        "symptom_text": "elevated blood pressure 145/90 with mild headache",
        "vitals": VITALS,
    })
    assert r.status_code == 200
    d = r.json()
    # All top-level keys present
    for key in ("sanitization", "triage", "classification", "structured_prediction",
                "rag", "hybrid_decision"):
        assert key in d, f"Missing key: {key}"

    hd = d["hybrid_decision"]
    assert hd["final_category"] in [
        "cardiac_emergency", "cardiac_chronic", "cardiac_arrhythmia",
        "cardiac_risk", "non_cardiac",
    ]
    assert isinstance(hd["is_emergency"], bool)
    assert 0.0 <= hd["confidence"] <= 1.0
    assert len(hd["score_breakdown"]) == 5


@pytest.mark.asyncio
async def test_hybrid_assess_no_vitals_still_works(client):
    """Without vitals the FNN prediction is absent but the route must not error."""
    r = await client.post("/api/hybrid/assess", json={
        "symptom_text": "shortness of breath and fatigue for two weeks",
    })
    assert r.status_code == 200
    d = r.json()
    assert d["structured_prediction"]["prediction"] is None
    assert "final_category" in d["hybrid_decision"]


@pytest.mark.asyncio
async def test_hybrid_assess_pii_is_stripped(client):
    """PII in the prompt is sanitized before being forwarded."""
    r = await client.post("/api/hybrid/assess", json={
        "symptom_text": "Dr. Jane Doe, phone 555-000-9999. Chest pain since 01/01/2024.",
    })
    assert r.status_code == 200
    san = r.json()["sanitization"]
    assert san["changed"] is True


@pytest.mark.asyncio
async def test_hybrid_assess_empty_prompt_rejected(client):
    r = await client.post("/api/hybrid/assess", json={"symptom_text": ""})
    assert r.status_code == 400


# ── /api/federated/update ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_federated_update_correct_dim(client):
    r = await client.post("/api/federated/update", json={
        "client_id": "pytest_node_1",
        "gradients": [0.001] * NN_WEIGHT_DIM,
    })
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "accepted"
    assert d["pending_count"] >= 1


@pytest.mark.asyncio
async def test_federated_update_wrong_dim_rejected(client):
    for bad_dim in [64, 128, 256, 1000]:
        r = await client.post("/api/federated/update", json={
            "client_id": "bad_node",
            "gradients": [0.0] * bad_dim,
        })
        assert r.status_code == 400, f"Expected 400 for dim={bad_dim}, got {r.status_code}"
        assert str(NN_WEIGHT_DIM) in r.json()["detail"]


@pytest.mark.asyncio
async def test_federated_update_empty_gradients_rejected(client):
    r = await client.post("/api/federated/update", json={
        "client_id": "empty_node",
        "gradients": [],
    })
    assert r.status_code == 400


# ── /api/federated/kb/update ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kb_update_accepted_and_indexed(client):
    # Get initial RAG doc count
    stats_r = await client.get("/api/rag/stats")
    initial_count = stats_r.json().get("total_documents", 0)

    r = await client.post("/api/federated/kb/update", json={
        "client_id": "hospital_node_pytest",
        "topic": "Acute Pulmonary Oedema Management",
        "content": (
            "Acute pulmonary oedema: IV furosemide 40-80mg, GTN infusion if SBP>100, "
            "CPAP or BiPAP for respiratory support. Morphine use is controversial."
        ),
        "source": "pytest_integration_test",
    })
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "accepted"
    assert d["rag_stats"]["total_documents"] == initial_count + 1


@pytest.mark.asyncio
async def test_kb_update_empty_topic_rejected(client):
    r = await client.post("/api/federated/kb/update", json={
        "client_id": "node_x",
        "topic": "",
        "content": "some content",
    })
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_kb_update_pii_in_content_is_stripped(client):
    r = await client.post("/api/federated/kb/update", json={
        "client_id": "node_pii_test",
        "topic": "Case insight with PII",
        "content": "Patient Mr. Smith, DOB 12/31/1970, called 555-123-4567 about his chest pain.",
        "source": "pytest",
    })
    assert r.status_code == 200
    # Content accepted (sanitized); verify via RAG retrieval that PII isn't stored
    rag_r = await client.post("/api/rag/retrieve", json={
        "symptom_text": "chest pain case insight"
    })
    assert rag_r.status_code == 200
    results_text = str(rag_r.json())
    assert "Mr. Smith" not in results_text
    assert "555-123-4567" not in results_text
