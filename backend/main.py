"""
MEDORBY FastAPI Backend — Main Application
Implements the 3-stage LLM Council with SSE streaming,
Red-Flag Engine, Federated Learning endpoints,
RAG-enhanced Knowledge Base, Medical Report Upload,
Local Symptom Classifier, and Hospital DB.
"""
import json
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# ── Modular imports ───────────────────────────────────────────────────────────
from core.red_flag_engine import evaluate as red_flag_evaluate
from council.orchestrator import run_divergence, run_convergence, run_synthesis, orchestrate
from council.groq_client import COUNCIL_MODELS
from rag.engine import get_rag_engine, rebuild_rag_engine
from rag.report_processor import process_report, get_all_reports, delete_report, get_report_text
from ml.symptom_classifier import get_classifier
from storage.hospital_db import (
    store_consultation, store_report_record,
    log_federated_contribution, get_records, get_db_stats,
)
from federated.aggregator import receive_update, aggregate, get_latest_adapter, get_status
from config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-initialize heavy singletons at startup so logs appear immediately."""
    get_classifier()   # prints [SymptomClassifier] Trained on N examples...
    get_rag_engine()   # prints [RAG] Indexed N documents.
    yield


app = FastAPI(
    title="MEDORBY API",
    description="Privacy-First Medical AI — LLM Council Orchestrator with RAG, Local ML, and Hospital DB",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "MEDORBY API", "version": "2.0.0", "docs": "/docs", "health": "/health"}


# ─── Request / Response Models ────────────────────────────────────────────────

class SymptomRequest(BaseModel):
    sanitized_prompt: str          # PII-free prompt built on the client
    vitals: Optional[dict] = None  # Optional vital signs dict


class FederatedUpdateRequest(BaseModel):
    client_id: str
    gradients: list[float]         # DP-noised adapter delta from client


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    rag = get_rag_engine()
    classifier = get_classifier()
    return {
        "status": "ok",
        "service": "MEDORBY API",
        "version": "2.0.0",
        "models": COUNCIL_MODELS,
        "features": {
            "rag_indexed": rag.get_stats()["total_documents"],
            "classifier_ready": classifier.is_trained,
            "hospital_db": get_db_stats(),
        },
    }


# ─── Red-Flag Triage (deterministic, no LLM) ─────────────────────────────────

@app.post("/api/triage")
async def triage(request: SymptomRequest):
    """
    Deterministic red-flag evaluation.
    Returns immediately — no cloud calls made.
    """
    result = red_flag_evaluate(
        symptom_text=request.sanitized_prompt,
        vitals=request.vitals or {},
    )
    return result


# ─── Local Symptom Classifier ─────────────────────────────────────────────────

@app.post("/api/classify")
async def classify_symptoms(request: SymptomRequest):
    """
    Run the local ML classifier on the symptom text.
    Returns predicted category, severity, confidence, and recommended action.
    No cloud calls made — runs entirely on-device.
    """
    classifier = get_classifier()
    prediction = classifier.predict(request.sanitized_prompt)
    return prediction


# ─── RAG Context Retrieval ────────────────────────────────────────────────────

@app.post("/api/rag/retrieve")
async def rag_retrieve(request: SymptomRequest):
    """
    Retrieve relevant medical context from the knowledge base and user reports.
    Returns matched documents with relevance scores.
    """
    rag = get_rag_engine()
    results = rag.retrieve(request.sanitized_prompt, top_k=5)
    return {
        "query": request.sanitized_prompt[:200],
        "results": results,
        "stats": rag.get_stats(),
    }


@app.get("/api/rag/stats")
async def rag_stats():
    """Return RAG engine statistics."""
    rag = get_rag_engine()
    return rag.get_stats()


# ─── Council Endpoint (SSE Streaming) — Now RAG-enhanced ─────────────────────

async def council_event_generator(sanitized_prompt: str):
    """Async generator that streams council stages as SSE events."""
    try:
        # Pre-stage: Classify symptoms locally
        classifier = get_classifier()
        classification = classifier.predict(sanitized_prompt)
        yield f"data: {json.dumps({'stage': 'classification', 'status': 'complete', 'data': classification})}\n\n"

        # Pre-stage: Retrieve RAG context
        rag = get_rag_engine()
        rag_context = rag.get_context_for_prompt(sanitized_prompt, top_k=3)
        rag_results = rag.retrieve(sanitized_prompt, top_k=3)
        yield f"data: {json.dumps({'stage': 'rag_retrieval', 'status': 'complete', 'data': {'documents_found': len(rag_results), 'topics': [r['topic'] for r in rag_results]}})}\n\n"

        # Augment the prompt with RAG context
        augmented_prompt = sanitized_prompt
        if rag_context:
            augmented_prompt = sanitized_prompt + "\n" + rag_context

        # Stage 1: Divergence
        yield f"data: {json.dumps({'stage': 'divergence', 'status': 'running'})}\n\n"
        divergence = await run_divergence(augmented_prompt)
        yield f"data: {json.dumps({'stage': 'divergence', 'status': 'complete', 'data': divergence})}\n\n"

        # Stage 2: Convergence
        yield f"data: {json.dumps({'stage': 'convergence', 'status': 'running'})}\n\n"
        convergence = await run_convergence(augmented_prompt, divergence)
        yield f"data: {json.dumps({'stage': 'convergence', 'status': 'complete', 'data': convergence['peer_review']})}\n\n"

        # Stage 3: Synthesis
        yield f"data: {json.dumps({'stage': 'synthesis', 'status': 'running'})}\n\n"
        synthesis = await run_synthesis(augmented_prompt, convergence)
        yield f"data: {json.dumps({'stage': 'synthesis', 'status': 'complete', 'data': synthesis})}\n\n"

        # Post-stage: Store in Hospital DB (anonymised)
        try:
            symptoms_hash = hashlib.sha256(sanitized_prompt.encode()).hexdigest()[:16]
            store_consultation(
                category=classification.get("category", "unknown"),
                severity=classification.get("severity", "unknown"),
                symptoms_hash=symptoms_hash,
                council_summary=synthesis.get("summary", "")[:500],
                confidence=synthesis.get("confidence", 0),
                metadata={
                    "rag_docs_used": len(rag_results),
                    "classification_confidence": classification.get("confidence", 0),
                },
            )
        except Exception as e:
            print(f"[HospitalDB] Error storing consultation: {e}")

        yield f"data: {json.dumps({'stage': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'stage': 'error', 'message': str(e)})}\n\n"


@app.post("/api/council")
async def council(request: SymptomRequest):
    """
    Main council endpoint — streams the 3-stage deliberation via SSE.
    Now includes pre-stages for classification and RAG retrieval.
    The client must first call /api/triage; this endpoint assumes no emergency.
    """
    if not request.sanitized_prompt.strip():
        raise HTTPException(status_code=400, detail="sanitized_prompt cannot be empty.")

    return StreamingResponse(
        council_event_generator(request.sanitized_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Medical Report Endpoints ────────────────────────────────────────────────

@app.post("/api/reports/upload")
async def upload_report(file: UploadFile = File(...)):
    """
    Upload a medical report (PDF, DOCX, TXT).
    Extracts text, stores locally, and re-indexes the RAG engine.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    result = process_report(file.filename, file_bytes)

    try:
        store_report_record(
            report_id=result["id"],
            category="user_report",
            summary=f"Uploaded report: {file.filename} ({result['word_count']} words)",
            metadata={"filename": file.filename, "word_count": result["word_count"]},
        )
    except Exception as e:
        print(f"[HospitalDB] Error storing report record: {e}")

    rebuild_rag_engine()
    return result


@app.get("/api/reports")
async def list_reports():
    """List all uploaded medical reports."""
    return {"reports": get_all_reports()}


@app.delete("/api/reports/{report_id}")
async def remove_report(report_id: str):
    """Delete a medical report and rebuild the index."""
    success = delete_report(report_id)
    if not success:
        raise HTTPException(status_code=404, detail="Report not found.")

    rebuild_rag_engine()
    return {"status": "deleted", "id": report_id}


# ─── Report Analysis Endpoint ────────────────────────────────────────────────

@app.post("/api/reports/analyze/{report_id}")
async def analyze_report(report_id: str):
    """
    Analyze a specific medical report using RAG + LLM.
    Retrieves the report text, finds relevant knowledge base context,
    and sends to the council for insights.
    """
    report_text = get_report_text(report_id)
    if not report_text:
        raise HTTPException(status_code=404, detail="Report not found.")

    rag = get_rag_engine()
    rag_context = rag.get_context_for_prompt(report_text[:1000], top_k=3)

    classifier = get_classifier()
    classification = classifier.predict(report_text[:500])

    analysis_prompt = (
        f"Medical Report Analysis:\n{report_text[:1500]}\n\n"
        f"Classification: {classification['label']} "
        f"(confidence: {classification['confidence']})\n"
        f"{rag_context}\n\n"
        f"Provide a clinical summary, key findings, risk assessment, "
        f"and recommended follow-up actions based on this report."
    )

    synthesis = await orchestrate(analysis_prompt)

    return {
        "report_id": report_id,
        "classification": classification,
        "analysis": synthesis.get("synthesis", {}),
        "rag_context_used": bool(rag_context),
    }


# ─── Hospital DB Endpoints ───────────────────────────────────────────────────

@app.get("/api/hospital/stats")
async def hospital_stats():
    """Get hospital database statistics."""
    return get_db_stats()


@app.get("/api/hospital/records")
async def hospital_records(record_type: Optional[str] = None, limit: int = 20):
    """Get medical records from the hospital DB."""
    records = get_records(record_type=record_type, limit=limit)
    return {"records": records, "count": len(records)}


# ─── Federated Learning Endpoints ────────────────────────────────────────────

@app.post("/api/federated/update")
async def federated_update(request: FederatedUpdateRequest):
    """
    Receive a DP-noised adapter delta from a client device.
    Automatically triggers aggregation when enough updates are buffered.
    """
    result = receive_update(request.client_id, request.gradients)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    try:
        gradient_hash = hashlib.sha256(
            json.dumps(request.gradients[:10]).encode()
        ).hexdigest()[:16]
        log_federated_contribution(
            record_id="",
            gradient_hash=gradient_hash,
            dp_noise_level=1.1,
        )
    except Exception as e:
        print(f"[HospitalDB] Error logging contribution: {e}")

    agg_result = aggregate(min_clients=3)
    if agg_result:
        result["aggregation"] = agg_result

    return result


@app.get("/api/federated/adapter")
async def get_adapter():
    """Download the latest global adapter weights."""
    adapter = get_latest_adapter()
    if not adapter:
        return {"status": "no_adapter", "message": "No global adapter available yet."}
    return adapter


@app.get("/api/federated/status")
async def federated_status():
    """Return current federated aggregator status."""
    return get_status()
