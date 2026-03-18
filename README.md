<p align="center">
  <h1 align="center">🏥 MEDORBY</h1>
  <p align="center"><strong>Privacy-First Medical AI with LLM Council, RAG, and Federated Learning</strong></p>
  <p align="center">
    <em>Your medical data never leaves your device. Period.</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Next.js-16-black?logo=next.js" />
  <img src="https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
  <img src="https://img.shields.io/badge/Privacy-First-brightgreen?logo=shield" />
</p>

---

## What is MEDORBY?

MEDORBY is a **privacy-first medical AI assistant** that uses a multi-model **LLM Council** to provide clinical reasoning about symptoms. Unlike traditional health chatbots, MEDORBY:

- **Never sends your raw medical data to the cloud** — all PII is stripped client-side
- **Runs a local ML classifier** on your device before any API calls
- **Retrieves relevant medical knowledge** from a FAISS-indexed knowledge base (RAG)
- **Lets you upload medical reports** that stay on your machine
- **Improves over time** through Federated Learning without sharing your data

> ⚠️ **Disclaimer**: MEDORBY is a research project and NOT a replacement for professional medical advice. Always consult a qualified healthcare provider for medical decisions.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM Council** | 3 AI models deliberate in parallel → peer review → chairman synthesis |
| 🔒 **Zero PII Exposure** | Names, dates, contacts stripped in-browser before any network call |
| 🔍 **RAG Knowledge Base** | FAISS-indexed AHA/ACC medical guidelines for context-aware responses |
| 🩺 **Local ML Classifier** | On-device symptom classification — no cloud calls needed |
| 🧬 **Federated Neural Network** | Lightweight structured-health predictor (12-feature local model) |
| 📄 **Medical Report Upload** | Upload PDF/DOCX/TXT reports, indexed locally for personalised context |
| 🏥 **Hospital DB** | SQLite edge storage for anonymised consultation records |
| 🤝 **Federated Learning** | Your feedback improves the global model via DP-noised gradients |
| 📚 **Federated Knowledge Base** | Edge nodes can contribute sanitized knowledge snippets for shared RAG context |
| 🔐 **Encrypted Storage** | Local medical history encrypted with AES-256-GCM |
| ⚡ **Real-Time Streaming** | Server-Sent Events for live council progress updates |
| 🚨 **Red-Flag Engine** | Instant emergency detection — no LLM needed for critical cases |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 16 (App Router), React 19, Vanilla CSS, Lucide Icons |
| **Backend** | FastAPI + Uvicorn (Python 3.12+) |
| **LLM Council** | Groq (Llama 3.3 70B, Llama 3.1 8B), OpenRouter (DeepSeek Chat), Mistral AI (Mistral Small) |
| **RAG Engine** | FAISS + TF-IDF Vectorizer (scikit-learn) |
| **ML Classifier** | TF-IDF + Logistic Regression (scikit-learn) |
| **Local Storage** | IndexedDB + AES-256-GCM (client), SQLite (server) |
| **Privacy** | Client-side PII sanitizer + Differential Privacy |
| **Federated Learning** | FedAvg aggregator + DP noise injection |
| **Streaming** | Server-Sent Events (SSE) |

---

## 📂 Project Structure

```
AarogyaAI/
├── README.md
├── .gitignore
│
├── backend/
│   ├── main.py                        ← FastAPI entry point (all API routes)
│   ├── config.py                      ← Pydantic settings (.env loader)
│   ├── requirements.txt               ← Python dependencies
│   ├── .env                           ← API keys (gitignored — see setup below)
│   │
│   ├── core/                          ← Core utilities
│   │   ├── red_flag_engine.py         ← Deterministic emergency detection
│   │   └── sanitizer.py              ← Server-side PII sanitizer
│   │
│   ├── council/                       ← LLM Council module
│   │   ├── groq_client.py             ← Multi-provider LLM client (Groq, OpenRouter, Mistral)
│   │   └── orchestrator.py            ← 3-stage orchestration (Divergence → Convergence → Synthesis)
│   │
│   ├── ml/                            ← Machine Learning module
│   │   ├── symptom_classifier.py      ← Local symptom classifier (TF-IDF + LR)
│   │   ├── federated_nn.py            ← Federated neural network (12→32→64→32→5)
│   │   └── feature_extractor.py       ← Structured vital sign feature extraction
│   │
│   ├── rag/                           ← RAG module
│   │   ├── engine.py                  ← FAISS-based retrieval engine
│   │   └── report_processor.py        ← PDF/DOCX/TXT report processor
│   │
│   ├── storage/                       ← Storage module
│   │   └── hospital_db.py             ← SQLite hospital database
│   │
│   ├── federated/                     ← Federated Learning module
│   │   ├── aggregator.py              ← FedAvg gradient aggregator
│   │   ├── dp_privacy.py              ← Differential privacy utilities
│   │   └── knowledge_sync.py          ← Federated knowledge base sync
│   │
│   ├── knowledge_base/                ← Curated medical knowledge (JSON)
│   │
│   └── tests/                         ← Backend test suite (55 tests)
│       ├── test_routes.py
│       ├── test_sanitizer.py
│       └── test_hybrid_fusion.py
│
└── frontend/
    ├── package.json
    ├── next.config.ts
    │
    └── src/
        ├── app/
        │   ├── layout.tsx             ← Root layout + fonts
        │   ├── page.tsx               ← Landing page + consultation UI
        │   └── globals.css            ← Full design system (vanilla CSS)
        ├── components/
        │   └── ChatInterface.tsx      ← Main consultation interface (SSE + council UI)
        └── lib/
            ├── sanitizer.ts           ← Client-side PII sanitizer
            ├── local_learning.ts      ← Federated learning client
            └── storage.ts             ← Encrypted IndexedDB storage
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** with `pip`
- **Node.js 18+** with `npm`
- API keys (at least one required):
  - **Groq** (free) — [console.groq.com](https://console.groq.com) — powers Member A, Chairman, Reviewer
  - **OpenRouter** — [openrouter.ai](https://openrouter.ai) — powers Member B (DeepSeek Chat)
  - **Mistral AI** — [console.mistral.ai](https://console.mistral.ai) — powers Member C

### 1. Clone the Repository

```bash
git clone https://github.com/googlecodernaman/ArogyaAI_v2-.git
cd ArogyaAI_v2-
```

### 2. Setup Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file inside the `backend/` directory:

```env
# ── LLM Council (required) ─────────────────────────
GROQ_API_KEY=gsk_your_groq_key_here
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key_here
MISTRAL_API_KEY=your_mistral_key_here

# ── Optional ────────────────────────────────────────
FEDERATED_SECRET_KEY=your_random_secret
```

> 🔒 The `.env` file is **gitignored** — your API keys will never be committed.

### 4. Start Backend

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

You should see:
```
[HospitalDB] Database initialized.
[SymptomClassifier] Trained on 64 examples, 5 categories.
[RAG] Indexed 65+ documents.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 5. Setup & Start Frontend

Open a **new terminal**:

```bash
cd frontend
npm install
npm run dev
```

### 6. Open the App

Visit **[http://localhost:3000](http://localhost:3000)** in your browser. 🎉

### Verify Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "MEDORBY API",
  "version": "2.0.0",
  "models": {
    "member_a": { "model": "llama-3.3-70b-versatile", "provider": "groq" },
    "member_b": { "model": "deepseek/deepseek-chat", "provider": "openrouter" },
    "member_c": { "model": "mistral-small-latest", "provider": "mistral" },
    "chairman": { "model": "llama-3.3-70b-versatile", "provider": "groq" },
    "reviewer": { "model": "llama-3.1-8b-instant", "provider": "groq" }
  }
}
```

---

## 🧠 LLM Council Architecture

The council uses a 3-stage protocol for robust clinical reasoning:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: DIVERGENCE                         │
│                                                                     │
│  Member A (Groq/Llama 70B) ────┐                                   │
│  Member B (OpenRouter/DeepSeek) ├──→ 3 independent clinical opinions│
│  Member C (Mistral/Small) ─────┘                                   │
├─────────────────────────────────────────────────────────────────────┤
│                        STAGE 2: CONVERGENCE                        │
│                                                                     │
│  Reviewer (Groq/Llama 8B) ──→ Ranks all 3 opinions blindly        │
├─────────────────────────────────────────────────────────────────────┤
│                        STAGE 3: SYNTHESIS                          │
│                                                                     │
│  Chairman (Groq/Llama 70B) ──→ Final consensus answer              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model info |
| `POST` | `/api/triage` | Deterministic red-flag check (no LLM) |
| `POST` | `/api/council` | Full council deliberation (SSE streaming) |
| `POST` | `/api/classify` | Local ML symptom classification |
| `POST` | `/api/fnn/predict` | Federated NN structured prediction from vitals |
| `POST` | `/api/rag/retrieve` | RAG knowledge base retrieval |
| `POST` | `/api/hybrid/assess` | Hybrid deterministic + NN + RAG + council assessment |
| `GET` | `/api/rag/stats` | RAG engine statistics |
| `POST` | `/api/reports/upload` | Upload medical report (PDF/DOCX/TXT) |
| `GET` | `/api/reports` | List uploaded reports |
| `DELETE` | `/api/reports/{id}` | Delete a report |
| `POST` | `/api/reports/analyze/{id}` | AI analysis of a report |
| `GET` | `/api/hospital/stats` | Hospital DB statistics |
| `GET` | `/api/hospital/records` | Anonymised consultation records |
| `POST` | `/api/federated/update` | Submit DP-noised model update |
| `GET` | `/api/federated/adapter` | Download latest global adapter |
| `GET` | `/api/federated/status` | Aggregator status |
| `POST` | `/api/federated/kb/update` | Submit sanitized federated KB entry |
| `GET` | `/api/federated/kb/status` | Federated KB + RAG status |

---

## 🧪 Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

Expected: **55 tests pass** covering routes, sanitizer, and hybrid fusion logic.

---

## 🔒 Privacy Guarantees

| Guarantee | How |
|-----------|-----|
| ✅ **No PII sent to cloud** | Client-side regex sanitizer strips names, emails, phone numbers, dates of birth before any API call |
| ✅ **Emergency detection is local** | Red-Flag Engine runs deterministic rules — zero cloud dependency |
| ✅ **ML classification is local** | TF-IDF + Logistic Regression model runs on your machine |
| ✅ **Reports stay on your device** | Uploaded PDFs/DOCX never leave `user_reports/` on your filesystem |
| ✅ **Encrypted local storage** | IndexedDB data encrypted with AES-256-GCM; only you hold the key |
| ✅ **Federated learning is private** | Gradient clipping + Gaussian DP noise before any update leaves the device |
| ✅ **Anonymised DB records** | Hospital DB stores symptom hashes, never raw text |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Built with ❤️ for privacy-respecting medical AI
</p>
