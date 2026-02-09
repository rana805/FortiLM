# FortiLM

**Secure LLM Integration Platform** — A full-stack security framework for Large Language Model deployments with prompt-to-output content filtering and real-time admin dashboards. Hybrid AI jailbreak detection and privacy preserver system.

## Overview

FortiLM provides:

- **Privacy Preserver** — PII detection and masking (BERT-based + rule-based)
- **Output Filter** — Toxicity, bias, and jailbreak detection with Block/Censor/Warn strategies
- **Admin Dashboards** — Real-time Privacy, Output Filter, and Unified Security views

**Architecture:** User Input → Privacy Preserver → LLM API → Output Filter → Response (with full audit trail)

## Tech Stack

| Layer      | Stack |
|-----------|--------|
| Backend   | Python, FastAPI, PostgreSQL, MongoDB |
| Frontend  | Next.js, React, TailwindCSS, Recharts |
| ML/NLP    | BERT, DistilBERT (Hugging Face Transformers) |
| LLM       | Groq API (Llama 3.1) |

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL and MongoDB (or use Docker)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Set .env (DATABASE_URL, MONGODB_URI, GROQ_API_KEY, etc.)
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Run both

From project root (if `start.sh` is present):

```bash
chmod +x start.sh && ./start.sh
```

Backend: `http://localhost:8000` · Frontend: `http://localhost:3000`

## Project Structure

```
fortilm-project/
├── backend/           # FastAPI app, modules, ML model loaders
│   ├── modules/       # Chat, admin, security (privacy, output filter)
│   ├── ml_models/     # PII, toxicity, bias detectors (wrappers + training scripts)
│   ├── models/        # SQLAlchemy DB models
│   └── utils/        # Model loading, helpers
├── frontend/          # Next.js app (chat, admin dashboards)
│   └── src/app/      # Chat, admin/output-filter, admin/privacy-preserver, admin/unified-security
├── docker-compose.yml
└── start.sh
```

## Model Training

Training scripts and guides live under `backend/ml_models/training/` (e.g. Colab-ready scripts for toxicity and bias detectors). Datasets and large artifacts are not included in the repo; see the training READMEs for data setup.

## License

MIT (or your chosen license).
