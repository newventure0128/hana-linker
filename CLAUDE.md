# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Linker — an end-to-end AI consultation agent for Hana Card (KPMG 6기 2팀 final project). Korean-language conversational system that triages customer inquiries, answers via RAG, and hands off to a human consultant when needed. The repo lives under `hana-linker/` — treat that subdirectory as the project root for all commands below.

## Common commands

All commands assume cwd = `hana-linker/`.

### Backend (Python / FastAPI)

```bash
# Install deps
pip install -r requirements.txt

# Run dev server (LangGraph workflow + WebSocket voice)
uvicorn app.main:app --reload --port 8000
# Swagger: http://localhost:8000/docs

# Initial / refresh vector DB indexing
python scripts/ingest_to_chromadb.py
python scripts/reset_and_ingest.py        # wipe + re-ingest
python scripts/view_chromadb.py           # inspect contents

# Fine-tune the intent classifier (KcELECTRA + LoRA, 38 categories)
python scripts/finetune_intent_classifier.py
```

### Frontends

```bash
# Customer voice chatbot (Vite + React 19, TS)
cd voice-chatbot-revision && npm install && npm run dev      # port 5173
cd voice-chatbot-revision && npm run build                   # tsc && vite build

# Consultant dashboard (CRA + React 19, TS, styled-components)
cd agent-dashboard && npm install && npm start               # port 3000
cd agent-dashboard && npm test                               # single test: npm test -- -t "<name>"
cd agent-dashboard && npm run build
```

### E2E evaluation pipeline

```bash
# Module-scoped or full evaluation, produces HTML+JSON in reports/
python -m e2e_evaluation_pipeline --mode quick
python -m e2e_evaluation_pipeline --mode full
python -m e2e_evaluation_pipeline --module stt        # stt|tts|intent|rag|slot_filling|summary|flow|e2e
python -m e2e_evaluation_pipeline --mode ci           # P0 metrics only
```

### Docker / DB

```bash
docker-compose up -d                                  # nginx + backend + frontend + mysql
mysql -u root -p < setup_database.sql                 # tables auto-created by SQLAlchemy on startup
```

## Architecture

### Two stacked product surfaces

1. **Customer chatbot** (`voice-chatbot-revision/`) — voice + text. Streams mic audio to backend via WebSocket, plays back TTS, supports barge-in (cut TTS when user speaks).
2. **Consultant dashboard** (`agent-dashboard/`) — real-time monitor of in-progress conversations, AI-generated summary, sentiment, recommended docs once a handover is triggered.

Both call the same FastAPI backend at `app/`.

### LangGraph conversation workflow

The single most important system to understand is `ai_engine/graph/workflow.py`. It compiles a `StateGraph` with two flows multiplexed by `is_human_required_flow` in `GraphState` (`ai_engine/graph/state.py`):

- **General flow** (`is_human_required_flow=False`)
  `triage_agent` → `answer_agent` → `chat_db_storage` → END
  `triage_agent` internally calls `intent_classification_tool` (38-category KcELECTRA+LoRA) and `rag_search_tool` (Hybrid vector+BM25, then reranker). Its `triage_decision` is one of `SIMPLE_ANSWER | AUTO_ANSWER | NEED_MORE_INFO | HUMAN_REQUIRED`. `HUMAN_REQUIRED` sets `is_human_required_flow=True` for subsequent turns.

- **Handover flow** (`is_human_required_flow=True`)
  Entry router → `consent_check` (rule-based "네"/"아니오") → `waiting_agent` (extracts one missing field per turn from history) → `chat_db_storage`. When `info_collection_complete=True`, the router from `chat_db_storage` goes to `summary_agent` → `human_transfer` → END.

State is preserved across turns by re-running the compiled graph with the previous `conversation_history`; routing flags (`is_human_required_flow`, `customer_consent_received`, `out_of_domain_count`, `unclear_count`) drive the entry router (`_entry_router` in `workflow.py`).

Nodes live in `ai_engine/graph/nodes/`; their shared "tools" (intent classifier, RAG search, chat history loader) live in `ai_engine/graph/tools/`. The FastAPI side talks to the graph via `app/services/workflow_service.py`, which also runs the External Validation Layer (rejects <2 chars or >2000 chars before invoking the graph).

### RAG

`ai_engine/vector_store.py` wraps ChromaDB. Retrieval is **Hybrid Search**: vector (`jhgan/ko-sroberta-multitask`, weight 0.6) + BM25 with Kiwi Korean tokenizer (weight 0.4), fused via RRF (`rrf_k=30`), then optionally reranked with `Dongjin-kr/ko-reranker` Cross-Encoder. Tuning constants live in `app/core/config.py` (`enable_hybrid_search`, `use_rrf`, `rerank_top_k`, `rerank_final_k`, `similarity_threshold`).

L2-distance → similarity uses `1/(1+distance)` (not `1-distance`); this was a deliberate fix — preserve it if touching `vector_store.py`.

### Voice pipeline

- **STT**: VITO (Return Zero) — `app/services/voice/stt_service.py`. Requires `VITO_CLIENT_ID` / `VITO_CLIENT_SECRET`.
- **TTS**: Google Cloud TTS — `app/services/voice/tts_service_google.py` (the active one; `tts_service.py` is OpenAI fallback, currently unused). Reads `GOOGLE_TTS_API_KEY` *or* `GEM_API_KEY` *or* `GOOGLE_APPLICATION_CREDENTIALS`.
- **VAD**: pluggable under `app/services/vad/` — `silero.py` (DL-based, default), `webrtc.py`, `hybrid.py`. Plus a thin Silero wrapper at `app/services/voice/silero_vad_service.py`.
- **WebSocket streaming**: `WS /api/v1/voice/streaming/{session_id}` (`app/api/v1/voice_ws.py`) — VAD detects 2s silence → triggers STT → workflow → TTS, with barge-in support. Customer frontend hook: `voice-chatbot-revision/src/hooks/useVoiceStream.ts`.

### Config loading quirk

`app/core/config.py` deliberately ignores process env vars for `OPENAI_API_KEY` and loads only from the `.env` at repo root (`hana-linker/.env`). It temporarily pops the env var, reads `.env` via `dotenv_values`, then re-injects. If you need to change OpenAI credentials, edit `.env` — exporting in the shell will not take effect. Other settings use normal pydantic-settings env loading.

### Persistence

MySQL via SQLAlchemy (`app/core/database.py`, `app/models/`). `chat_sessions` (with `collected_info` JSON) and `chat_messages` are written by the `chat_db_storage` node every turn. Schema is auto-created on startup; `setup_database.sql` only creates the database. A background task in `main.py` flags sessions inactive after 10 minutes of no activity (`session_manager.deactivate_inactive_sessions`).

### API surface

Routers under `app/api/v1/`: `chat`, `handover`, `session`, `voice`, `voice_ws`, `kms`. Frontend ↔ backend contract: `voice-chatbot-revision/src/services/api.ts` mirrors the Pydantic schemas in `app/schemas/`; if you change a schema, update the TS types in lockstep. Note: chat endpoint is `/api/v1/chat/message` (not `/api/v1/chat`).

## LangSmith

If `LANGSMITH_TRACING=true` in `.env` with a valid `LANGSMITH_API_KEY`, LangChain tracing activates automatically (see `config.py` bottom and the startup log in `main.py`). Use it to debug prompts inside the graph.

## Things that look like bugs but aren't

- `app/core/config.py` deliberately strips `OPENAI_API_KEY` from env before reloading from `.env` — see above.
- `vector_store.py` similarity formula is `1/(1+L2)`, not `1-L2` — intentional fix.
- `chat_db_storage` always runs even on the handover branch — it stores `collected_info` mid-collection; the conditional edge to `summary_agent` only fires once `info_collection_complete=True`.
