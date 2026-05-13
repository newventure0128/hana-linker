# Linker - AI 상담 서비스 플랫폼

> 심리스한 상담 경험을 제공하는 AI 상담 에이전트

---

## 📋 프로젝트 개요

- **프로젝트명**: Linker - End-to-End AI 상담 에이전트
- **목표**: 상담 맥락 단절을 해결하는 심리스한 AI 상담 서비스 구축
- **기간**: 2026년 4월 ~ 7월
---

## 🏗️ 시스템 아키텍처

<img width="1830" height="848" alt="최종 발표 PPT (1)" src="https://github.com/user-attachments/assets/966d7d8c-3ff0-4e0a-afa4-31276aafaf7c" />
<br/>
<br/>
<br/>
<img width="1830" height="916" alt="최종 발표 PPT (2)" src="https://github.com/user-attachments/assets/60979c4f-480c-485d-b814-bdd8dc1901bb" />


---

## ✨ 주요 기능

### 1. LangGraph 기반 대화 워크플로우
- **의도 분류 (Triage Agent)**: 38개 카테고리 분류 (KcELECTRA LoRA 파인튜닝)
- **RAG 기반 답변 생성**: Hybrid Search (벡터 + BM25) + Reranking
- **상담원 핸드오버**: 고객 동의 → 정보 수집 → 대화 요약 → 이관

### 2. 음성 인터페이스
- **STT (Speech-to-Text)**: VITO STT (Return Zero)
- **TTS (Text-to-Speech)**: Google Cloud TTS
- **VAD (Voice Activity Detection)**: Silero VAD (딥러닝 기반)
- **실시간 스트리밍**: WebSocket 기반 양방향 통신
- **Barge-in 지원**: TTS 재생 중 음성 입력 시 자동 중단

### 3. 상담원 대시보드
- 실시간 대화 모니터링
- AI 생성 요약 및 핵심 키워드
- 고객 감정 분석
- 추천 문서/FAQ 표시

### 4. E2E 평가 파이프라인
- STT/Intent/RAG/Slot Filling/Summary/Flow 메트릭
- HTML/JSON 리포트 자동 생성

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (`.env` 파일)

```env
# OpenAI API (필수)
OPENAI_API_KEY=sk-...

# VITO STT (음성 입력) - https://developers.vito.ai/
VITO_CLIENT_ID=...
VITO_CLIENT_SECRET=...

# Google TTS (음성 출력) - https://ai.google.dev/
GOOGLE_TTS_API_KEY=...
# 또는 서비스 계정 사용
GOOGLE_APPLICATION_CREDENTIALS=./gen-lang-client-xxxx.json

# 데이터베이스 (선택)
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/aicc_db?charset=utf8mb4

# LangSmith 추적 (디버깅용, 선택)
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=linker-aicc
LANGSMITH_TRACING=true
```

### 3. 벡터 DB 초기화

```bash
# ChromaDB에 문서 인덱싱
python scripts/ingest_to_chromadb.py
```

### 4. 서버 실행

```bash
# 백엔드 서버 (FastAPI)
uvicorn app.main:app --reload --port 8000

# 프론트엔드 (음성 챗봇)
cd voice-chatbot-revision
npm install
npm run dev

# 상담원 대시보드
cd agent-dashboard
npm install
npm start
```

---

## 📁 프로젝트 구조

```
fa06-fin-aicc/
├── app/                          # FastAPI 백엔드
│   ├── api/v1/                   # REST/WebSocket 엔드포인트
│   │   ├── chat.py               # 채팅 API
│   │   ├── handover.py           # 핸드오버 API
│   │   ├── session.py            # 세션 관리 API
│   │   ├── voice.py              # 음성 REST API
│   │   └── voice_ws.py           # 음성 WebSocket API
│   ├── core/                     # 설정 및 DB
│   │   ├── config.py             # 환경 설정
│   │   └── database.py           # DB 연결
│   ├── models/                   # SQLAlchemy 모델
│   ├── schemas/                  # Pydantic 스키마
│   └── services/                 # 비즈니스 로직
│       ├── vad/                  # VAD 서비스 (Silero, WebRTC, Hybrid)
│       ├── voice/                # STT/TTS 서비스
│       └── workflow_service.py   # LangGraph 워크플로우 래퍼
│
├── ai_engine/                    # AI 엔진
│   ├── graph/                    # LangGraph 워크플로우
│   │   ├── nodes/                # 노드 정의 (7개)
│   │   ├── tools/                # 도구 (의도분류, RAG 등)
│   │   ├── state.py              # 상태 정의
│   │   └── workflow.py           # 워크플로우 빌더
│   ├── ingestion/                # 문서 처리
│   │   └── bert_financial_intent_classifier/  # 의도 분류 모델
│   ├── prompts/                  # 프롬프트 템플릿
│   └── vector_store.py           # ChromaDB 래퍼
│
├── voice-chatbot-revision/       # 고객용 음성 챗봇 (React + Vite)
├── agent-dashboard/              # 상담원 대시보드 (React)
├── e2e_evaluation_pipeline/      # E2E 평가 파이프라인
├── models/                       # 학습된 모델 (의도 분류)
├── data/                         # 학습 데이터
├── scripts/                      # 유틸리티 스크립트
├── chroma_db/                    # 벡터 DB 저장소
└── docs/                         # 문서
```

---

## 🔌 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/chat/message` | 채팅 메시지 처리 |
| POST | `/api/v1/handover/analyze` | 핸드오버 분석 |
| POST | `/api/v1/handover/request` | 핸드오버 요청 |
| GET | `/api/v1/sessions` | 세션 목록 조회 |
| POST | `/api/v1/voice/stt` | 음성→텍스트 변환 |
| POST | `/api/v1/voice/tts` | 텍스트→음성 변환 |
| WS | `/api/v1/voice/streaming/{session_id}` | 실시간 음성 스트리밍 |
| GET | `/api/v1/voice/vad/status` | VAD 서비스 상태 확인 |
| GET | `/health` | 헬스체크 |

📄 **Swagger 문서**: `http://localhost:8000/docs`

---

## 🤖 AI 모델

### 의도 분류 모델
- **모델**: KcELECTRA + LoRA 파인튜닝
- **카테고리**: 38개 (하나카드 FAQ 기반)
- **경로**: `models/Final_Hana_Card_Classifier/`

### RAG 설정
- **임베딩**: `jhgan/ko-sroberta-multitask` (한국어 특화)
- **Reranker**: `Dongjin-kr/ko-reranker` (한국어 Cross-Encoder)
- **검색**: Hybrid Search (벡터 0.6 + BM25 0.4) + RRF Fusion

---

## 📊 평가 파이프라인

```bash
# E2E 평가 실행
python -m e2e_evaluation_pipeline --dataset datasets/test_set.json

# 결과 확인
# reports/ 폴더에 HTML/JSON 리포트 생성
```

---

## 🛠️ 개발 도구

### LangSmith 추적
`.env`에서 `LANGSMITH_TRACING=true` 설정 시 LangSmith에서 LLM 호출 추적 가능

### Docker 배포
```bash
docker-compose up -d
```

---

## 📚 참고 문서

- [설정 가이드](docs/setup/)
- [WebSocket API 문서](docs/websocket/)
- [트러블슈팅](docs/troubleshooting/)
- [분석 보고서](docs/analysis/)
- [대시보드 문서](docs/dashboard/)

---

## 📝 변경 이력

자세한 변경 사항은 [CHANGELOG.md](CHANGELOG.md)를 참조하세요.

---

## 📝 라이선스

