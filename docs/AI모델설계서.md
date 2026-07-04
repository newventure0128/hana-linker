# AI 모델 설계서 (AI Model Design Document)

**프로젝트:** Linker — 하나카드 AI 상담 에이전트
**대상:** 의도분류 · RAG 검색 · 생성 LLM · 음성(STT/TTS/VAD) 모델의 설계·학습·추론·평가
**문서 버전:** 1.0 (2026-07-03)
**작성 근거:** 현행 코드/모델 산출물 실측 (`hana-linker/`)

> 시스템 전반의 기능 요구사항은 `docs/REQUIREMENTS_DEFINITION.md`, 비용/용량은 `docs/COST_ESTIMATION.md`를 참조한다. 본 문서는 **AI 모델 계층**에 한정한다.

---

## 1. 개요

### 1.1 설계 목표
- 한국어 카드/금융 도메인에 특화된 **경량·온프레미스** AI 스택 구성.
- 소형 로컬 LLM의 불안정성을 **결정적 파이프라인 + 구조화 출력 + 전용 분류기**로 보완.
- 이관 미탐(HUMAN_REQUIRED FN) 최소화를 최우선 품질 목표로 함.

### 1.2 모델 인벤토리
| # | 모델 | 유형 | 역할 | 서빙 |
|---|---|---|---|---|
| 1 | KcELECTRA-base-v2022 + LoRA | 인코더 분류기 | 38-카테고리 의도분류 | 로컬(torch/PEFT) |
| 2 | jhgan/ko-sroberta-multitask | 문장 임베딩 | 벡터 검색 | 로컬(sentence-transformers) |
| 3 | Kiwi + BM25 | 통계 검색 | 키워드 보정 | 로컬 |
| 4 | Dongjin-kr/ko-reranker | Cross-Encoder | 문서 재정렬 | 로컬 |
| 5 | qwen3.5:9b | 생성 LLM | triage/answer/consent/extract/summary | Ollama(OpenAI-compat) |
| 6 | VITO (Return Zero) | STT | 음성→텍스트 | 외부 API |
| 7 | Google Neural2 (ko-KR) | TTS | 텍스트→음성 | 외부 API |
| 8 | Silero VAD | 음성활동감지 | 발화 종료 판정 | 로컬(DL) |

### 1.3 설계 원칙
1. **작업별 적정 모델**: 분류는 소형 인코더(빠르고 결정적), 생성만 LLM.
2. **결정성 우선**: LLM 자율 tool-call 대신 코드가 순서를 강제(intent→RAG→구조화 1콜).
3. **한국어 특화**: 임베딩·리랭커·분류기 모두 한국어 사전학습 모델.
4. **온프레미스 우선**: 컴플라이언스(데이터 미유출) 목적의 로컬 LLM.

---

## 2. AI 파이프라인 아키텍처

```
고객 발화(텍스트 또는 STT 결과)
   │
   ▼
① 의도분류(KcELECTRA+LoRA) ──▶ top-3 카테고리·신뢰도
   │
   ▼
② RAG 검색: 임베딩(벡터) + Kiwi·BM25 → RRF 융합 → Cross-Encoder 리랭크
   │
   ▼
③ Triage LLM(구조화 1콜) ──▶ 티켓 결정 ──(코드 오버라이드)──▶ 최종 결정
   │
   ├─ AUTO_ANSWER ▶ ④ Answer LLM(RAG 근거 기반, 2문장)
   ├─ HUMAN_REQUIRED ▶ 동의확인 LLM ▶ 슬롯추출 LLM ▶ 요약/감성 LLM
   └─ SIMPLE/NEED_MORE_INFO ▶ 규칙/LLM 응답
   │
   ▼
⑤ TTS(Google Neural2) ──▶ 음성 응답
```
공통 LLM은 항목별로 **temperature·프롬프트·출력 스키마**만 달리하여 재사용한다(`app/core/llm.py`).

---

## 3. 모델별 상세 설계

### 3.1 의도분류 모델 (KcELECTRA + LoRA)

**목적:** 발화를 9개 도메인 산하 **38개 카테고리**로 분류하여 슬롯 결정·triage 보조 근거 제공.

**구조**
- 백본: `beomi/KcELECTRA-base-v2022` (한국어 댓글·구어체 사전학습 ELECTRA)
- 헤드: `AutoModelForSequenceClassification`, `num_labels=38`, `id2label/label2id` 매핑
- 어댑터: **LoRA (PEFT)** — `r=8`, `alpha=16`, `dropout=0.1`, `target_modules=[query, value]`
  - 백본 동결 + 저랭크 어댑터만 학습 → 경량·빠른 파인튜닝, 과적합 억제

**학습 설정**
| 항목 | 값 |
|---|---|
| max_length | 128 토큰 |
| batch_size | 16 |
| learning_rate | 2e-4 |
| epochs | 5 (조기 최적 = 2~3) |
| warmup_ratio / weight_decay | 0.1 / 0.01 |
| grad clip | max_norm 1.0 |
| split | train/val 8:2 (stratified), seed 42 |

**학습 데이터:** 증강된 의도 학습셋(`data/augmented_intent_training_data*.json`) — train 8,172 / val 2,044 샘플.

**실측 성능** (`models/final_classifier_model/model_final/training_results.json`, best @ epoch 2)
| 지표 | 값 | KPI 목표 |
|---|---|---|
| Accuracy | **82.31%** | ≥ 75% |
| Macro F1 | **83.17%** | ≥ 65% |
| Weighted F1 | **85.5%** | ≥ 75% |

**산출물:** `best_model.pt`, `lora_adapter/`, `tokenizer/`, `id2intent.json`.
**추론:** `intent_classification_tool`이 백본+LoRA 어댑터를 로드하여 top-3 (intent, confidence) JSON 반환. 저신뢰(LABEL_x·미확신) 시 triage LLM이 문맥으로 보완.

### 3.2 RAG 검색 스택 (Hybrid Search + Reranking)

**설계 개요:** 벡터(의미) + BM25(키워드)를 RRF로 융합해 초기 후보를 뽑고, Cross-Encoder로 정밀 재정렬한다. 구어체 질의의 키워드 매칭 약점을 BM25가, 동의어/의미 매칭을 벡터가 보완한다.

```
query ─▶ (쿼리확장) ─┬─▶ 벡터검색(ko-sroberta)  weight 0.6 ┐
                     └─▶ BM25(Kiwi 토크나이저)   weight 0.4 ┘
                          └─▶ RRF 융합(k=30) ─▶ 상위 후보 ─▶ ko-reranker(Cross-Encoder)
                                                    └─▶ threshold 0.2 체크 ─▶ 최종 top-5
```

| 구성 | 모델/파라미터 |
|---|---|
| 임베딩 | `jhgan/ko-sroberta-multitask`, 벡터 가중 0.6 |
| 키워드 | Kiwi 형태소 토크나이저 + BM25, 가중 0.4 |
| 융합 | RRF, `rrf_k=30` (상위 순위 영향력 강화) |
| 리랭커 | `Dongjin-kr/ko-reranker` Cross-Encoder, rerank_top_k=10 → final_k=5 |
| 임계 | `similarity_threshold=0.2` (미만=저신뢰) |
| 거리→유사도 | `1/(1+L2)` (의도된 설계) |
| 벡터DB | ChromaDB(로컬, `chroma_db/`) |

> 메타 발화("상담원 연결" 등)는 RAG를 건너뛴다. 저장 문서에는 원문 스니펫·유사도가 포함되어 answer/summary 프롬프트에 근거로 주입된다.

### 3.3 생성 LLM (qwen3.5:9b)

**서빙:** Ollama의 OpenAI 호환 엔드포인트(`/v1/chat/completions`)를 `langchain_openai.ChatOpenAI`로 호출(`app/core/llm.py`). provider는 `settings.llm_provider`로 결정, 로컬 시 `base_url`·`model` 주입.

**핵심 제어**
- **Thinking 비활성화**: qwen3.5는 사고형 모델 → `reasoning_effort="none"`을 전달해 사고과정/토큰 폭증·수 분 지연을 방지(이것이 유일하게 동작하는 방식).
- **온도**: 대부분 `temperature=0.2`(결정성 위주).
- **구조화 출력**: `invoke_structured()`가 `with_structured_output(method="json_schema")`로 Pydantic 스키마를 강제 → JSON 파싱 실패 위험 제거.

**LLM 사용 에이전트**
| 에이전트 | 역할 | 출력 스키마 | 비고 |
|---|---|---|---|
| triage_agent | 4-티켓 분류 | `TriageResult{ticket, reason, customer_intent_summary}` | 구조화 1콜 + 코드 오버라이드 |
| answer_agent | RAG 근거 답변 | 자유 텍스트 | 마크다운 금지, 2문장, 구어체 |
| consent_check | 동의 판정 | `ConsentResult{classification, ai_message}` | CONSENT/REJECT/OUT_OF_DOMAIN/UNCLEAR |
| waiting_agent(extract) | 슬롯값 추출 | 자유 텍스트(값만) | 질문받은 슬롯은 focused 추출 |
| summary_agent | 요약·감성·키워드 | `SummaryResult{sentiment, summary, keywords≤5}` | 이관 리포트용 |

**Triage 결정성 보강 (설계 결정):**
소형 LLM은 "카드 분실(→AUTO) + 상담원 연결(→HUMAN)"처럼 규칙이 충돌하는 발화에서 판단이 흔들린다(실측 3/5 vs 2/5). 이를 **코드 오버라이드**로 확정한다.
1. 명시적 상담사 요청(상담사/상담원, 또는 사람·직원+연결) → **무조건 HUMAN_REQUIRED**.
2. 긴급 키워드(피싱/사기 등) → HUMAN_REQUIRED 유지 + 긴급 플래그(동의·수집 스킵).
3. 상담사 요청 없는 앱/ARS 처리가능 업무(분실·한도·재발급 등) → AUTO_ANSWER 강등.

### 3.4 음성 모델
| 모델 | 설계 |
|---|---|
| STT (VITO) | 외부 API. 실시간 스트리밍 STT, 자격증명(`VITO_CLIENT_ID/SECRET`) 필요. 금융용어 정확도가 KPI(≥95%). |
| TTS (Google Neural2 `ko-KR-Neural2-B`) | 외부 API, `speakingRate`·`pitch` 조정 가능. 답변 길이·속도가 발화시간을 좌우(§`COST_ESTIMATION.md §7`). |
| VAD (Silero) | DL 기반, 기본 VAD. WebRTC VAD와 Hybrid(mode=and) 구성 가능. 2초 무음 시 발화 종료 판정 → STT 트리거, 바지인 지원. |

---

## 4. 프롬프트 & 구조화 출력 설계

- **System/Human 분리**: 각 에이전트는 역할·규칙(system)과 컨텍스트(history·intent·docs·user)를 human 메시지로 분리 주입.
- **근거 주입**: triage/answer는 intent top-3 + RAG top-3 스니펫을 프롬프트에 포함.
- **출력 계약**: 분류/요약은 Pydantic `Literal`·필드로 스키마 강제 → 후처리 안정성 확보.
- **TTS 제약**: answer/consent/waiting 응답은 마크다운 기호 금지·구어체·간결 규칙을 프롬프트에 명시.
- **Focused 슬롯 추출**: 질문받은 슬롯은 "고객이 방금 이 항목을 답했다"는 프레이밍으로 현재 메시지에서 값만 추출 → 맨값("0433")·날짜 누락 방지.

---

## 5. 추론/서빙 설계

- **결정적 파이프라인**: triage 노드는 (1)intent 강제호출 → (2)RAG 강제호출 → (3)구조화 LLM 1콜 순서를 코드로 고정(자율 tool-call 루프 제거).
- **레이턴시 구성(음성 비스트리밍)**: STT + [rerank·검색·prefill·decode] + TTS 직렬. CPU-only 환경에서는 리랭커(수~십 초)·LLM decode가 지배적 → 데모 속도. GPU/스트리밍 TTS로 개선.
- **모델 로딩**: 임베딩·리랭커·의도분류 모델은 기동 시 로드(HF 캐시). ChromaDB·LLM 엔드포인트는 사전 인덱싱/기동 필요.
- **용량 산정**: GPU 대수는 decode 처리량이 결정(`COST_ESTIMATION.md §5`).

---

## 6. 학습 데이터 & 파인튜닝 방법론

- **데이터 증강**: 원본 FAQ/카테고리 시드를 증강(`scripts/augment_intent_training_data.py`)하여 클래스 불균형·구어체 다양성 보강 → 10,216 샘플.
- **파인튜닝**: LoRA로 KcELECTRA를 38-클래스에 적응(§3.1). 백본 동결로 소량 데이터·CPU/단일GPU에서도 학습 가능.
- **평가 분할**: stratified 8:2, macro/weighted F1로 소수 클래스 성능까지 확인.
- **재현성**: seed 42 고정, `training_results.json`에 config·지표 스냅샷 저장.

---

## 7. 평가 설계 (Evaluation)

**E2E 평가 파이프라인** (`e2e_evaluation_pipeline/`): 모듈별(stt/tts/intent/rag/slot_filling/summary/flow/e2e) 메트릭을 산출하고 산업 벤치마크 대비 등급(WORLD_CLASS~CRITICAL)과 P0/P1/P2 우선순위로 리포트(JSON+HTML).

| 모듈 | 핵심 지표(목표) |
|---|---|
| 의도분류 | Accuracy(≥75), Macro/Weighted F1, **HUMAN_REQUIRED Recall(≥90)**, Top-3(≥90) |
| Triage | Overall(≥85), **HUMAN_REQUIRED FN Rate(≤5)** |
| RAG | Precision@3(≥85), Recall@20(≥95), MRR(≥0.7), NDCG@3(≥0.85), Rerank 기여(≥20) |
| STT | CER(≤10)/WER(≤15), 금융용어(≥95), TTFB(≤300ms) |

> 최우선 관리 지표는 **HUMAN_REQUIRED 미탐(FN)** — 이관돼야 할 상담이 자동응답으로 새는 것을 방지(§3.3 결정적 오버라이드가 이를 보강).

---

## 8. 모델 선택 근거 (Design Rationale)

| 결정 | 근거 |
|---|---|
| 분류에 LLM 대신 KcELECTRA+LoRA | 38-클래스 분류는 인코더가 더 빠르고 결정적·저비용(97.5% 달성). LLM은 생성에만 사용. 많은 파라미터에 오랜 학습 시간이 소요되는 LLM 파인튜닝보다는 경량 모델을 모듈처럼 새로 출시되는 오픈소스 LLM에 부착하는 것이 낫다고 판단. |
| 한국어 특화 임베딩·리랭커 | 카드 구어체 질의 매칭 정확도. 벡터+BM25 하이브리드로 의미·키워드 동시 커버. |
| Cross-Encoder 리랭킹 | 초기 후보의 정밀 재정렬로 Precision@3·NDCG 향상(느리지만 top_k 제한). |
| 로컬 LLM(qwen3.5:9b) | 컴플라이언스(데이터 미유출). 로컬 전환은 비용절감이 아닌 규제 대응 목적. |
| 결정적 triage 파이프라인 | 소형 LLM의 tool-call·판단 흔들림을 코드 순서 강제 + 구조화 출력 + 키워드 오버라이드로 제거. |

---

## 9. 위험·한계 및 개선 방향

| 항목 | 현황/위험 | 개선 |
|---|---|---|
| LLM 판단 편차 | 규칙 충돌 발화에서 흔들림 | 코드 오버라이드로 결정성 확보(적용됨). 프롬프트 충돌 문구 정리. |
| 슬롯 추출 취약성 | 히스토리 기반 추출이 맨값 누락 | 질문 슬롯 focused 추출(적용됨). |
| CPU 추론 지연 | 리랭커·LLM decode 지배 | GPU 서빙, 문장단위 스트리밍 TTS, 리랭커 캐싱/축소. |
| STT/TTS 실측 부재 | 지연·정확도 추정치 | VITO·Google 실측으로 SLO 확정(`COST_ESTIMATION.md`). |
| 의도분류 도메인 드리프트 | 신규 상품/표현 | 프로덕션 오분류 수집 → 주기적 LoRA 재학습. |

---

*본 문서는 현행 모델 구성 기준이며, 모델·하이퍼파라미터·프롬프트 변경 시 해당 절을 갱신한다.*
