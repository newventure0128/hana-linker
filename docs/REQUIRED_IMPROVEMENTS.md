# Linker 필수 개선사항 (Required Improvements)

> 사업기획 + 개발 관점 코드 리뷰 결과. 우선순위순. 데모 완성도는 높으나
> "금융사 실서비스" 기준으로는 보안/컴플라이언스가 출시 차단 사유이고,
> LLM 호출 구조가 비용·지연의 최대 병목.
>
> 작성: 2026-06-12 / 기준 커밋: `e7f21c1`

---

## 우선순위 한눈에

| 순위 | 항목 | 영향 | 상태 |
|---|---|---|---|
| P0-1 | API 인증 부재 + CORS 전체 개방 | 타 세션 PII 조회 가능, 출시 불가 | ☐ |
| P0-2 | PII 평문 저장·평문 국외전송 | 개보법/신용정보법 위반 | ☐ |
| P0-3 | API 키 prefix 로그 출력 + `print()` 디버그 | 키 유출 단서 | ☐ |
| P0-4 | consent_check LLM 하드코딩(로컬 전환 불가 버그) | 로컬 전환 시 동의분류만 OpenAI 호출 | ✅ (로컬 전환 작업에서 수정) |
| P1-5 | 턴당 LLM 호출 팬아웃 과다 | 콜당 변동비·지연 | ◐ (triage 1회로 축소) |
| P1-6 | triage ReAct 에이전트 불필요 | 비결정적·로컬 모델 불안정 | ✅ (결정적 파이프라인 전환) |
| P1-7 | JSON 파싱 취약(조용한 폴백) | 로컬 모델에서 빈번한 오분류 | ✅ (structured output 전환) |
| P1-8 | 음성 파이프라인 비스트리밍 | 음성 UX 수 초 지연 | ☐ |
| P1-9 | 프롬프트 비대 + 코드 오버라이드 이중구조 | 고정 토큰비, 정책변경=배포 | ◐ |
| P2 | config 환경변수 무시 / 단일프로세스 / 동기DB / 레포 위생 / 테스트 0 | 운영·확장성 | ☐ |

범례: ☐ 미착수 · ◐ 부분 · ✅ 완료

---

## P0 — 실서비스 차단 사유 (출시 전 필수)

### P0-1. API 인증 전무 + CORS 전체 개방
- 위치: `app/main.py:31-37` (`allow_origins=["*"]` + `allow_credentials=True`)
- 모든 엔드포인트가 무인증 공개. `session_id`가 클라이언트 임의 지정이라
  **타인 세션의 대화 이력·수집 PII를 누구나 조회 가능**. 상담원 대시보드도 무인증.
- 조치: 고객용은 세션 토큰(서버 발급, 서명) + 상담원용은 별도 인증(JWT/OAuth).
  CORS는 운영 도메인 화이트리스트로 제한.

### P0-2. PII 평문 저장 + 평문 국외 전송
- `waiting_agent`가 이름·생년월일·카드 뒷자리 등을 수집 →
  `chat_sessions.collected_info`에 **평문 JSON 저장**, 동시에 발화 전체가 OpenAI로 전송.
- 조치: ① 저장 시 컬럼 암호화(또는 토큰화), ② 발화 내 PII 마스킹 후 LLM 전송,
  ③ 국외이전 동의 또는 **로컬 LLM 전환**(→ 본 문서 "로컬 LLM 선정" 참고).
- 💡 이 항목이 로컬 LLM 전환의 가장 강력한 사업적 근거.

### P0-3. API 키 prefix 로그 + 디버그 print
- `triage_agent.py:45`, `answer_agent.py:41`, `summary_agent.py:39`, `app/main.py:101` 등에서
  `openai_api_key[:20]`을 INFO 로그로 출력. `triage_agent.py:336`에 `print(content)` 잔존.
- 조치: 키 일부라도 로깅 금지. 로컬 전환 작업에서 함께 제거.

### P0-4. consent_check LLM 하드코딩 (로컬 전환 차단 버그)
- 위치: `consent_check_node.py:95` → `ChatOpenAI(model="gpt-4o-mini", ...)`
- 다른 노드와 달리 provider 분기가 없어 **로컬 전환해도 동의분류만 OpenAI 호출**.
  키 없으면 예외 → 폴백이 `CONSENT`(=LLM 실패 시 고객이 동의한 것으로 간주, `:148`)라 이중 위험.
- 조치: LLM 팩토리로 통일 + 폴백 기본값을 `UNCLEAR`로 변경. (✅ 로컬 전환 작업에서 처리)

---

## P1 — 비용·지연 구조 (로컬 전환 성패를 좌우)

### P1-5. 턴당 LLM 호출 팬아웃 과다
- 일반 플로우: triage(ReAct 1~3회 + 도구) + answer(1회) = **턴당 2~4회**
- 핸드오버 완료: `workflow_service._extract_slot_info_from_conversation()`이
  **슬롯 1개당 LLM 1회 직렬 호출** (`workflow_service.py:508-534`)
- "하루 50,000콜"이 턴 기준이면 실제 LLM 요청은 10만~20만 회.
- 조치: triage를 1회로 축소(✅), 슬롯 추출은 1회 배치 추출로 통합(☐, 향후).

### P1-6. triage ReAct 에이전트 불필요 → 결정적 파이프라인
- LLM이 도구를 안 부를까봐 intent를 **사전 강제 호출**(`triage_agent.py:217-227`),
  AUTO_ANSWER인데 RAG 없으면 **사후 강제 호출**(`:424-438`). 자율성을 코드로 두 번 덮어씀.
- 조치: `BERT 분류 → RAG 검색 → 구조화 출력 1회`의 결정적 파이프라인으로 전환.
  로컬 소형 모델은 ReAct 루프에서 더 자주 흔들리므로 전환 전 필수. (✅ 완료)

### P1-7. JSON 파싱 취약
- triage 출력을 `json.loads` 직접 파싱, 실패 시 조용히 `SIMPLE_ANSWER` 폴백(`:440-443`).
  summary는 정규식 줄 파싱(`summary_agent.py:98-147`).
- 조치: structured output(Ollama `format`/json_schema, vLLM guided decoding)으로 강제 +
  실패 시 JSON 추출 폴백. (✅ `app/core/llm.py:invoke_structured`)

### P1-8. 음성 파이프라인 비스트리밍
- VAD 2초 침묵 → STT → (LLM 2~4회 완료 대기) → TTS 전체 생성 → 재생, 완전 직렬.
- 조치: 문장 단위 LLM 스트리밍 → 문장별 TTS 파이프라이닝. 로컬은 토큰 속도가 느려
  스트리밍 없이는 음성 UX 붕괴. (☐)

### P1-9. 프롬프트 비대 + 코드 오버라이드 이중구조
- triage 시스템 프롬프트 ~2,500토큰, "🚨 절대 규칙" 반복·상충 + 키워드 코드 오버라이드(`:382-418`).
  BERT 38-카테고리 분류와 LLM 분류가 역할 중복.
- 조치: 분류 정책을 외부 설정으로 분리, BERT 결과와 통합. (◐ 프롬프트 축소만 진행)

---

## P2 — 운영·유지보수

- **config 환경변수 무시**(`config.py:15-30,123-130`): `.env`만 읽고 process env 무시 →
  Docker/K8s 시크릿 주입 불가, 12-factor 위반.
- **단일 프로세스 수직확장만 가능**: ChromaDB·BM25·임베딩·리랭커가 API 프로세스 내부 동작.
  uvicorn 워커 증설 시 모델 중복 로드. → 임베딩/리랭커/벡터DB를 별도 서비스로 분리.
- **동기 SQLAlchemy + 매 턴 히스토리 재조회**: 이벤트 루프 블로킹, DB 부하. → async + 캐시.
- **레포 위생**: `voice-chatbot-revision/backend-files/`에 백엔드 구버전 사본(드리프트),
  `bert-financial-intent-classifier`/`bert_financial_intent_classifier` 중복,
  `uvicorn.log`·`ingest.log`·`chroma_db/` 바이너리 트래킹. → `.gitignore` 정리.
- **백엔드 단위테스트 0 / CI 없음**: e2e 평가 파이프라인은 우수하나 회귀 방지 장치 부재.

---

## 서비스·사업 관점

1. **외부 API 3중 의존(OpenAI + VITO STT + Google TTS)**: 콜당 변동비·가용성 리스크 3중,
   서킷브레이커/폴백 없음.
2. **핸드오버 미완성**: `human_transfer`는 플래그 설정 수준. 상담원 큐잉/배분/수용량 관리 부재 →
   핵심 가치("심리스 이관")의 이관 이후가 비어 있음.
3. **운영 KPI 계측 부재**: containment rate(AI 종결률)·핸드오버율·턴당 지연·분류 정확도가
   프로덕션에서 미수집(LangSmith는 디버깅용). → "상담원 N명분 절감" 입증 불가.
4. **FAQ 캐싱 기회 방치**: BERT confidence 높은 상위 빈출 의도는 사전승인 템플릿으로 LLM 없이 응답 →
   비용 0원화 + 답변 일관성(금융 심의 유리).
5. **가드레일 부재**: 프롬프트 인젝션(RAG 문서 경유 포함), PII 마스킹, 금칙어 검증 없음.

---

## 권장 개선 순서

1. 인증 + CORS + 키 로깅 제거 (P0-1,3)
2. LLM 팩토리 통합 (+consent 버그 수정) — ✅ 로컬 전환 작업
3. triage 결정적 파이프라인 + structured output — ✅ 로컬 전환 작업
4. 문장 단위 스트리밍 TTS (P1-8)
5. KPI 계측 (사업성 입증)
6. FAQ 캐시 (비용 절감)
7. 상담원 큐 (핸드오버 완성)
8. PII 암호화/마스킹 + 로컬 LLM (P0-2)

---

## 로컬 LLM 선정 (2026-06-12 기준)

워크로드 특성: "분류 + RAG 근거 요약 + 슬롯 추출 + 한국어 구어체 생성".
초대형 모델 불필요. 기준 = 한국어 품질 · tool/JSON 출력 안정성 · 토큰 속도.

### A. 실사용 (하루 50,000콜)
- **1순위: Qwen3.6-35B-A3B (MoE, 활성 3B) — vLLM 서빙.** Apache 2.0(상용 무제한),
  MoE 고처리량으로 동시콜 배칭 유리, 한국어 포함 200+개 언어, tool/JSON 지원.
  HW: BF16 ~71GB → H100 80GB 1장 표준. 예산형 Q4(~21GB) RTX 4090/5090 1장도 가능(부하테스트 필수).
- **단순 대안:** Qwen3.6-27B(dense), 24GB GPU 1장 Q4.
- **한국어 최우선 대안: EXAONE 4.x(32B, LG)** — KMMLU 최상위권이나 **라이선스 비상업(NC)**,
  상용은 LG와 별도 계약. 하나카드/KPMG 맥락이면 국내 기업 간 계약이 현실적 → "기술 1순위 Qwen,
  조달·규제 카드로 EXAONE" 투트랙.
- **ROI 주의:** 50k콜/일이면 API 단가가 H100 임대보다 쌀 수 있음. 로컬 전환의 진짜 ROI는
  비용이 아니라 **PII 국외이전 차단·망분리 대응(컴플라이언스)**. 자가 구축(4090×2)이면 비용도 우위.

### B. 데모 (Ollama, RTX 4060 Laptop 8GB VRAM)
- **1순위: `qwen3.5:9b` (Q4_K_M, ~6.6GB).** 8GB 티어 표준, 한국어 우수, tool/JSON format 지원.
  8GB에 빠듯 → `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `num_ctx=8192` 권장.
- **음성 데모 차선: `qwen3.5:4b` (~2.5GB).** 전부 VRAM 상주 → TTFT 짧아 실시간 음성 반응성 우수.
  분류·슬롯은 충분, RAG 답변 품질만 소폭 하락.

### 전환 체크리스트
1. `consent_check_node.py:95` 하드코딩 수정 + LLM 팩토리 통합 (✅)
2. base_url을 Ollama(`http://localhost:11434/v1`)/vLLM 엔드포인트로, `llm_provider`로 일반화 (✅)
3. triage/summary/consent 출력 JSON 스키마 강제 (✅)
4. `create_agent`(ReAct) 의존 제거 (✅)
5. e2e 평가 파이프라인으로 gpt-4o-mini 골든 베이스라인 대비 모델별 회귀 평가 (☐, 사용자 실행)

참고 출처:
- Qwen3.6: github.com/QwenLM/Qwen3.6, huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.5 9B: ollama.com/library/qwen3.5:9b
- EXAONE 4.0: github.com/LG-AI-EXAONE/EXAONE-4.0 (NC 라이선스)
- Korea AI Leaderboard 2026: benchlm.ai/leaderboards/korean-llm
