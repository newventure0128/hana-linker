# 하이브리드 에이전트 설계 + 비용·지연 절감 실현 방안

> 고정 경로(triage→RAG→answer)는 **결정적**으로 유지하고, "진짜 분기가 필요한
> 트랜잭션 작업"만 **선택적 agentic tool-use**로 처리하는 하이브리드 설계와,
> 매 턴 비싼 작업(RAG/rerank/LLM)을 피하는 싸구려 필터 등 보완책.
> 대상 코드: `ai_engine/graph/workflow.py`, `ai_engine/graph/nodes/`,
> `ai_engine/graph/tools/`, `app/core/llm.py`, `app/schemas/common.py`
> 관련: `LOCAL_LLM_TRANSITION_DIAGNOSIS.md`, `REQUIRED_IMPROVEMENTS.md`

---

## 0. 설계 원칙

1. **라우팅은 결정적, 비결정성은 격리한다.** "FAQ냐 / 트랜잭션이냐 / 단답이냐"의 분기는
   LLM이 아니라 룰+BERT로 결정. agentic(가변 tool 루프)은 **action 노드 내부에만** 가둔다.
2. **매 턴 무조건 비싼 일을 하지 않는다.** trivial 턴은 RAG/rerank/LLM을 건너뛴다.
3. **점진 도입.** 백엔드 트랜잭션 API가 붙기 전까지는 결정적 경로만으로 동작. action 노드는
   필요 기능이 생길 때만 활성화.

---

## 1. 현재 구조 (baseline)

```
entry_router ─▶ triage_agent ─▶ answer_agent ─▶ chat_db_storage ─▶ END
                 │  (BERT intent → RAG(벡터+BM25+rerank) → 구조화 LLM 1콜)
                 └─ triage_decision ∈ {SIMPLE_ANSWER, AUTO_ANSWER, NEED_MORE_INFO, HUMAN_REQUIRED}
```

문제(진단서 참조): ① RAG+rerank가 **모든 일반 턴에 무조건** 실행(단답에도) ② intent·RAG **직렬**
③ 트랜잭션을 "실제로 처리"하는 경로 없음(안내만) ④ 사기 미이관·수치 환각.

---

## 2. 하이브리드 아키텍처

### 2.1 결정적 Fast Router (신규, LLM 없음)

`triage_agent` **앞단**에 룰/BERT 기반 라우터를 둔다. LLM 호출 0회.

```
entry ─▶ fast_router ─┬─ SIMPLE      ─▶ simple_answer (템플릿 또는 1콜, RAG 스킵)
                      ├─ FAQ_CACHE   ─▶ cache_answer  (사전승인 템플릿, LLM 0콜)
                      ├─ ACTION      ─▶ action_agent  (선택적 agentic)
                      └─ RAG         ─▶ triage_agent  (기존 결정적 경로)
```

분기 신호(전부 결정적):

| 분기 | 트리거 | 처리 |
|---|---|---|
| SIMPLE | 길이 ≤ N, 단답/감사/잡음 패턴("네","감사","…"), STT 오류 | RAG·rerank 스킵 |
| FAQ_CACHE | BERT intent confidence ≥ τ(예 0.9) AND intent ∈ 캐시 화이트리스트 | 템플릿 응답, LLM 0콜 |
| ACTION | intent ∈ **트랜잭션 의도 집합** (아래) AND 백엔드 연동 활성 | action_agent |
| RAG | 그 외 | 기존 triage→RAG→answer |

**트랜잭션 의도 집합 식별(38-카테고리 기반, 결정적 휴리스틱):** 라벨 접미사로 구분.
- `안내/조회` → 정보성 → **RAG 경로**
- `접수/처리/신청/취소/등록/변경/실행/출금` → 행위성 → **ACTION 후보**
  (예: "한도상향 접수/처리", "선결제/즉시출금", "가상계좌 예약/취소", "포인트/마일리지 전환등록")

> 백엔드 트랜잭션 API가 없으면 ACTION 집합을 비워두면 됨 → 전부 RAG 경로로 안전 폴백.

### 2.2 Action 노드 (선택적 agentic, 가드레일 필수)

"잔액조회 + 거래내역 + 분쟁접수"처럼 **여러 실제 도구를 동적으로 조합**해야 하는 의도만 담당.
여기**서만** tool-calling 루프(비결정적)를 허용하되, 강한 울타리를 친다.

구현 옵션: LangGraph `ToolNode` + 조건부 엣지(권장, 제어 명시적) 또는 `create_react_agent`.

```python
# ai_engine/graph/nodes/action_agent.py (스케치)
from langchain_core.tools import tool
from app.core.llm import get_chat_llm

@tool
def get_balance(card_last4: str) -> dict: ...      # 실제 백엔드 호출
@tool
def get_transactions(card_last4: str, days: int) -> list: ...
@tool
def file_dispute(tx_id: str, reason: str) -> dict: ...

ACTION_TOOLS = [get_balance, get_transactions, file_dispute]   # allowlist

def action_agent_node(state):
    llm = get_chat_llm(temperature=0.0).bind_tools(ACTION_TOOLS)
    # bounded loop: 아래 가드레일로 감싼다
    ...
```

**가드레일(모두 필수):**
- **tool allowlist** — 등록된 안전 도구만. 임의 코드/SQL/외부호출 금지.
- **호출 상한** — `max_tool_calls ≤ 4`, `max_steps ≤ 6`, 초과 시 강제 종료 → HUMAN_REQUIRED 폴백.
- **per-tool 타임아웃** + 실패 시 재시도 1회 후 폴백.
- **structured 최종 출력**(json_schema)로 답변/다음행동 강제.
- **PII·권한 검증** — 도구 실행 전 본인확인 상태 확인(슬롯), 미충족 시 NEED_MORE_INFO.
- **전수 감사 로깅** — 모든 tool 호출·인자·결과를 기록(규제 대응).
- **금전 변경 도구는 확인 게이트** — 분쟁접수/출금 등은 사용자 명시 확인 후 실행.

→ 이 노드만 비결정적이고, **진입 자체는 결정적**(fast_router)이므로 시스템 전체 SLO·테스트·감사는 유지된다.

### 2.3 LangGraph 통합 (구체)

- `GraphState`(`ai_engine/graph/state.py`)에 `route: str` 필드 추가.
- `TriageDecisionType`(`app/schemas/common.py`)에 **`ACTION_REQUIRED`** 추가(또는 route 필드로 분리).
- 신규 노드: `fast_router_node`(결정적), `cache_answer_node`(템플릿), `action_agent_node`(agentic).
- `workflow.py`:
  ```python
  graph.set_conditional_entry_point(_fast_router, {
      "simple_answer": "answer_agent",      # RAG 스킵 플래그 set
      "faq_cache":     "cache_answer",
      "action":        "action_agent",
      "rag":           "triage_agent",      # 기존 경로
      # 핸드오버 진행 중이면 기존 consent/waiting 분기 유지
  })
  graph.add_edge("action_agent", "chat_db_storage")
  graph.add_edge("cache_answer", "chat_db_storage")
  ```
- 기존 `triage_agent → answer_agent → chat_db_storage` 경로는 **그대로 유지**.

---

## 3. 비용·지연 절감 보완책 (싸구려 필터 등)

| # | 보완책 | 효과 | 하드웨어 | 위치 |
|---|---|---|---|---|
| F-1 | **Fast router로 trivial 턴 RAG 스킵** | 단답/감사 턴의 rerank(8초) 회피 | 불필요 | `fast_router_node` |
| F-2 | **FAQ 캐시** (고신뢰 빈출 의도 → 사전승인 템플릿) | 해당 턴 **LLM 0콜·비용 0**, 답변 일관성(금융심의 유리) | 불필요 | `cache_answer_node` |
| F-3 | **조건부 reranking 스킵** (하이브리드 최고점 ≥ θ면 rerank 생략) | rerank 7~8초 절약(고신뢰 건) | 불필요 | `vector_store.py` |
| F-4 | **passage truncation** (rerank 입력을 ~256토큰으로) | CPU에서도 rerank 대폭 단축(sweet spot) | 불필요 | `vector_store._rerank_documents` |
| F-5 | **intent ∥ RAG 병렬화** | 둘 중 짧은 쪽만큼 절약 | 불필요 | `triage_agent_node` |
| F-6 | **사기/부정결제 무조건 이관 하드게이트** (LLM 전·후) | card_004/079 미이관 차단(안전) | 불필요 | `triage_agent.py:187` 재배치 |
| F-7 | **수치 일관성 가드** (답변 숫자 ↔ 문서 숫자 대조, 왜곡 시 차단/재생성) | card_074류 환각 차단 | 불필요 | `answer_agent` 후처리 |
| F-8 | **reranker/임베더 GPU + 서비스 분리** | torch 스택 8.8초→0.7초(실측 12.5배) | GPU 필요 | 인프라 |
| F-9 | **vLLM guided decoding + 모델 풀 GPU 상주** | JSON 견고화 + 오프로드 페널티 소멸 | GPU 필요 | 인프라 |

**F-1 Fast router 의사코드:**
```python
def _fast_router(state) -> str:
    if state.get("is_human_required_flow"):      # 핸드오버 중이면 기존 분기
        return "consent_or_waiting"
    msg = (state["user_message"] or "").strip()
    if len(msg) <= 6 or _is_smalltalk_or_noise(msg):   # 룰
        state["skip_rag"] = True
        return "simple_answer"
    intent, conf = _bert_top1(msg)                # 17ms(GPU)/72ms(CPU)
    if conf >= 0.9 and intent in FAQ_CACHE_WHITELIST:
        return "faq_cache"
    if intent in ACTION_INTENTS and BACKEND_ENABLED:
        return "action"
    return "rag"
```

**F-6 사기 하드게이트(진단서 C-1 근본수정):** 현재 긴급 키워드 체크가
`if triage_decision == HUMAN_REQUIRED:` 안에 중첩되어 **LLM 판단에 종속**된다. 이를
**LLM 판단과 무관한 선/후처리**로 분리하고, 키워드뿐 아니라 BERT 부정사용 의도·패턴
("모르는 결제","쓰지 않은 결제")까지 포함해 무조건 `HUMAN_REQUIRED`로 강제.

---

## 4. 도입 로드맵 (우선순위)

**Phase 1 — 코드만, 무하드웨어 (즉시)**
- F-6 사기 하드게이트, F-7 수치 가드 (안전/정확성, 출시 차단 해소)
- F-1 fast_router + F-2 FAQ 캐시 + F-3/F-4 reranking 최적화 (지연·비용)
- F-5 intent∥RAG 병렬화

**Phase 2 — 트랜잭션 기능 도입 시**
- 2.2 action_agent + 가드레일, ACTION_INTENTS 활성화, 백엔드 도구 연동

**Phase 3 — 프로덕션 인프라**
- F-8 reranker/임베더 GPU·서비스 분리, F-9 vLLM + Qwen3.6-35B-A3B 풀 상주

---

## 5. 리스크 / 가드

- **action 노드 비결정성** → 호출 상한·타임아웃·allowlist·감사·HUMAN 폴백으로 통제. SLO는 노드별로 분리 측정.
- **FAQ 캐시 오답** → confidence 임계 + 캐시 화이트리스트 정기 검수 + 캐시 미스 시 RAG 폴백.
- **라우팅 오분류** → 기본값은 항상 보수적인 **RAG 경로**(애매하면 비싸더라도 안전하게).
- **트랜잭션 권한/PII** → 도구 실행 전 본인확인 슬롯 충족 필수, 금전 변경은 명시 확인 게이트.

---

## 6. 측정 / 검증

- `generate_review.py`로 **라우팅 정확도·에스컬레이션 정확도·groundedness** 회귀 평가(기존 도구 재사용).
- 신규 KPI: fast_router 분기 정확도, FAQ 캐시 적중률·오답률, action 노드 tool-call 성공률·평균 step 수·폴백률·평균 지연.
- Phase별로 골든셋(card_100_scenarios + 트랜잭션 시나리오 신설)으로 회귀.

---

## 7. 한 줄 요약

**라우팅은 결정적으로(싸구려 게이트), 실행만 필요한 곳에서 agentic으로.** 고정 FAQ 경로는
1콜 결정적 파이프라인을 유지하고, 트랜잭션 분기에만 가드레일 두른 tool-use를 더한다.
보완 필터(F-1~F-7)는 하드웨어 없이 지연·비용·안전을 동시에 개선하고, GPU/서빙(F-8/F-9)은
프로덕션에서 마무리한다.
