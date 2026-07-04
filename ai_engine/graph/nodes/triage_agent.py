# ai_engine/graph/nodes/triage_agent.py

"""Triage 노드 (결정적 파이프라인 + 구조화 출력).

과거 구조(ReAct create_agent)는 LLM이 도구를 호출할지 비결정적이라
intent를 사전 강제 호출하고 RAG를 사후 강제 호출하는 등 자율성을 코드로 두 번
덮어쓰고 있었다. 로컬 소형 모델에서는 tool-call 루프가 더 자주 흔들리므로
아래의 결정적 파이프라인으로 전환했다:

    1) intent_classification_tool 호출 (BERT 38-카테고리)
    2) rag_search_tool 호출 (Hybrid + Rerank)
    3) 단일 구조화 LLM 호출로 티켓 결정 (TriageResult)

티켓 결정 후 기존 키워드 코드 오버라이드(긴급/앱-ARS 처리 가능 업무)는 유지한다.
"""

from __future__ import annotations
import json
import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from ai_engine.graph.state import GraphState
from app.core.llm import invoke_structured, provider_label
from app.schemas.llm_outputs import TriageResult
from ai_engine.graph.tools import intent_classification_tool, rag_search_tool
from ai_engine.graph.tools.rag_search_tool import parse_rag_result
from app.schemas.common import IntentType, TriageDecisionType

logger = logging.getLogger(__name__)
logger.info(f"triage_agent LLM provider: {provider_label()}")


SYSTEM_PROMPT = """당신은 카드/금융 콜센터용 triage 분류기입니다.
고객의 최신 발화와 대화 맥락, 그리고 함께 제공되는 의도 분류 결과·문서 검색 결과를 보고
아래 네 가지 티켓 중 하나로 분류하세요. (고객에게 직접 답변하지 않습니다.)

- SIMPLE_ANSWER : 단순 반응/동의/감사/잡음/STT오류 (예: "네", "감사합니다", "...")
- AUTO_ANSWER   : 문서(FAQ/약관) 기반으로 자동 답변 가능한 질문
- NEED_MORE_INFO: 답변하려면 고객의 추가 정보가 필요한 질문
- HUMAN_REQUIRED: 상담사만 처리 가능한 업무 / 명시적 상담사 요청 / 반복 미해결 불만

[HUMAN_REQUIRED 인 경우]
- 보이스피싱·금융사기 신고, 해외 부정결제 신고, 결제 취소/환불 처리, 이의제기/분쟁,
  명시적 상담사 연결 요청, 반복 안내에도 미해결된 강한 불만.

[AUTO_ANSWER 로 처리 — 앱/ARS에서 고객이 직접 처리 가능]
- 카드 분실/도난/정지/해제/재발급 안내, 한도 변경/조회/상향 안내,
  결제일 변경/조회 안내, 개인정보 변경 안내, 이용내역·포인트 조회 안내.
- "정지해주세요", "재발급 받고 싶어요"처럼 요청형이어도 "방법 안내"로 충분하면 AUTO_ANSWER.
- 단, 위 업무라도 고객이 '상담사/상담원 연결'을 명시적으로 요청하면 HUMAN_REQUIRED 로 분류하세요.

[판단 보조]
- 의도 분류 결과(있으면)와 문서 검색 결과(있으면)를 근거로 판단하세요.
- 문서 검색 결과가 충분하면 AUTO_ANSWER, 부족하면 NEED_MORE_INFO 또는 HUMAN_REQUIRED.

반드시 ticket, reason, customer_intent_summary 세 필드를 채워 JSON으로만 응답하세요.
customer_intent_summary 는 모든 티켓에서 항상 1-2문장으로 작성합니다."""


def _format_history(conversation_history: List[Dict[str, Any]], limit: int = 10) -> str:
    """대화 히스토리를 프롬프트용 문자열로 (최근 limit개)."""
    recent = conversation_history[-limit:] if len(conversation_history) > limit else conversation_history
    lines: List[str] = []
    for msg in recent:
        role = msg.get("role")
        text = msg.get("message", "")
        if role == "user":
            lines.append(f"고객: {text}")
        elif role == "assistant":
            lines.append(f"상담봇: {text}")
    return "\n".join(lines)


def _format_intent(intent_top3_json: str | None) -> str:
    """intent 분류 JSON 문자열을 사람이 읽을 수 있는 요약으로."""
    if not intent_top3_json:
        return "(의도 분류 결과 없음)"
    try:
        items = json.loads(intent_top3_json)
        return ", ".join(
            f"{it.get('intent')}({it.get('confidence', 0):.2f})" for it in items[:3]
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        return "(의도 분류 파싱 실패)"


def _format_docs(retrieved_docs: List[Dict[str, Any]], limit: int = 3) -> str:
    """RAG 검색 결과를 프롬프트용 요약으로 (상위 limit개)."""
    if not retrieved_docs:
        return "(검색된 문서 없음)"
    parts: List[str] = []
    for i, doc in enumerate(retrieved_docs[:limit]):
        content = doc.get("content", "")
        snippet = content[:300]
        score = doc.get("rerank_score", doc.get("score", 0)) or 0
        parts.append(f"[문서 {i+1}] (유사도 {score:.2f}) {snippet}")
    return "\n".join(parts)


def triage_agent_node(state: GraphState) -> GraphState:
    """결정적 파이프라인으로 triage_decision을 산출한다.

    intent 분류 → RAG 검색 → 단일 구조화 LLM 호출 → (키워드 오버라이드)
    """
    user_message = state["user_message"]

    try:
        # 1) 의도 분류 (BERT 38-카테고리)
        intent_top3_json: str | None = None
        try:
            result = intent_classification_tool.invoke(user_message)
            if isinstance(result, str):
                intent_top3_json = result
        except Exception as e:
            logger.warning(f"의도 분류 실패 (계속 진행): {e}")

        # 2) RAG 검색 (Hybrid + Rerank)
        retrieved_docs: List[Dict[str, Any]] = []
        try:
            rag_result = rag_search_tool.invoke({"query": user_message})
            if isinstance(rag_result, str):
                retrieved_docs = parse_rag_result(rag_result)
        except Exception as e:
            logger.warning(f"RAG 검색 실패 (빈 결과로 진행): {e}")

        # intent 결과를 state에 저장
        if intent_top3_json:
            try:
                intent_list = json.loads(intent_top3_json)
                state["intent_classifications"] = intent_list
                if intent_list:
                    state["context_intent"] = intent_list[0].get("intent", "")
                    state["intent_confidence"] = intent_list[0].get("confidence", 0.0)
                else:
                    state["context_intent"] = None
                    state["intent_confidence"] = 0.0
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.error(f"Intent 결과 파싱 실패: {e}", exc_info=True)
                state["context_intent"] = None
                state["intent_confidence"] = 0.0
                state["intent_classifications"] = None
        else:
            state["context_intent"] = None
            state["intent_confidence"] = 0.0
            state["intent_classifications"] = None

        # RAG 결과를 state에 저장
        state["retrieved_documents"] = retrieved_docs
        if retrieved_docs:
            state["rag_best_score"] = max(
                doc.get("rerank_score", doc.get("score", 0)) for doc in retrieved_docs
            )
            state["rag_low_confidence"] = state["rag_best_score"] < 0.2
        else:
            state["rag_best_score"] = None
            state["rag_low_confidence"] = True

        # 3) 구조화 LLM 호출로 티켓 결정
        history_text = _format_history(state.get("conversation_history", []))
        history_section = f"\n[이전 대화 내용]\n{history_text}\n" if history_text else ""

        human = HumanMessage(content=f"""다음 정보를 보고 티켓을 분류하세요.
{history_section}
[현재 고객 발화]
{user_message}

[의도 분류 결과 (top3)]
{_format_intent(intent_top3_json)}

[문서 검색 결과 (top3)]
{_format_docs(retrieved_docs)}""")

        parsed: TriageResult = invoke_structured(
            TriageResult,
            [SystemMessage(content=SYSTEM_PROMPT), human],
            temperature=0.2,
        )

        ticket_str = parsed.ticket
        reason = parsed.reason
        customer_intent_summary = parsed.customer_intent_summary

        # 4) TriageDecisionType 변환 + 키워드 오버라이드
        try:
            triage_decision = TriageDecisionType(ticket_str)
            user_message_lower = user_message.lower() if user_message else ""

            # 명시적 상담사 연결 요청 감지 → LLM이 무엇으로 판단했든 HUMAN_REQUIRED 로 확정.
            # 주제가 분실/한도 등 AUTO 대상이면 프롬프트가 충돌(분실=AUTO vs 상담원요청=HUMAN)해
            # temp 0.2에서도 결정이 흔들린다(실측 3/5 vs 2/5). 코드로 결정성을 보장한다.
            explicit_agent_keywords = ["상담사", "상담원", "상담 직원", "상담직원"]
            is_explicit_agent_request = any(kw in user_message_lower for kw in explicit_agent_keywords)
            if not is_explicit_agent_request and any(k in user_message_lower for k in ["사람", "직원"]):
                is_explicit_agent_request = any(
                    v in user_message_lower for v in ["연결", "바꿔", "바꾸", "돌려", "넘겨"]
                )

            if is_explicit_agent_request and triage_decision != TriageDecisionType.HUMAN_REQUIRED:
                logger.info(
                    f"오버라이드: 명시적 상담사 요청 → HUMAN_REQUIRED (기존 {triage_decision.value}) "
                    f"- 세션={state.get('session_id', 'unknown')}"
                )
                triage_decision = TriageDecisionType.HUMAN_REQUIRED

            if triage_decision == TriageDecisionType.HUMAN_REQUIRED:
                # 긴급(보이스피싱/사기 등): HUMAN_REQUIRED 유지 + 긴급 플래그
                urgent_keywords = [
                    "보이스피싱", "보이스 피싱", "피싱", "사기", "금융사기", "금융 사기",
                    "해킹", "불법", "도용", "명의도용", "명의 도용",
                    "협박", "위협", "급하", "긴급", "응급",
                ]
                is_urgent_case = any(kw in user_message_lower for kw in urgent_keywords)

                if is_urgent_case:
                    logger.info(f"긴급 상황 감지: HUMAN_REQUIRED 유지 - 세션={state.get('session_id', 'unknown')}")
                    state["is_urgent_handover"] = True
                    reason = f"[긴급] {reason} → 보이스피싱/사기 의심으로 즉시 상담사 연결"
                elif not is_explicit_agent_request:
                    # 명시적 상담사 요청이 아니면서 앱/ARS에서 직접 처리 가능한 업무 → AUTO_ANSWER 로 강등
                    auto_answer_keywords = [
                        "분실", "도난", "잃어버", "정지", "해제",
                        "한도", "한도 변경", "한도 조회", "한도 상향",
                        "결제일", "결제일 변경", "결제일 조회",
                        "재발급",
                    ]
                    is_auto_answer_case = any(kw in user_message_lower for kw in auto_answer_keywords)

                    if is_auto_answer_case:
                        logger.info(f"오버라이드: HUMAN_REQUIRED → AUTO_ANSWER - 세션={state.get('session_id', 'unknown')}")
                        triage_decision = TriageDecisionType.AUTO_ANSWER
                        reason = f"[오버라이드] {reason} → 앱/ARS 직접 처리 가능 업무"

            state["triage_decision"] = triage_decision
            logger.info(f"Triage 결정: {triage_decision.value} - 세션={state.get('session_id', 'unknown')}")

        except ValueError:
            logger.warning("알 수 없는 ticket 타입: %s → SIMPLE_ANSWER 폴백", ticket_str)
            state["triage_decision"] = TriageDecisionType.SIMPLE_ANSWER

        state["handover_reason"] = reason
        state["customer_intent_summary"] = customer_intent_summary or None
        state["intent"] = IntentType.INFO_REQ

    except Exception as e:
        error_msg = str(e)
        logger.error(f"판단 에이전트 오류 - 세션: {state.get('session_id', 'unknown')}, 오류: {error_msg}", exc_info=True)
        state["triage_decision"] = TriageDecisionType.SIMPLE_ANSWER
        state["handover_reason"] = None
        state["customer_intent_summary"] = None
        state["intent"] = IntentType.INFO_REQ
        state["context_intent"] = None
        state["intent_confidence"] = 0.0
        state["intent_classifications"] = None
        state["retrieved_documents"] = []
        state["rag_best_score"] = None
        state["rag_low_confidence"] = True
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["decision_error"] = error_msg

    return state
