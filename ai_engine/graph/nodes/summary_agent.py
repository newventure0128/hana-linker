# ai_engine/graph/nodes/summary_agent.py
# 요약 에이전트

from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from ai_engine.graph.state import GraphState
from app.schemas.common import SentimentType
from app.schemas.llm_outputs import SummaryResult
from app.core.llm import invoke_structured, provider_label

logger = logging.getLogger(__name__)
logger.info(f"summary_agent LLM provider: {provider_label()}")


def summary_agent_node(state: GraphState) -> GraphState:
    """상담 내용을 요약하고 감정/키워드를 추출하는 노드"""
    
    # 🔧 이미 요약 정보가 있으면 재사용 (process_handover에서 전달된 경우)
    existing_summary = state.get("summary")
    existing_sentiment = state.get("customer_sentiment")
    existing_keywords = state.get("extracted_keywords", [])
    
    if existing_summary and existing_sentiment:
        logger.info(f"기존 요약 정보 재사용 - 세션: {state.get('session_id', 'unknown')}")
        logger.info(f"  📝 summary: {existing_summary}")
        logger.info(f"  😊 sentiment: {existing_sentiment}")
        logger.info(f"  🔑 keywords: {existing_keywords}")
        return state  # 이미 요약이 있으면 그대로 반환
    
    conversation_history = state.get("conversation_history", [])
    customer_intent_summary = state.get("customer_intent_summary")
    
    if not conversation_history:
        # 대화 이력이 없으면 기본값 설정
        state["summary"] = None
        state["customer_sentiment"] = None
        state["extracted_keywords"] = []
        return state
    
    # 대화 이력을 문자열로 변환
    conversation_text = "\n".join([
        f"[{msg.get('role', 'unknown')}] {msg.get('message', '')}"
        for msg in conversation_history
    ])
    
    # customer_intent_summary가 있으면 프롬프트에 포함
    intent_summary_hint = f"\n[고객 의도 요약 (triage_agent에서 생성)]: {customer_intent_summary}" if customer_intent_summary else ""

    system_message = SystemMessage(content="""당신은 카드사 상담 대화를 분석하는 어시스턴트입니다.
대화 기록을 보고 고객 감정(sentiment), 요약(summary, 3줄 정도), 핵심 키워드(keywords, 최대 5개)를
JSON으로 추출하세요. sentiment 는 POSITIVE/NEGATIVE/NEUTRAL 중 하나입니다.""")

    human_message = HumanMessage(content=f"""다음 상담 대화를 분석하세요.

[대화 기록]
{conversation_text}
{intent_summary_hint}""")

    try:
        logger.info(f"요약 에이전트 실행 - 세션: {state.get('session_id', 'unknown')}")
        result: SummaryResult = invoke_structured(
            SummaryResult, [system_message, human_message], temperature=0.2
        )

        # sentiment 매핑
        sentiment_map = {
            "POSITIVE": SentimentType.POSITIVE,
            "NEGATIVE": SentimentType.NEGATIVE,
            "NEUTRAL": SentimentType.NEUTRAL,
        }
        sentiment = sentiment_map.get(result.sentiment, SentimentType.NEUTRAL)
        summary = result.summary or None
        keywords = [k for k in result.keywords if k][:5]

        # 상태 업데이트
        state["summary"] = summary
        state["customer_sentiment"] = sentiment
        state["extracted_keywords"] = keywords

        logger.info(f"✅ 요약 생성 완료 - 세션: {state.get('session_id', 'unknown')}")
        logger.info(f"  📝 summary: {summary}")
        logger.info(f"  😊 sentiment: {sentiment}")
        logger.info(f"  🔑 keywords: {keywords}")

    except Exception as e:
        # 에러 발생 시 기본값 설정
        error_msg = str(e)
        logger.error(f"요약 에이전트 오류 - 세션: {state.get('session_id', 'unknown')}, 오류: {error_msg}", exc_info=True)
        state["summary"] = None
        state["customer_sentiment"] = None
        state["extracted_keywords"] = []
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["summary_error"] = error_msg
    
    return state

