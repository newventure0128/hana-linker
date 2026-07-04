# ai_engine/graph/nodes/answer_agent.py

"""챗봇 답변 생성 에이전트: GPT 모델 호출."""

from __future__ import annotations
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from ai_engine.graph.state import GraphState
from app.schemas.chat import SourceDocument
from app.core.llm import get_chat_llm, provider_label
from app.schemas.common import TriageDecisionType

logger = logging.getLogger(__name__)

# LLM 팩토리로 생성 (provider는 app/core/config.py 의 llm_provider 로 결정)
llm = get_chat_llm(temperature=0.2)
logger.info(f"answer_agent LLM provider: {provider_label()}")


def _handle_error(error_msg: str, state: GraphState) -> None:
    """공통 에러 처리"""
    if "quota" in error_msg.lower() or "429" in error_msg:
        state["ai_message"] = "죄송합니다. 현재 서비스 사용량이 초과되어 일시적으로 답변을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."
        logger.warning(f"API 할당량 초과 - 세션: {state.get('session_id', 'unknown')}")
    elif "api_key" in error_msg.lower() or "401" in error_msg:
        state["ai_message"] = "죄송합니다. API 설정 오류가 발생했습니다. 관리자에게 문의해주세요."
        logger.error(f"API 키 오류 - 세션: {state.get('session_id', 'unknown')}")
    elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
        state["ai_message"] = "죄송합니다. LM Studio 서버에 연결할 수 없습니다. LM Studio가 실행 중인지 확인해주세요."
        logger.error(f"LM Studio 연결 오류 - 세션: {state.get('session_id', 'unknown')}, 오류: {error_msg}")
    elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
        state["ai_message"] = "죄송합니다. 응답 생성에 시간이 오래 걸려 타임아웃이 발생했습니다. 더 간단한 질문으로 다시 시도해주세요."
        logger.warning(f"답변 생성 타임아웃 - 세션: {state.get('session_id', 'unknown')}")
    else:
        state["ai_message"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        logger.error(f"답변 생성 기타 오류 - 세션: {state.get('session_id', 'unknown')}, 오류: {error_msg}")
    
    if "metadata" not in state:
        state["metadata"] = {}
    state["metadata"]["answer_error"] = error_msg


def _format_conversation_history(conversation_history: list, max_turns: int = 5) -> str:
    """대화 히스토리를 프롬프트용 문자열로 포맷팅합니다.

    Args:
        conversation_history: 대화 히스토리 리스트
        max_turns: 포함할 최대 턴 수 (기본값: 5)

    Returns:
        포맷팅된 대화 히스토리 문자열
    """
    if not conversation_history:
        return ""

    # 최근 N개 턴만 사용
    recent_history = conversation_history[-max_turns * 2:] if len(conversation_history) > max_turns * 2 else conversation_history

    formatted_lines = []
    for msg in recent_history:
        role = msg.get("role", "unknown")
        message = msg.get("message", "")
        if role == "user":
            formatted_lines.append(f"고객: {message}")
        elif role == "assistant":
            formatted_lines.append(f"상담봇: {message}")

    return "\n".join(formatted_lines)


def answer_agent_node(state: GraphState) -> GraphState:
    """프롬프트를 구성해 LLM에게 답변 생성을 요청하고 상태를 갱신한다.

    triage_decision 값에 따라 다른 프롬프트를 사용하여 처리:
    - SIMPLE_ANSWER: 간단한 자연어 답변 생성
    - AUTO_ANSWER: RAG 문서 기반 답변 생성
    - NEED_MORE_INFO: 추가 정보 요청 질문 생성
    - HUMAN_REQUIRED: 상담사 연결 안내 메시지 생성
    """
    user_message = state["user_message"]
    triage_decision = state.get("triage_decision")
    retrieved_docs = state.get("retrieved_documents", [])
    customer_intent_summary = state.get("customer_intent_summary")
    conversation_history = state.get("conversation_history", [])
    
    # ========================================================================
    # 티켓별 프롬프트 생성 및 LLM 호출
    # ========================================================================
    try:
        # ------------------------------------------------------------------------
        # 티켓 1: SIMPLE_ANSWER - 간단한 자연어 답변 생성
        # 사용 정보: user_message, customer_intent_summary
        # RAG 검색 결과: 사용 안 함
        # ------------------------------------------------------------------------
        if triage_decision == TriageDecisionType.SIMPLE_ANSWER:
            # customer_intent_summary 정보 포함
            intent_summary_hint = f"\n[고객 의도 요약: {customer_intent_summary}]" if customer_intent_summary else ""

            # 대화 히스토리 포맷팅 (최근 5턴)
            history_text = _format_conversation_history(conversation_history, max_turns=5)
            history_section = f"\n\n[이전 대화 내용]\n{history_text}" if history_text else ""

            system_message = SystemMessage(content="""당신은 카드사 고객센터의 챗봇 어시스턴트입니다.
이전 대화 맥락을 고려하여 간단하고 자연스러운 응답을 생성하세요.

중요: 당신은 카드/금융 관련 상담만 제공합니다. 다른 주제는 정중히 거절하세요.

**가장 중요한 규칙: 이전 대화 맥락을 항상 고려하세요!**
- 이전 대화에서 진행 중인 업무(예: 카드 분실 신고)가 있다면, 고객의 현재 메시지는 그 맥락에서 해석해야 합니다.
- 예: 카드 분실 신고 중 이름/생년월일을 물었고 고객이 "홍길동이고 1990년 1월 1일"이라고 답하면
     → 이것은 분실 신고에 필요한 정보 제공이므로, 다음 단계(예: 카드 뒷자리 확인)로 진행해야 합니다.
     → "무엇을 도와드릴까요?"라는 응답은 절대 하지 마세요!

응답 규칙:
1) 진행 중인 업무의 정보 제공인 경우 (가장 중요!)
   - 이전 대화에서 특정 정보를 요청했고, 고객이 그 정보를 제공한 경우
   - → 정보를 확인하고 다음 단계로 진행
   - 예: "네, 홍길동님 확인되었습니다. 분실하신 카드의 뒷 4자리를 말씀해 주시겠어요?"

2) 단순 확인/동의/짧은 반응인 경우
   - 예: "네", "넵", "맞아요", "그거요", "알겠어요", "계속 해주세요"
   - → 이전 답변을 인정/확인하는 짧은 응답만 생성
   - 예: "네, 알겠습니다. 계속 진행하겠습니다."

3) 감사 인사/끝맺음인 경우
   - 예: "감사합니다", "덕분에 해결됐어요", "수고하세요", "고마워요"
   - → 간단한 인사로 대답
   - 예: "도움이 되어서 다행입니다. 추가 요청사항이 있으신가요?"

4) 시스템/잡음/의미 없는 입력인 경우 (대화 맥락이 없을 때만!)
   - 예: "", "…", "음", "아아", "ㅋㅋㅋ", "ㅎㅎㅎ", STT 오류로 보이는 텍스트
   - 단, 진행 중인 대화가 있으면 재확인 요청
   - 예: "죄송합니다. 잘 못 들었습니다. 다시 한 번 말씀해 주시겠어요?"

5) 카드/금융과 무관한 질문인 경우 (대화 맥락 없을 때)
   - 예: "피자 주문", "날씨", "주식 투자", "맛집 추천", "영화 추천" 등
   - → 정중히 거절하고 카드 관련 문의를 안내
   - 예: "죄송합니다. 저는 카드 및 금융 관련 상담만 도와드릴 수 있습니다."

6) 욕설/비속어/부적절한 표현인 경우
   - → 정중하게 대화 예절을 요청
   - 예: "원활한 상담을 위해 정중한 표현을 부탁드립니다."

7) 영어 또는 외국어 입력인 경우
   - → 한국어 사용을 요청
   - 예: "죄송합니다. 현재 한국어 상담만 지원하고 있습니다."

8) 직전 턴에 이미 충분히 답변이 끝난 경우
   - 추가 질문이 전혀 없는 단순 리액션
   - → 대화를 마무리하거나 짧게 응답
   - 예: "추가 질문이 없으시면, 언제든 다시 문의해 주세요.""")

            human_message = HumanMessage(content=f"""이전 대화 맥락을 고려하여 자연스러운 응답을 생성해주세요.
{history_section}

[현재 고객 메시지]
{user_message}
{intent_summary_hint}""")
            
            logger.info(f"SIMPLE_ANSWER 답변 생성 시작 - 세션: {state.get('session_id', 'unknown')}")
            response = llm.invoke([system_message, human_message])
            state["ai_message"] = response.content
            state["source_documents"] = []
            logger.info(f"SIMPLE_ANSWER 답변 생성 완료 - 세션: {state.get('session_id', 'unknown')}")
        
        # ------------------------------------------------------------------------
        # 티켓 2: AUTO_ANSWER - RAG 문서 기반 답변 생성
        # 사용 정보: user_message, customer_intent_summary, retrieved_docs
        # RAG 검색 결과: 사용 (필수)
        # ------------------------------------------------------------------------
        elif triage_decision == TriageDecisionType.AUTO_ANSWER:
            if not retrieved_docs:
                state["ai_message"] = "죄송합니다. 해당 업무는 상담원에게 문의해주세요."
                state["source_documents"] = []
                return state
            
            # customer_intent_summary 정보 포함
            intent_summary_hint = f"\n[고객 의도 요약: {customer_intent_summary}]" if customer_intent_summary else ""
            
            # 검색된 문서를 프롬프트에 포함
            context = "\n\n".join([
                f"[문서 {i+1}] {doc['content']} (출처: {doc['source']}, 페이지: {doc['page']}, 유사도: {doc['score']:.2f})"
                for i, doc in enumerate(retrieved_docs)
            ])

            system_message = SystemMessage(content="""당신은 고객 질문에 답변하는 음성 챗봇 어시스턴트입니다.
제공된 참고 문서를 기반으로 정확하고 예의바른 답변을 생성하세요.
근거 문서에 없는 내용은 추측하지 말고, 정확한 정보만 제공하세요.

⚠️ 중요: 이 답변은 TTS(음성 합성)로 읽혀집니다. 다음 규칙을 반드시 지켜주세요:
- 마크다운 기호 사용 금지: **, *, #, -, 번호(1. 2. 3.) 등의 기호 사용 금지
- 구어체로 자연스럽게 답변
- 간결하게 핵심만 전달 (2문장 이내)
- 불필요한 인사말이나 중복 안내는 생략하고, 질문에 대한 핵심 정보(방법·연락처·기간 등)를 담아 답변하세요""")

            human_message = HumanMessage(content=f"""참고 문서를 기반으로 고객 질문에 대한 답변을 생성해주세요.
마크다운 기호 없이 자연스러운 구어체로, 핵심만 2문장 이내로 답변하세요.

[참고 문서]
{context}
{intent_summary_hint}

[고객 질문]
{user_message}""")
            
            logger.info(f"AUTO_ANSWER 답변 생성 시작 - 세션: {state.get('session_id', 'unknown')}")
            response = llm.invoke([system_message, human_message])
            state["ai_message"] = response.content
            
            # retrieved_documents를 source_documents로 변환
            state["source_documents"] = [
                SourceDocument(
                    source=doc.get("source", "unknown"),
                    page=doc.get("page", 0),
                    score=doc.get("score", 0.0)
                )
                for doc in retrieved_docs
            ]
            logger.info(f"AUTO_ANSWER 답변 생성 완료 - 세션: {state.get('session_id', 'unknown')}")
        
        # ------------------------------------------------------------------------
        # 티켓 3: NEED_MORE_INFO - 추가 정보 요청 질문 생성
        # 사용 정보: user_message, customer_intent_summary
        # RAG 검색 결과: 사용 안 함
        # ------------------------------------------------------------------------
        elif triage_decision == TriageDecisionType.NEED_MORE_INFO:
            # customer_intent_summary 정보 포함
            intent_summary_hint = f"\n[고객 의도 요약: {customer_intent_summary}]" if customer_intent_summary else ""

            # 대화 히스토리 포맷팅 (최근 5턴)
            history_text = _format_conversation_history(conversation_history, max_turns=5)
            history_section = f"\n\n[이전 대화 내용]\n{history_text}" if history_text else ""

            system_message = SystemMessage(content="""당신은 고객에게 추가 정보를 요청하는 음성 챗봇 어시스턴트입니다.
이전 대화 맥락을 고려하여 필요한 추가 정보를 정중하게 질문하세요.

중요: 이전 대화에서 이미 수집한 정보는 다시 물어보지 마세요!
- 이전 대화에서 고객이 제공한 정보(예: 이름, 카드 종류, 생년월일 등)가 있다면 참고하세요.
- 아직 수집하지 않은 정보만 질문하세요.

질문 생성 시 주의사항:
- 한 번에 하나의 구체적인 질문만 하세요
- 예의바르고 친절한 톤을 유지하세요
- 고객이 쉽게 답변할 수 있는 질문으로 하세요
- 불필요한 정보를 요청하지 마세요
- 이미 제공받은 정보는 확인/인정하고 다음 단계로 진행하세요

⚠️ TTS 규칙: 마크다운 기호(**, *, #, -, 번호) 사용 금지. 구어체로 자연스럽게.""")

            human_message = HumanMessage(content=f"""이전 대화를 참고하여 필요한 추가 정보를 정중하게 질문해주세요.
{history_section}

[현재 고객 질문]
{user_message}
{intent_summary_hint}""")
            
            logger.info(f"NEED_MORE_INFO 질문 생성 시작 - 세션: {state.get('session_id', 'unknown')}")
            response = llm.invoke([system_message, human_message])
            state["ai_message"] = response.content
            state["source_documents"] = []
            logger.info(f"NEED_MORE_INFO 질문 생성 완료 - 세션: {state.get('session_id', 'unknown')}")
        
        # ------------------------------------------------------------------------
        # 티켓 4: HUMAN_REQUIRED - 상담사 연결 안내 메시지 생성
        # 사용 정보: user_message, customer_intent_summary
        # RAG 검색 결과: 사용 안 함
        # ------------------------------------------------------------------------
        elif triage_decision == TriageDecisionType.HUMAN_REQUIRED:
            # 긴급 상황 확인 (보이스피싱, 금융사기 등)
            is_urgent = state.get("is_urgent_handover", False)

            # customer_intent_summary 정보 포함
            intent_summary_hint = f"\n[고객 의도 요약: {customer_intent_summary}]" if customer_intent_summary else ""

            if is_urgent:
                # 🚨 긴급 상황: 슬롯 수집 없이 즉시 상담사 연결
                system_message = SystemMessage(content="""당신은 긴급 상황을 처리하는 음성 챗봇 어시스턴트입니다.
보이스피싱, 금융사기 등 긴급 상황으로 감지되었습니다.

안내 메시지 규칙:
- 고객의 상황에 공감하는 표현으로 시작
- "즉시 상담사에게 연결해 드리겠다"는 내용을 포함
- 추가 정보 요청 없이 바로 연결 안내
- 1-2문장으로 간결하게

⚠️ TTS 규칙: 마크다운 기호(**, *, #, -, 번호) 사용 금지. 구어체로 자연스럽게.""")

                human_message = HumanMessage(content=f"""긴급 상황에 대한 즉시 상담사 연결 안내 메시지를 생성해주세요.

[고객 메시지]
{user_message}
{intent_summary_hint}""")

                logger.info(f"🚨 긴급 HUMAN_REQUIRED 안내 메시지 생성 시작 (슬롯 수집 스킵) - 세션: {state.get('session_id', 'unknown')}")
                response = llm.invoke([system_message, human_message])
                state["ai_message"] = response.content
                state["source_documents"] = []

                # 긴급 상황: 동의 확인 건너뛰고 바로 waiting_agent로 이동
                state["is_human_required_flow"] = True
                state["customer_consent_received"] = True  # 동의 확인 스킵
                state["collected_info"] = {}
                state["info_collection_complete"] = True  # 슬롯 수집 스킵
                logger.info(f"🚨 긴급 HUMAN_REQUIRED 완료 - 세션: {state.get('session_id', 'unknown')}, 동의/슬롯수집 스킵 → waiting_agent로 직행")
            else:
                # 일반 상황: 동의 확인 + 슬롯 수집
                system_message = SystemMessage(content="""당신은 상담사 연결을 안내하는 음성 챗봇 어시스턴트입니다.
상담사가 이어받을 수 있도록 정중하고 공감 어린 어조로 안내 문장을 생성하세요.

안내 메시지 규칙:
- "상담사에게 연결해 드리겠다"는 내용을 포함
- 정중하고 공감 어린 어조 사용
- 1-2문장으로 간결하게

중요: 추가 정보 요청이나 문제 해결 시도 없이, 상담사 연결 안내만 생성하세요.
⚠️ TTS 규칙: 마크다운 기호(**, *, #, -, 번호) 사용 금지. 구어체로 자연스럽게.""")

                human_message = HumanMessage(content=f"""정중하고 공감 어린 상담사 연결 안내 메시지를 생성해주세요.

[고객 메시지]
{user_message}
{intent_summary_hint}""")

                logger.info(f"HUMAN_REQUIRED 안내 메시지 생성 시작 - 세션: {state.get('session_id', 'unknown')}")
                response = llm.invoke([system_message, human_message])
                # 상담사 연결 안내 메시지 뒤에 고정 메시지 추가
                fixed_message = "\n\n고객님의 문의를 빠르게 해결해드리기 위해 상담사 연결 전까지 정보를 수집할 예정입니다. 상담사 연결을 원하시면 '네'라고 말씀해주세요. 상담사 연결을 원하지 않으시면 '아니오'라고 말씀해주세요."
                state["ai_message"] = response.content + fixed_message
                state["source_documents"] = []

                # HUMAN_REQUIRED 플로우 진입 플래그 설정
                state["is_human_required_flow"] = True
                state["customer_consent_received"] = False
                state["collected_info"] = {}
                state["info_collection_complete"] = False
                logger.info(f"HUMAN_REQUIRED 안내 메시지 생성 완료 - 세션: {state.get('session_id', 'unknown')}, HUMAN_REQUIRED 플로우 진입")
        
        # ------------------------------------------------------------------------
        # Fallback: triage_decision이 없거나 예상치 못한 값
        # 문서가 있으면 AUTO_ANSWER 로직, 없으면 SIMPLE_ANSWER 로직 사용
        # ------------------------------------------------------------------------
        else:
            logger.warning(f"triage_decision이 없거나 예상치 못한 값: {triage_decision}, 기본 처리: 세션={state.get('session_id', 'unknown')}")
            if retrieved_docs:
                # AUTO_ANSWER 로직 사용
                intent_summary_hint = f"\n[고객 의도 요약: {customer_intent_summary}]" if customer_intent_summary else ""
                context = "\n\n".join([
                    f"[문서 {i+1}] {doc['content']} (출처: {doc['source']}, 페이지: {doc['page']}, 유사도: {doc['score']:.2f})"
                    for i, doc in enumerate(retrieved_docs)
                ])
                
                system_message = SystemMessage(content="""당신은 고객 질문에 답변하는 챗봇 어시스턴트입니다.
제공된 참고 문서를 기반으로 정확하고 예의바른 답변을 생성하세요.
근거 문서에 없는 내용은 추측하지 말고, 정확한 정보만 제공하세요.""")
                
                human_message = HumanMessage(content=f"""참고 문서를 기반으로 고객 질문에 대한 답변을 생성해주세요.

[참고 문서]
{context}
{intent_summary_hint}

[고객 질문]
{user_message}""")
                
                response = llm.invoke([system_message, human_message])
                state["ai_message"] = response.content
                state["source_documents"] = [
                    SourceDocument(
                        source=doc.get("source", "unknown"),
                        page=doc.get("page", 0),
                        score=doc.get("score", 0.0)
                    )
                    for doc in retrieved_docs
                ]
            else:
                # SIMPLE_ANSWER 로직 사용 (Fallback)
                intent_summary_hint = f"\n[고객 의도 요약: {customer_intent_summary}]" if customer_intent_summary else ""

                # 대화 히스토리 포맷팅 (최근 5턴)
                history_text = _format_conversation_history(conversation_history, max_turns=5)
                history_section = f"\n\n[이전 대화 내용]\n{history_text}" if history_text else ""

                system_message = SystemMessage(content="""당신은 카드사 고객센터의 챗봇 어시스턴트입니다.
이전 대화 맥락을 고려하여 간단하고 자연스러운 응답을 생성하세요.

중요: 당신은 카드/금융 관련 상담만 제공합니다. 다른 주제는 정중히 거절하세요.

**가장 중요한 규칙: 이전 대화 맥락을 항상 고려하세요!**
- 이전 대화에서 진행 중인 업무가 있다면, 고객의 현재 메시지는 그 맥락에서 해석해야 합니다.

응답 규칙:
1) 진행 중인 업무의 정보 제공인 경우
   - 이전 대화에서 특정 정보를 요청했고, 고객이 그 정보를 제공한 경우
   - → 정보를 확인하고 다음 단계로 진행

2) 카드/금융과 무관한 질문인 경우 (대화 맥락 없을 때)
   - → "죄송합니다. 저는 카드 및 금융 관련 상담만 도와드릴 수 있습니다."

3) 의미 없는 입력/잡음인 경우 (대화 맥락 없을 때)
   - → "무엇을 도와드릴까요? 카드 관련 문의사항이 있으시면 말씀해 주세요."

4) 욕설/비속어인 경우
   - → "원활한 상담을 위해 정중한 표현을 부탁드립니다."

5) 영어/외국어인 경우
   - → "죄송합니다. 현재 한국어 상담만 지원하고 있습니다." """)

                human_message = HumanMessage(content=f"""이전 대화 맥락을 고려하여 자연스러운 응답을 생성해주세요.
{history_section}

[현재 고객 메시지]
{user_message}
{intent_summary_hint}""")

                response = llm.invoke([system_message, human_message])
                state["ai_message"] = response.content
                state["source_documents"] = []
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"답변 생성 중 오류 발생 - 세션: {state.get('session_id', 'unknown')}, 티켓: {triage_decision}, 오류: {error_msg}", exc_info=True)
        _handle_error(error_msg, state)
        state["source_documents"] = []
    
    return state