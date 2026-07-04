# ai_engine/graph/nodes/consent_check_node.py

"""HUMAN_REQUIRED 플로우에서 고객 동의를 확인하는 노드.

룰 베이스 + LLM 기반으로 고객의 동의 여부를 확인합니다.
- 욕설/부정적 감정 (룰) → 상담사 연결 동의 (우선 처리)
- 명시적 거부 패턴 (룰) → is_human_required_flow=FALSE → triage_agent로 이동
- 이모티콘/감탄사/불명확 응답 (룰) → 다시 동의 질문
- 명시적 동의 패턴 (룰) → waiting_agent로 이동
- 그 외 (LLM 판단) → 동의/거부/도메인외/불명확 분류
"""

from __future__ import annotations
import logging
import re
from ai_engine.graph.state import GraphState
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.llm import invoke_structured
from app.schemas.llm_outputs import ConsentResult

logger = logging.getLogger(__name__)

# 욕설/부정적 감정 패턴 (고객이 화난 상태 → 상담사 연결 동의로 처리)
FRUSTRATION_PATTERNS = [
    "시발", "씨발", "ㅅㅂ", "씹", "개새", "ㄱㅅㄲ", "병신", "ㅂㅅ", "지랄", "ㅈㄹ",
    "미친", "ㅁㅊ", "존나", "ㅈㄴ", "빡쳐", "빡치", "열받", "짜증", "화나", "답답"
]

# 명시적 거부 패턴 정의 (룰 베이스)
NEGATIVE_PATTERNS = [
    "아니", "싫", "안 해", "안해", "괜찮아", "됐어", "필요 없", "필요없",
    "상담사 안", "연결 안", "그냥 됐", "취소", "안 할", "안할", "말고",
    "않아도", "안 해도", "안해도", "하지 않", "하지않", "않을", "안 할게", "안할게",
    "연결하지", "상담사 필요", "no", "nope", "ㄴㄴ", "ㄴ"
]

# 이모티콘/감탄사/불명확 응답 패턴 (다시 동의 질문 필요)
UNCLEAR_PATTERNS = [
    "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ", "ㄷㄷ", "ㅇㅇ", "ㅇㅋ", "ㄱㄱ",
    "ㅋ", "ㅎ", "ㅠ", "ㅜ",
    ";;", "...", "..", "?", "??", "???",
    "음", "흠", "아", "어", "오", "우", "에",
    "헐", "헐ㅋ", "헉", "앗", "엥", "잉", "윙",
    "lol", "ㅇㅅㅇ", "ㅇㅁㅇ", "ㅡㅡ", "^^", "^_^",
]

# 명시적 동의 패턴
POSITIVE_PATTERNS = [
    "네", "예", "응", "좋아", "알겠", "그래", "부탁", "해주", "해 주", "연결해",
    "빨리", "급해", "도와", "yes", "ok", "okay"
]


def _is_unclear_only(message: str) -> bool:
    """메시지가 불명확한 응답만으로 이루어져 있는지 확인."""
    cleaned = message.replace(" ", "")

    if not cleaned:
        return True

    # 반복되는 자음/모음만 있는 경우
    if re.match(r'^[ㅋㅎㅠㅜㄷㅇㄱ]+$', cleaned):
        return True

    # 반복되는 완성형 글자
    if re.match(r'^(크|킄|킥|키|ㅋ)+$', cleaned):
        return True
    if re.match(r'^(하|히|허|호|흐|ㅎ)+$', cleaned):
        return True
    if re.match(r'^(ㅠ|ㅜ|우|유|으)+$', cleaned):
        return True

    # 특수문자/이모티콘만 있는 경우
    if re.match(r'^[.;?!~^_\-=+]+$', cleaned):
        return True

    # UNCLEAR_PATTERNS 중 하나와 정확히 일치
    for pattern in UNCLEAR_PATTERNS:
        if cleaned == pattern or cleaned == pattern * (len(cleaned) // len(pattern)):
            return True

    return False


def _classify_with_llm(user_message: str, session_id: str) -> dict:
    """LLM을 사용하여 고객 메시지를 분류합니다.

    Returns:
        dict: {
            "classification": "CONSENT" | "REJECT" | "OUT_OF_DOMAIN" | "UNCLEAR",
            "ai_message": str (OUT_OF_DOMAIN/UNCLEAR인 경우 응답 메시지)
        }
    """
    try:
        system_prompt = """당신은 카드사 고객센터 AI 상담사입니다.
현재 고객에게 "상담사 연결을 원하시나요?"라고 질문한 상태입니다.

고객의 응답을 아래 4가지 중 하나로 분류하세요:

1. CONSENT (동의): 상담사 연결을 원하는 경우
   - "네", "예", "연결해주세요", "빨리 연결해", "상담사 바꿔줘" 등

2. REJECT (거부): 상담사 연결을 원하지 않는 경우
   - "아니요", "괜찮아요", "필요없어요", "연결 안 해도 돼" 등

3. OUT_OF_DOMAIN (도메인 외): 카드사 업무와 완전히 무관한 말
   - 성적 농담, 개인적인 이야기, 날씨, 음식 주문 등
   - "오늘 밤 나 너무 외로워", "배고파", "뭐해?" 등

4. UNCLEAR (불명확): 동의/거부를 판단할 수 없는 경우
   - 카드 관련이지만 동의/거부가 명확하지 않은 경우

JSON 형식으로 응답하세요:
{
    "classification": "CONSENT" | "REJECT" | "OUT_OF_DOMAIN" | "UNCLEAR",
    "ai_message": "OUT_OF_DOMAIN이나 UNCLEAR인 경우에만 고객에게 보낼 응답 메시지"
}

OUT_OF_DOMAIN 응답 예시:
- "죄송합니다, 저는 카드 관련 상담만 도와드릴 수 있습니다. 상담사 연결을 원하시면 '네'라고 말씀해 주세요."

UNCLEAR 응답 예시:
- "죄송합니다, 다시 한번 말씀해 주시겠어요? 상담사 연결을 원하시면 '네' 또는 '연결해 주세요'라고 말씀해 주세요."
"""

        human_prompt = f"고객 메시지: \"{user_message}\""

        result: ConsentResult = invoke_structured(
            ConsentResult,
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
            temperature=0.0,
        )
        logger.info(f"LLM 분류 결과 - 세션: {session_id}, 분류: {result.classification}")
        return {"classification": result.classification, "ai_message": result.ai_message}

    except Exception as e:
        logger.error(f"LLM 분류 실패 - 세션: {session_id}, 에러: {e}")
        # 실패 시 기본값: UNCLEAR (재질문) — 'CONSENT'로 처리하면 LLM 오류 시
        # 고객 동의 없이 정보 수집이 진행되는 위험이 있으므로 안전하게 재질문한다.
        return {
            "classification": "UNCLEAR",
            "ai_message": "죄송합니다, 다시 한번 말씀해 주시겠어요? 상담사 연결을 원하시면 '네' 또는 '연결해 주세요'라고 말씀해 주세요.",
        }


MAX_OUT_OF_DOMAIN_COUNT = 3  # 도메인 외 질문 최대 허용 횟수
MAX_UNCLEAR_COUNT = 3  # 불명확 응답 최대 허용 횟수


def consent_check_node(state: GraphState) -> GraphState:
    """고객 동의를 확인하는 노드 (룰 베이스 + LLM).

    판단 순서:
    1. 욕설/부정적 감정 패턴 (룰) → 상담사 연결 동의 (우선 처리)
    2. 명시적 거부 패턴 (룰) → triage_agent로 이동
    3. 이모티콘/감탄사/불명확 응답 (룰) → 다시 동의 질문
    4. 명시적 동의 패턴 (룰) → waiting_agent로 이동
    5. 그 외 (LLM 판단) → 동의/거부/도메인외/불명확 분류

    도메인 외 질문이 3회 이상 반복되면 세션을 종료합니다.
    """
    user_message = state.get("user_message", "").strip().lower()
    session_id = state.get("session_id", "unknown")
    out_of_domain_count = state.get("out_of_domain_count", 0)
    unclear_count = state.get("unclear_count", 0)

    # 1. 욕설/부정적 감정 패턴 확인 (최우선 - 룰 베이스)
    is_frustrated = any(p in user_message for p in FRUSTRATION_PATTERNS)

    # 2. 명시적 거부 패턴 확인 (룰 베이스)
    is_rejection = any(p in user_message for p in NEGATIVE_PATTERNS)

    # 3. 불명확한 응답 확인 (룰 베이스)
    is_unclear = _is_unclear_only(user_message)

    # 4. 명시적 동의 패턴 확인 (룰 베이스)
    is_positive = any(p in user_message for p in POSITIVE_PATTERNS)

    if is_frustrated:
        # 고객이 화난 상태 → 상담사 연결 동의로 처리
        state["customer_consent_received"] = True
        if not state.get("collected_info"):
            state["collected_info"] = {}
        logger.info(f"고객 부정적 감정 감지 - 세션: {session_id}, 상담사 연결 동의로 처리 → waiting_agent로 이동")

    elif is_rejection:
        # 명시적 거부 → triage_agent로 이동
        state["is_human_required_flow"] = False
        state["customer_declined_handover"] = True
        state["customer_consent_received"] = False
        state["handover_status"] = None
        state["ai_message"] = "알겠습니다. 다른 도움이 필요하시면 말씀해 주세요."
        logger.info(f"고객 명시적 거부 - 세션: {session_id}, HUMAN_REQUIRED 플로우 종료 → triage_agent로 이동")

    elif is_unclear:
        # 불명확한 응답 → 카운터 증가 후 다시 동의 질문 또는 세션 종료
        unclear_count += 1
        state["unclear_count"] = unclear_count

        if unclear_count >= MAX_UNCLEAR_COUNT:
            # 3회 이상 불명확 응답 → 세션 종료
            state["is_human_required_flow"] = False
            state["customer_consent_received"] = False
            state["is_session_end"] = True
            state["ai_message"] = "죄송합니다. 명확한 응답을 확인하지 못해 상담을 종료합니다. 추가 문의가 있으시면 다시 연락해 주세요. 감사합니다."
            logger.info(f"불명확 응답 {unclear_count}회 초과 - 세션: {session_id}, 세션 종료")
        else:
            state["customer_consent_received"] = False
            state["ai_message"] = "죄송합니다, 다시 한번 말씀해 주시겠어요? 상담사 연결을 원하시면 '네' 또는 '연결해 주세요'라고 말씀해 주세요."
            logger.info(f"고객 불명확 응답 ({unclear_count}회) - 세션: {session_id}, 다시 동의 질문 → consent_check_node 유지")

    elif is_positive:
        # 명시적 동의 → waiting_agent로 이동
        state["customer_consent_received"] = True
        if not state.get("collected_info"):
            state["collected_info"] = {}
        logger.info(f"고객 명시적 동의 - 세션: {session_id}, 동의로 처리 → waiting_agent로 이동")

    else:
        # 그 외 → LLM으로 판단
        logger.info(f"LLM 분류 시작 - 세션: {session_id}, 메시지: {user_message}")
        llm_result = _classify_with_llm(user_message, session_id)
        classification = llm_result.get("classification", "CONSENT")

        if classification == "CONSENT":
            state["customer_consent_received"] = True
            if not state.get("collected_info"):
                state["collected_info"] = {}
            logger.info(f"LLM 판단: 동의 - 세션: {session_id} → waiting_agent로 이동")

        elif classification == "REJECT":
            state["is_human_required_flow"] = False
            state["customer_declined_handover"] = True
            state["customer_consent_received"] = False
            state["handover_status"] = None
            state["ai_message"] = "알겠습니다. 다른 도움이 필요하시면 말씀해 주세요."
            logger.info(f"LLM 판단: 거부 - 세션: {session_id} → triage_agent로 이동")

        elif classification == "OUT_OF_DOMAIN":
            # 도메인 외 질문 카운터 증가
            out_of_domain_count += 1
            state["out_of_domain_count"] = out_of_domain_count

            if out_of_domain_count >= MAX_OUT_OF_DOMAIN_COUNT:
                # 3회 이상 도메인 외 질문 → 세션 종료
                state["is_human_required_flow"] = False
                state["customer_consent_received"] = False
                state["is_session_end"] = True
                state["ai_message"] = "죄송합니다. 카드 관련 상담 외 문의가 계속되어 상담을 종료합니다. 카드 관련 문의가 있으시면 다시 연락해 주세요. 감사합니다."
                logger.info(f"도메인 외 질문 {out_of_domain_count}회 초과 - 세션: {session_id}, 세션 종료")
            else:
                state["customer_consent_received"] = False
                state["ai_message"] = llm_result.get(
                    "ai_message",
                    "죄송합니다, 저는 카드 관련 상담만 도와드릴 수 있습니다. 상담사 연결을 원하시면 '네'라고 말씀해 주세요."
                )
                logger.info(f"LLM 판단: 도메인 외 ({out_of_domain_count}회) - 세션: {session_id}, 다시 동의 질문")

        else:  # UNCLEAR
            # 불명확 응답 카운터 증가
            unclear_count += 1
            state["unclear_count"] = unclear_count

            if unclear_count >= MAX_UNCLEAR_COUNT:
                # 3회 이상 불명확 응답 → 세션 종료
                state["is_human_required_flow"] = False
                state["customer_consent_received"] = False
                state["is_session_end"] = True
                state["ai_message"] = "죄송합니다. 명확한 응답을 확인하지 못해 상담을 종료합니다. 추가 문의가 있으시면 다시 연락해 주세요. 감사합니다."
                logger.info(f"LLM 판단: 불명확 {unclear_count}회 초과 - 세션: {session_id}, 세션 종료")
            else:
                state["customer_consent_received"] = False
                state["ai_message"] = llm_result.get(
                    "ai_message",
                    "죄송합니다, 다시 한번 말씀해 주시겠어요? 상담사 연결을 원하시면 '네' 또는 '연결해 주세요'라고 말씀해 주세요."
                )
                logger.info(f"LLM 판단: 불명확 ({unclear_count}회) - 세션: {session_id}, 다시 동의 질문")

    return state
