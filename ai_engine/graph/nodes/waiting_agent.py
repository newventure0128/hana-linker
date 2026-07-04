# ai_engine/graph/nodes/waiting_agent.py

"""HUMAN_REQUIRED 플로우에서 정보 수집을 담당하는 에이전트.

동의 확인은 consent_check_node에서 룰 베이스로 처리됩니다.
이 노드는 동의 후 정보 수집만 담당합니다:
1. triage_agent에서 분류한 카테고리(context_intent)를 기반으로 필요한 슬롯 결정
2. 대화 히스토리에서 이미 언급된 정보 추출
3. 부족한 정보에 대해서만 질문 생성
4. 정보 수집 완료 시 summary_agent로 이동

도메인별 슬롯 시스템:
- slot_definitions.json: 카테고리별 필수/선택 슬롯 정의
- slot_metadata.json: 슬롯별 라벨, 질문, 검증 규칙
"""

from __future__ import annotations
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from ai_engine.graph.state import GraphState
from ai_engine.graph.utils.slot_loader import get_slot_loader
from app.core.llm import get_chat_llm, provider_label

logger = logging.getLogger(__name__)

# 슬롯 수집 중/완료 후 거부 패턴 (상담사 연결 취소 의사)
CANCEL_PATTERNS = ["됐어", "괜찮아", "취소", "그만", "안 할래", "필요 없", "연결 안", "상담사 안", "그냥 됐", "아니요", "아니", "싫어", "안해", "안 해"]

# "모르겠어요" 패턴 (해당 정보를 모를 때 - 다음 슬롯으로 진행)
UNKNOWN_PATTERNS = [
    "모르겠", "모름", "몰라", "모르는", "잘 모르", "잘모르",
    "기억 안", "기억이 안", "기억안", "기억나지 않", "기억이 나지",
    "확인 못", "확인이 안", "확인못", "확인을 못", "확인 안",
    "없어요", "없는데", "없습니다", "없어", "없음",
    "아직", "나중에", "다음에",
    "pass", "패스", "스킵", "skip", "넘어가", "넘겨"
]

# 불명확 응답 최대 허용 횟수 (consent_check_node와 동일)
MAX_UNCLEAR_COUNT = 3


def _check_collection_cancel(user_message: str) -> bool:
    """슬롯 수집 중/완료 후 거부 의사를 감지합니다.

    Args:
        user_message: 사용자 메시지

    Returns:
        거부 의사가 감지되면 True
    """
    user_message_lower = user_message.strip().lower()
    return any(pattern in user_message_lower for pattern in CANCEL_PATTERNS)


def _check_unknown_response(user_message: str) -> bool:
    """고객이 해당 정보를 모른다고 응답했는지 확인합니다.

    Args:
        user_message: 사용자 메시지

    Returns:
        "모르겠어요" 패턴이 감지되면 True
    """
    user_message_lower = user_message.strip().lower()
    return any(pattern in user_message_lower for pattern in UNKNOWN_PATTERNS)


def _is_unclear_response(message: str) -> bool:
    """메시지가 불명확한 응답인지 확인합니다 (consent_check_node와 동일한 로직).

    Args:
        message: 사용자 메시지

    Returns:
        불명확한 응답이면 True
    """
    import re
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

    # 불명확 패턴 목록
    unclear_patterns = [
        "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ", "ㄷㄷ", "ㅇㅇ", "ㅇㅋ", "ㄱㄱ",
        "ㅋ", "ㅎ", "ㅠ", "ㅜ",
        ";;", "...", "..", "?", "??", "???",
        "음", "흠", "아", "어", "오", "우", "에",
        "헐", "헐ㅋ", "헉", "앗", "엥", "잉", "윙",
        "lol", "ㅇㅅㅇ", "ㅇㅁㅇ", "ㅡㅡ", "^^", "^_^",
    ]

    for pattern in unclear_patterns:
        if cleaned == pattern or cleaned == pattern * (len(cleaned) // max(len(pattern), 1)):
            return True

    return False


# LLM 팩토리로 생성 (provider는 app/core/config.py 의 llm_provider 로 결정)
llm = get_chat_llm(temperature=0.2)
logger.info(f"waiting_agent LLM provider: {provider_label()}")


def _format_conversation_history(conversation_history: List[Dict]) -> str:
    """대화 히스토리를 문자열로 포맷팅합니다."""
    if not conversation_history:
        return "대화 이력 없음"

    formatted_lines = []
    for msg in conversation_history:
        role = msg.get("role", "unknown")
        message = msg.get("message", "")
        role_label = "고객" if role == "user" else "상담봇"
        formatted_lines.append(f"{role_label}: {message}")

    return "\n".join(formatted_lines)


def _extract_single_field(conversation_history: List[Dict], current_message: str, field_name: str, field_label: str) -> str | None:
    """대화에서 단일 필드 값을 추출합니다.

    LLM에게 JSON이 아닌 단순 텍스트로 응답받아 파싱 실패 위험을 줄입니다.
    고객(사용자) 메시지에서만 정보를 추출합니다.

    Args:
        conversation_history: 대화 히스토리
        current_message: 현재 사용자 메시지
        field_name: 추출할 필드명 (예: "customer_name")
        field_label: 필드 라벨 (예: "고객명")

    Returns:
        추출된 값 또는 None
    """
    # 고객(사용자) 메시지만 필터링 - AI 답변은 제외
    user_messages_only = [
        msg for msg in conversation_history
        if msg.get("role") == "user"
    ]
    history_text = _format_conversation_history(user_messages_only)

    system_message = SystemMessage(content=f"""당신은 대화 내용에서 특정 정보를 추출하는 어시스턴트입니다.

추출할 정보: {field_label}

중요 규칙:
1. **오직 고객이 직접 말한 내용에서만** 정보를 추출하세요.
2. 상담봇/AI의 답변 내용은 절대 추출하지 마세요.
3. '{field_label}' 정보가 고객 메시지에서 명확하게 언급되었으면 그 값만 추출하세요.
4. 추측하지 마세요. 확실하지 않으면 "없음"이라고 응답하세요.
5. 추출된 값만 간단히 응답하세요. 설명이나 다른 텍스트는 포함하지 마세요.

예시:
- 고객명을 추출하는 경우: "홍길동" (O), "고객님의 이름은 홍길동입니다" (X)
- 카드 뒤 4자리: "1234" (O), "카드번호 뒤 4자리는 1234입니다" (X)
- 정보가 없는 경우: "없음" (O)""")

    human_message = HumanMessage(content=f"""[고객 메시지 기록]
{history_text}

[현재 고객 메시지]
{current_message}

위 고객 메시지에서 '{field_label}' 정보를 추출해주세요. 값만 응답하세요.""")

    try:
        response = llm.invoke([system_message, human_message])
        extracted_value = response.content.strip()

        # "없음", "null", 빈 문자열 등은 None으로 처리
        if not extracted_value or extracted_value.lower() in ["없음", "null", "none", "n/a", "-"]:
            return None

        logger.debug(f"필드 '{field_name}' 추출 성공: {extracted_value}")
        return extracted_value

    except Exception as e:
        logger.error(f"필드 '{field_name}' 추출 중 오류: {str(e)}")
        return None


def _extract_answer_for_slot(current_message: str, field_label: str) -> str | None:
    """현재 질문 중인 슬롯에 대한 고객의 '직전 답변'에서 값을 추출한다.

    _extract_single_field는 봇의 질문을 제외한 고객 히스토리만 보므로, "0433"이나
    "2025년 12월 15일" 같은 맨값 답변이 무슨 항목인지 맥락을 못 잡아 '없음'으로 흘리기 쉽다.
    이 함수는 "고객이 방금 이 항목을 묻는 질문에 답했다"는 프레이밍으로 현재 메시지에서
    값을 직접 뽑는다. 답변이 항목과 무관하면 None을 반환한다.
    """
    system_message = SystemMessage(content=(
        f"고객이 '{field_label}'을(를) 묻는 질문을 받고 답했습니다. "
        f"고객 답변에서 '{field_label}'에 해당하는 값만 정확히 뽑아 그대로 출력하세요. "
        f"설명·문장 없이 값만. 답변이 '{field_label}'과(와) 전혀 무관하면 '없음'이라고만 답하세요."
    ))
    human_message = HumanMessage(
        content=f"질문 항목: {field_label}\n고객 답변: \"{current_message}\"\n\n'{field_label}' 값:"
    )
    try:
        val = llm.invoke([system_message, human_message]).content.strip()
        if not val or val.lower() in ["없음", "null", "none", "n/a", "-"]:
            return None
        return val
    except Exception as e:
        logger.error(f"슬롯 답변 추출 오류: {str(e)}")
        return None


def _extract_info_from_conversation(
    conversation_history: List[Dict],
    current_message: str,
    collected_info: Dict[str, Any],
    required_slots: List[str],
    slot_loader
) -> Dict[str, Any]:
    """대화 전체 히스토리에서 필요한 슬롯 정보를 추출합니다.

    각 슬롯을 개별적으로 추출하여 JSON 파싱 실패 위험을 줄입니다.
    이미 수집된 슬롯은 건너뜁니다.
    """
    if collected_info is None:
        collected_info = {}

    updated_info = collected_info.copy()

    # 아직 수집되지 않은 슬롯만 추출 시도
    for slot_name in required_slots:
        # 이미 값이 있으면 건너뜀
        if updated_info.get(slot_name):
            continue
        # inquiry_detail(자유서술 문의내용)은 자동 추출하지 않는다.
        # 핸드오버 트리거 문장("상담원 연결")에서 잘못 추출돼 즉시 완료되는 것을 막고,
        # 고객이 질문에 실제로 답한 내용을 노드에서 직접 캡처한다. (direction b)
        if slot_name == "inquiry_detail":
            continue

        # 슬롯 라벨 가져오기
        slot_label = slot_loader.get_slot_label(slot_name)

        # 단일 슬롯 추출
        extracted_value = _extract_single_field(
            conversation_history,
            current_message,
            slot_name,
            slot_label
        )

        if extracted_value:
            updated_info[slot_name] = extracted_value
            logger.info(f"슬롯 '{slot_name}' 수집 완료: {extracted_value}")

    return updated_info


def _generate_collection_question(
    missing_slots: List[str],
    collected_info: Dict[str, Any],
    slot_loader,
    user_message: str = "",
    current_category: str = "",
    is_first_entry: bool = False
) -> str:
    """부족한 슬롯에 대한 질문을 생성합니다.

    고객이 다른 문의를 한 경우, 정중하게 안내하면서 현재 진행 중인 슬롯 수집을 계속합니다.

    Args:
        is_first_entry: 첫 번째 waiting_agent 진입 여부. True면 단순 질문만 반환.
    """
    if not missing_slots:
        return "필요한 정보를 모두 수집했습니다."

    # 첫 번째 부족한 슬롯에 대해 질문
    next_slot = missing_slots[0]
    base_question = slot_loader.get_slot_question(next_slot)
    slot_label = slot_loader.get_slot_label(next_slot)

    # 첫 번째 진입 시에는 단순 질문만 반환 (intro_message와 함께 사용됨)
    if is_first_entry:
        return base_question

    # 이미 수집된 정보가 있으면 맥락에 맞게 질문 생성
    collected_labels = []
    for k, v in collected_info.items():
        if v and not k.startswith("_"):  # 내부 필드(_로 시작)는 제외
            label = slot_loader.get_slot_label(k)
            collected_labels.append(f"{label}: {v}")

    collected_summary = ", ".join(collected_labels)

    # 고객이 다른 문의를 했는지 감지하여 정중한 응대 생성
    system_message = SystemMessage(content=f"""당신은 친절하고 공감 능력이 뛰어난 금융 고객 상담 챗봇입니다.
현재 '{current_category}' 관련 상담사 연결을 위해 고객 정보를 수집하고 있습니다.

고객의 현재 메시지: "{user_message}"

당신의 역할:
1. 고객의 메시지가 카드사 업무와 관련이 있는지 먼저 판단하세요.
2. 카드사 업무 범위: 카드 발급/분실/도난, 결제/청구, 한도, 포인트/마일리지, 대출, 연회비, 할부 등
3. 카드사 업무와 완전히 무관한 질문(음식 주문, 날씨, 택배 등)에는 "죄송합니다. 저는 카드 관련 상담만 도와드릴 수 있습니다."라고 안내하세요.
4. 카드사 업무와 관련 있지만 현재 진행 중인 '{current_category}'와 다른 문의인 경우, 상담사 연결 후 안내해 드리겠다고 안내하세요.
5. 그 후 자연스럽게 현재 필요한 정보 수집을 계속하세요.

응답 규칙:
1. 항상 친절하고 공손한 어조를 유지하세요.
2. 이모티콘, 감탄사, 짧은 반응(ㅋㅋ, ㅎㅎ, ㅠㅠ, 아, 음, 네, 예 등)은 무시하고 바로 다음 질문만 하세요.
3. 카드사 업무 외 질문: "죄송합니다. 저는 카드 관련 상담만 도와드릴 수 있습니다. {base_question}"
4. 카드 관련 다른 문의: "말씀하신 [문의 내용]에 대해서는 상담사 연결 후 자세히 안내해 드리겠습니다. [다음 질문]"
5. 2-3문장으로 간결하게 작성하세요.
6. 고객의 메시지가 현재 카테고리와 관련 있거나 슬롯 정보 제공이면, 단순히 다음 정보를 요청하세요.

예시 (이모티콘/감탄사/짧은 반응):
고객: "ㅋㅋㅋ" → "{base_question}"
고객: "ㅠㅠ" → "{base_question}"
고객: "아..." → "{base_question}"

예시 (카드사 업무 외 질문):
"죄송합니다. 저는 카드 관련 상담만 도와드릴 수 있습니다. {base_question}"

예시 (카드 관련 다른 문의):
"결제일에 대해 궁금하신 부분은 상담사 연결 후 자세히 안내해 드리겠습니다. 먼저 {base_question}"

예시 (관련 있는 경우):
"감사합니다. {base_question}"
""")

    human_message = HumanMessage(content=f"""현재 카테고리: {current_category}
이미 수집된 정보: {collected_summary if collected_summary else "없음"}
다음으로 수집할 정보: {slot_label}
기본 질문: {base_question}
고객의 현재 메시지: {user_message}

고객에게 응답할 메시지를 생성해주세요.""")

    try:
        response = llm.invoke([system_message, human_message])
        return response.content.strip()
    except Exception as e:
        logger.error(f"질문 생성 중 오류: {str(e)}")
        return base_question


def _determine_category(state: GraphState) -> str:
    """상태에서 카테고리를 결정합니다.

    우선순위:
    1. context_intent (triage_agent에서 분류한 카테고리) - slot_definitions에 있는 경우만
    2. 대화 내용에서 키워드 기반 카테고리 추론
    3. collected_info.inquiry_detail (기존 방식 호환)
    4. 기본값: "기타 문의"
    """
    slot_loader = get_slot_loader()

    # 1. context_intent 확인 - slot_definitions에 있는 카테고리인 경우만 사용
    context_intent = state.get("context_intent")
    if context_intent and context_intent != "기타":
        # LABEL_X 형태이거나 slot_definitions에 없으면 무시
        if not context_intent.startswith("LABEL_"):
            domain_code = slot_loader.get_domain_by_category(context_intent)
            if domain_code:
                return context_intent

    # 2. 대화 내용에서 키워드 기반 카테고리 추론
    conversation_history = state.get("conversation_history", [])
    user_message = state.get("user_message", "")

    # 모든 사용자 메시지 합치기
    all_user_text = user_message + " " + " ".join([
        msg.get("message", "") for msg in conversation_history
        if msg.get("role") == "user"
    ])
    all_user_text_lower = all_user_text.lower()

    # 키워드 기반 카테고리 매핑 (slot_definitions.json의 38개 카테고리명과 일치)
    keyword_category_map = {
        # SEC_CARD (인증/보안/카드관리)
        ("분실", "도난", "잃어버", "카드를 잃"): "도난/분실 신청/해제",
        ("긴급 배송", "빨리 받고", "급하게 카드"): "긴급 배송 신청",
        # LIMIT_AUTH (한도/승인)
        ("한도 안내", "한도 조회", "한도가 얼마"): "한도 안내",
        ("한도 상향", "한도 올려", "한도 증액"): "한도상향 접수/처리",
        ("승인취소", "매출취소", "결제취소"): "승인취소/매출취소 안내",
        ("심사", "진행사항", "심사 결과"): "심사 진행사항 안내",
        ("신용공여", "공여기간"): "신용공여기간 안내",
        # PAY_BILL (결제/청구)
        ("결제대금", "이번달 결제", "얼마 나왔"): "결제대금 안내",
        ("결제일 변경", "결제일 안내"): "결제일 안내/변경",
        ("결제계좌", "계좌 변경"): "결제계좌 안내/변경",
        ("이용내역", "사용내역"): "이용내역 안내",
        ("입금내역", "입금 확인"): "입금내역 안내",
        ("가상계좌 안내", "가상계좌 번호"): "가상계좌 안내",
        ("가상계좌 예약", "가상계좌 취소"): "가상계좌 예약/취소",
        ("청구지", "청구서 주소"): "청구지 안내/변경",
        ("쇼핑케어", "구매안심"): "쇼핑케어",
        ("매출구분", "일시불 할부", "할부 변경"): "매출구분 변경",
        ("이용방법", "어떻게 사용"): "이용방법 안내",
        # DELINQ (연체/수납)
        ("연체대금", "연체 금액", "연체료"): "연체대금 안내",
        ("연체 출금", "연체금 즉시"): "연체대금 즉시출금",
        ("선결제", "즉시출금", "미리 결제"): "선결제/즉시출금",
        ("이월약정 안내", "리볼빙 안내"): "일부결제대금이월약정 안내",
        ("이월약정 해지", "리볼빙 해지"): "일부결제대금이월약정 해지",
        # LOAN (대출)
        ("단기카드대출", "현금서비스", "단기대출"): "단기카드대출 안내/실행",
        ("장기카드대출", "카드론", "장기대출"): "장기카드대출 안내",
        ("오토할부", "오토캐쉬백"): "오토할부/오토캐쉬백 안내/신청/취소",
        # BENEFIT (포인트/혜택/바우처)
        ("포인트", "마일리지", "적립금"): "포인트/마일리지 안내",
        ("포인트 전환", "마일리지 전환"): "포인트/마일리지 전환등록",
        ("정부지원", "바우처", "등유", "임신"): "정부지원 바우처 (등유, 임신 등)",
        ("프리미엄 바우처",): "프리미엄 바우처 안내/발급",
        ("이벤트", "행사", "프로모션"): "이벤트 안내",
        ("연회비", "연회비 안내"): "연회비 안내",
        # DOC_TAX (증명/세금)
        ("증명서", "확인서", "발급"): "증명서/확인서 발급",
        ("교육비", "학비"): "교육비",
        ("금리인하", "금리인하요구권"): "금리인하요구권 안내/신청",
        # UTILITY (공과금)
        ("도시가스", "가스요금"): "도시가스",
        ("전화요금", "통신요금"): "전화요금",
    }

    for keywords, category in keyword_category_map.items():
        if any(kw in all_user_text_lower for kw in keywords):
            logger.info(f"키워드 기반 카테고리 결정: {category}")
            return category

    # 3. collected_info에서 확인
    collected_info = state.get("collected_info", {})
    inquiry_detail = collected_info.get("inquiry_detail")
    if inquiry_detail and inquiry_detail != "기타 문의":
        domain_code = slot_loader.get_domain_by_category(inquiry_detail)
        if domain_code:
            return inquiry_detail

    # 4. 기본값
    return "기타 문의"


def waiting_agent_node(state: GraphState) -> GraphState:
    """HUMAN_REQUIRED 플로우에서 정보 수집을 담당하는 노드.

    동의 확인은 consent_check_node에서 처리되므로,
    이 노드에 진입했다는 것은 이미 동의한 상태입니다.

    역할:
    1. 카테고리(context_intent)에 따른 필수 슬롯 결정
    2. 대화 전체 히스토리에서 필요한 정보 추출
    3. 아직 수집되지 않은 정보에 대해 질문 생성
    4. 모든 필수 슬롯 수집 완료 시 info_collection_complete 플래그 설정
    """
    user_message = state.get("user_message", "")
    session_id = state.get("session_id", "unknown")
    conversation_history = state.get("conversation_history", [])
    unclear_count = state.get("unclear_count", 0)

    # 수집된 정보 가져오기 (없으면 빈 딕셔너리)
    collected_info: Dict[str, Any] = state.get("collected_info", {})

    logger.info(f"Waiting Agent 실행 - 세션: {session_id}")

    # ========== 정보 수집 완료 후 대기 중 추가 메시지 처리 ==========
    # info_collection_complete=True 상태에서 추가 메시지가 오면 대기 안내만 출력
    info_collection_complete = state.get("info_collection_complete", False)
    if info_collection_complete:
        logger.info(f"정보 수집 완료 상태에서 추가 메시지 - 세션: {session_id}, 메시지: {user_message}")

        # 거부 패턴 체크 (대기 중에도 취소 가능)
        if _check_collection_cancel(user_message):
            logger.info(f"대기 중 거부 감지 - 세션: {session_id}")
            state["is_human_required_flow"] = False
            state["customer_consent_received"] = False
            state["info_collection_complete"] = False
            state["handover_status"] = "cancelled"
            state["ai_message"] = "알겠습니다. 상담사 연결을 취소했습니다. 다른 도움이 필요하시면 말씀해 주세요."
            state["source_documents"] = []
            state["collected_info"] = {}
            return state

        # 대기 중 안내 메시지 (완료 메시지 반복 방지)
        state["ai_message"] = "현재 상담사 연결을 기다리고 있습니다. 잠시만 기다려 주세요. 상담사가 곧 연결될 예정입니다."
        state["source_documents"] = []
        return state

    # ========== 불명확 응답 3회 이상 시 세션 종료 ==========
    if _is_unclear_response(user_message):
        unclear_count += 1
        state["unclear_count"] = unclear_count

        if unclear_count >= MAX_UNCLEAR_COUNT:
            # 3회 이상 불명확 응답 → 세션 종료
            state["is_human_required_flow"] = False
            state["customer_consent_received"] = False
            state["info_collection_complete"] = False
            state["is_session_end"] = True
            state["ai_message"] = "죄송합니다. 명확한 응답을 확인하지 못해 상담을 종료합니다. 추가 문의가 있으시면 다시 연락해 주세요. 감사합니다."
            state["source_documents"] = []
            logger.info(f"불명확 응답 {unclear_count}회 초과 - 세션: {session_id}, 세션 종료")
            return state
        else:
            # 불명확 응답이지만 아직 3회 미만 → 재질문 메시지 반환
            logger.info(f"불명확 응답 감지 ({unclear_count}회) - 세션: {session_id}, 메시지: {user_message}")
            remaining = MAX_UNCLEAR_COUNT - unclear_count
            state["ai_message"] = f"죄송합니다. 말씀을 정확히 이해하지 못했습니다. 다시 한 번 말씀해 주시겠어요?"
            state["source_documents"] = []
            return state

    # ========== 슬롯 수집 중/완료 후 거부 감지 ==========
    # "아니요 그냥 됐어요", "취소할게요" 등의 거부 의사 감지 시 플로우 종료
    if _check_collection_cancel(user_message):
        logger.info(f"슬롯 수집 중 거부 감지 - 세션: {session_id}, 메시지: {user_message}")

        # 플로우 상태 초기화
        state["is_human_required_flow"] = False
        state["customer_consent_received"] = False
        state["info_collection_complete"] = False
        state["handover_status"] = "cancelled"  # 취소 상태로 설정

        # 거부 메시지 설정
        state["ai_message"] = "알겠습니다. 상담사 연결을 취소했습니다. 다른 도움이 필요하시면 말씀해 주세요."
        state["source_documents"] = []

        # collected_info 초기화 (내부 플래그 제거)
        state["collected_info"] = {}

        return state

    try:
        # 슬롯 로더 가져오기
        slot_loader = get_slot_loader()

        # 1. 카테고리 결정
        category = _determine_category(state)
        logger.info(f"카테고리 결정: {category}")

        # 도메인 정보 저장 (UI 표시용)
        domain_code = slot_loader.get_domain_by_category(category)
        domain_name = slot_loader.get_domain_name(domain_code) if domain_code else "기타"

        # collected_info에 도메인/카테고리 정보 저장
        collected_info["_domain_code"] = domain_code or "_DEFAULT"
        collected_info["_domain_name"] = domain_name
        collected_info["_category"] = category

        # 2. 필수 슬롯 결정 (호환성 필드 세팅 전에 계산해야 함)
        required_slots, optional_slots = slot_loader.get_slots_for_category(category)
        logger.info(f"필수 슬롯: {required_slots}, 선택 슬롯: {optional_slots}")

        # 기존 호환성을 위해 inquiry_type, inquiry_detail도 설정
        # 단, inquiry_detail이 '수집해야 할 필수 슬롯'인 경우(예: '기타 문의')에는
        # 카테고리명으로 자동 채우지 않는다 → 고객에게 실제로 질문해서 받는다. (direction b)
        if not collected_info.get("inquiry_type"):
            collected_info["inquiry_type"] = domain_name
        if not collected_info.get("inquiry_detail") and "inquiry_detail" not in required_slots:
            collected_info["inquiry_detail"] = category

        # 2-1. "모르겠어요" 패턴 감지 시 현재 질문 중인 슬롯에 "미확인" 저장
        current_asking_slot = collected_info.get("_current_asking_slot")
        if current_asking_slot and _check_unknown_response(user_message):
            # 해당 슬롯에 "미확인" 값 저장 → 다음 슬롯으로 진행
            collected_info[current_asking_slot] = "미확인"
            logger.info(f"'모르겠어요' 패턴 감지 - 슬롯 '{current_asking_slot}'에 '미확인' 저장")

        # 2-2. 현재 질문 중인 슬롯에 대한 답변을 '현재 메시지'에서 직접 캡처한다.
        #      히스토리 기반 자동추출(_extract_single_field)은 봇 질문을 제외해 맥락이 없어
        #      "0433"·날짜 같은 맨값을 '없음'으로 흘리기 쉽다 → 같은 질문을 반복하게 됨.
        #      질문받은 슬롯은 현재 메시지를 그 답변으로 보고 focused 추출/캡처한다.
        if (
            current_asking_slot
            and not collected_info.get(current_asking_slot)
            and not _check_unknown_response(user_message)
            and user_message and len(user_message.strip()) >= 1
        ):
            if current_asking_slot == "inquiry_detail":
                captured = user_message.strip()  # 자유서술 문의내용은 그대로
            else:
                captured = _extract_answer_for_slot(
                    user_message, slot_loader.get_slot_label(current_asking_slot)
                )
            if captured:
                collected_info[current_asking_slot] = captured
                logger.info(
                    f"질문 슬롯 '{current_asking_slot}' 답변 직접 캡처 - 세션: {session_id}, 값: {captured}"
                )

        # 3. 대화 히스토리에서 정보 추출
        collected_info = _extract_info_from_conversation(
            conversation_history,
            user_message,
            collected_info,
            required_slots,
            slot_loader
        )
        state["collected_info"] = collected_info

        logger.info(f"정보 추출 완료 - 세션: {session_id}, 수집된 정보: {collected_info}")

        # 4. 부족한 슬롯 확인
        missing_slots = slot_loader.get_missing_required_slots(category, collected_info)

        # 5. 모든 정보 수집 완료 여부 확인
        # 첫 번째 waiting_agent 진입인지 확인 (안내 메시지 표시 여부)
        # _waiting_intro_shown 플래그로 안내 메시지가 이미 표시되었는지 추적
        waiting_intro_shown = collected_info.get("_waiting_intro_shown", False)

        if not missing_slots:
            # 모든 정보 수집 완료 - 상담사 대기 중임을 안내
            # 시나리오: 상담사가 수락하면 고객에게 연결 확인 모달이 뜨므로 여기서는 대기 안내만
            state["info_collection_complete"] = True
            state["ai_message"] = "감사합니다. 필요한 정보가 모두 확인되었습니다. 상담사 연결을 기다리고 계세요."
            # handover_status를 pending으로 유지 (voice-chatbot에서 폴링 시작 트리거)
            state["handover_status"] = "pending"
            logger.info(f"정보 수집 완료 - 세션: {session_id}, 수집된 정보: {collected_info}, handover_status: pending")
        else:
            # 부족한 정보에 대해 질문 생성
            state["info_collection_complete"] = False

            # 현재 질문할 슬롯 저장 (다음 턴에서 "모르겠어요" 패턴 감지 시 사용)
            next_asking_slot = missing_slots[0] if missing_slots else None
            if next_asking_slot:
                collected_info["_current_asking_slot"] = next_asking_slot
                state["collected_info"] = collected_info

            if not waiting_intro_shown:
                # 첫 번째 waiting_agent 진입 - 단순 질문만 반환 (intro_message와 함께)
                question = _generate_collection_question(
                    missing_slots,
                    collected_info,
                    slot_loader,
                    user_message=user_message,
                    current_category=category,
                    is_first_entry=True  # 첫 진입 시 LLM 없이 단순 질문
                )
                # 상담사 확인 안내 메시지 + 질문
                intro_message = "현재 응대 가능한 상담사가 있는지 확인해 보겠습니다. 잠시만 기다려 주시기 바랍니다."
                state["ai_message"] = f"{intro_message} {question}"
                # 안내 메시지 표시 완료 플래그 설정
                collected_info["_waiting_intro_shown"] = True
                state["collected_info"] = collected_info
                # 첫 번째 슬롯 수집 시작 = 상담사 대기 시작 (handover_status를 pending으로)
                state["handover_status"] = "pending"
                logger.info(f"핸드오버 대기 시작 - 세션: {session_id}, handover_status: pending, 질문 슬롯: {next_asking_slot}")
            else:
                # 직전에 물어본 슬롯이 방금 채워졌으면(정상 답변) → 다음 슬롯 질문만 간단히 잇는다.
                # (LLM 문맥해석을 태우면 "보안상 말할 수 없다"류 엉뚱한 응답이 나오기 쉬움)
                # 채워지지 않았으면(무관/오답) → 문맥 기반 LLM 응답으로 정중히 안내한다.
                prev_slot_filled = bool(current_asking_slot and collected_info.get(current_asking_slot))
                question = _generate_collection_question(
                    missing_slots,
                    collected_info,
                    slot_loader,
                    user_message=user_message,
                    current_category=category,
                    is_first_entry=prev_slot_filled,  # 정상 진행이면 단순 다음 질문
                )
                state["ai_message"] = f"감사합니다. {question}" if prev_slot_filled else question

            logger.info(f"정보 수집 질문 생성 - 세션: {session_id}, 부족한 슬롯: {missing_slots}, 현재 질문 슬롯: {next_asking_slot}")

        # 6. source_documents는 빈 리스트로 설정 (정보 수집에는 RAG 사용 안 함)
        state["source_documents"] = []

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Waiting Agent 오류 - 세션: {session_id}, 오류: {error_msg}", exc_info=True)

        # 에러 발생 시 기본 메시지
        state["ai_message"] = "죄송합니다. 정보 수집 중 오류가 발생했습니다. 상담사에게 연결해 드리겠습니다."
        state["info_collection_complete"] = True  # 에러 시 상담사 연결로 이동
        state["source_documents"] = []

        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["waiting_agent_error"] = error_msg

    return state
