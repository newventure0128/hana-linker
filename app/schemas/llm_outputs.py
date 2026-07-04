"""LLM 구조화 출력 스키마.

각 노드가 LLM에서 받아야 하는 JSON 구조를 Pydantic으로 정의한다.
app/core/llm.py:invoke_structured() 와 함께 사용한다.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class TriageResult(BaseModel):
    """triage_agent 의 분류 결과."""

    ticket: Literal[
        "SIMPLE_ANSWER", "AUTO_ANSWER", "NEED_MORE_INFO", "HUMAN_REQUIRED"
    ] = Field(description="네 가지 티켓 중 하나")
    reason: str = Field(description="내부 판단 근거 (1-2문장)")
    customer_intent_summary: str = Field(
        default="", description="고객의 핵심 의도 요약 (1-2문장)"
    )


class ConsentResult(BaseModel):
    """consent_check_node 의 동의 분류 결과."""

    classification: Literal["CONSENT", "REJECT", "OUT_OF_DOMAIN", "UNCLEAR"] = Field(
        description="고객 응답 분류"
    )
    ai_message: str = Field(
        default="", description="OUT_OF_DOMAIN/UNCLEAR 인 경우 고객에게 보낼 응답 메시지"
    )


class SummaryResult(BaseModel):
    """summary_agent 의 상담 요약 결과."""

    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description="고객 감정 상태"
    )
    summary: str = Field(description="상담 내용 요약 (3줄 정도)")
    keywords: List[str] = Field(default_factory=list, description="핵심 키워드 (최대 5개)")
