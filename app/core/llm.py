"""LLM 팩토리.

provider 설정(app/core/config.py)에 따라 OpenAI / Ollama / LM Studio / vLLM 용
ChatOpenAI 인스턴스를 생성한다.

⚠️ 각 노드는 직접 `ChatOpenAI(...)`를 만들지 말고 반드시 이 모듈을 사용한다.
   (과거 노드마다 LLM 초기화 블록이 복붙되어 consent_check_node 가 provider 분기를
    누락한 버그가 있었음 → 단일 팩토리로 통합)

구조화 출력은 `invoke_structured()`를 사용한다. 1순위로 json_schema 구조화 출력을
시도하고, 로컬 모델이 지원하지 않거나 실패하면 평문 호출 후 JSON 추출로 폴백한다.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Type, TypeVar

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 로컬(OpenAI 호환) provider 집합
_LOCAL_PROVIDERS = {"ollama", "lm_studio", "lmstudio", "local", "vllm"}


def _is_local_provider() -> bool:
    """로컬/OpenAI호환 엔드포인트 사용 여부."""
    if settings.use_lm_studio:  # 하위 호환
        return True
    return (settings.llm_provider or "openai").lower() in _LOCAL_PROVIDERS


def _local_kwargs() -> dict:
    """로컬 provider용 ChatOpenAI kwargs."""
    base_url = settings.llm_base_url
    model = settings.llm_model
    # use_lm_studio=True 하위 호환: lm_studio_* 값 우선
    if settings.use_lm_studio:
        base_url = settings.lm_studio_base_url
        model = settings.lm_studio_model
    kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key": settings.llm_api_key or "not-needed",
        "timeout": settings.llm_timeout,
    }
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    return kwargs


def get_chat_llm(temperature: float = 0.2, **overrides: Any) -> ChatOpenAI:
    """provider 설정에 맞는 ChatOpenAI 인스턴스를 반환한다.

    Args:
        temperature: 샘플링 온도
        **overrides: ChatOpenAI 생성자 추가/덮어쓰기 인자
    """
    if _is_local_provider():
        kwargs = _local_kwargs()
    else:
        if not settings.openai_api_key:
            raise ValueError(
                "❌ OpenAI API 키가 설정되지 않았습니다.\n"
                "   .env에 OPENAI_API_KEY=sk-... 를 추가하거나,\n"
                "   로컬 모델을 쓰려면 .env에 LLM_PROVIDER=ollama 를 설정하세요."
            )
        kwargs = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
            "timeout": settings.llm_timeout,
        }

    kwargs["temperature"] = temperature
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


def provider_label() -> str:
    """현재 LLM provider를 사람이 읽을 수 있는 문자열로 (로깅용, 키 노출 없음)."""
    if _is_local_provider():
        k = _local_kwargs()
        return f"LOCAL[{k['model']} @ {k['base_url']}]"
    return f"OpenAI[{settings.openai_model}]"


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _content_to_text(content: Any) -> str:
    """LangChain 메시지 content(문자열 또는 ContentBlock 리스트)를 텍스트로 평탄화."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for block in content:
            if isinstance(block, dict):
                pieces.append(block.get("text", ""))
            elif hasattr(block, "text"):
                pieces.append(getattr(block, "text") or "")
        return "".join(pieces)
    return str(content)


def invoke_structured(
    schema: Type[T],
    messages: List[Any],
    *,
    temperature: float = 0.2,
) -> T:
    """구조화 출력 호출 (provider 무관, robust).

    1순위: with_structured_output(method="json_schema")
    폴백:  평문 호출 → 응답에서 첫 JSON 객체 추출 → schema 검증

    Args:
        schema: 출력 Pydantic 모델
        messages: LangChain 메시지 리스트
        temperature: 샘플링 온도

    Returns:
        schema 인스턴스

    Raises:
        ValueError: 폴백에서도 JSON을 얻지 못한 경우
    """
    llm = get_chat_llm(temperature=temperature)

    # 1) json_schema 구조화 출력 시도
    try:
        structured = llm.with_structured_output(schema, method="json_schema")
        result = structured.invoke(messages)
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)
    except Exception as exc:
        logger.warning("structured output 실패 → 평문 JSON 파싱 폴백: %s", exc)

    # 2) 폴백: 평문 호출 + JSON 추출
    fallback_messages = list(messages)
    fallback_messages.append(
        HumanMessage(content="위 지시에 따라 JSON 객체 하나만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.")
    )
    raw = llm.invoke(fallback_messages)
    text = _content_to_text(getattr(raw, "content", raw))
    match = _JSON_RE.search(text or "")
    if not match:
        raise ValueError(f"LLM 응답에서 JSON 객체를 찾을 수 없습니다: {text!r}")
    data = json.loads(match.group(0))
    return schema.model_validate(data)
