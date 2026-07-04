"""애플리케이션 설정
.env 파일에서만 설정을 로드합니다. 환경 변수는 무시합니다.
"""

from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


# .env 파일 경로
env_path = Path(__file__).parent.parent.parent / ".env"

# 환경 변수에서 OPENAI_API_KEY를 임시로 제거 (나중에 복원)
original_env_key = os.environ.pop("OPENAI_API_KEY", None)

# .env 파일에서 직접 읽기
env_values = {}
if env_path.exists():
    from dotenv import dotenv_values
    env_values = dotenv_values(env_path)
    # .env 파일의 값을 환경 변수에 설정 (다른 설정들도 읽을 수 있도록)
    for key, value in env_values.items():
        if value:  # 빈 값이 아닌 경우만
            os.environ[key] = value
else:
    # .env 파일이 없으면 기본 로드 시도
    load_dotenv(override=True)


class Settings(BaseSettings):
    """애플리케이션 설정
    
    .env 파일에서만 설정을 로드합니다.
    환경 변수는 무시됩니다.
    """
    
    # 애플리케이션 설정
    app_name: str = "Bank AICC Dev Server"
    debug: bool = False
    
    # 데이터베이스 설정
    database_url: str = "mysql+pymysql://root:password@localhost:3306/aicc_db?charset=utf8mb4"
    
    # OpenAI 설정
    openai_api_key: Optional[str] = None
    
    # ========== STT/TTS 음성 서비스 설정 ==========
    
    # VITO STT 설정 (Return Zero)
    # https://developers.vito.ai/ 에서 발급
    vito_client_id: Optional[str] = None
    vito_client_secret: Optional[str] = None
    vito_stt_timeout: int = 60  # STT 요청 타임아웃 (초)
    
    # OpenAI TTS 설정
    # 음성 종류: alloy, echo, fable, onyx, nova, shimmer
    tts_voice: str = "alloy"
    # 모델: tts-1 (빠름, 저품질), tts-1-hd (느림, 고품질)
    tts_model: str = "tts-1"
    tts_timeout: int = 30  # TTS 요청 타임아웃 (초)
    
    # ========== LLM 설정 ==========

    # 통합 LLM provider 설정 (app/core/llm.py 팩토리에서 사용)
    # provider: "openai" | "ollama" | "lm_studio" | "vllm"
    #   - openai  : OpenAI API (openai_model, openai_api_key 사용)
    #   - ollama  : 로컬 Ollama (llm_base_url=http://localhost:11434/v1)
    #   - lm_studio/vllm: OpenAI 호환 엔드포인트 (llm_base_url 지정)
    llm_provider: str = "openai"
    llm_base_url: str = "http://localhost:11434/v1"  # Ollama 기본 엔드포인트
    llm_model: str = "qwen3.5:9b"                    # 로컬 모델명
    llm_api_key: str = "ollama"                      # 로컬은 더미 키 (인증 불필요)
    # thinking 모델(qwen3.5 등)의 추론 토큰 제어: "none"이면 thinking 비활성화.
    # Ollama OpenAI 호환 엔드포인트는 "think" 필드를 무시하므로 reasoning_effort로 제어해야 함.
    llm_reasoning_effort: Optional[str] = None       # None이면 요청에 포함하지 않음
    openai_model: str = "gpt-4o-mini"                # provider=openai 일 때 사용할 모델

    # LM Studio 설정 (하위 호환 유지 - use_lm_studio=True면 아래 값 우선 사용)
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "openai/gpt-oss-20b"
    use_lm_studio: bool = False  # True면 LM Studio 사용 (하위 호환), 신규는 llm_provider 사용
    llm_timeout: int = 60  # LLM 호출 타임아웃 (초)

    # ========== LangSmith 설정 (프롬프트 디버깅용) ==========
    # LangSmith는 LangChain과 자동으로 통합됩니다.
    # 환경 변수만 설정하면 자동으로 추적이 시작됩니다.
    langsmith_api_key: Optional[str] = None  # LangSmith API 키
    langsmith_project: Optional[str] = None  # LangSmith 프로젝트 이름 (선택사항)
    langsmith_tracing: bool = False  # True면 LangSmith 추적 활성화

    # 벡터 DB 설정 (ChromaDB)
    vector_db_path: str = "./chroma_db"  # ChromaDB 저장 경로
    # 한국어 특화 임베딩 모델 (한국어 금융 문서에 최적화)
    embedding_model: str = "jhgan/ko-sroberta-multitask"  # 한국어 특화 모델
    # 다른 옵션: "BM-K/KoSimCSE-roberta-multitask" 또는 "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    collection_name: str = "financial_documents"  # ChromaDB 컬렉션 이름
    similarity_threshold: float = 0.2  # RAG 검색 유사도 임계값 (0.0 ~ 1.0, 이 값 이상이어야 챗봇으로 처리)
    enable_query_expansion: bool = True  # 쿼리 확장 활성화 (동의어, 관련어 추가)
    
    # Hybrid Search 설정 (벡터 검색 + BM25)
    enable_hybrid_search: bool = True  # Hybrid Search 활성화 여부
    # RRF (Reciprocal Rank Fusion) 사용 여부
    use_rrf: bool = True  # True면 RRF 사용, False면 가중치 결합 사용
    rrf_k: int = 30  # RRF 상수 k (60→30으로 낮춰서 상위 순위 문서 영향력 강화)
    # 가중치 결합 방식 (use_rrf=False일 때 사용)
    vector_search_weight: float = 0.6  # 벡터 검색 가중치 (0.0 ~ 1.0)
    bm25_search_weight: float = 0.4  # BM25 검색 가중치 (0.0 ~ 1.0, vector_weight + bm25_weight = 1.0 권장) - 구어체 키워드 매칭 강화
    # BM25 한국어 토크나이저
    # None: 기본 정규표현식 토크나이저
    # "kiwi": Kiwi 형태소 분석기 사용 (권장, 높은 정확도)
    # "konlpy": KoNLPy 형태소 분석기 (향후 지원)
    # "mecab": MeCab 형태소 분석기 (향후 지원)
    bm25_korean_tokenizer: Optional[str] = "kiwi"  # 기본값을 Kiwi로 설정
    
    # Reranking 설정
    enable_reranking: bool = True  # Reranking 활성화 (RAG Score 향상 목적)
    rerank_top_k: int = 10  # Reranking할 상위 문서 수 (5→10 확장)
    rerank_final_k: int = 5  # 최종 반환할 문서 수 (3→5 확장)
    # 한국어 최적화: 기본값을 한국어 reranker로 설정
    reranker_model: str = "Dongjin-kr/ko-reranker"  # 한국어 특화 Cross-Encoder 모델
    # 다른 옵션: "sigridjineth/ko-reranker-v1.1" (더 큰 모델, 더 정확하지만 느림)
    # 영어 모델: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # pydantic-settings는 자동으로 openai_api_key -> OPENAI_API_KEY로 변환합니다
        env_file_encoding = "utf-8"
        extra = "ignore"  # 추가 필드 무시


# 전역 설정 인스턴스
settings = Settings()

# .env 파일에서 직접 OPENAI_API_KEY를 읽어서 강제로 설정 (환경 변수 무시)
if env_path.exists() and "OPENAI_API_KEY" in env_values and env_values["OPENAI_API_KEY"]:
    settings.openai_api_key = env_values["OPENAI_API_KEY"]
    # 환경 변수도 .env 값으로 업데이트 (다른 모듈에서 사용할 수 있도록)
    os.environ["OPENAI_API_KEY"] = env_values["OPENAI_API_KEY"]
elif original_env_key:
    # .env 파일에 없으면 원래 환경 변수 복원 (하지만 사용하지 않음)
    os.environ["OPENAI_API_KEY"] = original_env_key

# LangSmith 초기화 (프롬프트 디버깅용)
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    print(f"✅ LangSmith 추적 활성화됨 - 프로젝트: {settings.langsmith_project or 'default'}")
elif settings.langsmith_tracing:
    print("⚠️ LangSmith 추적이 활성화되어 있지만 API 키가 설정되지 않았습니다.")