import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import chat, handover, session, voice, voice_ws
from app.core.database import init_db, engine
from app.core.config import settings
from app.services.session_manager import session_manager
from sqlalchemy import text

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bank AICC Dev Server",
    description="AI 상담 챗봇 서버 - LangGraph 기반 워크플로우",
    version="1.0.0"
)

# 무활동 세션 정리 태스크
inactivity_cleanup_task: Optional[asyncio.Task] = None

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 데이터베이스 초기화 (테이블 생성)
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 DB 테이블 생성 및 연결 확인"""
    import os

    # LangSmith 트레이싱 확인
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        logger.info(f"✅ LangSmith 추적 활성화됨 - 프로젝트: {project}")

    try:
        logger.info("데이터베이스 초기화 시작...")
               
        # 비동기로 실행하여 블로킹 방지
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        # DB 초기화를 별도 스레드에서 실행 (타임아웃 10초)
        try:
            await asyncio.wait_for(
                loop.run_in_executor(executor, init_db),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("⚠️ 데이터베이스 초기화 타임아웃 (10초) - 서버는 계속 시작됩니다")
        except Exception as e:
            logger.warning(f"⚠️ 데이터베이스 초기화 실패: {str(e)} - 서버는 계속 시작됩니다")
        
        # DB 연결 확인 (타임아웃 5초)
        def _check_db_connection():
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.fetchone()
        
        try:
            await asyncio.wait_for(
                loop.run_in_executor(executor, _check_db_connection),
                timeout=5.0
            )
            logger.info("✅ 데이터베이스 연결 성공")
        except asyncio.TimeoutError:
            logger.warning("⚠️ 데이터베이스 연결 확인 타임아웃 (5초) - MySQL이 실행 중인지 확인하세요")
            logger.warning("   서버는 계속 시작되지만 데이터베이스 기능은 사용할 수 없습니다")
        except Exception as e:
            logger.warning(f"⚠️ 데이터베이스 연결 확인 실패: {str(e)}")
            logger.warning("   서버는 계속 시작되지만 데이터베이스 기능은 사용할 수 없습니다")
        
        executor.shutdown(wait=False)
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}", exc_info=True)
        # DB 연결 실패해도 서버는 시작 (나중에 재시도 가능)
    
    # LLM 설정 확인 (키 값은 절대 로깅하지 않음)
    from app.core.llm import provider_label, _is_local_provider
    logger.info(f"✅ LLM provider: {provider_label()}")
    if _is_local_provider():
        logger.info("   로컬 LLM 서버가 실행 중인지 확인하세요 (예: Ollama → http://localhost:11434)")
    elif not settings.openai_api_key:
        logger.error("❌ OpenAI API 키가 설정되지 않았습니다! .env 에 OPENAI_API_KEY 를 추가하거나 LLM_PROVIDER=ollama 로 설정하세요.")
    
    # 벡터 DB 초기화 확인 (비동기, 타임아웃 30초)
    try:
        def _init_vector_store():
            from ai_engine.vector_store import get_vector_store
            return get_vector_store()
        
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        try:
            vector_store = await asyncio.wait_for(
                loop.run_in_executor(executor, _init_vector_store),
                timeout=30.0
            )
            logger.info(f"✅ 벡터 DB 초기화 완료 - 경로: {settings.vector_db_path}, 컬렉션: {settings.collection_name}")
        except asyncio.TimeoutError:
            logger.warning("⚠️ 벡터 DB 초기화 타임아웃 (30초) - RAG 검색은 빈 결과 반환될 수 있습니다")
        except Exception as e:
            logger.warning(f"⚠️ 벡터 DB 초기화 실패 (RAG 검색은 빈 결과 반환): {str(e)}")
            logger.warning("   벡터 DB는 선택사항입니다. 문서가 없으면 RAG 검색 결과가 비어있을 수 있습니다.")
        finally:
            executor.shutdown(wait=False)
    except Exception as e:
        logger.warning(f"⚠️ 벡터 DB 초기화 중 오류: {str(e)}")
        logger.warning("   벡터 DB는 선택사항입니다. 문서가 없으면 RAG 검색 결과가 비어있을 수 있습니다.")
    
    # Final Classifier 모델 확인 (LoRA 기반 KcELECTRA) - 비동기, 타임아웃 60초
    try:
        def _init_classifier():
            from ai_engine.ingestion.bert_financial_intent_classifier.scripts.inference import IntentClassifier
            return IntentClassifier()
        
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        try:
            classifier = await asyncio.wait_for(
                loop.run_in_executor(executor, _init_classifier),
                timeout=60.0
            )
            if classifier is not None:
                logger.info("✅ Final Classifier 의도 분류 모델 로드 완료 (38개 카테고리)")
            else:
                logger.warning("⚠️ Final Classifier 모델을 찾을 수 없습니다.")
                logger.warning("   모델 위치: fa06-fin-aicc/models/final_classifier_model/model_final/ 폴더를 확인하세요.")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Final Classifier 모델 로드 타임아웃 (60초) - 의도 분류 기능이 제한될 수 있습니다")
            logger.warning("   모델 위치: fa06-fin-aicc/models/final_classifier_model/model_final/ 폴더를 확인하세요.")
        except Exception as e:
            logger.warning(f"⚠️ Final Classifier 모델 로드 실패: {str(e)}")
            logger.warning("   모델 위치: fa06-fin-aicc/models/final_classifier_model/model_final/ 폴더를 확인하세요.")
        finally:
            executor.shutdown(wait=False)
    except Exception as e:
        logger.warning(f"⚠️ Final Classifier 모델 로드 중 오류: {str(e)}")
        logger.warning("   모델 위치: fa06-fin-aicc/models/final_classifier_model/model_final/ 폴더를 확인하세요.")
    
    # STT/TTS 음성 서비스 설정 확인
    logger.info("음성 서비스 설정 확인 중...")
    
    # VITO STT 설정 확인
    if settings.vito_client_id and settings.vito_client_secret:
        logger.info(f"✅ VITO STT 설정 확인 완료 (Client ID: {settings.vito_client_id[:8]}...)")
    else:
        logger.warning("⚠️ VITO STT 설정이 없습니다. 음성 입력 기능을 사용하려면 설정이 필요합니다.")
        logger.warning("   .env 파일에 VITO_CLIENT_ID와 VITO_CLIENT_SECRET을 추가하세요.")
        logger.warning("   발급: https://developers.vito.ai/")
    
    # Google TTS 설정 확인 (메인 TTS 서비스)
    import os
    google_tts_api_key = os.getenv("GOOGLE_TTS_API_KEY") or os.getenv("GEM_API_KEY")
    google_tts_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    google_tts_voice = os.getenv("GOOGLE_TTS_VOICE", "ko-KR-Neural2-B")
    google_tts_language = os.getenv("GOOGLE_TTS_LANGUAGE", "ko-KR")
    
    if google_tts_api_key or google_tts_credentials:
        logger.info(f"✅ Google TTS 설정 확인 완료 (voice: {google_tts_voice}, language: {google_tts_language})")
    else:
        logger.error("❌ Google TTS 설정이 없습니다. TTS 기능을 사용할 수 없습니다.")
        logger.error("   .env 파일에 GOOGLE_TTS_API_KEY 또는 GOOGLE_APPLICATION_CREDENTIALS를 추가하세요.")
        logger.error("   Google TTS API 키 발급: https://ai.google.dev/")
    
    # OpenAI TTS 설정 확인 (선택사항) - 주석 처리됨 (Google TTS 사용)
    # if settings.openai_api_key:
    #     logger.info(f"ℹ️  OpenAI TTS 설정 확인 완료 (voice: {settings.tts_voice}, model: {settings.tts_model}) - 현재 미사용")
    # else:
    #     logger.debug("OpenAI TTS 설정 없음 (현재 미사용)")

    # Reranker 모델 사전 로딩 (첫 요청 지연 방지) - 비동기, 타임아웃 60초
    if settings.enable_reranking:
        try:
            def _init_reranker():
                from sentence_transformers import CrossEncoder
                logger.info(f"Reranker 모델 로딩 중... ({settings.reranker_model})")
                return CrossEncoder(settings.reranker_model)

            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)

            try:
                reranker_model = await asyncio.wait_for(
                    loop.run_in_executor(executor, _init_reranker),
                    timeout=60.0
                )
                # 모델을 _rerank_documents 함수에 캐시
                from ai_engine.vector_store import _rerank_documents
                _rerank_documents._model = reranker_model
                logger.info(f"✅ Reranker 모델 사전 로딩 완료 ({settings.reranker_model})")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Reranker 모델 로드 타임아웃 (60초) - 첫 요청 시 로드됩니다")
            except Exception as e:
                logger.warning(f"⚠️ Reranker 모델 사전 로딩 실패: {str(e)} - 첫 요청 시 로드됩니다")
            finally:
                executor.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"⚠️ Reranker 모델 사전 로딩 중 오류: {str(e)}")

    # 무활동 세션 정리 태스크 시작 (10분 무활동, 60초 주기)
    global inactivity_cleanup_task
    if inactivity_cleanup_task is None:
        inactivity_cleanup_task = asyncio.create_task(_inactive_session_cleanup_worker())
        logger.info("무활동 세션 정리 태스크 시작 (10분 무활동 시 is_active=0, 60초 주기)")


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리 작업"""
    logger.info("애플리케이션 종료 중...")
    global inactivity_cleanup_task
    if inactivity_cleanup_task:
        inactivity_cleanup_task.cancel()
        inactivity_cleanup_task = None


# 라우터 등록 (부품 조립)
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(handover.router, prefix="/api/v1/handover", tags=["Handover"])
app.include_router(session.router, prefix="/api/v1/sessions", tags=["Sessions"])
app.include_router(voice.router, prefix="/api/v1/voice", tags=["Voice"])
app.include_router(voice_ws.router, prefix="/api/v1/voice", tags=["Voice WebSocket"])


@app.get("/")
def root():
    """루트 엔드포인트 - 서버 상태 확인"""
    return {
        "status": "Server is running properly!",
        "service": "Bank AICC Dev Server",
        "version": "1.0.0"
    }


async def _inactive_session_cleanup_worker():
    """무활동 세션을 주기적으로 비활성화"""
    while True:
        try:
            session_manager.deactivate_inactive_sessions(threshold_minutes=10)
        except Exception as e:
            logger.error(f"무활동 세션 정리 태스크 오류: {str(e)}", exc_info=True)
        await asyncio.sleep(60)


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트 - 서버 및 DB 상태 확인"""
    try:
        # DB 연결 확인
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "database": "connected",
                "service": "Bank AICC Dev Server"
            }
        )
    except Exception as e:
        logger.error(f"헬스체크 실패: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )