# 채팅 담당 직원 - LangGraph 워크플로우 연결

import logging
import json
from typing import Dict
from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.workflow_service import process_chat_message

logger = logging.getLogger(__name__)
router = APIRouter()


# WebSocket 연결 관리자
class ConnectionManager:
    """WebSocket 연결을 관리하는 클래스"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        # 최근 생성된 리포트 캐시 (세션ID -> 리포트 데이터)
        self.recent_reports: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """WebSocket 연결 수락 및 등록"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket 연결 수립 - 세션: {session_id}, 현재 연결 수: {len(self.active_connections)}")
    
    def disconnect(self, session_id: str):
        """WebSocket 연결 해제"""
        if session_id in self.active_connections:
            self.active_connections.pop(session_id)
            logger.info(f"WebSocket 연결 해제 - 세션: {session_id}, 현재 연결 수: {len(self.active_connections)}")
    
    async def send_message(self, session_id: str, message: dict):
        """특정 세션에 메시지 전송"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"메시지 전송 실패 - 세션: {session_id}, 오류: {str(e)}")
                self.disconnect(session_id)
    
    async def send_error(self, session_id: str, error_message: str):
        """에러 메시지 전송"""
        await self.send_message(session_id, {
            "type": "error",
            "message": error_message
        })
    
    async def broadcast_to_consultants(self, message: dict):
        """상담원 대시보드로 브로드캐스트"""
        consultant_sessions = [sid for sid in self.active_connections.keys() if sid.startswith("consultant_dashboard")]
        logger.info(f"상담원 대시보드로 브로드캐스트 - 대상: {len(consultant_sessions)}개")
        
        # 리포트를 캐시에 저장 (최대 50개까지 유지)
        if message.get("type") == "handover_report":
            session_id = message.get("session_id")
            if session_id:
                self.recent_reports[session_id] = message
                # 최대 50개까지만 유지
                if len(self.recent_reports) > 50:
                    # 가장 오래된 항목 제거
                    oldest_key = next(iter(self.recent_reports))
                    self.recent_reports.pop(oldest_key)
                logger.debug(f"리포트 캐시 저장 - 세션: {session_id}, 캐시 크기: {len(self.recent_reports)}")
        
        for session_id in consultant_sessions:
            await self.send_message(session_id, message)
    
    async def send_recent_reports_to_consultant(self, consultant_session_id: str):
        """상담원 대시보드 연결 시 최근 리포트 전송"""
        if self.recent_reports:
            logger.info(f"상담원 대시보드에 최근 리포트 전송 - 대상: {consultant_session_id}, 리포트 수: {len(self.recent_reports)}")
            for session_id, report in self.recent_reports.items():
                await self.send_message(consultant_session_id, report)
        else:
            logger.info(f"전송할 최근 리포트 없음 - 대상: {consultant_session_id}")
    
    def is_connected(self, session_id: str) -> bool:
        """연결 상태 확인"""
        return session_id in self.active_connections


# 전역 연결 관리자 인스턴스
manager = ConnectionManager()


async def broadcast_handover_report(session_id: str) -> dict:
    """HANDOVER 완료 시 상담원 리포트를 생성해 대시보드로 브로드캐스트(+캐시)한다.

    전송 경로(REST /chat/message · 채팅 WS · 음성 WS)와 무관하게 재사용된다.
    broadcast_to_consultants가 recent_reports에도 캐시하므로, 대시보드가 아직
    접속 전이어도 이후 접속 시 리포트를 받는다.
    """
    from app.schemas.handover import HandoverRequest
    from app.services.workflow_service import process_handover

    handover_response = await process_handover(
        HandoverRequest(session_id=session_id, trigger_reason="AUTO_DETECTED")
    )

    report_data = {
        "type": "handover_report",
        "session_id": session_id,
        "data": {
            "status": handover_response.status,
            "customer_sentiment": handover_response.analysis_result.customer_sentiment.value,
            "summary": handover_response.analysis_result.summary,
            "extracted_keywords": handover_response.analysis_result.extracted_keywords,
            "kms_recommendations": [
                {
                    "title": rec.title,
                    "url": str(rec.url),
                    "relevance_score": rec.relevance_score,
                }
                for rec in handover_response.analysis_result.kms_recommendations
            ],
        },
    }

    logger.info(
        f"📦 상담원 리포트 브로드캐스트 - 세션: {session_id}, "
        f"sentiment={report_data['data']['customer_sentiment']}, "
        f"kms={len(report_data['data']['kms_recommendations'])}개"
    )
    await manager.broadcast_to_consultants(report_data)
    return report_data


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    고객 메시지를 받아 LangGraph 워크플로우를 통해 처리하고 응답을 반환
    
    - **session_id**: 세션 ID (필수)
    - **user_message**: 사용자 메시지 (필수)
    
    처리 흐름:
    1. DB에서 이전 대화 이력 로드
    2. LangGraph 워크플로우 실행
    3. DB에 메시지 저장
    4. 응답 반환
    """
    logger.info(f"=== API 엔드포인트 도달: /api/v1/chat/message ===")
    logger.info(f"요청 본문: session_id={request.session_id}, user_message={request.user_message[:50]}")
    try:
        # 입력 검증
        if not request.session_id or not request.session_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id는 필수입니다."
            )
        
        if not request.user_message or not request.user_message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_message는 필수입니다."
            )
        
        logger.info(f"채팅 메시지 수신 - 세션: {request.session_id}, 메시지: {request.user_message[:50]}")
        
        # 워크플로우 실행
        response = await process_chat_message(request)
        
        # 응답 메시지 확인
        if "오류" in response.ai_message or "error" in response.ai_message.lower() or "죄송합니다" in response.ai_message:
            logger.warning(f"채팅 메시지 처리 완료 (에러 응답) - 세션: {request.session_id}, 응답: {response.ai_message[:100]}")
        else:
            logger.info(f"채팅 메시지 처리 완료 - 세션: {request.session_id}, intent: {response.intent}, action: {response.suggested_action}")

        # 상담원 이관이 확정된 시점(action=HANDOVER)에만, 세션당 1회 대시보드로 리포트 브로드캐스트.
        # (handover_status="pending"은 수집 '시작' 때도 세팅되므로 게이트로 쓰면 조기 발화된다)
        if response.suggested_action.value == "HANDOVER" and request.session_id not in manager.recent_reports:
            try:
                await broadcast_handover_report(request.session_id)
            except Exception as e:
                logger.error(f"상담원 리포트 브로드캐스트 실패 - 세션: {request.session_id}: {e}", exc_info=True)

        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"채팅 메시지 처리 중 오류 발생 - 세션: {request.session_id}, 오류: {str(e)}", exc_info=True)
        # 에러 발생 시에도 응답 반환 (워크플로우 서비스에서 처리)
        return await process_chat_message(request)


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket 엔드포인트 - 실시간 양방향 통신
    
    - **session_id**: 세션 ID (URL 파라미터)
    
    메시지 포맷:
    - 클라이언트 → 서버:
      {
        "type": "message",
        "user_message": "사용자 메시지"
      }
    
    - 서버 → 클라이언트:
      {
        "type": "response",
        "data": {
          "ai_message": "AI 응답",
          "intent": "INFO_REQ",
          "suggested_action": "CONTINUE",
          "source_documents": [...]
        }
      }
      
    - 에러 메시지:
      {
        "type": "error",
        "message": "에러 메시지"
      }
    
    - 연결 상태:
      {
        "type": "status",
        "message": "connected",
        "session_id": "sess_123"
      }
    """
    # WebSocket 연결 수립
    await manager.connect(websocket, session_id)
    
    # 연결 확인 메시지 전송
    await manager.send_message(session_id, {
        "type": "status",
        "message": "connected",
        "session_id": session_id
    })
    
    # 상담원 대시보드인 경우 최근 리포트 전송
    if session_id.startswith("consultant_dashboard"):
        await manager.send_recent_reports_to_consultant(session_id)
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_json()
            
            logger.info(f"WebSocket 메시지 수신 - 세션: {session_id}, 타입: {data.get('type')}")
            
            # 메시지 타입 확인
            if data.get("type") == "message":
                user_message = data.get("user_message", "").strip()
                
                if not user_message:
                    await manager.send_error(session_id, "user_message는 필수입니다.")
                    continue
                
                # 처리 중 상태 전송
                await manager.send_message(session_id, {
                    "type": "processing",
                    "message": "메시지를 처리하고 있습니다..."
                })
                
                try:
                    # ChatRequest 생성
                    request = ChatRequest(
                        session_id=session_id,
                        user_message=user_message
                    )
                    
                    # 워크플로우 실행
                    response = await process_chat_message(request)
                    
                    # 응답 전송
                    await manager.send_message(session_id, {
                        "type": "response",
                        "data": {
                            "ai_message": response.ai_message,
                            "intent": response.intent.value,
                            "suggested_action": response.suggested_action.value,
                            "source_documents": [
                                {
                                    "source": doc.source,
                                    "page": doc.page,
                                    "score": doc.score
                                }
                                for doc in response.source_documents
                            ]
                        }
                    })
                    
                    logger.info(f"WebSocket 응답 전송 완료 - 세션: {session_id}, 액션: {response.suggested_action.value}")
                    
                    # 상담원 연결이 필요한 경우 자동으로 리포트 생성
                    # (정보 수집이 완료되었을 때만 HANDOVER 액션이 반환됨)
                    if response.suggested_action.value == "HANDOVER" and session_id not in manager.recent_reports:
                        logger.info(f"상담원 이관 감지 (정보 수집 완료) - 자동 리포트 생성 시작: {session_id}")
                        
                        # 리포트 생성 중 알림
                        await manager.send_message(session_id, {
                            "type": "handover_processing",
                            "message": "상담원 연결을 준비하고 있습니다..."
                        })
                        
                        try:
                            # 상담원 리포트 생성 + 대시보드 브로드캐스트 (공통 함수)
                            report_data = await broadcast_handover_report(session_id)
                            # 고객(현재 세션)에게도 리포트 전송
                            await manager.send_message(session_id, report_data)
                            logger.info(f"상담원 리포트 자동 생성 완료 - 세션: {session_id}")
                            
                        except Exception as handover_error:
                            logger.error(f"상담원 리포트 생성 오류 - 세션: {session_id}, 오류: {str(handover_error)}", exc_info=True)
                            await manager.send_message(session_id, {
                                "type": "handover_error",
                                "message": "상담원 리포트 생성 중 오류가 발생했습니다."
                            })
                    
                except Exception as e:
                    logger.error(f"WebSocket 메시지 처리 중 오류 - 세션: {session_id}, 오류: {str(e)}", exc_info=True)
                    await manager.send_error(session_id, "메시지 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            
            elif data.get("type") == "request_handover":
                # 수동 상담원 이관 요청
                logger.info(f"수동 상담원 이관 요청 - 세션: {session_id}")
                
                await manager.send_message(session_id, {
                    "type": "handover_processing",
                    "message": "상담원 연결을 준비하고 있습니다..."
                })
                
                try:
                    from app.schemas.handover import HandoverRequest
                    from app.services.workflow_service import process_handover
                    
                    handover_request = HandoverRequest(
                        session_id=session_id,
                        trigger_reason="USER_REQUEST"
                    )
                    
                    handover_response = await process_handover(handover_request)
                    
                    # 상담원 리포트 전송
                    await manager.send_message(session_id, {
                        "type": "handover_report",
                        "data": {
                            "status": handover_response.status,
                            "customer_sentiment": handover_response.analysis_result.customer_sentiment.value,
                            "summary": handover_response.analysis_result.summary,
                            "extracted_keywords": handover_response.analysis_result.extracted_keywords,
                            "kms_recommendations": [
                                {
                                    "title": rec.title,
                                    "url": str(rec.url),
                                    "relevance_score": rec.relevance_score
                                }
                                for rec in handover_response.analysis_result.kms_recommendations
                            ]
                        }
                    })
                    
                    logger.info(f"수동 상담원 리포트 생성 완료 - 세션: {session_id}")
                    
                except Exception as e:
                    logger.error(f"수동 상담원 리포트 생성 오류 - 세션: {session_id}, 오류: {str(e)}", exc_info=True)
                    await manager.send_message(session_id, {
                        "type": "handover_error",
                        "message": "상담원 리포트 생성 중 오류가 발생했습니다."
                    })
            
            elif data.get("type") == "ping":
                # Ping/Pong으로 연결 유지
                await manager.send_message(session_id, {
                    "type": "pong",
                    "timestamp": data.get("timestamp")
                })
            
            else:
                logger.warning(f"알 수 없는 메시지 타입 - 세션: {session_id}, 타입: {data.get('type')}")
                await manager.send_error(session_id, f"알 수 없는 메시지 타입: {data.get('type')}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket 연결 종료 (클라이언트 연결 해제) - 세션: {session_id}")
        manager.disconnect(session_id)
    
    except Exception as e:
        logger.error(f"WebSocket 오류 - 세션: {session_id}, 오류: {str(e)}", exc_info=True)
        manager.disconnect(session_id)