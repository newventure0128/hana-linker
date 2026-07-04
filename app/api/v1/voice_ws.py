"""WebSocket 실시간 음성 스트리밍 API

실시간 양방향 음성 통신을 위한 WebSocket 엔드포인트.

사용 흐름:
1. 클라이언트가 WebSocket 연결 (/api/v1/voice/ws/{session_id})
2. audio_start 메시지로 음성 전송 시작 알림
3. audio_chunk 메시지로 음성 데이터 청크 전송
4. audio_end 메시지로 음성 전송 종료
5. 서버가 STT → 워크플로우 → TTS 처리 후 응답

메시지 형식:
- 클라이언트 → 서버: JSON {"type": "...", "data": {...}}
- 서버 → 클라이언트: JSON {"type": "...", "data": {...}}
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.chat import ChatRequest
from app.schemas.voice import WSMessageType
from app.services.workflow_service import process_chat_message
from app.services.voice.stt_service import AICCSTTService, STTError, pcm_to_wav
from app.services.voice.tts_service_google import AICCGoogleTTSService, TTSError
from app.services.vad import HybridVADStream, SileroVADStream
from app.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class VoiceWebSocketManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        # session_id → WebSocket 매핑
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """연결 수락 및 등록"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"[WS] 연결됨 - 세션: {session_id}")
    
    def disconnect(self, session_id: str):
        """연결 해제"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"[WS] 연결 해제 - 세션: {session_id}")
    
    async def send_json(self, session_id: str, message: dict):
        """JSON 메시지 전송"""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)
    
    async def send_message(
        self, 
        session_id: str, 
        msg_type: str, 
        data: Optional[dict] = None
    ):
        """타입이 있는 메시지 전송"""
        message = {
            "type": msg_type,
            "data": data or {},
            "timestamp": time.time(),
        }
        await self.send_json(session_id, message)


# 전역 매니저 인스턴스
ws_manager = VoiceWebSocketManager()


@router.websocket("/ws/{session_id}")
async def voice_websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    🎤 실시간 음성 WebSocket 엔드포인트
    
    연결 URL: ws://localhost:8000/api/v1/voice/ws/{session_id}
    
    클라이언트 → 서버 메시지:
    - audio_start: 음성 전송 시작 {"type": "audio_start", "data": {"language": "ko", "tts_voice": "ko-KR-Neural2-B"}}
    - audio_chunk: 음성 데이터 {"type": "audio_chunk", "data": {"audio_base64": "..."}}
    - audio_end: 음성 전송 종료 {"type": "audio_end"}
    - text_message: 텍스트 직접 전송 {"type": "text_message", "data": {"text": "..."}}
    - ping: 연결 확인 {"type": "ping"}
    
    서버 → 클라이언트 메시지:
    - connected: 연결 완료
    - stt_result: STT 결과
    - ai_response: AI 응답
    - tts_audio: TTS 음성
    - error: 에러
    - pong: Ping 응답
    """
    await ws_manager.connect(session_id, websocket)
    
    # 연결 완료 메시지
    await ws_manager.send_message(
        session_id,
        WSMessageType.CONNECTED,
        {"session_id": session_id, "message": "연결되었습니다."}
    )
    
    # 음성 데이터 버퍼
    audio_buffer: list[bytes] = []
    audio_settings: dict = {
        "language": "ko",
        "tts_voice": "ko-KR-Neural2-B",  # Google TTS 기본 음성
        "diarize": False,
    }
    
    try:
        while True:
            # 메시지 수신
            raw_message = await websocket.receive()
            
            # 연결 종료 확인
            if raw_message.get("type") == "websocket.disconnect":
                break
            
            # 바이너리 데이터 (직접 오디오 청크)
            if "bytes" in raw_message:
                audio_buffer.append(raw_message["bytes"])
                continue
            
            # JSON 메시지
            if "text" in raw_message:
                try:
                    message = json.loads(raw_message["text"])
                except json.JSONDecodeError:
                    await ws_manager.send_message(
                        session_id,
                        WSMessageType.ERROR,
                        {"error": "잘못된 JSON 형식입니다."}
                    )
                    continue
                
                msg_type = message.get("type", "")
                msg_data = message.get("data", {})
                
                # ========== 메시지 타입별 처리 ==========
                
                if msg_type == WSMessageType.PING:
                    # Ping-Pong
                    await ws_manager.send_message(session_id, WSMessageType.PONG)
                
                elif msg_type == WSMessageType.AUDIO_START:
                    # 음성 전송 시작
                    audio_buffer.clear()
                    audio_settings.update({
                        "language": msg_data.get("language", "ko"),
                        "tts_voice": msg_data.get("tts_voice", "ko-KR-Neural2-B"),  # Google TTS 기본 음성
                        "diarize": msg_data.get("diarize", False),
                    })
                    logger.info(f"[WS] 음성 시작 - 세션: {session_id}, 설정: {audio_settings}")
                
                elif msg_type == WSMessageType.AUDIO_CHUNK:
                    # 음성 데이터 청크 (Base64)
                    audio_base64 = msg_data.get("audio_base64", "")
                    if audio_base64:
                        try:
                            audio_bytes = base64.b64decode(audio_base64)
                            audio_buffer.append(audio_bytes)
                        except Exception as e:
                            logger.warning(f"[WS] Base64 디코딩 실패: {e}")
                
                elif msg_type == WSMessageType.AUDIO_END:
                    # 음성 전송 종료 → 처리 시작
                    if audio_buffer:
                        await process_audio_and_respond(
                            session_id=session_id,
                            audio_data=b"".join(audio_buffer),
                            settings=audio_settings,
                        )
                        audio_buffer.clear()
                    else:
                        await ws_manager.send_message(
                            session_id,
                            WSMessageType.ERROR,
                            {"error": "음성 데이터가 없습니다."}
                        )
                
                elif msg_type == WSMessageType.TEXT_MESSAGE:
                    # 텍스트 직접 전송 (STT 건너뛰기)
                    text = msg_data.get("text", "").strip()
                    if text:
                        await process_text_and_respond(
                            session_id=session_id,
                            text=text,
                            tts_voice=audio_settings.get("tts_voice", "ko-KR-Neural2-B"),
                        )
                    else:
                        await ws_manager.send_message(
                            session_id,
                            WSMessageType.ERROR,
                            {"error": "텍스트가 비어있습니다."}
                        )
                
                else:
                    await ws_manager.send_message(
                        session_id,
                        WSMessageType.ERROR,
                        {"error": f"알 수 없는 메시지 타입: {msg_type}"}
                    )
    
    except WebSocketDisconnect:
        logger.info(f"[WS] 클라이언트 연결 종료 - 세션: {session_id}")
    except Exception as e:
        logger.error(f"[WS] 오류 발생 - 세션: {session_id}, 오류: {e}", exc_info=True)
        try:
            await ws_manager.send_message(
                session_id,
                WSMessageType.ERROR,
                {"error": f"서버 오류: {str(e)}"}
            )
        except:
            pass
    finally:
        ws_manager.disconnect(session_id)


async def process_audio_and_respond(
    session_id: str,
    audio_data: bytes,
    settings: dict,
):
    """음성 데이터 처리 및 응답 (STT → 워크플로우 → TTS)"""
    
    try:
        # ========== 1. STT ==========
        logger.info(f"[WS] STT 시작 - 세션: {session_id}, 크기: {len(audio_data)} bytes")
        
        try:
            stt_service = AICCSTTService.get_instance()
            stt_result = stt_service.transcribe(
                audio_data,
                language=settings.get("language", "ko"),
                diarize=settings.get("diarize", False),
            )
            transcribed_text = stt_result.text
        except STTError as e:
            await ws_manager.send_message(
                session_id,
                WSMessageType.ERROR,
                {"error": f"음성 인식 실패: {str(e)}"}
            )
            return
        
        if not transcribed_text.strip():
            await ws_manager.send_message(
                session_id,
                WSMessageType.ERROR,
                {"error": "음성에서 텍스트를 인식할 수 없습니다."}
            )
            return
        
        # STT 결과 전송
        await ws_manager.send_message(
            session_id,
            WSMessageType.STT_RESULT,
            {
                "text": transcribed_text,
                "is_final": True,
                "language": stt_result.language,
            }
        )
        
        # ========== 2. 워크플로우 + TTS ==========
        await process_text_and_respond(
            session_id=session_id,
            text=transcribed_text,
            tts_voice=settings.get("tts_voice", "ko-KR-Neural2-B"),
        )
        
    except Exception as e:
        logger.error(f"[WS] 처리 오류 - 세션: {session_id}, 오류: {e}", exc_info=True)
        await ws_manager.send_message(
            session_id,
            WSMessageType.ERROR,
            {"error": f"처리 중 오류: {str(e)}"}
        )


async def process_text_and_respond(
    session_id: str,
    text: str,
    tts_voice: str = "ko-KR-Neural2-B",
):
    """텍스트 처리 및 응답 (워크플로우 → TTS)

    이관 모드인 경우 AI 워크플로우를 스킵하고 STT 결과만 전송합니다.
    """

    try:
        # ========== 이관 상태 확인 ==========
        if session_manager.is_handover_mode(session_id):
            logger.info(f"[WS] 이관 모드 - AI 워크플로우 스킵 - 세션: {session_id}")
            # 이관 모드에서는 AI 응답 없이 완료 (프론트엔드에서 상담원에게 메시지 전송)
            await ws_manager.send_message(
                session_id,
                WSMessageType.AI_RESPONSE,
                {
                    "text": "",  # 빈 응답
                    "intent": "HANDOVER_MODE",
                    "suggested_action": "HANDOVER",
                    "is_handover_mode": True,  # 이관 모드 표시
                }
            )
            return

        # ========== 워크플로우 실행 ==========
        logger.info(f"[WS] 워크플로우 시작 - 세션: {session_id}, 텍스트: {text[:30]}...")

        chat_request = ChatRequest(
            session_id=session_id,
            user_message=text,
        )
        chat_response = await process_chat_message(chat_request)

        # AI 응답 전송 (handover_status, is_human_required_flow, is_session_end 포함)
        await ws_manager.send_message(
            session_id,
            WSMessageType.AI_RESPONSE,
            {
                "text": chat_response.ai_message,
                "intent": chat_response.intent.value if hasattr(chat_response.intent, 'value') else str(chat_response.intent),
                "suggested_action": chat_response.suggested_action.value if hasattr(chat_response.suggested_action, 'value') else str(chat_response.suggested_action),
                "handover_status": chat_response.handover_status,  # 핸드오버 상태 추가
                "is_human_required_flow": chat_response.is_human_required_flow,  # HUMAN_REQUIRED 플로우 여부
                "is_session_end": chat_response.is_session_end,  # 세션 종료 여부
            }
        )

        # ========== 상담원 이관 확정 시 대시보드로 리포트 브로드캐스트 ==========
        # 기존엔 상담원 알림(리포트 생성+broadcast)이 chat WS 경로에만 있었으나,
        # 실제 고객 프론트는 음성 WS를 쓰므로 여기에도 배선한다.
        # 게이트는 action=HANDOVER(정보 수집 완료 확정)이며 세션당 1회만 발화한다.
        _sa = chat_response.suggested_action
        _sa_val = _sa.value if hasattr(_sa, "value") else str(_sa)
        if _sa_val == "HANDOVER":
            try:
                from app.api.v1.chat import broadcast_handover_report, manager
                if session_id not in manager.recent_reports:
                    await broadcast_handover_report(session_id)
                    logger.info(f"[WS] 상담원 리포트 브로드캐스트 완료 - 세션: {session_id}")
            except Exception as e:
                logger.error(f"[WS] 상담원 리포트 브로드캐스트 실패 - 세션: {session_id}: {e}", exc_info=True)

        # ========== TTS ==========
        logger.info(f"[WS] TTS 시작 - 세션: {session_id}")

        try:
            tts_service = AICCGoogleTTSService.get_instance()
            tts_audio = tts_service.synthesize(
                chat_response.ai_message,
                voice=tts_voice,
            )

            # TTS 음성 전송
            await ws_manager.send_message(
                session_id,
                WSMessageType.TTS_AUDIO,
                {
                    "audio_base64": base64.b64encode(tts_audio).decode("utf-8"),
                    "format": "mp3",
                    "is_final": True,
                }
            )

            logger.info(f"[WS] 응답 완료 - 세션: {session_id}")

        except TTSError as e:
            logger.warning(f"[WS] TTS 실패 - 세션: {session_id}, 오류: {e}")
            # TTS 실패해도 텍스트 응답은 이미 전송됨

    except Exception as e:
        logger.error(f"[WS] 처리 오류 - 세션: {session_id}, 오류: {e}", exc_info=True)
        await ws_manager.send_message(
            session_id,
            WSMessageType.ERROR,
            {"error": f"처리 중 오류: {str(e)}"}
        )


# ========== 연결 상태 확인 API ==========

@router.get("/ws/connections")
async def get_active_connections():
    """현재 활성 WebSocket 연결 목록 (디버깅용)"""
    return {
        "active_sessions": list(ws_manager.active_connections.keys()),
        "count": len(ws_manager.active_connections),
    }


# ========== VAD 기반 실시간 스트리밍 엔드포인트 ==========

class VoiceStreamSession:
    """VAD 기반 음성 스트리밍 세션 관리 (Hybrid VAD: WebRTC + Silero)"""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket

        # Hybrid VAD 초기화: WebRTC + Silero (이어폰/전화 환경 기준)
        silero_vad = SileroVADStream(
            sample_rate=16000,
            frame_ms=40,  # Silero는 40ms 프레임
            threshold=0.35,  # 음성 확률 임계값 (이어폰 환경 - 음성 신호 명확)
        )
        self.vad = HybridVADStream(
            silero_vad,
            sample_rate=16000,
            frame_ms=30,  # WebRTC는 30ms 프레임
            aggressiveness=2,  # 중간 민감도 (이어폰이면 충분)
            min_speech_ms=250,  # 최소 250ms 음성 (짧은 노이즈 필터링)
            max_silence_ms=700,  # 0.7초 침묵 후 음성 종료 (빠른 응답)
            mode="and",  # WebRTC와 Silero 모두 음성으로 판단해야 함
        )

        self.is_active = True
        self.is_speaking = False
        self.audio_buffer: list[bytes] = []
        self.last_activity_time = time.time()
        self.audio_settings: dict = {
            "language": "ko",
            "tts_voice": "ko-KR-Neural2-B",
        }
        self._processing_task: asyncio.Task | None = None  # 진행 중인 STT/AI/TTS 작업

    async def send_message(self, msg_type: str, data: dict):
        """클라이언트에 메시지 전송"""
        if not self.is_active:
            return  # 세션이 비활성화되면 전송 안함
        try:
            await self.websocket.send_json({
                'type': msg_type,
                'data': data,
                'timestamp': time.time() * 1000
            })
        except Exception as e:
            logger.error(f"메시지 전송 오류: {e}")
            self.is_active = False  # 전송 실패 시 세션 비활성화

    async def process_audio(self, audio_data: bytes):
        """오디오 데이터 처리 및 Hybrid VAD 수행 (WebRTC + Silero)"""
        if not self.is_active:
            return  # 세션이 비활성화되면 처리 안함
        try:
            # 이전 VAD 상태 저장 (speech_start 감지용)
            was_in_speech = self.vad._in_speech

            # Hybrid VAD로 음성 세그먼트 감지
            segments = self.vad.feed(audio_data)

            # 현재 VAD 상태
            is_in_speech = self.vad._in_speech

            # 음성 시작 감지 (이전: 침묵 → 현재: 음성)
            if not was_in_speech and is_in_speech:
                self.is_speaking = True
                self.audio_buffer = []  # 새 음성 시작 시 버퍼 초기화
                logger.info(f"[{self.session_id}] 음성 시작")
                await self.send_message('vad_result', {
                    'is_speech': True,
                    'speech_prob': self.vad.last_silero_prob,  # 실제 Silero 확률값
                    'event': 'speech_start',
                })

            # 음성 중이면 버퍼에 저장
            if is_in_speech or self.is_speaking:
                self.audio_buffer.append(audio_data)
                # 음성 지속 상태 전송 (실제 VAD 상태 반영)
                if not segments:
                    await self.send_message('vad_result', {
                        'is_speech': is_in_speech,  # 실제 VAD 상태
                        'speech_prob': self.vad.last_silero_prob,  # 실제 Silero 확률값
                        'event': 'speech_continue' if is_in_speech else 'silence_in_buffer',
                    })

            # 음성 세그먼트 완료 시 (2초 침묵 후)
            for segment in segments:
                if segment.is_speech:
                    logger.info(f"[{self.session_id}] 음성 세그먼트 완료 - "
                              f"start: {segment.start_ms}ms, end: {segment.end_ms}ms, "
                              f"duration: {segment.end_ms - segment.start_ms}ms")

                    # 음성 종료 이벤트 전송
                    await self.send_message('vad_result', {
                        'is_speech': False,
                        'speech_prob': segment.score or 0.0,
                        'event': 'speech_end',
                    })

                    # 자동 전송 이벤트
                    await self.send_message('auto_send', {
                        'reason': 'silence_detected',
                        'buffer_chunks': len(self.audio_buffer),
                        'duration_ms': segment.end_ms - segment.start_ms,
                    })

                    # 버퍼에 데이터가 있으면 STT/AI/TTS 처리
                    if self.audio_buffer:
                        audio_data_combined = b"".join(self.audio_buffer)
                        self.audio_buffer = []

                        # 비동기로 처리 시작 (Task 추적)
                        self._processing_task = asyncio.create_task(self._process_speech(audio_data_combined))

                    self.is_speaking = False

            # 침묵 상태 (음성 중이 아닐 때)
            if not is_in_speech and not self.is_speaking and not segments:
                await self.send_message('vad_result', {
                    'is_speech': False,
                    'speech_prob': self.vad.last_silero_prob,  # 실제 Silero 확률값
                    'event': 'silence',
                })

            self.last_activity_time = time.time()

        except Exception as e:
            logger.error(f"오디오 처리 오류: {e}")

    async def _process_speech(self, audio_data: bytes):
        """음성 데이터 STT → AI → TTS 처리

        이관 모드인 경우 AI 워크플로우를 스킵하고 STT 결과만 전송합니다.
        """
        try:
            # 1. STT
            logger.info(f"[{self.session_id}] STT 시작 - 크기: {len(audio_data)} bytes")

            try:
                # Raw INT16 PCM을 WAV 형식으로 변환 (VITO STT 요구사항)
                wav_data = pcm_to_wav(audio_data, sample_rate=16000, channels=1, sample_width=2)
                logger.info(f"[{self.session_id}] PCM → WAV 변환 완료 - 크기: {len(wav_data)} bytes")

                stt_service = AICCSTTService.get_instance()
                stt_result = stt_service.transcribe(
                    wav_data,
                    language=self.audio_settings.get("language", "ko"),
                )
                transcribed_text = stt_result.text
            except STTError as e:
                await self.send_message('error', {"error": f"음성 인식 실패: {str(e)}"})
                return

            if not transcribed_text.strip():
                await self.send_message('error', {"error": "음성에서 텍스트를 인식할 수 없습니다."})
                return

            # STT 결과 전송
            await self.send_message('stt_result', {
                'text': transcribed_text,
                'is_final': True,
            })

            # ========== 이관 상태 확인 ==========
            if session_manager.is_handover_mode(self.session_id):
                logger.info(f"[{self.session_id}] 이관 모드 - AI 워크플로우 스킵")
                # 이관 모드에서는 AI 응답 없이 완료 (프론트엔드에서 상담원에게 메시지 전송)
                await self.send_message('ai_response', {
                    'text': '',  # 빈 응답
                    'intent': 'HANDOVER_MODE',
                    'suggested_action': 'HANDOVER',
                    'is_handover_mode': True,  # 이관 모드 표시
                })
                await self.send_message('completed', {
                    'message': '이관 모드 - STT만 처리',
                    'final_text': transcribed_text,
                    'is_handover_mode': True,
                })
                return

            # 2. AI 워크플로우
            logger.info(f"[{self.session_id}] 워크플로우 시작 - 텍스트: {transcribed_text[:30]}...")

            chat_request = ChatRequest(
                session_id=self.session_id,
                user_message=transcribed_text,
            )
            chat_response = await process_chat_message(chat_request)

            # AI 응답 전송 (handover_status, is_human_required_flow, is_session_end 포함)
            await self.send_message('ai_response', {
                'text': chat_response.ai_message,
                'intent': chat_response.intent.value if hasattr(chat_response.intent, 'value') else str(chat_response.intent),
                'suggested_action': chat_response.suggested_action.value if hasattr(chat_response.suggested_action, 'value') else str(chat_response.suggested_action),
                'handover_status': chat_response.handover_status,  # 핸드오버 상태 추가
                'is_human_required_flow': chat_response.is_human_required_flow,  # HUMAN_REQUIRED 플로우 여부
                'is_session_end': chat_response.is_session_end,  # 세션 종료 여부
            })

            # 3. TTS (Google TTS 사용)
            logger.info(f"[{self.session_id}] TTS 시작")

            try:
                tts_service = AICCGoogleTTSService.get_instance()
                tts_audio = tts_service.synthesize(
                    chat_response.ai_message,
                    voice=self.audio_settings.get("tts_voice", "ko-KR-Neural2-B"),
                )

                # TTS 음성 전송
                await self.send_message('tts_chunk', {
                    'audio_base64': base64.b64encode(tts_audio).decode("utf-8"),
                    'format': 'mp3',
                    'chunk_index': 0,
                })

                logger.info(f"[{self.session_id}] 응답 완료")

            except TTSError as e:
                logger.warning(f"[{self.session_id}] TTS 실패: {e}")

            # 완료 메시지
            await self.send_message('completed', {
                'message': '처리 완료',
                'final_text': transcribed_text,
            })

        except Exception as e:
            logger.error(f"[{self.session_id}] 처리 오류: {e}", exc_info=True)
            await self.send_message('error', {"error": f"처리 중 오류: {str(e)}"})

    def reset(self):
        """세션 상태 초기화"""
        self.vad.reset()
        self.is_speaking = False
        self.audio_buffer = []


@router.websocket("/streaming/{session_id}")
async def voice_streaming(websocket: WebSocket, session_id: str):
    """
    VAD 기반 양방향 음성 스트리밍 WebSocket 엔드포인트

    클라이언트 → 서버:
    - binary: INT16 PCM 오디오 데이터 (16kHz)
    - text "EOS": 스트리밍 종료
    - text "RESET": VAD 상태 초기화

    서버 → 클라이언트:
    - vad_result: VAD 감지 결과 {is_speech, speech_prob, event}
    - auto_send: 자동 전송 트리거 (2초 침묵)
    - stt_result: STT 결과 {text, is_final}
    - ai_response: AI 응답 {text, intent, suggested_action}
    - tts_chunk: TTS 오디오 {audio_base64, format}
    - completed: 처리 완료
    - error: 오류 메시지
    """
    await websocket.accept()

    session = VoiceStreamSession(session_id, websocket)
    logger.info(f"[{session_id}] WebSocket 연결됨 (VAD 스트리밍)")

    try:
        # 연결 성공 메시지
        await session.send_message('connected', {
            'session_id': session_id,
            'message': 'Hybrid VAD (WebRTC + Silero) 스트리밍 연결 완료',
            'vad_config': {
                'engine': 'hybrid',
                'mode': session.vad.mode,
                'webrtc_aggressiveness': 2,
                'silero_threshold': session.vad.silero_threshold,
                'sample_rate': session.vad.sample_rate,
                'min_speech_ms': session.vad.min_speech_ms,
                'max_silence_ms': session.vad.max_silence_ms,
            }
        })

        while session.is_active:
            try:
                # 메시지 수신 (타임아웃 60초)
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=60.0
                )

                if message['type'] == 'websocket.disconnect':
                    break

                # 바이너리 오디오 데이터
                if 'bytes' in message:
                    await session.process_audio(message['bytes'])

                # 텍스트 명령
                elif 'text' in message:
                    text = message['text'].strip()

                    if text == 'EOS':
                        logger.info(f"[{session_id}] EOS 수신")

                        # 버퍼에 남은 데이터 처리 (완료까지 대기)
                        if session.audio_buffer:
                            audio_data_combined = b"".join(session.audio_buffer)
                            session.audio_buffer = []
                            await session._process_speech(audio_data_combined)
                        elif session._processing_task and not session._processing_task.done():
                            # VAD가 이미 처리를 시작한 경우, 완료될 때까지 대기
                            logger.info(f"[{session_id}] 진행 중인 처리 완료 대기...")
                            await session._processing_task
                        else:
                            # 버퍼도 비어있고 진행 중인 작업도 없으면 완료 메시지 전송
                            await session.send_message('completed', {
                                'message': 'EOS 처리 완료 (버퍼 없음)'
                            })

                        session.reset()

                    elif text == 'RESET':
                        logger.info(f"[{session_id}] RESET 수신")
                        session.reset()
                        await session.send_message('reset', {
                            'message': 'VAD 상태 초기화 완료'
                        })

                    elif text == 'ping':
                        await session.send_message('pong', {})

            except asyncio.TimeoutError:
                # 타임아웃 시 ping 전송
                await session.send_message('ping', {})

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] 클라이언트 연결 해제")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket 오류: {e}")
        await session.send_message('error', {'message': str(e)})
    finally:
        session.is_active = False
        logger.info(f"[{session_id}] 세션 종료")


@router.get("/vad/status")
async def vad_status():
    """Hybrid VAD 서비스 상태 확인"""
    try:
        # Hybrid VAD 테스트 인스턴스 생성
        silero_vad = SileroVADStream(
            sample_rate=16000,
            frame_ms=40,
            threshold=0.3,
        )
        hybrid_vad = HybridVADStream(
            silero_vad,
            sample_rate=16000,
            frame_ms=30,
            aggressiveness=3,
            mode="and",
        )
        return {
            'status': 'ok',
            'engine': 'hybrid',
            'mode': hybrid_vad.mode,
            'silero_threshold': silero_vad.threshold,
            'webrtc_aggressiveness': 3,
            'sample_rate': hybrid_vad.sample_rate,
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

