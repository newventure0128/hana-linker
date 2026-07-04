import { useState, useCallback, useRef, useEffect } from 'react';
import VoiceButton from './components/VoiceButton';
import ChatMessage, { Message } from './components/ChatMessage';
import { useVoiceStream } from './hooks/useVoiceStream';
import { useVoiceRecording } from './hooks/useVoiceRecording';
import { voiceApi, getOrCreateSessionId, resetSessionId, formatSessionIdForDisplay, HandoverResponse } from './services/api';
import './App.css';

// 인사 메시지 설정 (새 상담 버튼 클릭 시에만 표시)
const WELCOME_MESSAGE = "안녕하세요 고객님 무엇을 도와 드릴까요?";

// 핸드오버 관련 설정
const HANDOVER_POLL_INTERVAL_MS = 2000; // 2초마다 상담사 수락 여부 폴링
const HANDOVER_TIMEOUT_MS = 60000; // 60초 타임아웃 (실제로는 필요시 조정)
const HANDOVER_WAIT_TIME_MESSAGE = "현재 모든 상담사가 상담 중입니다. 예상 대기 시간은 약 10분입니다.";

// 고객 비활성 리마인더 설정
const INACTIVITY_REMINDER_MS = 30000; // 30초 비활성 시 리마인더
const INACTIVITY_REMINDER_MESSAGE = "고객님, 아직 계시나요? 상담사 연결을 원하시면 '네'라고 말씀해 주세요.";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState(() => getOrCreateSessionId());
  const [isSessionStarted, setIsSessionStarted] = useState(false);  // 새 상담 시작 여부 (세션 번호 표시용)
  const [_handoverData, setHandoverData] = useState<HandoverResponse | null>(null);
  void _handoverData;  // 미사용 (향후 확장용)
  const [isHandoverMode, setIsHandoverMode] = useState(false);  // 상담원 연결 모드 (실제 상담 중)
  const [_isHandoverLoading, setIsHandoverLoading] = useState(false);  // 상담원 연결 로딩 상태 (미사용)
  void _isHandoverLoading;
  const [isWaitingForAgent, setIsWaitingForAgent] = useState(false);  // 상담사 수락 대기 중
  const [isHumanRequiredFlow, setIsHumanRequiredFlow] = useState(false);  // HUMAN_REQUIRED 플로우 진입 여부 (consent_check 포함)
  const [_handoverTimeoutReached, setHandoverTimeoutReached] = useState(false);  // 타임아웃 도달 여부 (미사용, 향후 확장용)
  void _handoverTimeoutReached;
  const handoverPollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const handoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startHandoverPollingRef = useRef<(() => void) | null>(null);  // 클로저 문제 해결용
  const handoverAcceptedProcessingRef = useRef<boolean>(false);  // accepted 상태 처리 중복 방지
  const [textInput, setTextInput] = useState('');  // 텍스트 입력 상태
  const [isTextSending, setIsTextSending] = useState(false);  // 텍스트 전송 중 상태
  const [isRecordingMode, setIsRecordingMode] = useState(false);  // true: 녹음 모드, false: 실시간 스트리밍 모드 (기본: 실시간)
  const [isStopped, setIsStopped] = useState(false);  // 중지 상태 표시용
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textInputRef = useRef<HTMLInputElement>(null);
  const isHandoverModeRef = useRef(false);  // 클로저 문제 해결용 ref
  const isHumanRequiredFlowRef = useRef(false);  // 클로저 문제 해결용 ref
  const isWaitingForAgentRef = useRef(false);  // 클로저 문제 해결용 ref (상담사 수락 대기 중)
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lastMessageIdRef = useRef<number>(0);  // 마지막 메시지 ID (폴링용)
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isConfirmingHandoverRef = useRef(false);  // confirmHandover 중복 호출 방지용
  const inactivityTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);  // 고객 비활성 리마인더 타이머
  const lastActivityTimeRef = useRef<number>(Date.now());  // 마지막 활동 시간

  // 양방향 스트리밍 훅 사용 (STT + AI + TTS 통합)
  const {
    isRecording: isStreamRecording,
    isConnected: _isConnected,
    isProcessing: isStreamProcessing,
    isPlayingTTS,
    transcript,
    finalTranscript,
    error: _sttError,
    startRecording: startStreamRecording,
    stopRecording: stopStreamRecording,
    disconnect: disconnectStream,  // 실시간 모드 WebSocket 연결 해제
    stopTTS: _stopTTS,
    setOnAutoStop,
    setOnTTSComplete,
    setOnBargeIn,
    setExternalTTSPlaying,  // 외부 TTS 재생 상태 설정 (첫 인사 Barge-in용)
  } = useVoiceStream(sessionId);
  // Suppress unused variable warnings
  void _isConnected;
  void _sttError;
  void _stopTTS;

  // 녹음 모드 훅 (Hybrid VAD 적용)
  const {
    isRecording: isRecordRecording,
    isProcessing: isRecordProcessing,
    error: _recordError,
    recordingTime,
    isSpeaking: isRecordSpeaking,      // Hybrid VAD: 음성 감지 여부
    speechProb: recordSpeechProb,      // Silero VAD: 음성 확률
    vadEvent: recordVadEvent,          // VAD 이벤트
    isPlayingTTS: isRecordPlayingTTS,  // 녹음 모드 TTS 재생 중 여부
    startRecording: startRecordRecording,
    stopRecording: stopRecordRecording,
    disconnect: disconnectRecord,  // 녹음 모드 WebSocket 연결 해제
    stopTTS: stopRecordTTS,  // 녹음 모드 TTS 중지 (Barge-in용)
    setOnBargeIn: setOnRecordBargeIn,  // 녹음 모드 Barge-in 콜백 설정
    setOnAiResponse: setOnRecordAiResponse,  // 녹음 모드 추가 AI 응답 콜백 설정
  } = useVoiceRecording(sessionId);
  void _recordError;
  void recordVadEvent;  // 향후 UI 표시용
  void stopRecordTTS;  // 향후 사용 (현재 내부에서 자동 호출)

  // 현재 모드에 따른 상태 통합
  const isRecording = isRecordingMode ? isRecordRecording : isStreamRecording;
  const isProcessing = isRecordingMode ? isRecordProcessing : isStreamProcessing;
  // TTS 재생 상태 통합 (녹음 모드 또는 실시간 모드)
  const isAnyTTSPlaying = isRecordingMode ? isRecordPlayingTTS : isPlayingTTS;

  // 통합 녹음 시작/중지 함수
  const startRecording = isRecordingMode ? startRecordRecording : startStreamRecording;
  const stopRecording = isRecordingMode ? stopRecordRecording : stopStreamRecording;

  // 연속 대화 모드 (첫 녹음 후 활성화)
  const [isContinuousMode, setIsContinuousMode] = useState(false);

  // 연속 빈 입력 카운터 (무한 루프 방지)
  const emptyInputCountRef = useRef<number>(0);
  const MAX_EMPTY_INPUTS = 2;  // 연속 2회 빈 입력 시 연속 대화 일시 중지

  // 현재 표시할 텍스트 (확정 + 부분)
  const currentTranscript = finalTranscript + (transcript ? ` ${transcript}` : '');

  // isHandoverMode가 변경될 때 ref도 업데이트 (클로저 문제 해결)
  useEffect(() => {
    isHandoverModeRef.current = isHandoverMode;
    console.log('[App] isHandoverMode 변경:', isHandoverMode);
  }, [isHandoverMode]);

  // isHumanRequiredFlow가 변경될 때 ref도 업데이트 (클로저 문제 해결)
  useEffect(() => {
    isHumanRequiredFlowRef.current = isHumanRequiredFlow;
    console.log('[App] isHumanRequiredFlow 변경:', isHumanRequiredFlow);
  }, [isHumanRequiredFlow]);

  // isWaitingForAgent가 변경될 때 ref도 업데이트 (클로저 문제 해결)
  useEffect(() => {
    isWaitingForAgentRef.current = isWaitingForAgent;
  }, [isWaitingForAgent]);

  // 실시간 모드 전환 시 자동 녹음 시작 (문제 1, 3 해결)
  // 새 상담 시 disconnect 후 약간의 딜레이 필요
  const isResettingRef = useRef(false);  // 새 상담 진행 중 플래그

  useEffect(() => {
    // 세션이 시작된 후에만 자동 녹음 시작 (새 상담 전에는 토글 전환 가능)
    if (isSessionStarted && !isRecordingMode && !isRecording && !isProcessing && !isHandoverMode && !isResettingRef.current) {
      console.log('[App] 실시간 모드 전환 - 자동 녹음 시작');
      startStreamRecording().catch((err) => {
        console.error('[App] 실시간 모드 자동 녹음 시작 실패:', err);
      });
    }
  }, [isSessionStarted, isRecordingMode, isRecording, isProcessing, isHandoverMode, startStreamRecording]);

  // 메시지 추가 시 스크롤
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentTranscript]);

  // base64 → Blob 변환
  const base64ToBlob = (base64: string, mimeType: string): Blob => {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
  };

  // 자동 인사 메시지 제거 - 이제 '새 상담' 버튼을 눌러야만 인사 메시지가 표시됨
  // 앱 로드 시에는 빈 화면에서 마이크 버튼 클릭 대기

  // TTS 오디오 재생
  const playAudio = useCallback((base64Audio: string) => {
    try {
      const audioBlob = base64ToBlob(base64Audio, 'audio/mp3');
      const audioUrl = URL.createObjectURL(audioBlob);

      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
      }
    } catch (err) {
      console.error('오디오 재생 실패:', err);
    }
  }, []);

  // 고객 비활성 리마인더 타이머 정리
  const clearInactivityTimer = useCallback(() => {
    if (inactivityTimerRef.current) {
      clearTimeout(inactivityTimerRef.current);
      inactivityTimerRef.current = null;
    }
  }, []);

  // 고객 비활성 리마인더 타이머 시작
  const startInactivityTimer = useCallback(() => {
    clearInactivityTimer();
    lastActivityTimeRef.current = Date.now();

    inactivityTimerRef.current = setTimeout(async () => {
      console.log('[App] 고객 비활성 리마인더 발생');

      // 리마인더 메시지 표시
      const reminderMessage: Message = {
        id: `msg_${Date.now()}_reminder`,
        role: 'assistant',
        content: INACTIVITY_REMINDER_MESSAGE,
        timestamp: new Date(),
        isNew: true,
      };
      setMessages((prev) => [...prev, reminderMessage]);

      // TTS 재생
      try {
        const ttsResponse = await voiceApi.requestTTS(INACTIVITY_REMINDER_MESSAGE);
        if (ttsResponse.audio_base64) {
          playAudio(ttsResponse.audio_base64);
        }
      } catch (ttsErr) {
        console.warn('리마인더 TTS 재생 실패:', ttsErr);
      }

      // 리마인더 후 타이머 재시작 (고객이 계속 응답하지 않으면 다시 리마인더)
      // 단, 동의 대기 단계(isHumanRequiredFlow=true, 아직 수집/대기 아님)에서만.
      // 정보 수집 시작(isWaitingForAgent) 이후엔 "…원하시면 '네'" 문구가 맞지 않으므로 멈춘다.
      if (isHumanRequiredFlowRef.current && !isHandoverModeRef.current && !isWaitingForAgentRef.current) {
        console.log('[App] 리마인더 후 타이머 재시작');
        startInactivityTimer();
      }
    }, INACTIVITY_REMINDER_MS);
  }, [clearInactivityTimer, playAudio]);

  // 고객 활동 시 타이머 리셋 (동의 대기 단계에서만; 수집/수락대기 중엔 리마인더 안 함)
  const resetInactivityTimer = useCallback(() => {
    if (isHumanRequiredFlow && !isHandoverMode && !isWaitingForAgent) {
      lastActivityTimeRef.current = Date.now();
      startInactivityTimer();
    }
  }, [isHumanRequiredFlow, isHandoverMode, isWaitingForAgent, startInactivityTimer]);

  // HUMAN_REQUIRED 플로우 상태 변경 시 리마인더 타이머 관리
  // 중요: TTS 재생 중에는 타이머를 시작하지 않음
  useEffect(() => {
    if (isHumanRequiredFlow && !isHandoverMode && !isWaitingForAgent && !isAnyTTSPlaying) {
      // 동의 대기 단계 진입 + TTS 재생 완료 → 리마인더 타이머 시작
      console.log('[App] 동의 대기 + TTS 완료 - 비활성 리마인더 타이머 시작');
      startInactivityTimer();
    } else if (!isHumanRequiredFlow || isHandoverMode || isWaitingForAgent) {
      // 동의 대기 단계가 아니거나(수집/수락대기/이관모드) → 타이머 정리
      clearInactivityTimer();
    }
    // isAnyTTSPlaying이 true일 때는 타이머를 시작하지 않고 대기
    // TTS가 끝나면 isAnyTTSPlaying이 false가 되어 다시 이 effect가 실행됨

    return () => {
      clearInactivityTimer();
    };
  }, [isHumanRequiredFlow, isHandoverMode, isWaitingForAgent, isAnyTTSPlaying, startInactivityTimer, clearInactivityTimer]);

  // 상담원 메시지 폴링 (이관 모드일 때만)
  // 이미 처리한 메시지 ID를 추적하는 Set (중복 방지)
  const processedAgentMessageIdsRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    // 중지 상태이거나 핸드오버 모드가 아니면 폴링 안함
    if (!isHandoverMode || isStopped) return;

    // 폴링 중복 실행 방지 플래그
    let isPolling = false;

    const pollAgentMessages = async () => {
      // 이미 폴링 중이면 스킵
      if (isPolling) return;
      isPolling = true;

      try {
        const params = new URLSearchParams();
        if (lastMessageIdRef.current > 0) {
          params.append('after_id', lastMessageIdRef.current.toString());
        }
        params.append('after_handover', 'true');

        const response = await fetch(`/api/v1/sessions/${sessionId}/messages?${params.toString()}`);
        if (!response.ok) return;

        const newMessages = await response.json();

        if (newMessages.length > 0) {
          // 마지막 메시지 ID 먼저 업데이트 (다음 폴링에서 중복 방지)
          const maxId = Math.max(...newMessages.map((m: any) => m.id));
          lastMessageIdRef.current = maxId;

          // 상담원 메시지만 필터링 (role === 'assistant')
          const agentMessages = newMessages.filter((m: any) => m.role === 'assistant');

          // 아직 처리하지 않은 메시지만 필터링
          const unprocessedMessages = agentMessages.filter(
            (msg: any) => !processedAgentMessageIdsRef.current.has(msg.id)
          );

          if (unprocessedMessages.length === 0) return;

          // 처리할 메시지 ID들을 먼저 등록 (중복 처리 방지)
          unprocessedMessages.forEach((msg: any) => {
            processedAgentMessageIdsRef.current.add(msg.id);
          });

          // 새 메시지들을 한 번에 추가 (배치 처리)
          const newChatMessages: Message[] = unprocessedMessages.map((msg: any) => ({
            id: `msg_agent_${msg.id}`,
            role: 'assistant' as const,
            content: msg.message,
            timestamp: new Date(msg.created_at),
            isAgent: true,
            isNew: true,
          }));

          setMessages((prev) => {
            // 기존 메시지 ID 집합
            const existingIds = new Set(prev.map((m) => m.id));
            // 중복되지 않은 새 메시지만 필터링
            const uniqueNewMessages = newChatMessages.filter((m) => !existingIds.has(m.id));
            if (uniqueNewMessages.length === 0) return prev;
            return [...prev, ...uniqueNewMessages];
          });

          // TTS 재생 (첫 번째 메시지만, 순차적으로)
          for (const msg of unprocessedMessages) {
            try {
              const ttsResponse = await fetch('/api/v1/voice/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: msg.message }),
              });

              if (ttsResponse.ok) {
                const ttsData = await ttsResponse.json();
                if (ttsData.audio_base64) {
                  playAudio(ttsData.audio_base64);
                }
              }
            } catch (ttsErr) {
              console.warn('TTS 실패:', ttsErr);
            }
          }
        }
      } catch (err) {
        console.error('상담원 메시지 폴링 실패:', err);
      } finally {
        isPolling = false;
      }
    };

    // 2초마다 폴링
    pollingIntervalRef.current = setInterval(pollAgentMessages, 2000);
    pollAgentMessages(); // 즉시 한 번 실행

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [isHandoverMode, sessionId, playAudio, isStopped]);

  // 녹음 중지 및 메시지 처리 (공통 로직)
  const processStopRecording = useCallback(async () => {
    console.log('[App] ========== processStopRecording 시작 ==========');
    console.log('[App] isHandoverMode:', isHandoverMode, 'isRecordingMode:', isRecordingMode, 'isWaitingForAgent:', isWaitingForAgent);

    // 고객 활동 감지 - 리마인더 타이머 리셋
    resetInactivityTimer();

    const result = await stopRecording();
    console.log('[App] stopRecording 결과 - result 존재:', !!result);
    if (result) {
      console.log('[App] stopRecording 결과 - userText:', result.userText);
      console.log('[App] stopRecording 결과 - aiResponse:', JSON.stringify(result.aiResponse, null, 2));
    }

    if (result) {
      // 사용자 메시지 추가 (서버에서 받은 final_text 사용)
      if (result.userText?.trim()) {
        console.log('[App] 사용자 메시지 추가:', result.userText);
        // 성공적인 입력 시 빈 입력 카운터 초기화
        emptyInputCountRef.current = 0;

        const userMessageContent = result.userText.trim();
        const userMessage: Message = {
          id: `msg_${Date.now()}_user_${Math.random().toString(36).substring(2, 11)}`,
          role: 'user',
          content: userMessageContent,
          timestamp: new Date(),
          isVoice: true,
          isNew: true,
        };

        // 중복 메시지 방지: 최근 5개 메시지 중 동일 content가 있으면 추가하지 않음
        setMessages((prev) => {
          const recentMessages = prev.slice(-5);
          const isDuplicate = recentMessages.some(
            (m) => m.role === 'user' && m.content === userMessageContent
          );
          if (isDuplicate) {
            console.log('[App] 중복 사용자 메시지 무시:', userMessageContent);
            return prev;
          }
          return [...prev, userMessage];
        });

        // 이관 모드일 때: AI 응답 무시하고 상담원에게 메시지 전송
        if (isHandoverMode) {
          console.log('[App] 이관 모드 - 상담원에게 메시지 전송:', result.userText);
          try {
            await voiceApi.sendCustomerMessage(sessionId, result.userText.trim());
            console.log('[App] 상담원에게 메시지 전송 완료');
          } catch (err) {
            console.error('[App] 상담원에게 메시지 전송 실패:', err);
          }
          return; // AI 응답 처리 건너뛰기
        }
      } else {
        console.log('[App] userText가 비어있음');
      }

      // AI 응답 메시지 추가 (이관 모드가 아닐 때만)
      if (!isHandoverMode && result.aiResponse?.text) {
        // 디버그: AI 응답 전체 내용 출력
        console.log('[App] 녹음 모드 - AI 응답 전체:', JSON.stringify(result.aiResponse, null, 2));

        // HUMAN_REQUIRED 플로우 감지 (백엔드에서 받은 is_human_required_flow 사용)
        const backendHumanRequiredFlow = result.aiResponse.isHumanRequiredFlow || false;
        console.log('[App] 녹음 모드 AI 응답 - suggestedAction:', result.aiResponse.suggestedAction, ', handoverStatus:', result.aiResponse.handoverStatus, ', isHumanRequiredFlow:', backendHumanRequiredFlow);

        // 세션 종료 감지 (불명확 응답/도메인 외 질문 3회 이상)
        const isSessionEnd = result.aiResponse.isSessionEnd || false;
        if (isSessionEnd) {
          console.log('[App] 녹음 모드 - 세션 종료 감지 (is_session_end=true)');
          setIsHumanRequiredFlow(false);
          setIsWaitingForAgent(false);
          cleanupHandoverPolling();
        }
        // HUMAN_REQUIRED 플로우 진입/종료 감지
        else if (backendHumanRequiredFlow && !isHumanRequiredFlow) {
          console.log('[App] 녹음 모드 - HUMAN_REQUIRED 플로우 진입 감지 (백엔드)');
          setIsHumanRequiredFlow(true);
        } else if (!backendHumanRequiredFlow && isHumanRequiredFlow) {
          console.log('[App] 녹음 모드 - 고객 상담사 연결 거부 - HUMAN_REQUIRED 플로우 종료');
          setIsHumanRequiredFlow(false);
        }

        // HANDOVER 감지 (녹음 모드에서 - 메시지 표시용)
        const isHandoverSuggested =
          result.aiResponse.suggestedAction === 'HANDOVER' ||
          result.aiResponse.suggestedAction === 'handover' ||
          result.aiResponse.text.includes('상담사에게 연결해 드리겠습니다') ||
          result.aiResponse.text.includes('상담원에게 연결');

        // handover_status가 pending이면 상담사 수락 대기 폴링 시작
        // 안내 메시지는 백엔드(waiting_agent)에서 이미 ai_message에 포함됨
        console.log('[App] 녹음 모드 - isWaitingForAgent 상태:', isWaitingForAgent);
        if (result.aiResponse.handoverStatus === 'pending' && !isWaitingForAgent) {
          console.log('[App] 녹음 모드 - handover_status=pending 감지 - 상담사 수락 대기 폴링 시작');
          setIsWaitingForAgent(true);
          setHandoverTimeoutReached(false);
          startHandoverPolling();
        }

        if (isHandoverSuggested) {
          console.log('[App] 녹음 모드 - HANDOVER 감지, 메시지 표시 및 TTS 재생 시작');
          console.log('[App] 녹음 모드 - 표시할 메시지:', result.aiResponse.text);

          // AI 응답 메시지만 표시 (고객에게 동의 요청)
          // 고객이 "네"라고 응답하면 백엔드의 consent_check_node가 처리함
          const aiMessageContent = result.aiResponse.text;
          const aiMessage: Message = {
            id: `msg_${Date.now()}_assistant_${Math.random().toString(36).substring(2, 11)}`,
            role: 'assistant',
            content: aiMessageContent,
            timestamp: new Date(),
            isNew: true,
          };
          setMessages((prev) => [...prev, aiMessage]);

          // TTS는 WebSocket을 통해 실시간으로 재생됨 (별도 처리 불필요)
          console.log('[App] 녹음 모드 - TTS는 WebSocket을 통해 실시간 재생됨');

          // 고객이 "네"라고 응답할 때까지 대기
          // 다음 메시지에서 백엔드가 consent_check_node → waiting_agent 플로우를 처리함
        } else {
          // 일반 AI 응답 메시지 추가 (HANDOVER가 아닌 경우)
          console.log('[App] 녹음 모드 - 일반 AI 응답 (HANDOVER 아님), 메시지 표시 시작');
          console.log('[App] AI 응답 메시지 추가:', result.aiResponse.text);
          const aiMessageContent = result.aiResponse.text;
          const assistantMessage: Message = {
            id: `msg_${Date.now()}_assistant_${Math.random().toString(36).substring(2, 11)}`,
            role: 'assistant',
            content: aiMessageContent,
            timestamp: new Date(),
            isNew: true,
          };

          // 중복 메시지 방지: 최근 5개 메시지 중 동일 content가 있으면 추가하지 않음
          setMessages((prev) => {
            const recentMessages = prev.slice(-5);
            const isDuplicate = recentMessages.some(
              (m) => m.role === 'assistant' && m.content === aiMessageContent
            );
            if (isDuplicate) {
              console.log('[App] 중복 AI 메시지 무시:', aiMessageContent.substring(0, 30) + '...');
              return prev;
            }
            return [...prev, assistantMessage];
          });

          // TTS는 WebSocket을 통해 실시간으로 재생됨 (녹음 모드도 동일)
          console.log('[App] TTS는 WebSocket을 통해 실시간 재생됨');
        }
      } else if (!isHandoverMode) {
        console.log('[App] aiResponse가 없거나 text가 비어있음');
      }
    } else {
      // result가 null인 경우 (빈 입력 등)
      emptyInputCountRef.current += 1;
      console.log(`[App] result가 null - 빈 입력 횟수: ${emptyInputCountRef.current}/${MAX_EMPTY_INPUTS}`);

      // 녹음 모드에서는 연속 녹음 안 함
      if (!isRecordingMode && isContinuousMode && !isHandoverMode && emptyInputCountRef.current < MAX_EMPTY_INPUTS) {
        // 연속 빈 입력이 아니면 다시 녹음 시작 (스트리밍 모드만)
        console.log('[App] 다시 녹음 시작...');
        setTimeout(() => {
          startRecording().catch((err: Error) => {
            console.error('[App] 재녹음 시작 실패:', err);
          });
        }, 500);
      } else if (emptyInputCountRef.current >= MAX_EMPTY_INPUTS) {
        // 연속 빈 입력 시 연속 대화 일시 중지
        console.log('[App] 연속 빈 입력 감지 - 연속 대화 일시 중지, 버튼 클릭 대기');
        emptyInputCountRef.current = 0;  // 카운터 초기화
      }
    }
  }, [stopRecording, isContinuousMode, isHandoverMode, isHumanRequiredFlow, startRecording, sessionId, isRecordingMode, playAudio, resetInactivityTimer]);

  // VAD 자동 중지 콜백 설정 (2초 침묵 시 자동 전송)
  // 백엔드에서 처리 완료 후 결과를 직접 받음 (EOS 전송 없이)
  // 중요: isHandoverModeRef.current를 사용하여 클로저 문제 해결
  useEffect(() => {
    setOnAutoStop(async (result) => {
      const currentHandoverMode = isHandoverModeRef.current;
      console.log('[App] VAD 자동 중지 결과:', result, 'isHandoverMode (ref):', currentHandoverMode);

      if (result) {
        // 사용자 메시지 추가
        if (result.userText?.trim()) {
          console.log('[App] 사용자 메시지 추가:', result.userText);
          emptyInputCountRef.current = 0;

          const userMessageContent = result.userText.trim();
          const userMessage: Message = {
            id: `msg_${Date.now()}_user_${Math.random().toString(36).substring(2, 11)}`,
            role: 'user',
            content: userMessageContent,
            timestamp: new Date(),
            isVoice: true,
            isNew: true,
          };

          // 중복 메시지 방지: 최근 5개 메시지 중 동일 content가 있으면 추가하지 않음
          setMessages((prev) => {
            const recentMessages = prev.slice(-5);
            const isDuplicate = recentMessages.some(
              (m) => m.role === 'user' && m.content === userMessageContent
            );
            if (isDuplicate) {
              console.log('[App] VAD - 중복 사용자 메시지 무시:', userMessageContent);
              return prev;
            }
            return [...prev, userMessage];
          });

          // 이관 모드일 때: AI 응답 무시하고 상담원에게 메시지 전송
          if (currentHandoverMode) {
            console.log('[App] 이관 모드 - 상담원에게 메시지 전송:', result.userText);
            try {
              await voiceApi.sendCustomerMessage(sessionId, result.userText.trim());
              console.log('[App] 상담원에게 메시지 전송 완료');
            } catch (err) {
              console.error('[App] 상담원에게 메시지 전송 실패:', err);
            }
            return; // AI 응답 처리 건너뛰기
          }
        } else {
          console.log('[App] userText가 비어있음');
        }

        // AI 응답 메시지 추가 (이관 모드가 아닐 때만)
        if (!currentHandoverMode && result.aiResponse?.text) {
          // HUMAN_REQUIRED 플로우 감지 (백엔드에서 받은 is_human_required_flow 사용)
          const backendHumanRequiredFlow = result.aiResponse.isHumanRequiredFlow || false;
          console.log('[App] AI 응답 처리 - suggestedAction:', result.aiResponse.suggestedAction, ', handoverStatus:', result.aiResponse.handoverStatus, ', isHumanRequiredFlow:', backendHumanRequiredFlow);

          // 세션 종료 감지 (불명확 응답/도메인 외 질문 3회 이상)
          const isSessionEnd = result.aiResponse.isSessionEnd || false;
          if (isSessionEnd) {
            console.log('[App] 스트리밍 모드 - 세션 종료 감지 (is_session_end=true)');
            setIsHumanRequiredFlow(false);
            setIsWaitingForAgent(false);
          }
          // HUMAN_REQUIRED 플로우 진입/종료 감지
          const currentHumanRequiredFlow = isHumanRequiredFlowRef.current;
          if (!isSessionEnd && backendHumanRequiredFlow && !currentHumanRequiredFlow) {
            console.log('[App] 스트리밍 모드 - HUMAN_REQUIRED 플로우 진입 감지 (백엔드)');
            setIsHumanRequiredFlow(true);
          } else if (!isSessionEnd && !backendHumanRequiredFlow && currentHumanRequiredFlow) {
            console.log('[App] 스트리밍 모드 - 고객 상담사 연결 거부 - HUMAN_REQUIRED 플로우 종료');
            setIsHumanRequiredFlow(false);
          }

          // HANDOVER 감지: suggestedAction 또는 메시지 내용으로 판단 (메시지 표시용)
          const isHandoverSuggested =
            result.aiResponse.suggestedAction === 'HANDOVER' ||
            result.aiResponse.suggestedAction === 'handover' ||
            result.aiResponse.text.includes('상담사에게 연결해 드리겠습니다') ||
            result.aiResponse.text.includes('상담원에게 연결');

          // handover_status가 pending이면 상담사 수락 대기 폴링 시작
          // 안내 메시지는 백엔드(waiting_agent)에서 이미 ai_message에 포함됨
          if (result.aiResponse.handoverStatus === 'pending') {
            console.log('[App] 음성 모드 - handover_status=pending 감지 - 상담사 수락 대기 폴링 시작');
            setIsWaitingForAgent(true);
            setHandoverTimeoutReached(false);
            startHandoverPollingRef.current?.();
          }

          // AI가 HANDOVER를 권장한 경우: AI 응답만 표시하고 고객 동의 대기
          if (isHandoverSuggested) {
            console.log('[App] AI가 HANDOVER 권장 - 고객 동의 대기');

            // AI 응답 메시지 표시 (고객에게 동의 요청)
            // 고객이 "네"라고 응답하면 백엔드의 consent_check_node가 처리함
            const assistantMessage: Message = {
              id: `msg_${Date.now()}_assistant`,
              role: 'assistant',
              content: result.aiResponse.text,
              timestamp: new Date(),
              isNew: true,
            };
            setMessages((prev) => [...prev, assistantMessage]);

            // TTS는 백엔드에서 이미 재생 중이므로 중단하지 않음
            // 고객이 "네"라고 응답할 때까지 대기
            // 다음 메시지에서 백엔드가 consent_check_node → waiting_agent 플로우를 처리함
          } else {
            // 일반 AI 응답 메시지 표시
            console.log('[App] AI 응답 메시지 추가:', result.aiResponse.text);
            const assistantMessage: Message = {
              id: `msg_${Date.now()}_assistant`,
              role: 'assistant',
              content: result.aiResponse.text,
              timestamp: new Date(),
              isNew: true,
            };
            setMessages((prev) => [...prev, assistantMessage]);
          }
        } else if (!currentHandoverMode) {
          console.log('[App] aiResponse가 없거나 text가 비어있음');
        }
      } else {
        // result가 null인 경우 (빈 입력 등)
        emptyInputCountRef.current += 1;
        console.log(`[App] result가 null - 빈 입력 횟수: ${emptyInputCountRef.current}/${MAX_EMPTY_INPUTS}`);

        const currentHandoverMode = isHandoverModeRef.current;
        if (isContinuousMode && !currentHandoverMode && emptyInputCountRef.current < MAX_EMPTY_INPUTS) {
          setTimeout(() => {
            startRecording().catch((err) => {
              console.error('[App] 재녹음 시작 실패:', err);
            });
          }, 500);
        } else if (emptyInputCountRef.current >= MAX_EMPTY_INPUTS) {
          console.log('[App] 연속 빈 입력 감지 - 연속 대화 일시 중지');
          emptyInputCountRef.current = 0;
        }
      }
    });
  }, [setOnAutoStop, isContinuousMode, startRecording, sessionId]);

  // TTS 재생 완료 콜백 설정 (실시간 모드에서 자동 녹음 시작)
  // 실시간 모드: TTS 완료 후 항상 자동으로 마이크 활성화 (문제 1, 3 해결)
  // 중요: isHandoverModeRef.current를 사용하여 클로저 문제 해결
  useEffect(() => {
    setOnTTSComplete(() => {
      const currentHandoverMode = isHandoverModeRef.current;
      // 실시간 모드에서는 항상 자동 녹음, 녹음 모드에서는 isContinuousMode 필요
      if (!currentHandoverMode && !isRecordingMode) {
        console.log('[App] TTS 완료 - 실시간 모드 자동 녹음 시작');
        startRecording().catch((err) => {
          console.error('[App] 자동 녹음 시작 실패:', err);
        });
      } else if (!currentHandoverMode && isRecordingMode && isContinuousMode) {
        console.log('[App] TTS 완료 - 녹음 모드 연속 대화 자동 녹음 시작');
        startRecording().catch((err) => {
          console.error('[App] 자동 녹음 시작 실패:', err);
        });
      }
    });
  }, [setOnTTSComplete, isContinuousMode, isRecordingMode, startRecording]);

  // Barge-in 시 마지막 AI 메시지에 "[고객 발화로 중단]" 표시 추가
  // 텍스트가 너무 길면 앞부분만 보여주고 "..." 추가
  const markLastAiMessageAsInterrupted = useCallback(() => {
    setMessages((prev) => {
      // 마지막 AI 메시지 찾기
      const lastAiIndex = prev.map((m, i) => ({ m, i }))
        .filter(({ m }) => m.role === 'assistant')
        .pop()?.i;

      if (lastAiIndex === undefined) return prev;

      const lastAiMessage = prev[lastAiIndex];
      // 이미 "[고객 발화로 중단]"이 붙어있으면 스킵
      if (lastAiMessage.content.endsWith('[고객 발화로 중단]')) return prev;

      // 텍스트 길이 제한 (50자 초과 시 자르기)
      const MAX_LENGTH = 50;
      let truncatedContent = lastAiMessage.content;
      if (truncatedContent.length > MAX_LENGTH) {
        truncatedContent = truncatedContent.substring(0, MAX_LENGTH) + '...';
      }

      // 새 배열 생성 후 마지막 AI 메시지 수정
      const newMessages = [...prev];
      newMessages[lastAiIndex] = {
        ...lastAiMessage,
        content: truncatedContent + ' [고객 발화로 중단]',
      };
      return newMessages;
    });
  }, []);

  // Barge-in 콜백 설정 (TTS 재생 중 사용자가 말하면 TTS 중단 + 녹음 시작)
  // 실시간 스트리밍 모드용
  useEffect(() => {
    setOnBargeIn(() => {
      console.log('[App] Barge-in 감지 (스트리밍 모드) - 녹음 시작');

      // 첫 인사 TTS (외부 audioRef)도 중단
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        console.log('[App] 첫 인사 TTS 중단 (Barge-in)');
      }

      // 외부 TTS 재생 상태 해제
      setExternalTTSPlaying(false);

      markLastAiMessageAsInterrupted();
      startStreamRecording().catch((err) => {
        console.error('[App] Barge-in 녹음 시작 실패:', err);
      });
    });
  }, [setOnBargeIn, startStreamRecording, markLastAiMessageAsInterrupted, setExternalTTSPlaying]);

  // 녹음 모드용 Barge-in 콜백 설정
  useEffect(() => {
    setOnRecordBargeIn(() => {
      console.log('[App] Barge-in 감지 (녹음 모드) - 녹음 시작');

      // 첫 인사 TTS (외부 audioRef)도 중단
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        console.log('[App] 첫 인사 TTS 중단 (Barge-in - 녹음 모드)');
      }

      markLastAiMessageAsInterrupted();
      startRecordRecording().catch((err) => {
        console.error('[App] Barge-in 녹음 시작 실패:', err);
      });
    });
  }, [setOnRecordBargeIn, startRecordRecording, markLastAiMessageAsInterrupted]);

  // 녹음 모드용 추가 AI 응답 콜백 설정 (첫 번째 응답 후 추가 응답 처리)
  useEffect(() => {
    setOnRecordAiResponse((response) => {
      if (!response || !response.text) return;

      console.log('[App] 추가 AI 응답 수신 (녹음 모드):', response.text);

      const aiMessageContent = response.text;
      const assistantMessage: Message = {
        id: `msg_${Date.now()}_assistant_${Math.random().toString(36).substring(2, 11)}`,
        role: 'assistant',
        content: aiMessageContent,
        timestamp: new Date(),
        isNew: true,
      };

      // 중복 메시지 방지: 최근 5개 메시지 중 동일 content가 있으면 추가하지 않음
      setMessages((prev) => {
        const recentMessages = prev.slice(-5);
        const isDuplicate = recentMessages.some(
          (m) => m.role === 'assistant' && m.content === aiMessageContent
        );
        if (isDuplicate) {
          console.log('[App] 중복 AI 메시지 무시 (추가 응답):', aiMessageContent.substring(0, 30) + '...');
          return prev;
        }
        return [...prev, assistantMessage];
      });
    });
  }, [setOnRecordAiResponse]);

  // 마이크 버튼 클릭 핸들러
  const handleVoiceButtonClick = useCallback(async () => {
    console.log('[App] 버튼 클릭, isRecording:', isRecording);

    if (isRecording) {
      // 수동 녹음 중지
      await processStopRecording();
    } else {
      // 녹음 시작
      console.log('[App] 녹음 시작...');
      try {
        await startRecording();
        // 첫 녹음 시작 시 연속 대화 모드 활성화
        if (!isContinuousMode) {
          setIsContinuousMode(true);
          console.log('[App] 연속 대화 모드 활성화');
        }
        console.log('[App] 녹음 시작 완료');
      } catch (err) {
        console.error('녹음 시작 실패:', err);
      }
    }
  }, [isRecording, processStopRecording, startRecording, isContinuousMode]);

  // 핸드오버 폴링 정리 (먼저 선언 - handleForceStop에서 사용)
  const cleanupHandoverPolling = useCallback(() => {
    if (handoverPollIntervalRef.current) {
      clearInterval(handoverPollIntervalRef.current);
      handoverPollIntervalRef.current = null;
    }
    if (handoverTimeoutRef.current) {
      clearTimeout(handoverTimeoutRef.current);
      handoverTimeoutRef.current = null;
    }
  }, []);

  // 새 상담 시작 (대화 초기화 + 인사 메시지 + TTS)
  const handleResetSession = useCallback(async () => {
    if (window.confirm('새로운 상담을 시작하시겠습니까?')) {
      // 자동 녹음 시작 방지 플래그 설정
      isResettingRef.current = true;

      // 기존 WebSocket 연결 해제 (세션 ID 동기화 문제 방지)
      console.log('[App] 새 상담 시작 - 기존 WebSocket 연결 해제');
      disconnectStream();
      disconnectRecord();

      // 핸드오버 폴링 정리
      cleanupHandoverPolling();

      // 상태 초기화
      setIsContinuousMode(false);
      setIsHandoverMode(false);
      setIsWaitingForAgent(false);
      setIsHumanRequiredFlow(false);  // HUMAN_REQUIRED 플로우 초기화
      setIsStopped(false);  // 중지 상태 해제
      setHandoverData(null);
      emptyInputCountRef.current = 0;
      lastMessageIdRef.current = 0;
      processedAgentMessageIdsRef.current.clear();
      handoverAcceptedProcessingRef.current = false;  // 핸드오버 처리 플래그 초기화

      // 세션 ID 리셋 및 새 세션 ID 생성
      resetSessionId();
      const newSessionId = getOrCreateSessionId();
      setSessionId(newSessionId);
      setIsSessionStarted(true);  // 세션 시작됨 표시

      // 인사 메시지 표시
      const greetingMessage: Message = {
        id: `msg_${Date.now()}_greeting`,
        role: 'assistant',
        content: WELCOME_MESSAGE,
        timestamp: new Date(),
        isNew: true,
      };
      setMessages([greetingMessage]);

      // 인사 TTS 재생 (Barge-in 지원)
      try {
        const ttsResponse = await voiceApi.requestTTS(WELCOME_MESSAGE);
        if (ttsResponse.audio_base64) {
          console.log('[App] 인사 TTS 재생 시작 - Barge-in 감지 활성화');

          // 오디오 재생
          const audioBlob = base64ToBlob(ttsResponse.audio_base64, 'audio/mp3');
          const audioUrl = URL.createObjectURL(audioBlob);

          if (audioRef.current) {
            // Barge-in 감지를 위한 마이크 시작 (TTS 재생과 동시에)
            isResettingRef.current = false;  // 녹음 허용

            // 실시간 모드에서 WebSocket 연결 및 VAD 시작
            if (!isRecordingMode) {
              console.log('[App] 인사 TTS 중 Barge-in 감지를 위해 마이크 시작');

              // 먼저 마이크 시작하여 WebSocket 연결
              startStreamRecording().then(() => {
                // 연결 완료 후 외부 TTS 재생 상태 설정 (Barge-in 감지 활성화)
                setExternalTTSPlaying(true);
                console.log('[App] WebSocket 연결 완료 - 외부 TTS 재생 상태 설정');
              }).catch((err) => {
                console.error('[App] 인사 TTS 중 마이크 시작 실패:', err);
              });
            }

            // 오디오 재생
            audioRef.current.src = audioUrl;
            audioRef.current.play();

            // 오디오 종료 시 처리 (Barge-in으로 중단되지 않은 경우)
            audioRef.current.onended = () => {
              console.log('[App] 인사 TTS 재생 완료');
              URL.revokeObjectURL(audioUrl);
              // 외부 TTS 재생 상태 해제
              setExternalTTSPlaying(false);
            };
          }
        }
      } catch (err) {
        console.warn('인사 TTS 재생 실패:', err);
        // TTS 실패 시에도 녹음은 시작
        isResettingRef.current = false;
        if (!isRecordingMode) {
          setTimeout(() => {
            startStreamRecording().catch((err) => {
              console.error('[App] 새 상담 후 자동 녹음 시작 실패:', err);
            });
          }, 500);
        }
      }
    }
  }, [disconnectStream, disconnectRecord, cleanupHandoverPolling, isRecordingMode, startStreamRecording, setExternalTTSPlaying]);

  // 강제 종료 (현재 상담 중지 - 대기 상태로 전환)
  const handleForceStop = useCallback(() => {
    console.log('[App] 강제 종료 - 모든 처리 중단');

    // 자동 녹음 시작 방지 플래그 설정
    isResettingRef.current = true;

    // TTS 재생 중지
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    // WebSocket 연결 해제 (녹음/STT/TTS 모두 중단)
    disconnectStream();
    disconnectRecord();

    // 핸드오버 폴링 중지
    cleanupHandoverPolling();
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }

    // 핸드오버 관련 상태 초기화 (대기 중 상태만 해제, 연결된 상태는 유지)
    setIsWaitingForAgent(false);
    setIsHumanRequiredFlow(false);
    // isHandoverMode는 유지 - 상담사 연결된 상태에서 중지 후 재시작 시 계속 대화 가능하도록

    // 연속 대화 모드 해제
    setIsContinuousMode(false);

    // 중지 상태 표시
    setIsStopped(true);

    // 강제 종료 후에는 자동 녹음 시작하지 않음 (대기 상태 유지)
    // isResettingRef.current는 true로 유지하여 자동 녹음 방지
    console.log('[App] 강제 종료 완료 - 마이크 버튼을 눌러 다시 시작하세요');
  }, [disconnectStream, disconnectRecord, cleanupHandoverPolling]);

  // 마이크 버튼 클릭 시 isResettingRef 해제 (강제 종료 후 재시작 허용)
  const handleVoiceButtonClickWithReset = useCallback(async () => {
    // 강제 종료 상태였다면 해제
    if (isResettingRef.current) {
      isResettingRef.current = false;
    }
    // 중지 상태 해제 -> useEffect 의존성으로 폴링 자동 재시작
    setIsStopped(false);
    await handleVoiceButtonClick();
  }, [handleVoiceButtonClick]);

  // 상담사 수락 상태 폴링 시작
  const startHandoverPolling = useCallback(() => {
    // 기존 폴링 정리
    cleanupHandoverPolling();
    // accepted 처리 플래그 초기화 (새로운 폴링 시작)
    handoverAcceptedProcessingRef.current = false;

    // 상담사 수락 여부 폴링
    handoverPollIntervalRef.current = setInterval(async () => {
      // 이미 accepted 처리 중이면 스킵 (중복 방지)
      if (handoverAcceptedProcessingRef.current) {
        console.log('[App] 핸드오버 accepted 처리 중 - 스킵');
        return;
      }

      try {
        const status = await voiceApi.getHandoverStatus(sessionId);
        console.log('[App] 핸드오버 상태:', status.handover_status);

        if (status.handover_status === 'accepted') {
          // 중복 처리 방지 플래그 설정
          if (handoverAcceptedProcessingRef.current) {
            console.log('[App] 핸드오버 accepted 이미 처리됨 - 스킵');
            return;
          }
          handoverAcceptedProcessingRef.current = true;

          // 상담사가 수락함 → 모달 없이 바로 연결
          cleanupHandoverPolling();
          setIsWaitingForAgent(false);
          setIsHumanRequiredFlow(false);  // HUMAN_REQUIRED 플로우 종료
          setIsHandoverMode(true);  // 상담사 메시지 폴링 시작

          // 연결 완료 안내 메시지
          const connectedMessage = '상담사에게 연결되었습니다. 상담을 시작합니다.';
          const aiMessage: Message = {
            id: `msg_${Date.now()}_agent_connected`,
            role: 'assistant',
            content: connectedMessage,
            timestamp: new Date(),
            isNew: true,
          };
          setMessages((prev) => [...prev, aiMessage]);

          // TTS로 연결 완료 메시지 재생
          try {
            const ttsResponse = await voiceApi.requestTTS(connectedMessage);
            if (ttsResponse.audio_base64) {
              playAudio(ttsResponse.audio_base64);
            }
          } catch (ttsErr) {
            console.warn('TTS 재생 실패:', ttsErr);
          }
        } else if (status.handover_status === 'declined') {
          // 고객이 상담사 연결을 거부함
          console.log('[App] 핸드오버 거부됨 - 폴링 중지');
          cleanupHandoverPolling();
          setIsWaitingForAgent(false);
          setIsHumanRequiredFlow(false);  // HUMAN_REQUIRED 플로우 종료
          // isHandoverMode는 false로 유지 (일반 대화로 복귀)
        }
      } catch (err) {
        console.error('[App] 핸드오버 상태 폴링 실패:', err);
      }
    }, HANDOVER_POLL_INTERVAL_MS);

    // 타임아웃 설정 - 메시지만 표시하고 폴링은 계속 유지
    handoverTimeoutRef.current = setTimeout(async () => {
      // 폴링은 계속 유지 (cleanupHandoverPolling 호출 안 함)
      // isWaitingForAgent도 true로 유지

      // 타임아웃 안내 메시지 (채팅창에 표시)
      const timeoutMessage: Message = {
        id: `msg_${Date.now()}_timeout`,
        role: 'assistant',
        content: HANDOVER_WAIT_TIME_MESSAGE + ' 계속 기다리시려면 잠시만 기다려 주세요. 추가 문의가 있으시면 말씀해 주세요.',
        timestamp: new Date(),
        isNew: true,
      };
      setMessages((prev) => [...prev, timeoutMessage]);

      // TTS 재생
      try {
        const ttsResponse = await voiceApi.requestTTS(timeoutMessage.content);
        if (ttsResponse.audio_base64) {
          playAudio(ttsResponse.audio_base64);
        }
      } catch (ttsErr) {
        console.warn('TTS 재생 실패:', ttsErr);
      }
    }, HANDOVER_TIMEOUT_MS);
  }, [sessionId, cleanupHandoverPolling, playAudio]);

  // startHandoverPolling ref 업데이트 (클로저 문제 해결)
  useEffect(() => {
    startHandoverPollingRef.current = startHandoverPolling;
  }, [startHandoverPolling]);

  // 상담원 연결 요청 (새로운 시나리오)
  const handleRequestHandover = useCallback(async () => {
    setIsHandoverLoading(true);
    try {
      // 1단계: 핸드오버 요청 → pending 상태로 설정
      const response = await voiceApi.requestHandoverWithStatus(sessionId);
      console.log('[App] 핸드오버 요청 응답:', response);

      // 안내 메시지 표시
      const waitingMessage: Message = {
        id: `msg_${Date.now()}_waiting`,
        role: 'assistant',
        content: '현재 응대 가능한 상담사가 있는지 확인을 해 보겠습니다. 잠시만 기다려 주시기 바랍니다.',
        timestamp: new Date(),
        isNew: true,
      };
      setMessages((prev) => [...prev, waitingMessage]);

      // TTS 재생
      try {
        const ttsResponse = await voiceApi.requestTTS(waitingMessage.content);
        if (ttsResponse.audio_base64) {
          playAudio(ttsResponse.audio_base64);
        }
      } catch (ttsErr) {
        console.warn('TTS 재생 실패:', ttsErr);
      }

      // 상담사 수락 대기 시작
      setIsWaitingForAgent(true);
      setHandoverTimeoutReached(false);
      startHandoverPolling();

    } catch (err) {
      console.error('상담원 이관 요청 실패:', err);
      alert('상담원 이관 요청 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setIsHandoverLoading(false);
    }
  }, [sessionId, playAudio, startHandoverPolling]);
  void handleRequestHandover; // Reserved for future use

  // 고객이 상담사 연결 확인 (미사용 - 모달 제거로 자동 연결됨)
  const _handleConfirmAgentConnection = useCallback(async () => {
    // 중복 호출 방지
    if (isConfirmingHandoverRef.current) {
      console.log('[App] confirmHandover 이미 진행 중 - 스킵 (버튼)');
      return;
    }
    isConfirmingHandoverRef.current = true;
    setIsHandoverLoading(true);

    try {
      // 기존 handover/request 호출하여 분석 결과 가져오기
      const response = await voiceApi.confirmHandover(sessionId);
      setHandoverData(response);
      setIsHandoverMode(true);  // 실제 상담원 모드 활성화 → 메시지 폴링 시작

      // 연결 완료 메시지
      const connectedMessage: Message = {
        id: `msg_${Date.now()}_connected`,
        role: 'assistant',
        content: '상담원에게 연결되었습니다. 상담을 시작합니다.',
        timestamp: new Date(),
        isNew: true,
      };
      setMessages((prev) => [...prev, connectedMessage]);

      // TTS 재생
      try {
        const ttsResponse = await voiceApi.requestTTS(connectedMessage.content);
        if (ttsResponse.audio_base64) {
          playAudio(ttsResponse.audio_base64);
        }
      } catch (ttsErr) {
        console.warn('TTS 재생 실패:', ttsErr);
      }
    } catch (err) {
      console.error('상담원 연결 확인 실패:', err);
      alert('상담원 연결 중 오류가 발생했습니다.');
    } finally {
      setIsHandoverLoading(false);
      isConfirmingHandoverRef.current = false;
    }
  }, [sessionId, playAudio]);
  void _handleConfirmAgentConnection;

  // 고객이 상담사 연결 거부 (미사용 - 모달 제거됨)
  const _handleDeclineAgentConnection = useCallback(async () => {
    setIsWaitingForAgent(false);
    setHandoverTimeoutReached(false);
    setIsHandoverMode(false);  // 상담사 메시지 폴링 중지
    cleanupHandoverPolling();

    // 안내 메시지
    const declineMessage: Message = {
      id: `msg_${Date.now()}_decline`,
      role: 'assistant',
      content: '알겠습니다. 추가로 문의하실 내용이 있으시면 말씀해 주세요.',
      timestamp: new Date(),
      isNew: true,
    };
    setMessages((prev) => [...prev, declineMessage]);

    // TTS 재생
    try {
      const ttsResponse = await voiceApi.requestTTS(declineMessage.content);
      if (ttsResponse.audio_base64) {
        playAudio(ttsResponse.audio_base64);
      }
    } catch (ttsErr) {
      console.warn('TTS 재생 실패:', ttsErr);
    }
  }, [cleanupHandoverPolling, playAudio]);
  void _handleDeclineAgentConnection;

  // 타임아웃 후 계속 대기 선택 (향후 확장용)
  const _handleContinueWaiting = useCallback(() => {
    setHandoverTimeoutReached(false);
    startHandoverPolling();  // 폴링 재시작

    const waitMessage: Message = {
      id: `msg_${Date.now()}_continue_wait`,
      role: 'assistant',
      content: '네, 계속 대기하겠습니다. 상담사가 응대 가능해지면 안내해 드리겠습니다.',
      timestamp: new Date(),
      isNew: true,
    };
    setMessages((prev) => [...prev, waitMessage]);
  }, [startHandoverPolling]);
  void _handleContinueWaiting;

  // 컴포넌트 언마운트 시 핸드오버 폴링 정리
  useEffect(() => {
    return () => {
      cleanupHandoverPolling();
    };
  }, [cleanupHandoverPolling]);

  // 모달 닫기 (미사용 - 향후 확장용)
  const _handleCloseModal = useCallback(() => {
    setHandoverData(null);
  }, []);
  void _handleCloseModal;

  // 텍스트 메시지 전송
  const handleTextSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();

    const trimmedInput = textInput.trim();
    if (!trimmedInput || isTextSending) return;

    // 중지 상태 해제 -> useEffect 의존성으로 폴링 자동 재시작
    if (isStopped) {
      setIsStopped(false);
    }

    // 고객 활동 감지 - 리마인더 타이머 리셋
    resetInactivityTimer();

    setIsTextSending(true);
    setTextInput('');

    // 사용자 메시지 추가
    const userMessage: Message = {
      id: `msg_${Date.now()}_user`,
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
      isVoice: false,
      isNew: true,
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      // 이관 모드일 때: 상담원에게 메시지 전송
      if (isHandoverMode) {
        await voiceApi.sendCustomerMessage(sessionId, trimmedInput);
        console.log('[App] 상담원에게 텍스트 메시지 전송 완료');
      } else {
        // 일반 모드: AI 응답 받기
        const response = await voiceApi.sendTextMessage(sessionId, trimmedInput);

        console.log('[App] AI 응답:', {
          suggested_action: response.suggested_action,
          handover_status: response.handover_status,
          info_collection_complete: response.info_collection_complete
        });

        // AI 응답 메시지 추가
        const assistantMessage: Message = {
          id: `msg_${Date.now()}_assistant`,
          role: 'assistant',
          content: response.ai_message,
          timestamp: new Date(),
          isNew: true,
        };
        setMessages((prev) => [...prev, assistantMessage]);

        // TTS 재생
        try {
          const ttsResponse = await voiceApi.requestTTS(response.ai_message);
          if (ttsResponse.audio_base64) {
            playAudio(ttsResponse.audio_base64);
          }
        } catch (ttsErr) {
          console.warn('TTS 재생 실패:', ttsErr);
        }

        // 세션 종료 감지 (불명확 응답/도메인 외 질문 3회 이상)
        const isSessionEnd = response.is_session_end || false;
        console.log('[App] is_human_required_flow:', response.is_human_required_flow, ', is_session_end:', isSessionEnd, ', 현재 상태:', isHumanRequiredFlow);

        if (isSessionEnd) {
          console.log('[App] 텍스트 모드 - 세션 종료 감지 (is_session_end=true)');
          setIsHumanRequiredFlow(false);
          setIsWaitingForAgent(false);
          cleanupHandoverPolling();
        } else if (response.is_human_required_flow && !isHumanRequiredFlow) {
          console.log('[App] HUMAN_REQUIRED 플로우 진입 감지 (백엔드)');
          setIsHumanRequiredFlow(true);
        } else if (!response.is_human_required_flow && isHumanRequiredFlow) {
          // 고객이 상담사 연결을 거부한 경우 (is_human_required_flow가 false로 변경됨)
          console.log('[App] 고객 상담사 연결 거부 - HUMAN_REQUIRED 플로우 종료');
          setIsHumanRequiredFlow(false);
        }

        // handover_status가 pending이면 상담사 수락 대기 폴링 시작
        // 안내 메시지는 백엔드(waiting_agent)에서 이미 ai_message에 포함되어 있으므로 추가 표시 불필요
        if (response.handover_status === 'pending' && !isWaitingForAgent) {
          console.log('[App] handover_status=pending 감지 - 상담사 수락 대기 폴링 시작');
          setIsWaitingForAgent(true);
          setHandoverTimeoutReached(false);
          startHandoverPolling();
        }
      }
    } catch (err) {
      console.error('텍스트 메시지 전송 실패:', err);
      // 에러 메시지 표시
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        role: 'assistant',
        content: '죄송합니다. 메시지 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
        timestamp: new Date(),
        isNew: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsTextSending(false);
      textInputRef.current?.focus();
    }
  }, [textInput, isTextSending, isHandoverMode, sessionId, isWaitingForAgent, isHumanRequiredFlow, startHandoverPolling, playAudio, resetInactivityTimer, isStopped]);

  // Enter 키 핸들러
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleTextSubmit();
    }
  }, [handleTextSubmit]);

  const hasMessages = messages.length > 0;

  return (
    <div className="app">
      <div className="chat-window">
        {/* 헤더 */}
        <div className="chat-header">
          <div className="chat-header-content">
            <h1>상담 보이스봇</h1>
            <p>카드사 AI 에이전트</p>
          </div>
          <div className="header-actions">
            {/* 음성 모드 토글 */}
            <div className="voice-mode-toggle">
              <label className="toggle-label">
                <span className={!isRecordingMode ? 'active' : ''}>실시간</span>
                <input
                  type="checkbox"
                  checked={isRecordingMode}
                  onChange={(e) => setIsRecordingMode(e.target.checked)}
                  disabled={isRecording || isProcessing}
                />
                <span className="toggle-slider"></span>
                <span className={isRecordingMode ? 'active' : ''}>녹음</span>
              </label>
            </div>
            {/* 세션 정보 및 버튼들 */}
            <div className="session-info-header">
              {/* 세션 시작 후에만 세션 번호 표시 */}
              {isSessionStarted && (
                <span className="session-id">세션: {formatSessionIdForDisplay(sessionId)}</span>
              )}
              {/* 강제 종료 버튼 (세션 시작 후 항상 표시) */}
              {isSessionStarted && (
                <button onClick={handleForceStop} className="stop-button-header">
                  중지
                </button>
              )}
              <button onClick={handleResetSession} className="reset-button-header">
                새 상담
              </button>
            </div>
          </div>
        </div>

        {/* 메시지 영역 */}
        <div className="chat-messages">
          {/* 초기 화면 - 메시지 없을 때 안내 */}
          {!hasMessages && (
            <div className="welcome-prompt">
              <p>'새 상담' 버튼을 눌러 상담을 시작하세요.</p>
            </div>
          )}

          {/* 메시지 목록 */}
          {hasMessages && (
            <>
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}

              {/* 실시간 STT 표시 */}
              {isRecording && currentTranscript && (
                <div className="realtime-transcript">
                  <div className="transcript-content">
                    <span className="transcript-icon">🎤</span>
                    <span className="transcript-text">
                      {currentTranscript}
                      <span className="cursor-blink">|</span>
                    </span>
                  </div>
                </div>
              )}

              {/* 녹음 중 표시 (텍스트 없을 때) */}
              {isRecording && !currentTranscript && (
                <div className="recording-indicator">
                  <div className="recording-animation">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <p>
                    {isRecordingMode
                      ? `녹음 중... ${recordingTime}초 ${isRecordSpeaking ? '🎤 음성 감지' : ''} (버튼을 눌러 전송)`
                      : '듣고 있습니다...'}
                  </p>
                  {/* 녹음 모드에서 VAD 음성 확률 표시 */}
                  {isRecordingMode && recordSpeechProb > 0 && (
                    <div className="vad-indicator" style={{
                      marginTop: '8px',
                      fontSize: '12px',
                      color: isRecordSpeaking ? '#4CAF50' : '#999',
                    }}>
                      음성 확률: {(recordSpeechProb * 100).toFixed(0)}%
                    </div>
                  )}
                </div>
              )}

              {isProcessing && (
                <div className="processing-message">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}

              {/* TTS 재생 중 표시 */}
              {isPlayingTTS && !isRecording && (
                <div className="tts-playing-indicator">
                  <span className="speaker-icon">🔊</span>
                  <span>응답 재생 중... (말씀하시면 중단됩니다)</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* 하단 입력 영역 */}
        <div className="chat-input-area">
          {/* 텍스트 입력 */}
          <form className="text-input-form" onSubmit={handleTextSubmit}>
            <input
              ref={textInputRef}
              type="text"
              className="text-input"
              placeholder={isHandoverMode ? "상담원에게 메시지 입력..." : "메시지를 입력하세요..."}
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isTextSending || isRecording || isProcessing}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!textInput.trim() || isTextSending || isRecording || isProcessing}
            >
              {isTextSending ? '전송중...' : '전송'}
            </button>
          </form>

          {/* 음성 버튼 */}
          <VoiceButton
            isRecording={isRecording}
            isProcessing={isProcessing}
            onClick={handleVoiceButtonClickWithReset}
            size="small"
          />
        </div>
      </div>

      {/* 숨겨진 오디오 플레이어 */}
      <audio ref={audioRef} style={{ display: 'none' }} />

      {/* 모달들이 제거됨 - 상담사 수락 시 자동 연결 */}

      {/* 상담사 대기 중 표시 */}
      {isWaitingForAgent && (
        <div className="waiting-indicator">
          <div className="waiting-spinner"></div>
          <span>상담사 연결 대기 중...</span>
        </div>
      )}

      {/* 중지 상태 표시 */}
      {isStopped && (
        <div className="stopped-indicator">
          <span>상담이 일시 중지 상태입니다. 계속하시려면 마이크 버튼을 눌러주세요.</span>
        </div>
      )}
    </div>
  );
}

export default App;
