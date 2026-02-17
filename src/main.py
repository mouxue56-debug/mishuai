"""Fukuraku AI Secretary - Main Entry Point.

Starts all subsystems and runs the main event loop.

Key architectural fixes (v0.3):
- Vision input: Camera → Gemini Flash for scene understanding
- Proactive scheduler: Time/idle/vision-triggered speech
- Speaker ID: Real voiceprint extraction (resemblyzer / 3D-Speaker)
- Unified output consumer: Both voice and text modes share the same TTS/WS path
- Audio sent via WebSocket as base64 for frontend playback (OBS-compatible)

Usage:
    # Full system (voice + vision + Live2D frontend)
    python -m src.main

    # Text-only mode (no microphone/camera, WebSocket chat only)
    python -m src.main --text-only

    # Test LLM connection
    python -m src.main --test-llm
"""

import argparse
import asyncio
import base64
import re
import signal
import subprocess
import uuid

from src.api.config_api import ConfigAPI
from src.api.knowledge_api import KnowledgeAPI
from src.api.memory_api import MemoryAPI
from src.api.speaker_api import SpeakerAPI
from src.core.interrupt_handler import PipelineState
from src.core.pipeline import DialoguePipeline
from src.core.proactive_scheduler import ProactiveScheduler
from src.audio.microphone import MicrophoneInput
from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.audio.speaker_id import SpeakerIdentifier
from src.audio.wake_word import WakeWordDetector
from src.display.websocket_server import WebSocketServer
from src.vision.camera import VisionInput
from src.mcp.tools.memo_reminder import set_memory_manager
from src.utils.config_loader import load_env, get_main_config, validate_configs
from src.utils.logger import setup_logger, get_logger


logger = get_logger("main")


class FukurakuSecretary:
    """Main application class that orchestrates all subsystems.

    Architecture (v0.3):
        Text/Voice Input ──→ Pipeline (LLM + Tools) → Output Queue
        Camera (Vision) ──→ Proactive Scheduler ──────↗     ↓
        Time/Idle triggers ─────────────────────────↗  Output Consumer
                                                       ├── TTS synthesis
                                                       ├── WS: speech_start + emotion
                                                       ├── WS: audio (base64) + lip_sync
                                                       ├── WS: speech_end
                                                       └── State: SPEAKING → IDLE
    """

    def __init__(self, text_only: bool = False):
        self.text_only = text_only
        self.pipeline = DialoguePipeline()
        self.ws_server = WebSocketServer()
        self.tts = TTSEngine()
        self.scheduler = ProactiveScheduler()
        self.config_api = ConfigAPI()
        self.knowledge_api = KnowledgeAPI()
        self.memory_api = None  # Initialized after pipeline.initialize()
        self.speaker_api = None  # Initialized after pipeline.initialize()

        # Audio components (disabled in text-only mode)
        self.microphone = None
        self.asr = None
        self.speaker_id = None
        self.wake_word = None

        # ASR and Speaker ID are always initialized (needed for browser audio input)
        self.asr = ASREngine()
        self.speaker_id = SpeakerIdentifier()

        # Vision (enabled independently of text-only mode)
        self.vision = VisionInput()

        if not text_only:
            self.microphone = MicrophoneInput()
            self.wake_word = WakeWordDetector()

        self._running = False

        # Error recovery components (py-xiaozhi inspired)
        from src.utils.error_recovery import (
            SilencePeriod, CircuitBreaker, TaskLifecycle
        )
        self._silence_period = SilencePeriod(duration_ms=250)  # Anti-echo after TTS
        self._tts_circuit = CircuitBreaker(
            failure_threshold=3, cooldown_sec=30, name="tts"
        )
        self._llm_circuit = CircuitBreaker(
            failure_threshold=3, cooldown_sec=60, name="llm"
        )
        self._task_lifecycle = TaskLifecycle()

        # Voice enrollment state
        self._enrollment_mode = False
        self._enrollment_speaker_id = None   # who is being enrolled (new person)
        self._enrollment_inviter = None      # who triggered enrollment (registered person)
        self._enrollment_samples: list[bytes] = []
        self._enrollment_step = 0
        self._enrollment_prompts = [
            "今日はいい天気ですね。お散歩日和です。",
            "福楽キャッテリーへようこそ。猫ちゃんたちが待っていますよ。",
            "お気に入りの猫ちゃんが見つかるといいですね。",
        ]

    async def start(self):
        """Initialize and start all subsystems."""
        logger.info("=" * 50)
        logger.info("  Fukuraku AI Secretary v0.3.0")
        logger.info("  福楽キャッテリー AI秘書 「ミケ」")
        logger.info("=" * 50)

        # Initialize pipeline (memory + tools)
        await self.pipeline.initialize()

        # Share memory manager with MCP tools
        set_memory_manager(self.pipeline.memory)

        # Initialize TTS
        await self.tts.initialize()

        # Initialize ASR + Speaker ID (needed for browser audio input even in text-only)
        await self.asr.initialize()
        if self.speaker_id:
            await self.speaker_id.initialize()

        # Initialize microphone and wake word (only in full mode)
        if not self.text_only:
            if self.wake_word:
                await self.wake_word.initialize()

        # Initialize vision
        await self.vision.initialize()

        # Set up vision callbacks → scheduler
        self.vision.on_person_detected(self._on_person_detected)
        self.vision.on_scene_change(self._on_scene_change)

        # Set up proactive scheduler → pipeline output queue
        self.scheduler.on_proactive_speak(self._handle_proactive_speak)

        # Register subsystems with ConfigAPI for runtime config updates
        self.config_api.register_subsystems(
            tts=self.tts,
            llm=self.pipeline.llm,
            scheduler=self.scheduler,
            vision=self.vision,
            memory=self.pipeline.memory,
        )

        # Initialize MemoryAPI for frontend CRUD operations
        self.memory_api = MemoryAPI(self.pipeline.memory)

        # Initialize SpeakerAPI for voice print management
        self.speaker_api = SpeakerAPI(
            self.pipeline.memory.long_term,
            self.speaker_id,
        )

        # Auto-register boss speaker if no speakers exist
        await self._ensure_boss_speaker()

        # Bind KnowledgeAPI to the CatteryKnowledgeTool for index invalidation
        if hasattr(self.pipeline, 'tools'):
            kt = self.pipeline.tools.get_tool('cattery_knowledge')
            if kt:
                self.knowledge_api.set_knowledge_tool(kt)

        # Set up WebSocket server
        self.ws_server.on_text_input(self._handle_text_input)
        self.ws_server.on_command(self._handle_command)
        self.ws_server.on_audio_input(self._handle_audio_input)
        self.pipeline.set_ws_broadcast(self.ws_server.broadcast)

        # Start WebSocket server
        await self.ws_server.start()

        # Start main loop
        self._running = True
        logger.info("System ready! Waiting for input...")
        logger.info(f"Mode: {'Text-only' if self.text_only else 'Full (Voice + Text)'}")
        logger.info(f"Vision: {'enabled' if self.vision.enabled else 'disabled'}")
        logger.info(f"Speaker ID: {'enabled' if (self.speaker_id and self.speaker_id.enabled) else 'disabled'}")
        logger.info(f"WebSocket: ws://localhost:{self.ws_server.port}")
        logger.info(f"Frontend: Open frontend/index.html in a browser")

        # Run all tasks concurrently
        tasks = [
            self.pipeline.run(),
            self._output_consumer(),
            self._reminder_checker(),
            self.scheduler.start(),
        ]

        if not self.text_only and self.microphone:
            tasks.append(self._voice_loop())

        if self.vision.enabled:
            tasks.append(self.vision.start())

        await asyncio.gather(*tasks)

    async def stop(self):
        """Gracefully shut down all subsystems."""
        logger.info("Shutting down...")
        self._running = False

        # Cancel all tracked tasks first (py-xiaozhi lifecycle pattern)
        await self._task_lifecycle.shutdown(timeout=3.0)

        if self.microphone:
            await self.microphone.stop()

        await self.vision.stop()
        await self.scheduler.stop()
        await self.ws_server.stop()
        await self.pipeline.shutdown()
        logger.info("Goodbye!")

    # ----------------------------------------------------------------
    # Vision Callbacks
    # ----------------------------------------------------------------

    async def _on_person_detected(self, description: str):
        """Called when the camera detects a person."""
        logger.info(f"[Vision] Person detected: {description}")
        # Feed to proactive scheduler for greeting
        self.scheduler.notify_vision_event(description)

    async def _on_scene_change(self, description: str):
        """Called when the camera scene changes significantly."""
        logger.debug(f"[Vision] Scene: {description}")

    # ----------------------------------------------------------------
    # Proactive Speech Handler
    # ----------------------------------------------------------------

    async def _handle_proactive_speak(self, context: str, trigger: str):
        """Handle proactive speech from the scheduler.

        Sends the context through the LLM so the response has personality,
        rather than using a hardcoded message.
        """
        request_id = f"proactive-{trigger}-{str(uuid.uuid4())[:4]}"
        logger.info(f"[{request_id}] Proactive trigger: {trigger}")

        # Build a prompt for the LLM to generate a natural proactive message
        if trigger == "vision":
            prompt = (
                f"[カメラ入力] 受付カメラで以下が検出されました: {context}\n"
                "来客に気づいたように自然に声をかけてください。"
            )
        elif trigger == "greeting":
            prompt = f"[時間挨拶] {context}"
        elif trigger == "idle":
            prompt = f"[待機中] {context}"
        else:
            prompt = context

        # Enqueue as a special "proactive" input
        await self.pipeline.input_queue.put({
            "type": "text",
            "text": prompt,
            "speaker_id": None,
            "request_id": request_id,
            "intent": "proactive",
        })

    # ----------------------------------------------------------------
    # Unified Output Consumer
    # ----------------------------------------------------------------

    async def _output_consumer(self):
        """Unified output consumer - handles both streaming and legacy responses.

        Two paths:
        1. stream_start: Streaming pipeline (Pipecat-inspired)
           - LLM streams tokens → sentence splitter → TTS per sentence → audio chunks
           - First audio plays in <1s while LLM/TTS continue in background
        2. response: Legacy non-streaming (enrollment, tools, reminders)
           - Full response → single TTS → single audio delivery

        Key design (N.E.K.O speech_id + Pipecat streaming):
        - Each speech turn gets a unique speech_id
        - Multiple audio chunks share the same speech_id
        - Frontend queues audio chunks for gapless playback
        - Interruption cancels remaining sentences
        """
        import time as _time

        while self._running:
            try:
                result = await asyncio.wait_for(
                    self.pipeline.output_queue.get(), timeout=1.0
                )

                result_type = result.get("type")

                if result_type == "stream_start":
                    await self._handle_streaming_output(result)
                elif result_type == "response":
                    await self._handle_legacy_output(result)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Output consumer error: {e}")
                self.pipeline.interrupt.set_state(PipelineState.IDLE)
                self.pipeline.interrupt.current_speech_id = None
                if self.microphone:
                    self.microphone.stop_suppress()
                continue

    async def _handle_streaming_output(self, result: dict):
        """Handle streaming pipeline output (Pipecat-inspired).

        Flow:
        1. Start streaming LLM → sentence splitter
        2. For each complete sentence:
           a. Synthesize TTS for that sentence
           b. Send audio_chunk to frontend (queued playback)
           c. Continue LLM streaming in parallel
        3. After all sentences delivered, wait for playback to finish
        4. Send speech_end
        """
        import time as _time

        request_id = result.get("request_id", "?")
        speech_id = f"sp-{str(uuid.uuid4())[:8]}"

        logger.info(f"[{request_id}] Streaming output (speech_id={speech_id})")

        # Notify scheduler
        self.scheduler.notify_interaction()

        # Set SPEAKING state
        self.pipeline.interrupt.set_state(PipelineState.SPEAKING)
        self.pipeline.interrupt.clear_interrupt()
        self.pipeline.interrupt.current_speech_id = speech_id
        if self.microphone:
            self.microphone.suppress_echo()

        # Notify frontend: speech starting (streaming mode)
        await self.ws_server.broadcast({
            "type": "speech_start",
            "text": "",
            "emotion": "neutral",
            "speech_id": speech_id,
            "streaming": True,
        })
        await self.ws_server.send_status("speaking")

        # Run the streaming pipeline
        total_duration = 0.0
        total_bytes = 0
        audio_format = "wav"
        turn = None
        full_text = ""
        emotion = "neutral"
        sentences_sent = 0

        try:
            async for chunk in self.pipeline.process_text_input_stream(
                text=result.get("text", ""),
                speaker_id=result.get("speaker_id"),
                request_id=request_id,
                intent=result.get("intent", "chat"),
                session_id=result.get("session_id"),
            ):
                if chunk["type"] == "sentence":
                    sentence = chunk["sentence"]
                    turn = chunk["turn"]

                    # Check for interruption
                    if self.pipeline.interrupt.should_stop():
                        logger.info(f"[{request_id}] Streaming interrupted at sentence {sentences_sent}")
                        if turn:
                            turn.aborted = True
                        break

                    # Synthesize this sentence (streaming mode for lower latency)
                    audio_data = await self.tts.synthesize_cosyvoice_streaming(sentence)

                    if not audio_data:
                        continue

                    # Check interruption after TTS
                    if self.pipeline.interrupt.should_stop():
                        if turn:
                            turn.aborted = True
                        break

                    # Send audio chunk to frontend
                    audio_b64 = base64.b64encode(audio_data).decode("ascii")
                    fmt = "wav" if audio_data[:4] == b"RIFF" else "mp3"
                    audio_format = fmt

                    await self.ws_server.broadcast({
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "format": fmt,
                        "speech_id": speech_id,
                        "chunk_index": sentences_sent,
                        "sentence": sentence,
                    })

                    # Update subtitle incrementally
                    full_text += sentence
                    await self.ws_server.send_subtitle(full_text, "assistant")

                    duration = self._calculate_audio_duration(audio_data, fmt)
                    total_duration += duration
                    total_bytes += len(audio_data)
                    sentences_sent += 1

                    logger.info(
                        f"[{request_id}] Chunk {sentences_sent}: '{sentence[:30]}' "
                        f"({len(audio_data)}b, {duration:.1f}s)"
                    )

                elif chunk["type"] == "final":
                    turn = chunk.get("turn")
                    full_text = chunk.get("full_text", full_text)
                    emotion = chunk.get("emotion", "neutral")

                    # Handle tool call responses (synthesize tool result)
                    if chunk.get("tool_calls") and chunk.get("response"):
                        tool_response = chunk["response"]
                        if tool_response.text and tool_response.text != full_text:
                            audio_data = await self.tts.synthesize(
                                tool_response.text, emotion=tool_response.emotion
                            )
                            if audio_data and not self.pipeline.interrupt.should_stop():
                                audio_b64 = base64.b64encode(audio_data).decode("ascii")
                                fmt = "wav" if audio_data[:4] == b"RIFF" else "mp3"
                                await self.ws_server.broadcast({
                                    "type": "audio_chunk",
                                    "data": audio_b64,
                                    "format": fmt,
                                    "speech_id": speech_id,
                                    "chunk_index": sentences_sent,
                                    "sentence": tool_response.text,
                                })
                                duration = self._calculate_audio_duration(audio_data, fmt)
                                total_duration += duration
                                total_bytes += len(audio_data)
                                sentences_sent += 1
                                full_text = tool_response.text

                    # Update emotion on frontend
                    await self.ws_server.broadcast({
                        "type": "emotion",
                        "emotion": emotion,
                        "expression": emotion,
                        "motion_group": "Idle",
                    })

        except Exception as e:
            logger.error(f"[{request_id}] Streaming pipeline error: {e}")

        # Send full text as chat message
        if full_text:
            await self.ws_server.broadcast({
                "type": "speech_text",
                "text": full_text,
                "emotion": emotion,
                "speech_id": speech_id,
            })

        # Brief pause before sending speech_end to give frontend time to
        # enqueue the last audio chunk. Frontend now uses speechEndReceived
        # flag to finalize only after all audio finishes playing.
        if total_duration > 0 and not self.pipeline.interrupt.should_stop():
            wait_time = min(total_duration * 0.15, 2.0)
            try:
                interrupted = await self.pipeline.interrupt.wait_for_interrupt(
                    timeout=wait_time
                )
                if interrupted and turn:
                    turn.aborted = True
            except asyncio.CancelledError:
                pass

        # Finalize turn
        was_interrupted = self.pipeline.interrupt.should_stop()
        if turn:
            turn.tts_engine = self.tts.get_engine()
            turn.audio_bytes = total_bytes
            turn.audio_format = audio_format
            turn.audio_duration_sec = total_duration
            turn.finished_at = _time.time()
            logger.info(f"Turn complete: {turn.summary()}")
            try:
                await self.pipeline.memory.long_term.save_turn_log(turn)
            except Exception as log_err:
                logger.warning(f"Failed to save turn log: {log_err}")

        # Always send speech_end (with interrupted flag if applicable)
        await self.ws_server.broadcast({
            "type": "speech_end",
            "speech_id": speech_id,
            "interrupted": was_interrupted,
        })
        await self.ws_server.send_status("idle")
        self.pipeline.interrupt.set_state(PipelineState.IDLE)
        self.pipeline.interrupt.clear_interrupt()
        self.pipeline.interrupt.current_speech_id = None

        if self.microphone:
            self.microphone.stop_suppress()

        # Start anti-echo silence period (py-xiaozhi pattern)
        if not was_interrupted:
            self._silence_period.start()

        logger.info(
            f"[{request_id}] Streaming complete: {sentences_sent} chunks, "
            f"{total_bytes}b, {total_duration:.1f}s total"
            f"{' (interrupted)' if was_interrupted else ''}"
        )

    async def _handle_legacy_output(self, result: dict):
        """Handle legacy non-streaming output (enrollment, reminders, etc.)."""
        import time as _time

        response = result.get("response")
        request_id = result.get("request_id", "?")
        turn = result.get("turn")

        if not response or not response.text:
            return

        speech_id = f"sp-{str(uuid.uuid4())[:8]}"
        logger.info(f"[{request_id}] Legacy output (speech_id={speech_id})")

        self.scheduler.notify_interaction()

        self.pipeline.interrupt.set_state(PipelineState.SPEAKING)
        self.pipeline.interrupt.clear_interrupt()
        self.pipeline.interrupt.current_speech_id = speech_id
        if self.microphone:
            self.microphone.suppress_echo()

        await self.ws_server.broadcast({
            "type": "speech_start",
            "text": response.text,
            "emotion": response.emotion or "neutral",
            "speech_id": speech_id,
        })
        await self.ws_server.send_subtitle(response.text, "assistant")
        await self.ws_server.send_status("speaking")

        tts_start = _time.time()
        audio_data = await self.tts.synthesize(response.text, emotion=response.emotion)
        tts_end = _time.time()

        if turn:
            turn.tts_started_at = tts_start
            turn.tts_finished_at = tts_end
            turn.tts_engine = self.tts.get_engine()

        if self.pipeline.interrupt.should_stop():
            if turn:
                turn.aborted = True
            audio_data = None

        if audio_data:
            volumes = await self.tts.get_audio_for_lip_sync(audio_data)
            await self.ws_server.send_lip_sync(volumes)

            audio_b64 = base64.b64encode(audio_data).decode("ascii")
            audio_format = "wav" if audio_data[:4] == b"RIFF" else "mp3"
            await self.ws_server.broadcast({
                "type": "audio",
                "data": audio_b64,
                "format": audio_format,
                "request_id": request_id,
                "speech_id": speech_id,
            })

            if turn:
                turn.audio_bytes = len(audio_data)
                turn.audio_format = audio_format

            duration_sec = self._calculate_audio_duration(audio_data, audio_format)
            if turn:
                turn.audio_duration_sec = duration_sec

            try:
                interrupted = await self.pipeline.interrupt.wait_for_interrupt(
                    timeout=duration_sec + 0.5
                )
                if interrupted and turn:
                    turn.aborted = True
            except asyncio.CancelledError:
                pass

        was_interrupted = self.pipeline.interrupt.should_stop()
        if turn:
            turn.finished_at = _time.time()
            logger.info(f"Turn complete: {turn.summary()}")
            try:
                await self.pipeline.memory.long_term.save_turn_log(turn)
            except Exception as log_err:
                logger.warning(f"Failed to save turn log: {log_err}")

        # Always send speech_end (with interrupted flag if applicable)
        await self.ws_server.broadcast({
            "type": "speech_end",
            "speech_id": speech_id,
            "interrupted": was_interrupted,
        })
        await self.ws_server.send_status("idle")
        self.pipeline.interrupt.set_state(PipelineState.IDLE)
        self.pipeline.interrupt.clear_interrupt()
        self.pipeline.interrupt.current_speech_id = None

        if self.microphone:
            self.microphone.stop_suppress()

        # Start anti-echo silence period (py-xiaozhi pattern)
        self._silence_period.start()

    @staticmethod
    def _calculate_audio_duration(audio_data: bytes, audio_format: str) -> float:
        """Calculate actual audio duration from WAV header or estimate for MP3.

        For WAV: parse header to get exact duration.
        For MP3: estimate from compressed size.

        Returns:
            Duration in seconds.
        """
        if audio_format == "wav" and len(audio_data) > 44 and audio_data[:4] == b"RIFF":
            try:
                import struct
                # WAV header: bytes 24-27 = sample rate, bytes 34-35 = bits per sample
                # bytes 28-31 = byte rate (sample_rate * channels * bits_per_sample / 8)
                byte_rate = struct.unpack_from('<I', audio_data, 28)[0]
                # Data size = total size - header (44 bytes typically)
                data_size = len(audio_data) - 44
                if byte_rate > 0:
                    return data_size / byte_rate
            except Exception:
                pass
            # Fallback: CosyVoice default is 22050Hz 16-bit mono = 44100 bytes/sec
            return max((len(audio_data) - 44) / 44100.0, 0.5)
        else:
            # MP3: rough estimate ~12000 bytes/sec at 128kbps
            return max(len(audio_data) / 12000.0, 0.5)

    # ----------------------------------------------------------------
    # Boss auto-registration
    # ----------------------------------------------------------------

    async def _ensure_boss_speaker(self):
        """Auto-create the boss speaker on first startup and check voice enrollment."""
        speakers = await self.pipeline.memory.long_term.get_all_speakers()
        if not speakers:
            logger.info("No speakers registered — auto-creating boss 'will'")
            await self.pipeline.memory.long_term.register_speaker(
                speaker_id="will",
                display_name="ウィルさん",
                role="boss",
                language="mixed",
                permissions='["all"]',
                style="casual",
                notes="Auto-registered boss account",
            )

        # In full mode, check if boss has a voice embedding
        if not self.text_only and self.speaker_id and self.speaker_id.enabled:
            boss = await self.pipeline.memory.long_term.get_speaker("will")
            if boss and not self.speaker_id.has_embedding("will"):
                logger.info("Boss 'will' has no voice embedding — enrollment needed")
                # Prompt naturally — ミケ will mention it during first interaction
                await self.pipeline.input_queue.put({
                    "type": "text",
                    "text": (
                        "[システム] オーナーのウィルさんの声紋がまだ登録されていません。"
                        "自然な挨拶をして、会話の中で声紋登録を勧めてください。"
                        "「声紋登録」と言えば開始できると伝えてください。"
                        "登録後は声で自動的に認識できるようになります。"
                    ),
                    "speaker_id": None,
                    "request_id": "enrollment-prompt",
                    "intent": "proactive",
                })

    # ----------------------------------------------------------------
    # Input Handlers
    # ----------------------------------------------------------------

    async def _handle_text_input(self, text: str, websocket=None, source: str = "chat"):
        """Handle text input from the WebSocket frontend.

        In full mode, requires voice authentication first (speaker_id set via
        voice recognition or authenticate_as command). In text-only mode,
        allows unauthenticated input for demo purposes.

        Interrupt policy by source:
            - 'chat': User text input — CAN interrupt current AI speech.
            - 'danmaku': Live stream comments — NEVER interrupts, queued only.
            - 'voice': Voice ASR result — interrupt handled by VAD barge-in.

        Args:
            text: User message text.
            websocket: The WebSocket client that sent the message.
            source: Input source ('chat' | 'danmaku' | 'voice').
        """
        client_info = self.ws_server.get_client_info(websocket) if websocket else {}
        speaker_id = client_info.get("speaker_id")
        session_id = client_info.get("session_id")
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Text input: '{text}' (source={source}, speaker={speaker_id})")

        # Access control: in full mode, require authenticated speaker
        if not self.text_only and not client_info.get("authenticated", False):
            await self.ws_server.send_to_client(websocket, {
                "type": "access_denied",
                "message": "声紋認証が必要です。マイクで話しかけてください。",
            })
            return

        # Interrupt policy: chat can interrupt, danmaku NEVER interrupts
        if source == "chat" and self.pipeline.interrupt.is_speaking:
            old_speech_id = self.pipeline.interrupt.current_speech_id
            await self.pipeline.interrupt.interrupt()
            logger.info(f"[{request_id}] Text barge-in: interrupted speech {old_speech_id}")
            # Give the streaming output handler time to notice the interrupt
            await asyncio.sleep(0.1)
        elif source == "danmaku" and self.pipeline.interrupt.is_speaking:
            logger.info(f"[{request_id}] Danmaku queued (no interrupt): '{text[:30]}'")

        # Notify scheduler (resets idle timer)
        self.scheduler.notify_interaction()

        # Show user's message as subtitle
        await self.ws_server.send_subtitle(text, "user")

        # Enqueue for pipeline processing with speaker context
        await self.pipeline.input_queue.put({
            "type": "text",
            "text": text,
            "speaker_id": speaker_id,
            "request_id": request_id,
            "session_id": session_id,
            "source": source,
        })

    async def _handle_command(self, command: str, data=None, websocket=None):
        """Handle commands from the frontend.

        Args:
            command: Command name.
            data: Command data payload.
            websocket: The WebSocket client that sent the command.
        """
        logger.info(f"Command: {command} ({data})")

        # --- Config API commands (new settings UI) ---
        if command == "get_config":
            sections = data.get("sections") if data else None
            result = await self.config_api.handle_get_config(sections)
            await self.ws_server.broadcast({"type": "config_data", **result})

        elif command == "set_config":
            updates = data.get("updates", []) if data else []
            result = await self.config_api.handle_set_config(updates)
            await self.ws_server.broadcast({"type": "config_update_result", **result})

        elif command == "clear_cache":
            result = await self.config_api.handle_clear_cache()
            await self.ws_server.broadcast({"type": "config_update_result", **result})

        # --- Persona commands ---
        elif command == "get_persona":
            result = await self.config_api.handle_get_persona()
            await self.ws_server.broadcast({"type": "persona_data", **result})

        elif command == "set_persona":
            updates = data.get("updates", []) if data else []
            result = await self.config_api.handle_set_persona(updates)
            await self.ws_server.broadcast({"type": "persona_update_result", **result})

        # --- API Key commands ---
        elif command == "get_api_key_status":
            result = await self.config_api.handle_get_api_key_status()
            await self.ws_server.broadcast({"type": "api_key_status", **result})

        elif command == "set_api_key":
            provider = data.get("provider", "") if data else ""
            key = data.get("key", "") if data else ""
            result = await self.config_api.handle_set_api_key(provider, key)
            await self.ws_server.broadcast({"type": "api_key_update_result", **result})

        # --- Memory CRUD commands ---
        elif command == "get_memos":
            result = await self.memory_api.handle_get_memos(
                query=data.get("query", "") if data else "",
                limit=data.get("limit", 20) if data else 20,
            )
            await self.ws_server.broadcast({"type": "memos_data", **result})

        elif command == "save_memo":
            result = await self.memory_api.handle_save_memo(
                content=data.get("content", "") if data else "",
                category=data.get("category", "general") if data else "general",
            )
            await self.ws_server.broadcast({"type": "memo_saved", **result})

        elif command == "update_memo":
            result = await self.memory_api.handle_update_memo(
                memo_id=data.get("memo_id") if data else None,
                content=data.get("content", "") if data else "",
                category=data.get("category") if data else None,
            )
            await self.ws_server.broadcast({"type": "memo_updated", **result})

        elif command == "archive_memo":
            result = await self.memory_api.handle_archive_memo(
                memo_id=data.get("memo_id") if data else None,
            )
            await self.ws_server.broadcast({"type": "memo_archived", **result})

        elif command == "get_reminders":
            result = await self.memory_api.handle_get_reminders(
                include_completed=data.get("include_completed", False) if data else False,
            )
            await self.ws_server.broadcast({"type": "reminders_data", **result})

        elif command == "save_reminder":
            result = await self.memory_api.handle_save_reminder(
                content=data.get("content", "") if data else "",
                remind_at=data.get("remind_at", "") if data else "",
                repeat=data.get("repeat", "none") if data else "none",
            )
            await self.ws_server.broadcast({"type": "reminder_saved", **result})

        elif command == "complete_reminder":
            result = await self.memory_api.handle_complete_reminder(
                reminder_id=data.get("reminder_id") if data else None,
            )
            await self.ws_server.broadcast({"type": "reminder_completed", **result})

        elif command == "delete_reminder":
            result = await self.memory_api.handle_delete_reminder(
                reminder_id=data.get("reminder_id") if data else None,
            )
            await self.ws_server.broadcast({"type": "reminder_deleted", **result})

        elif command == "get_history":
            result = await self.memory_api.handle_get_history(
                limit=data.get("limit", 10) if data else 10,
            )
            await self.ws_server.broadcast({"type": "history_data", **result})

        elif command == "clear_short_term":
            result = await self.memory_api.handle_clear_short_term()
            await self.ws_server.broadcast({"type": "short_term_cleared", **result})

        # --- Knowledge base commands ---
        elif command == "get_knowledge_files":
            result = await self.knowledge_api.handle_get_knowledge_files()
            await self.ws_server.broadcast({"type": "knowledge_files", **result})

        elif command == "get_knowledge_content":
            category = data.get("category", "") if data else ""
            result = await self.knowledge_api.handle_get_knowledge_content(category)
            await self.ws_server.broadcast({"type": "knowledge_content", **result})

        elif command == "save_knowledge_content":
            category = data.get("category", "") if data else ""
            content = data.get("content", "") if data else ""
            result = await self.knowledge_api.handle_save_knowledge_content(category, content)
            await self.ws_server.broadcast({"type": "knowledge_save_result", **result})

        elif command == "rebuild_knowledge_index":
            result = await self.knowledge_api.handle_rebuild_index()
            await self.ws_server.broadcast({"type": "knowledge_rebuild_result", **result})

        # --- Speaker API commands ---
        elif command == "register_speaker":
            result = await self.speaker_api.handle_register_speaker(
                speaker_id=data.get("speaker_id", "") if data else "",
                display_name=data.get("display_name", "") if data else "",
                role=data.get("role", "visitor") if data else "visitor",
                language=data.get("language", "japanese") if data else "japanese",
                permissions=data.get("permissions") if data else None,
                style=data.get("style", "formal") if data else "formal",
                notes=data.get("notes", "") if data else "",
                invited_by=data.get("invited_by") if data else None,
                relationship=data.get("relationship", "") if data else "",
                audio_samples_b64=data.get("audio_samples") if data else None,
            )
            await self.ws_server.send_to_client(websocket, {
                "type": "speaker_registered", **result,
            })
            # If registration is successful, also broadcast updated speakers list
            if result.get("success"):
                speakers_result = await self.speaker_api.handle_get_speakers()
                await self.ws_server.broadcast({
                    "type": "speakers_data", **speakers_result,
                })

        elif command == "get_speakers":
            result = await self.speaker_api.handle_get_speakers()
            await self.ws_server.send_to_client(websocket, {
                "type": "speakers_data", **result,
            })

        elif command == "update_speaker":
            speaker_id = data.get("speaker_id", "") if data else ""
            updates = data.get("updates", {}) if data else {}
            result = await self.speaker_api.handle_update_speaker(speaker_id, updates)
            await self.ws_server.send_to_client(websocket, {
                "type": "speaker_updated", **result,
            })

        elif command == "reenroll_speaker":
            # Re-register voice print from frontend settings
            speaker_id = data.get("speaker_id", "") if data else ""
            audio_samples_b64 = data.get("audio_samples", []) if data else []
            if speaker_id and audio_samples_b64:
                import base64 as _b64
                webm_samples = [_b64.b64decode(s) for s in audio_samples_b64]
                success = await self.speaker_id.register_speaker_from_webm(
                    speaker_id, webm_samples
                )
                await self.ws_server.send_to_client(websocket, {
                    "type": "speaker_updated",
                    "success": success,
                    "message": "声紋を更新しました" if success else "声紋の更新に失敗しました",
                })
            else:
                await self.ws_server.send_to_client(websocket, {
                    "type": "speaker_updated",
                    "success": False,
                    "message": "speaker_idとaudio_samplesが必要です",
                })

        elif command == "delete_speaker":
            speaker_id = data.get("speaker_id", "") if data else ""
            result = await self.speaker_api.handle_delete_speaker(speaker_id)
            await self.ws_server.send_to_client(websocket, {
                "type": "speaker_deleted", **result,
            })
            # Broadcast updated speakers list
            if result.get("success"):
                speakers_result = await self.speaker_api.handle_get_speakers()
                await self.ws_server.broadcast({
                    "type": "speakers_data", **speakers_result,
                })

        elif command == "authenticate_as":
            # Text-only mode: manual speaker authentication
            speaker_id = data.get("speaker_id", "") if data else ""
            speaker = await self.speaker_api.handle_check_access(speaker_id)
            if speaker:
                # Create session
                session_id = await self.pipeline.get_or_create_session(speaker_id)
                self.ws_server.set_client_speaker(websocket, speaker_id, session_id)
                await self.ws_server.send_to_client(websocket, {
                    "type": "speaker_authenticated",
                    "success": True,
                    "speaker_id": speaker_id,
                    "speaker_name": speaker["display_name"],
                    "session_id": session_id,
                    "role": speaker["role"],
                })
            else:
                await self.ws_server.send_to_client(websocket, {
                    "type": "speaker_authenticated",
                    "success": False,
                    "error": f"Speaker '{speaker_id}' not found or inactive",
                })

        # --- Audit commands ---
        elif command == "get_audit_sessions":
            speaker_id = data.get("speaker_id") if data else None
            limit = data.get("limit", 20) if data else 20
            result = await self.speaker_api.handle_get_audit_sessions(
                speaker_id=speaker_id, limit=limit
            )
            await self.ws_server.send_to_client(websocket, {
                "type": "audit_sessions_data", **result,
            })

        elif command == "get_session_detail":
            session_id = data.get("session_id", "") if data else ""
            result = await self.speaker_api.handle_get_session_detail(session_id)
            await self.ws_server.send_to_client(websocket, {
                "type": "session_detail_data", **result,
            })

        # --- Interrupt command (barge-in) ---
        elif command == "interrupt":
            if self.pipeline.interrupt.is_speaking:
                speech_id = self.pipeline.interrupt.current_speech_id
                await self.pipeline.interrupt.interrupt()
                logger.info(f"Speech interrupted by frontend (speech_id={speech_id})")
                # Notify frontend to stop playback for this specific speech
                await self.ws_server.broadcast({
                    "type": "speech_end",
                    "speech_id": speech_id,
                    "interrupted": True,
                })

        # --- Legacy commands (kept for backward compatibility) ---
        elif command == "set_llm_mode":
            from src.core.llm_router import LLMMode
            mode_str = data.get("mode", "balanced") if data else "balanced"
            self.pipeline.llm.set_mode(LLMMode(mode_str))

        elif command == "set_language":
            lang = data.get("language", "japanese") if data else "japanese"
            self.tts.active_language = lang
            logger.info(f"Language manually set to: {lang}")

        elif command == "set_tts_engine":
            engine = data.get("engine", "voicevox") if data else "voicevox"
            self.tts.set_engine(engine)
            logger.info(f"TTS engine set to: {engine}")

        elif command == "mute":
            muted = data.get("muted", False) if data else False
            logger.info(f"Mute: {muted}")

    # ----------------------------------------------------------------
    # Frontend Audio Input (browser MediaRecorder → backend ASR + SID)
    # ----------------------------------------------------------------

    async def _handle_audio_input(self, audio_b64: str, audio_format: str, websocket=None):
        """Handle audio input from the browser frontend.

        The frontend sends raw audio (base64-encoded webm/opus) via WebSocket.
        We convert it to PCM, run ASR + speaker ID in parallel, then process.

        Important: We do NOT auto-interrupt here. The frontend handles barge-in
        detection (VAD with high threshold) and sends an explicit 'interrupt'
        command. Auto-interrupting on every audio input would cut off speech
        when the audio is just background noise or echo.

        Args:
            audio_b64: Base64-encoded audio data.
            audio_format: MIME type (e.g., 'audio/webm').
            websocket: The WebSocket client that sent the audio.
        """

        # Anti-echo: skip audio received during silence period after TTS
        # (py-xiaozhi inspired: prevents TTS echo from being picked up as input)
        if self._silence_period.is_active():
            logger.debug("Audio input during silence period — suppressed (anti-echo)")
            return

        # If AI is currently speaking, the frontend should have already sent
        # an 'interrupt' command via barge-in. We do NOT auto-interrupt here
        # because the audio might be noise or echo remnants.
        # Just log and proceed — the interrupt handler takes care of stopping speech.
        was_speaking = self.pipeline.interrupt.is_speaking
        if was_speaking:
            logger.debug("Audio input received while AI speaking — processing without auto-interrupt")

        await self.ws_server.send_status("processing")
        self.scheduler.notify_interaction()

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            logger.error(f"Audio base64 decode error: {e}")
            await self.ws_server.send_status("idle")
            return

        logger.debug(f"Received audio from frontend: {len(audio_bytes)} bytes ({audio_format})")

        # Convert webm/opus → PCM int16 16kHz mono using ffmpeg
        # Use -err_detect ignore_err to handle slightly malformed WebM from browser
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                [
                    "ffmpeg",
                    "-err_detect", "ignore_err",
                    "-i", "pipe:",
                    "-f", "s16le", "-ar", "16000", "-ac", "1",
                    "-loglevel", "error",
                    "pipe:",
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=10,
            )
        except FileNotFoundError:
            logger.error("ffmpeg not found — install ffmpeg for audio conversion")
            await self.ws_server.send_status("idle")
            return
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg conversion timeout")
            await self.ws_server.send_status("idle")
            return

        if proc.returncode != 0 or not proc.stdout:
            # Retry with more lenient flags
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "ffmpeg",
                        "-f", "webm", "-i", "pipe:",
                        "-f", "s16le", "-ar", "16000", "-ac", "1",
                        "-loglevel", "error",
                        "pipe:",
                    ],
                    input=audio_bytes,
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass

            if not proc or proc.returncode != 0 or not proc.stdout:
                logger.warning(f"ffmpeg conversion failed ({len(audio_bytes)}b): "
                              f"{proc.stderr[:200] if proc and proc.stderr else 'no output'}")
                await self.ws_server.send_status("idle")
                return

        pcm_data = proc.stdout
        logger.debug(f"Converted to PCM: {len(pcm_data)} bytes ({len(pcm_data)/32000:.1f}s)")

        # Run ASR and speaker ID concurrently
        asr_task = asyncio.create_task(self.asr.transcribe(pcm_data))
        sid_task = None
        if self.speaker_id and self.speaker_id.enabled:
            sid_task = asyncio.create_task(self.speaker_id.identify(pcm_data))

        text = await asr_task
        speaker_id = await sid_task if sid_task else None

        if not text:
            logger.debug("No text from ASR — dropping")
            await self.ws_server.send_status("idle")
            return

        # Detect input language for language-follow mode
        detected_lang = self._detect_language(text)
        if detected_lang != "auto":
            # Switch TTS output language to match user's input
            if self.tts.active_language != detected_lang:
                logger.info(f"Language follow: switching TTS {self.tts.active_language} → {detected_lang}")
                self.tts.active_language = detected_lang

        # Update websocket client with recognized speaker
        if speaker_id and websocket:
            session_id = await self.pipeline.get_or_create_session(speaker_id)
            self.ws_server.set_client_speaker(websocket, speaker_id, session_id)

        await self._process_voice_input(pcm_data, text, speaker_id, detected_lang)

    async def _process_voice_input(
        self, audio_data: bytes, text: str, speaker_id: str = None,
        detected_lang: str = "auto",
    ):
        """Shared voice input processing logic.

        Used by both:
        - _handle_audio_input (frontend browser audio)
        - on_utterance (server microphone)

        Handles enrollment mode, access control, and normal conversation.

        Args:
            audio_data: Raw PCM audio bytes (int16, 16kHz mono).
            text: ASR transcription result.
            speaker_id: Identified speaker, or None if unknown.
            detected_lang: Detected input language ("japanese", "chinese", or "auto").
        """
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] ASR: '{text}' | Speaker: {speaker_id}")

        # ─── Layer 1: Active enrollment — collect samples ───
        if self._enrollment_mode:
            logger.info(f"[{request_id}] Enrollment sample: '{text[:30]}'")
            await self.ws_server.send_subtitle(text, "user")
            await self._handle_enrollment_sample(
                audio_data, text, request_id, speaker_id
            )
            return

        await self.ws_server.send_subtitle(text, "user")

        # ─── Access control + enrollment trigger ───
        if speaker_id is None:
            # Special case: boss has no embedding yet → allow self-enrollment
            if self._is_enrollment_trigger(text) and self.speaker_id:
                boss = await self.pipeline.memory.long_term.get_speaker("will")
                if boss and not self.speaker_id.has_embedding("will"):
                    logger.info(f"[{request_id}] Boss self-enrollment trigger (no embedding yet)")
                    await self._start_boss_self_enrollment(audio_data, text, request_id)
                    return

            # Special case: re-enrollment trigger from unrecognized but registered speaker
            if self._is_reenroll_trigger(text) and self.speaker_id:
                # Check if "will" exists (most likely the unrecognized speaker)
                boss = await self.pipeline.memory.long_term.get_speaker("will")
                if boss:
                    logger.info(f"[{request_id}] Re-enrollment trigger from unrecognized speaker (assuming boss)")
                    await self._start_reenrollment("will", audio_data, request_id)
                    return

            # Unknown speaker → denied
            logger.info(f"[{request_id}] Unknown speaker — denied")
            await self.pipeline.input_queue.put({
                "type": "text",
                "text": (
                    f"[システム] 声紋未登録の方が話しかけています。"
                    f"丁寧にお断りして、オーナーに声紋登録をお願いするよう案内してください。"
                    f"相手の発言:「{text}」"
                ),
                "speaker_id": None,
                "request_id": request_id,
                "intent": "access_control",
            })
            return

        # ─── Registered speaker: check re-enrollment (update own voice) ───
        if self._is_reenroll_trigger(text):
            logger.info(f"[{request_id}] Re-enrollment trigger by: {speaker_id}")
            await self._start_reenrollment(speaker_id, audio_data, request_id)
            return

        # ─── Registered speaker: check enrollment trigger (new person) ───
        if self._is_enrollment_trigger(text):
            logger.info(f"[{request_id}] Enrollment trigger by registered speaker: {speaker_id}")
            await self._start_enrollment(speaker_id, request_id)
            return

        # Recognized speaker → update all connected clients' speaker info
        session_id = await self.pipeline.get_or_create_session(speaker_id)
        for client in list(self.ws_server._clients):
            self.ws_server.set_client_speaker(client, speaker_id, session_id)

        # Build input text with language hint if needed
        input_text = text
        if detected_lang == "chinese":
            input_text = f"[lang:zh] {text}"
        elif detected_lang == "japanese":
            input_text = f"[lang:ja] {text}"

        await self.pipeline.input_queue.put({
            "type": "text",
            "text": input_text,
            "speaker_id": speaker_id,
            "request_id": request_id,
            "session_id": session_id,
        })

    # ----------------------------------------------------------------
    # Voice Loop (server microphone path)
    # ----------------------------------------------------------------

    async def _voice_loop(self):
        """Main voice capture and processing loop.

        Two-layer logic:
        1. If enrollment mode is active → collect voice samples
        2. Normal mode → identify speaker transparently, pass to pipeline
           - Enrollment trigger (「声紋登録」) starts enrollment
           - speaker_id=None means unknown person — pipeline still processes normally
        """
        if not self.microphone or not self.asr:
            return

        async def on_utterance(audio_data: bytes):
            """Called when a complete utterance is detected from server mic.

            Runs ASR and speaker identification in parallel,
            then delegates to the shared _process_voice_input().
            """
            await self.ws_server.send_status("processing")
            self.scheduler.notify_interaction()

            # Run ASR and speaker ID concurrently
            asr_task = asyncio.create_task(self.asr.transcribe(audio_data))
            sid_task = None
            if self.speaker_id:
                sid_task = asyncio.create_task(self.speaker_id.identify(audio_data))

            text = await asr_task
            speaker_id = await sid_task if sid_task else None

            if not text:
                await self.ws_server.send_status("idle")
                return

            await self._process_voice_input(audio_data, text, speaker_id)

        async def on_speech_start():
            await self.ws_server.send_status("listening")
            # Interrupt if AI is currently speaking
            if self.pipeline.interrupt.is_speaking:
                speech_id = self.pipeline.interrupt.current_speech_id
                await self.pipeline.interrupt.interrupt()
                await self.ws_server.broadcast({
                    "type": "speech_end",
                    "speech_id": speech_id,
                    "interrupted": True,
                })

        async def on_speech_end():
            pass

        self.microphone.on_utterance(on_utterance)
        self.microphone.on_speech_start(on_speech_start)
        self.microphone.on_speech_end(on_speech_end)

        await self.microphone.start()

    # ----------------------------------------------------------------
    # Voice Enrollment
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Language Detection
    # ----------------------------------------------------------------

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect primary language from text.

        Simple heuristic based on Unicode ranges:
        - CJK Unified Ideographs alone are ambiguous (shared by zh/ja)
        - Hiragana/Katakana → Japanese
        - If CJK chars but no kana → Chinese
        - Otherwise → "auto" (let TTS decide)

        Returns:
            "japanese", "chinese", or "auto"
        """
        if not text:
            return "auto"

        has_hiragana = bool(re.search(r'[\u3040-\u309F]', text))
        has_katakana = bool(re.search(r'[\u30A0-\u30FF]', text))
        has_cjk = bool(re.search(r'[\u4E00-\u9FFF]', text))

        if has_hiragana or has_katakana:
            return "japanese"
        elif has_cjk:
            return "chinese"
        else:
            return "auto"

    def _is_enrollment_trigger(self, text: str) -> bool:
        """Check if the user said the enrollment trigger phrase.

        Trigger phrases mean "remember this person" — the registered speaker
        who says this will initiate voice enrollment for the next person.
        Special case: "声紋登録" is also supported for boss self-enrollment
        when no voice embeddings exist yet.
        """
        triggers = [
            # Japanese: "remember this person"
            "この人を覚えて", "この人覚えて", "覚えて",
            "このひとをおぼえて", "おぼえて",
            # Chinese: "remember this person"
            "记住这个人", "记住他", "记住她",
            # Fallback: explicit enrollment command
            "声紋登録", "声紋を登録",
        ]
        normalized = text.lower().replace(" ", "").replace("　", "")
        return any(t in normalized for t in triggers)

    def _is_reenroll_trigger(self, text: str) -> bool:
        """Check if the user wants to re-register their own voice print.

        This allows an already-registered speaker to update their voice
        embedding (e.g., if the original was recorded from a different mic).
        """
        triggers = [
            # Japanese
            "声紋を更新", "声紋更新", "声を覚え直して", "声覚え直して",
            "声紋を再登録", "声紋再登録", "声を登録し直して",
            # Chinese
            "更新声纹", "重新注册声纹", "重新登记声纹",
            "重新记住我的声音",
        ]
        normalized = text.lower().replace(" ", "").replace("　", "")
        return any(t in normalized for t in triggers)

    async def _start_boss_self_enrollment(
        self, audio_data: bytes, text: str, request_id: str
    ):
        """Special enrollment: boss registers themselves when no embeddings exist.

        This only happens when the boss has a DB record but no voice embedding,
        and there are no other registered speakers to act as inviter.
        The boss's trigger utterance audio is used as the first sample.
        """
        self._enrollment_mode = True
        self._enrollment_speaker_id = "will"
        self._enrollment_inviter = None  # self-enrollment
        self._enrollment_samples = [audio_data]  # trigger audio = sample 1
        self._enrollment_step = 1  # already have 1 sample

        prompt_text = self._enrollment_prompts[1]  # start from prompt 2

        logger.info("Starting boss self-enrollment for 'will'")

        await self.pipeline.input_queue.put({
            "type": "text",
            "text": (
                "[システム] ウィルさんの声紋登録を開始します。"
                "声を覚えるために、あと2つの文章を読んでもらってください。"
                f"次の文章を読んでください:「{prompt_text}」"
            ),
            "speaker_id": "will",
            "request_id": request_id,
            "intent": "enrollment",
        })

    async def _start_enrollment(self, inviter_id: str, request_id: str):
        """Start voice enrollment for a NEW person, triggered by a registered speaker.

        Flow:
        1. Registered speaker (inviter) says 「この人を覚えて」
        2. System enters enrollment mode, waiting for an UNKNOWN person to speak
        3. The next unknown speaker's voice gets collected as enrollment samples
        4. After 3 samples, the new person is registered

        Args:
            inviter_id: The registered speaker who triggered enrollment.
            request_id: Request tracking ID.
        """
        self._enrollment_mode = True
        self._enrollment_speaker_id = None  # don't know yet — will be assigned
        self._enrollment_inviter = inviter_id
        self._enrollment_samples = []
        self._enrollment_step = 0

        logger.info(f"Enrollment mode activated by {inviter_id} — waiting for new speaker")

        await self.pipeline.input_queue.put({
            "type": "text",
            "text": (
                f"[システム] {inviter_id}が新しい人の声紋登録を開始しました。"
                f"新しい方に話しかけてもらってください。"
                f"まず、次の文章を読んでもらってください:「{self._enrollment_prompts[0]}」"
            ),
            "speaker_id": inviter_id,
            "request_id": request_id,
            "intent": "enrollment",
        })

    async def _start_reenrollment(
        self, speaker_id: str, trigger_audio: bytes, request_id: str
    ):
        """Start voice re-enrollment for an existing registered speaker.

        The speaker wants to update their voice print (e.g., different mic,
        voice changed, etc.). Uses the same enrollment flow but targets
        the existing speaker_id and overwrites the old embedding.

        Args:
            speaker_id: The registered speaker requesting re-enrollment.
            trigger_audio: Audio from the trigger utterance (used as sample 1).
            request_id: Request tracking ID.
        """
        self._enrollment_mode = True
        self._enrollment_speaker_id = speaker_id
        self._enrollment_inviter = None  # self re-enrollment
        self._enrollment_samples = [trigger_audio]  # trigger audio = sample 1
        self._enrollment_step = 1

        prompt_text = self._enrollment_prompts[1]  # start from prompt 2

        logger.info(f"Starting voice re-enrollment for '{speaker_id}'")

        await self.pipeline.input_queue.put({
            "type": "text",
            "text": (
                f"[システム] {speaker_id}の声紋を更新します。"
                f"声を覚え直すために、あと2つの文章を読んでもらってください。"
                f"次の文章を読んでください:「{prompt_text}」"
            ),
            "speaker_id": speaker_id,
            "request_id": request_id,
            "intent": "enrollment",
        })

    async def _handle_enrollment_sample(
        self, audio_data: bytes, text: str, request_id: str,
        speaker_id: str = None,
    ):
        """Handle an audio sample during enrollment mode.

        Two flows:
        1. Boss self-enrollment: all audio is from the boss → collect directly
        2. Inviter flow: skip audio from the inviter (registered), only
           collect from the new (unknown) person

        Args:
            audio_data: Raw audio bytes.
            text: ASR transcription.
            request_id: Request tracking ID.
            speaker_id: Already-identified speaker (from parallel SID), or None.

        Collects 3 samples, then finalizes registration.
        """
        # If inviter flow, skip audio from the registered inviter
        if self._enrollment_inviter:
            if speaker_id == self._enrollment_inviter:
                logger.info(f"Enrollment: skipping audio from inviter '{identified}'")
                await self.pipeline.input_queue.put({
                    "type": "text",
                    "text": (
                        "[システム] 登録する方の声が必要です。"
                        "新しい方に次の文章を読んでもらってください。"
                    ),
                    "speaker_id": self._enrollment_inviter,
                    "request_id": request_id,
                    "intent": "enrollment",
                })
                return

        self._enrollment_samples.append(audio_data)
        self._enrollment_step += 1

        logger.info(
            f"Enrollment sample {self._enrollment_step}/3 "
            f"for {self._enrollment_speaker_id or 'new_speaker'} ({len(audio_data)} bytes)"
        )

        if self._enrollment_step < 3:
            # Need more samples
            prompt_text = self._enrollment_prompts[self._enrollment_step]
            await self.pipeline.input_queue.put({
                "type": "text",
                "text": (
                    f"[システム] ありがとうございます。{self._enrollment_step}つ目完了。"
                    f"次の文章を読んでください:「{prompt_text}」"
                ),
                "speaker_id": self._enrollment_speaker_id,
                "request_id": request_id,
                "intent": "enrollment",
            })
        else:
            # All 3 samples collected — finalize
            await self._complete_enrollment(request_id)

    async def _complete_enrollment(self, request_id: str):
        """Finalize voice enrollment: extract embeddings and save.

        For inviter flow (new person), auto-generates a speaker_id and
        creates a DB record. For boss self-enrollment, uses existing 'will'.
        """
        speaker_id = self._enrollment_speaker_id
        inviter = self._enrollment_inviter
        samples = self._enrollment_samples

        # For inviter flow, assign a speaker_id for the new person
        if speaker_id is None:
            speaker_id = f"guest_{uuid.uuid4().hex[:6]}"
            self._enrollment_speaker_id = speaker_id
            # Create a DB record for the new speaker
            await self.pipeline.memory.long_term.register_speaker(
                speaker_id=speaker_id,
                display_name=speaker_id,  # placeholder — can be updated later
                role="guest",
                language="ja",
                permissions='["chat"]',
                style="polite",
                notes=f"Voice-enrolled by {inviter}",
            )
            logger.info(f"Created new speaker record: {speaker_id} (invited by {inviter})")

        logger.info(f"Completing enrollment for {speaker_id} with {len(samples)} samples")

        success = await self.speaker_id.register_speaker(speaker_id, samples)

        # Reset enrollment state
        self._enrollment_mode = False
        self._enrollment_speaker_id = None
        self._enrollment_inviter = None
        self._enrollment_samples = []
        self._enrollment_step = 0

        if success:
            logger.info(f"Voice enrollment completed for {speaker_id}")
            await self.pipeline.input_queue.put({
                "type": "text",
                "text": (
                    f"[システム] {speaker_id}の声紋登録が完了しました！"
                    f"これからはお声で自動的に認識します。喜んでお伝えください。"
                ),
                "speaker_id": speaker_id,
                "request_id": request_id,
                "intent": "enrollment",
            })
            # Notify frontend
            await self.ws_server.broadcast({
                "type": "speaker_enrolled",
                "speaker_id": speaker_id,
                "success": True,
            })
        else:
            logger.error(f"Voice enrollment failed for {speaker_id}")
            notify_speaker = inviter or speaker_id
            await self.pipeline.input_queue.put({
                "type": "text",
                "text": (
                    f"[システム] 声紋登録に失敗しました。音声が短すぎた可能性があります。"
                    f"もう一度「この人を覚えて」と言ってやり直してください。"
                ),
                "speaker_id": notify_speaker,
                "request_id": request_id,
                "intent": "enrollment",
            })

    # ----------------------------------------------------------------
    # Reminder Checker
    # ----------------------------------------------------------------

    async def _reminder_checker(self):
        """Periodically check for due reminders."""
        while self._running:
            try:
                reminders = await self.pipeline.memory.get_pending_reminders()
                for reminder in reminders:
                    text = f"リマインダーです！「{reminder['content']}」"
                    logger.info(f"Reminder triggered: {reminder['content']}")

                    from src.core.llm_router import LLMResponse
                    await self.pipeline.output_queue.put({
                        "type": "response",
                        "request_id": f"reminder-{reminder['id']}",
                        "response": LLMResponse(
                            text=text,
                            emotion="excited",
                            model="system",
                        ),
                    })

                    await self.pipeline.memory.long_term.complete_reminder(reminder["id"])

            except Exception as e:
                logger.error(f"Reminder check error: {e}")

            await asyncio.sleep(30)


async def test_llm():
    """Quick test of LLM connectivity."""
    from src.core.llm_router import LLMRouter, LLMMessage, LLMTask

    logger.info("Testing LLM connection...")
    router = LLMRouter()

    messages = [LLMMessage(role="user", content="こんにちは！自己紹介してください。")]

    try:
        response = await router.chat(messages, LLMTask.CONVERSATION)
        logger.info(f"Model: {response.model}")
        logger.info(f"Emotion: {response.emotion}")
        logger.info(f"Response: {response.text}")
        logger.info("LLM test passed!")
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        raise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Fukuraku AI Secretary")
    parser.add_argument(
        "--text-only", action="store_true",
        help="Run in text-only mode (no microphone, WebSocket chat only)"
    )
    parser.add_argument(
        "--test-llm", action="store_true",
        help="Test LLM connection and exit"
    )
    args = parser.parse_args()

    # Load environment
    load_env()
    setup_logger()

    # Validate configs on startup
    valid, config_errors = validate_configs()
    if not valid:
        for err in config_errors:
            print(f"  Config warning: {err}")

    if args.test_llm:
        asyncio.run(test_llm())
        return

    # Create and run secretary
    secretary = FukurakuSecretary(text_only=args.text_only)

    # Handle shutdown signals
    loop = asyncio.new_event_loop()

    def signal_handler():
        loop.create_task(secretary.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(secretary.start())
    except KeyboardInterrupt:
        loop.run_until_complete(secretary.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
