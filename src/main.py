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
import signal
import uuid

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

        # Audio components (disabled in text-only mode)
        self.microphone = None
        self.asr = None
        self.speaker_id = None
        self.wake_word = None

        # Vision (enabled independently of text-only mode)
        self.vision = VisionInput()

        if not text_only:
            self.microphone = MicrophoneInput()
            self.asr = ASREngine()
            self.speaker_id = SpeakerIdentifier()
            self.wake_word = WakeWordDetector()

        self._running = False

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

        # Initialize audio if not text-only
        if not self.text_only:
            if self.asr:
                await self.asr.initialize()
            if self.speaker_id:
                await self.speaker_id.initialize()
            if self.wake_word:
                await self.wake_word.initialize()

        # Initialize vision
        await self.vision.initialize()

        # Set up vision callbacks → scheduler
        self.vision.on_person_detected(self._on_person_detected)
        self.vision.on_scene_change(self._on_scene_change)

        # Set up proactive scheduler → pipeline output queue
        self.scheduler.on_proactive_speak(self._handle_proactive_speak)

        # Set up WebSocket server
        self.ws_server.on_text_input(self._handle_text_input)
        self.ws_server.on_command(self._handle_command)
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
        })

    # ----------------------------------------------------------------
    # Unified Output Consumer
    # ----------------------------------------------------------------

    async def _output_consumer(self):
        """Unified output consumer - handles all assistant responses."""
        while self._running:
            try:
                result = await asyncio.wait_for(
                    self.pipeline.output_queue.get(), timeout=1.0
                )

                if result.get("type") != "response":
                    continue

                response = result.get("response")
                request_id = result.get("request_id", "?")

                if not response or not response.text:
                    continue

                logger.info(f"[{request_id}] Output consumer: delivering response")

                # Notify scheduler that interaction occurred
                self.scheduler.notify_interaction()

                # 1. Set SPEAKING state
                self.pipeline.interrupt.set_state(PipelineState.SPEAKING)
                self.pipeline.interrupt.clear_interrupt()

                # 2. Notify frontend: speech starting
                await self.ws_server.send_speech_start(response.text, response.emotion)
                await self.ws_server.send_subtitle(response.text, "assistant")
                await self.ws_server.send_status("speaking")

                # 3. Synthesize TTS audio
                audio_data = await self.tts.synthesize(response.text)

                if audio_data and not self.pipeline.interrupt.should_stop():
                    # 4. Extract lip sync data (legacy, frontend now uses real-time)
                    volumes = await self.tts.get_audio_for_lip_sync(audio_data)
                    await self.ws_server.send_lip_sync(volumes)

                    # 5. Send audio to frontend as base64 for playback
                    audio_b64 = base64.b64encode(audio_data).decode("ascii")
                    await self.ws_server.broadcast({
                        "type": "audio",
                        "data": audio_b64,
                        "format": "mp3",  # Edge TTS outputs MP3
                        "request_id": request_id,
                    })

                    # 6. Wait for approximate playback duration
                    # MP3 is compressed; estimate ~4x compression ratio
                    duration_sec = max(len(audio_data) / 12000, 0.5)
                    try:
                        interrupted = await self.pipeline.interrupt.wait_for_interrupt(
                            timeout=duration_sec
                        )
                        if interrupted:
                            logger.info(f"[{request_id}] Speech interrupted by user")
                    except asyncio.CancelledError:
                        pass

                # 7. Speech ended
                await self.ws_server.send_speech_end()
                await self.ws_server.send_status("idle")
                self.pipeline.interrupt.set_state(PipelineState.IDLE)
                self.pipeline.interrupt.clear_interrupt()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Output consumer error: {e}")
                self.pipeline.interrupt.set_state(PipelineState.IDLE)
                continue

    # ----------------------------------------------------------------
    # Input Handlers
    # ----------------------------------------------------------------

    async def _handle_text_input(self, text: str):
        """Handle text input from the WebSocket frontend."""
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Text input: '{text}'")

        # Notify scheduler (resets idle timer)
        self.scheduler.notify_interaction()

        # Show user's message as subtitle
        await self.ws_server.send_subtitle(text, "user")

        # Enqueue for pipeline processing
        await self.pipeline.input_queue.put({
            "type": "text",
            "text": text,
            "speaker_id": None,
            "request_id": request_id,
        })

    async def _handle_command(self, command: str, data=None):
        """Handle commands from the frontend."""
        logger.info(f"Command: {command} ({data})")

        if command == "set_llm_mode":
            from src.core.llm_router import LLMMode
            mode_str = data.get("mode", "balanced") if data else "balanced"
            self.pipeline.llm.set_mode(LLMMode(mode_str))

        elif command == "set_language":
            lang = data.get("language", "japanese") if data else "japanese"
            self.tts.active_language = lang

        elif command == "mute":
            muted = data.get("muted", False) if data else False
            logger.info(f"Mute: {muted}")

    # ----------------------------------------------------------------
    # Voice Loop
    # ----------------------------------------------------------------

    async def _voice_loop(self):
        """Main voice capture and processing loop."""
        if not self.microphone or not self.asr:
            return

        async def on_utterance(audio_data: bytes):
            """Called when a complete utterance is detected."""
            await self.ws_server.send_status("processing")

            # Notify scheduler (resets idle timer)
            self.scheduler.notify_interaction()

            # Speaker identification
            speaker_id = None
            if self.speaker_id:
                speaker_id = await self.speaker_id.identify(audio_data)

            # ASR
            text = await self.asr.transcribe(audio_data)
            if not text:
                await self.ws_server.send_status("idle")
                return

            request_id = str(uuid.uuid4())[:8]
            logger.info(f"[{request_id}] ASR: '{text}' (speaker={speaker_id})")
            await self.ws_server.send_subtitle(text, "user")

            # Enqueue for pipeline (output consumer will handle response)
            await self.pipeline.input_queue.put({
                "type": "text",
                "text": text,
                "speaker_id": speaker_id,
                "request_id": request_id,
            })

        async def on_speech_start():
            await self.ws_server.send_status("listening")
            # Interrupt if AI is currently speaking
            if self.pipeline.interrupt.is_speaking:
                await self.pipeline.interrupt.interrupt()

        async def on_speech_end():
            pass

        self.microphone.on_utterance(on_utterance)
        self.microphone.on_speech_start(on_speech_start)
        self.microphone.on_speech_end(on_speech_end)

        await self.microphone.start()

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
