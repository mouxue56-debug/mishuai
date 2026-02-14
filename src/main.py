"""Fukuraku AI Secretary - Main Entry Point.

Starts all subsystems and runs the main event loop.

Usage:
    # Full system (voice + Live2D frontend)
    python -m src.main

    # Text-only mode (no microphone, chat via WebSocket frontend)
    python -m src.main --text-only

    # Test LLM connection
    python -m src.main --test-llm
"""

import argparse
import asyncio
import signal
import sys

from src.core.pipeline import DialoguePipeline
from src.audio.microphone import MicrophoneInput
from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.audio.speaker_id import SpeakerIdentifier
from src.audio.wake_word import WakeWordDetector
from src.display.websocket_server import WebSocketServer
from src.mcp.tools.memo_reminder import set_memory_manager
from src.utils.config_loader import load_env, get_main_config
from src.utils.logger import setup_logger, get_logger


logger = get_logger("main")


class FukurakuSecretary:
    """Main application class that orchestrates all subsystems."""

    def __init__(self, text_only: bool = False):
        self.text_only = text_only
        self.pipeline = DialoguePipeline()
        self.ws_server = WebSocketServer()
        self.tts = TTSEngine()

        # Audio components (disabled in text-only mode)
        self.microphone = None
        self.asr = None
        self.speaker_id = None
        self.wake_word = None

        if not text_only:
            self.microphone = MicrophoneInput()
            self.asr = ASREngine()
            self.speaker_id = SpeakerIdentifier()
            self.wake_word = WakeWordDetector()

        self._running = False

    async def start(self):
        """Initialize and start all subsystems."""
        logger.info("=" * 50)
        logger.info("  Fukuraku AI Secretary v0.1.0")
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
        logger.info(f"WebSocket: ws://localhost:{self.ws_server.port}")
        logger.info(f"Frontend: Open frontend/index.html in a browser")

        # Run all tasks concurrently
        tasks = [
            self.pipeline.run(),
            self._reminder_checker(),
        ]

        if not self.text_only and self.microphone:
            tasks.append(self._voice_loop())

        await asyncio.gather(*tasks)

    async def stop(self):
        """Gracefully shut down all subsystems."""
        logger.info("Shutting down...")
        self._running = False

        if self.microphone:
            await self.microphone.stop()

        await self.ws_server.stop()
        await self.pipeline.shutdown()
        logger.info("Goodbye!")

    async def _handle_text_input(self, text: str):
        """Handle text input from the WebSocket frontend."""
        logger.info(f"Text input: '{text}'")

        # Send to pipeline
        await self.pipeline.input_queue.put({
            "type": "text",
            "text": text,
            "speaker_id": None,
        })

        # Wait for response
        try:
            result = await asyncio.wait_for(self.pipeline.output_queue.get(), timeout=30)
            response = result.get("response")

            if response:
                # Send subtitle
                await self.ws_server.send_subtitle(text, "user")
                await self.ws_server.send_speech_start(response.text, response.emotion)

                # Synthesize and play TTS
                audio = await self.tts.synthesize(response.text)
                if audio:
                    # Send lip sync data
                    volumes = await self.tts.get_audio_for_lip_sync(audio)
                    await self.ws_server.send_lip_sync(volumes)

                    # Play audio (would need audio playback here)
                    # For now, the frontend handles audio

                await self.ws_server.send_speech_end()

        except asyncio.TimeoutError:
            logger.error("Pipeline response timeout")

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

    async def _voice_loop(self):
        """Main voice capture and processing loop."""
        if not self.microphone or not self.asr:
            return

        async def on_utterance(audio_data: bytes):
            """Called when a complete utterance is detected."""
            await self.ws_server.send_status("processing")

            # Speaker identification
            speaker_id = None
            if self.speaker_id:
                speaker_id = await self.speaker_id.identify(audio_data)

            # ASR
            text = await self.asr.transcribe(audio_data)
            if not text:
                await self.ws_server.send_status("idle")
                return

            logger.info(f"ASR: '{text}' (speaker={speaker_id})")
            await self.ws_server.send_subtitle(text, "user")

            # Send to pipeline
            await self.pipeline.input_queue.put({
                "type": "text",
                "text": text,
                "speaker_id": speaker_id,
            })

        async def on_speech_start():
            await self.ws_server.send_status("listening")
            # Check for interruption
            if self.pipeline.interrupt.is_speaking:
                await self.pipeline.interrupt.interrupt()

        async def on_speech_end():
            pass

        self.microphone.on_utterance(on_utterance)
        self.microphone.on_speech_start(on_speech_start)
        self.microphone.on_speech_end(on_speech_end)

        await self.microphone.start()

    async def _reminder_checker(self):
        """Periodically check for due reminders."""
        while self._running:
            try:
                reminders = await self.pipeline.memory.get_pending_reminders()
                for reminder in reminders:
                    text = f"リマインダーです！「{reminder['content']}」"
                    logger.info(f"Reminder triggered: {reminder['content']}")

                    # Announce via TTS and WebSocket
                    await self.ws_server.send_speech_start(text, "excited")
                    audio = await self.tts.synthesize(text)
                    if audio:
                        volumes = await self.tts.get_audio_for_lip_sync(audio)
                        await self.ws_server.send_lip_sync(volumes)
                    await self.ws_server.send_speech_end()

                    # Mark as completed
                    await self.pipeline.memory.long_term.complete_reminder(reminder["id"])

            except Exception as e:
                logger.error(f"Reminder check error: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds


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
            # Windows doesn't support add_signal_handler
            pass

    try:
        loop.run_until_complete(secretary.start())
    except KeyboardInterrupt:
        loop.run_until_complete(secretary.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
