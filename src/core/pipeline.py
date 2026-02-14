"""Main dialogue pipeline: Audio Input → ASR → LLM → TTS → Audio Output.

Orchestrates the full conversation flow, inspired by Pipecat's pipeline pattern.
"""

import asyncio
from datetime import datetime
from typing import Optional

from src.core.emotion_analyzer import EmotionAnalyzer
from src.core.interrupt_handler import InterruptHandler, PipelineState
from src.core.llm_router import LLMMessage, LLMResponse, LLMRouter, LLMTask
from src.memory.memory_manager import MemoryManager
from src.mcp.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger("pipeline")


class DialoguePipeline:
    """Main dialogue pipeline orchestrating all components.

    Flow:
        1. Audio captured from microphone
        2. VAD detects speech boundaries
        3. ASR converts speech to text
        4. Speaker identification (optional)
        5. LLM generates response (with emotion tag)
        6. MCP tools executed if needed
        7. TTS converts response to speech
        8. Audio played + Live2D expression updated
        9. Memory updated
    """

    def __init__(self):
        self.llm = LLMRouter()
        self.emotion = EmotionAnalyzer()
        self.interrupt = InterruptHandler()
        self.memory = MemoryManager()
        self.tools = ToolRegistry()

        self._running = False
        self._ws_broadcast = None  # Set by WebSocket server

        # Event queues
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()

    async def initialize(self):
        """Initialize all subsystems."""
        logger.info("Initializing dialogue pipeline...")
        await self.memory.initialize()
        await self.tools.initialize()
        logger.info("Pipeline initialized successfully")

    async def shutdown(self):
        """Gracefully shut down the pipeline."""
        self._running = False
        await self.memory.shutdown()
        logger.info("Pipeline shut down")

    def set_ws_broadcast(self, broadcast_fn):
        """Set the WebSocket broadcast function for Live2D updates."""
        self._ws_broadcast = broadcast_fn

    async def _broadcast_to_frontend(self, message: dict):
        """Send a message to the Live2D frontend via WebSocket."""
        if self._ws_broadcast:
            await self._ws_broadcast(message)

    async def process_text_input(
        self,
        text: str,
        speaker_id: Optional[str] = None,
    ) -> LLMResponse:
        """Process a text input through the pipeline.

        This is the main entry point for both voice (post-ASR) and text input.

        Args:
            text: The user's text input.
            speaker_id: Optional speaker identifier from voice print.

        Returns:
            LLMResponse with text, emotion, and tool results.
        """
        logger.info(f"Processing input: '{text[:50]}...' (speaker={speaker_id})")
        self.interrupt.set_state(PipelineState.PROCESSING)

        # Get speaker profile
        speaker_profile = None
        if speaker_id:
            persona = self.llm.persona
            profiles = persona.get("speaker_profiles", {})
            speaker_profile = profiles.get(speaker_id, profiles.get("unknown"))

        # Add to short-term memory
        self.memory.add_message("user", text)

        # Build conversation context
        conversation = self.memory.get_conversation_context()
        messages = [LLMMessage(role=m["role"], content=m["content"]) for m in conversation]

        # Call LLM
        response = await self.llm.chat(messages, LLMTask.CONVERSATION, speaker_profile)

        # Update emotion
        emotion_data = self.emotion.update_emotion(response.emotion)
        await self._broadcast_to_frontend(self.emotion.get_emotion_for_frontend())

        # Execute tool calls if any
        if response.tool_calls:
            tool_results = await self._execute_tools(response.tool_calls)
            if tool_results:
                # Send tool results back to LLM for natural response
                tool_context = "\n".join(
                    f"[ツール結果: {r['name']}] {r['result']}" for r in tool_results
                )
                messages.append(LLMMessage(role="assistant", content=response.text))
                messages.append(LLMMessage(role="user", content=f"ツール実行結果:\n{tool_context}\nこの結果を元に自然に返答してください。"))
                response = await self.llm.chat(messages, LLMTask.CONVERSATION, speaker_profile)
                emotion_data = self.emotion.update_emotion(response.emotion)

        # Add assistant response to memory
        self.memory.add_message("assistant", response.text)

        # Broadcast speaking state to frontend
        await self._broadcast_to_frontend({
            "type": "speech",
            "text": response.text,
            "emotion": response.emotion,
            "is_speaking": True,
        })

        logger.info(f"Response: '{response.text[:50]}...' (emotion={response.emotion})")
        return response

    async def _execute_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Execute MCP tool calls.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'.

        Returns:
            List of results from each tool execution.
        """
        results = []
        for call in tool_calls:
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})
            logger.info(f"Executing tool: {tool_name}({arguments})")

            # Show thinking expression while tool runs
            self.emotion.update_emotion("thinking")
            await self._broadcast_to_frontend(self.emotion.get_emotion_for_frontend())

            try:
                result = await self.tools.execute(tool_name, arguments)
                results.append({
                    "name": tool_name,
                    "result": str(result),
                    "success": True,
                })
            except Exception as e:
                logger.error(f"Tool execution error ({tool_name}): {e}")
                results.append({
                    "name": tool_name,
                    "result": f"エラー: {e}",
                    "success": False,
                })

        return results

    async def process_voice_input(self, audio_data: bytes) -> Optional[LLMResponse]:
        """Process raw audio through the full pipeline.

        This method handles:
        1. ASR (speech-to-text)
        2. Speaker identification
        3. Text processing via process_text_input
        4. TTS (text-to-speech)

        Args:
            audio_data: Raw audio bytes from microphone.

        Returns:
            LLMResponse if successfully processed, None otherwise.
        """
        # ASR and speaker ID would be handled by the audio subsystem
        # This method is called after ASR with transcribed text
        # See audio/microphone.py for the audio capture loop
        pass

    async def run(self):
        """Main pipeline loop - processes items from the input queue."""
        self._running = True
        logger.info("Pipeline running - waiting for input...")

        while self._running:
            try:
                # Wait for input (text or audio)
                item = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)

                if item.get("type") == "text":
                    response = await self.process_text_input(
                        item["text"],
                        item.get("speaker_id"),
                    )
                    await self.output_queue.put({
                        "type": "response",
                        "response": response,
                    })

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                continue
