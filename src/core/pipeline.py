"""Main dialogue pipeline: Audio Input → ASR → LLM → TTS → Audio Output.

Orchestrates the full conversation flow.
Inspired by Pipecat's frame-based pipeline and py-xiaozhi's state machine.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from src.core.emotion_analyzer import EmotionAnalyzer
from src.core.interrupt_handler import InterruptHandler, PipelineState
from src.core.llm_router import LLMMessage, LLMResponse, LLMRouter, LLMTask
from src.memory.memory_manager import MemoryManager
from src.mcp.tool_registry import ToolRegistry
from src.utils.config_loader import get_persona_config
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
        6. MCP tools executed if needed (with permission check)
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

        # Event queues (with request_id tracking to prevent race conditions)
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # Current speaker context
        self._current_speaker_id: Optional[str] = None
        self._current_speaker_profile: Optional[dict] = None

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

    def _get_speaker_profile(self, speaker_id: Optional[str]) -> Optional[dict]:
        """Get speaker profile from persona config."""
        if not speaker_id:
            return None
        persona = get_persona_config()
        profiles = persona.get("speaker_profiles", {})
        return profiles.get(speaker_id, profiles.get("unknown"))

    def _check_tool_permission(self, tool_name: str, speaker_profile: Optional[dict]) -> bool:
        """Check if the current speaker has permission to use a tool.

        Args:
            tool_name: Name of the tool.
            speaker_profile: Speaker profile dict with 'permissions' field.

        Returns:
            True if permitted, False otherwise.
        """
        if not speaker_profile:
            # Unknown speakers: only allow knowledge base
            return tool_name in ("cattery_knowledge",)

        permissions = speaker_profile.get("permissions", [])
        if "all" in permissions:
            return True

        return tool_name in permissions

    async def process_text_input(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        """Process a text input through the pipeline.

        Args:
            text: The user's text input.
            speaker_id: Optional speaker identifier from voice print.
            request_id: Optional request ID for tracking.

        Returns:
            LLMResponse with text, emotion, and tool results.
        """
        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        logger.info(f"[{request_id}] Processing: '{text[:50]}' (speaker={speaker_id})")
        self.interrupt.set_state(PipelineState.PROCESSING)
        await self._broadcast_to_frontend({"type": "status", "status": "processing"})

        # Resolve speaker profile
        speaker_profile = self._get_speaker_profile(speaker_id)
        self._current_speaker_id = speaker_id
        self._current_speaker_profile = speaker_profile

        # Add to short-term memory
        self.memory.add_message("user", text)

        # Build conversation context (include memory context if available)
        memory_context = await self.memory.get_context_for_prompt()
        conversation = self.memory.get_conversation_context()
        messages = []

        # Inject memory context as a system-level message if available
        if memory_context:
            messages.append(LLMMessage(
                role="user",
                content=f"[コンテキスト情報]\n{memory_context}\n---\n以上は参考情報です。ユーザーの質問に答えてください。"
            ))
            messages.append(LLMMessage(role="assistant", content="はい、承知しました。"))

        for m in conversation:
            messages.append(LLMMessage(role=m["role"], content=m["content"]))

        # Call LLM
        response = await self.llm.chat(messages, LLMTask.CONVERSATION, speaker_profile)

        # Check for interruption
        if self.interrupt.should_stop():
            logger.info(f"[{request_id}] Interrupted during LLM call")
            self.interrupt.clear_interrupt()
            return LLMResponse(text="", emotion="neutral", model="interrupted")

        # Update emotion
        self.emotion.update_emotion(response.emotion)
        await self._broadcast_to_frontend(self.emotion.get_emotion_for_frontend())

        # Execute tool calls if any (with permission check)
        if response.tool_calls:
            tool_results = await self._execute_tools(response.tool_calls, speaker_profile)
            if tool_results:
                # Send tool results back to LLM for natural response
                tool_context = "\n".join(
                    f"[ツール結果: {r['name']}] {r['result']}" for r in tool_results
                )
                messages.append(LLMMessage(role="assistant", content=response.text))
                messages.append(LLMMessage(
                    role="user",
                    content=f"ツール実行結果:\n{tool_context}\nこの結果を元に自然に返答してください。"
                ))
                response = await self.llm.chat(messages, LLMTask.CONVERSATION, speaker_profile)
                self.emotion.update_emotion(response.emotion)

        # Add assistant response to memory
        self.memory.add_message("assistant", response.text)

        logger.info(f"[{request_id}] Response: '{response.text[:50]}' (emotion={response.emotion})")
        return response

    async def _execute_tools(
        self, tool_calls: list[dict], speaker_profile: Optional[dict] = None
    ) -> list[dict]:
        """Execute MCP tool calls with permission checking.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'.
            speaker_profile: Current speaker's profile for permission check.

        Returns:
            List of results from each tool execution.
        """
        results = []
        for call in tool_calls:
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})

            # Permission check
            if not self._check_tool_permission(tool_name, speaker_profile):
                logger.warning(f"Tool '{tool_name}' denied for speaker role")
                results.append({
                    "name": tool_name,
                    "result": "この操作は権限がありません。",
                    "success": False,
                })
                continue

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

    async def run(self):
        """Main pipeline loop - processes items from the input queue."""
        self._running = True
        logger.info("Pipeline running - waiting for input...")

        while self._running:
            try:
                # Wait for input (text or audio)
                item = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)

                if item.get("type") == "text":
                    request_id = item.get("request_id", str(uuid.uuid4())[:8])
                    response = await self.process_text_input(
                        item["text"],
                        item.get("speaker_id"),
                        request_id,
                    )
                    await self.output_queue.put({
                        "type": "response",
                        "request_id": request_id,
                        "response": response,
                    })

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                continue
