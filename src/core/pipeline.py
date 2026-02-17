"""Main dialogue pipeline: Audio Input → ASR → LLM → TTS → Audio Output.

Orchestrates the full conversation flow.
Inspired by Pipecat's frame-based pipeline and py-xiaozhi's state machine.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.core.emotion_analyzer import EmotionAnalyzer
from src.core.interrupt_handler import InterruptHandler, PipelineState
from src.core.llm_router import LLMMessage, LLMResponse, LLMRouter, LLMTask
from src.memory.memory_manager import MemoryManager
from src.mcp.tool_registry import ToolContext, ToolRegistry
from src.utils.config_loader import get_persona_config
from src.utils.logger import get_logger

logger = get_logger("pipeline")


@dataclass
class TurnContext:
    """Structured context for a single dialogue turn.

    Captures every piece of information flowing through the pipeline
    for a single user→assistant exchange. Enables:
    - Debug & replay: full state of each turn is preserved
    - Cost tracking: LLM usage per turn
    - Analytics: response time, tool usage patterns
    - Abort tracking: interrupted turns are kept, marked as aborted
    """

    # --- Identity ---
    request_id: str = ""
    intent: str = "chat"  # "chat" | "tool" | "proactive" | "reminder"

    # --- Input ---
    input_text: str = ""
    input_type: str = "text"  # "text" | "voice"
    speaker_id: Optional[str] = None
    session_id: Optional[str] = None
    speaker_profile: Optional[dict] = None

    # --- Memory ---
    memory_context: str = ""
    memory_refs: list[str] = field(default_factory=list)

    # --- LLM ---
    llm_response: Optional[LLMResponse] = None
    llm_model: str = ""
    llm_usage: dict = field(default_factory=dict)  # {"input_tokens": N, "output_tokens": N}

    # --- Tools ---
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)

    # --- Output ---
    response_text: str = ""
    emotion: str = "neutral"
    tts_engine: str = ""
    audio_bytes: int = 0
    audio_format: str = ""  # "wav" | "mp3"
    audio_duration_sec: float = 0.0

    # --- Timing ---
    started_at: float = field(default_factory=time.time)
    llm_started_at: float = 0.0
    llm_finished_at: float = 0.0
    tts_started_at: float = 0.0
    tts_finished_at: float = 0.0
    finished_at: float = 0.0

    # --- Status ---
    aborted: bool = False
    error: Optional[str] = None

    @property
    def total_duration_ms(self) -> float:
        """Total turn duration in milliseconds."""
        if self.finished_at and self.started_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0

    @property
    def llm_duration_ms(self) -> float:
        """LLM call duration in milliseconds."""
        if self.llm_finished_at and self.llm_started_at:
            return (self.llm_finished_at - self.llm_started_at) * 1000
        return 0.0

    @property
    def tts_duration_ms(self) -> float:
        """TTS synthesis duration in milliseconds."""
        if self.tts_finished_at and self.tts_started_at:
            return (self.tts_finished_at - self.tts_started_at) * 1000
        return 0.0

    def summary(self) -> str:
        """One-line summary for logging."""
        status = "ABORTED" if self.aborted else ("ERROR" if self.error else "OK")
        return (
            f"[{self.request_id}] {status} "
            f"total={self.total_duration_ms:.0f}ms "
            f"llm={self.llm_duration_ms:.0f}ms "
            f"tts={self.tts_duration_ms:.0f}ms "
            f"model={self.llm_model} "
            f"emotion={self.emotion} "
            f"tools={len(self.tool_results)} "
            f"audio={self.audio_bytes}b/{self.audio_format}"
        )


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

        # Per-speaker session isolation
        from src.memory.short_term import ShortTermMemory
        self._speaker_sessions: dict[str, str] = {}   # speaker_id → session_id
        self._speaker_memory: dict[str, ShortTermMemory] = {}  # speaker_id → separate context

    async def initialize(self):
        """Initialize all subsystems."""
        logger.info("Initializing dialogue pipeline...")
        await self.memory.initialize()
        await self.tools.initialize()
        # Initialize preference extractor (needs both memory and LLM)
        self.memory.init_preference_extractor(self.llm)
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

    def _get_speaker_memory(self, speaker_id: Optional[str]):
        """Get or create per-speaker short-term memory.

        For session isolation: each speaker has their own conversation buffer.
        Falls back to the shared memory if speaker_id is None.
        """
        if not speaker_id:
            return self.memory.short_term

        if speaker_id not in self._speaker_memory:
            from src.memory.short_term import ShortTermMemory
            self._speaker_memory[speaker_id] = ShortTermMemory()
        return self._speaker_memory[speaker_id]

    async def get_or_create_session(self, speaker_id: str) -> str:
        """Get or create a conversation session for a speaker.

        Args:
            speaker_id: Speaker to get/create session for.

        Returns:
            Session UUID.
        """
        if speaker_id in self._speaker_sessions:
            return self._speaker_sessions[speaker_id]

        session_id = await self.memory.long_term.create_session(speaker_id)
        self._speaker_sessions[speaker_id] = session_id
        return session_id

    async def process_text_input(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        request_id: Optional[str] = None,
        intent: str = "chat",
        session_id: Optional[str] = None,
    ) -> tuple[LLMResponse, TurnContext]:
        """Process a text input through the pipeline.

        Args:
            text: The user's text input.
            speaker_id: Optional speaker identifier from voice print.
            request_id: Optional request ID for tracking.
            intent: Turn intent ("chat", "proactive", "reminder").
            session_id: Optional session ID for audit tracking.

        Returns:
            Tuple of (LLMResponse, TurnContext) with full turn tracking.
        """
        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        # --- Session management ---
        if speaker_id and not session_id:
            session_id = await self.get_or_create_session(speaker_id)

        # --- Build TurnContext ---
        speaker_profile = self._get_speaker_profile(speaker_id)
        turn = TurnContext(
            request_id=request_id,
            intent=intent,
            input_text=text,
            speaker_id=speaker_id,
            session_id=session_id,
            speaker_profile=speaker_profile,
        )

        logger.info(f"[{request_id}] Processing: '{text[:50]}' (speaker={speaker_id}, session={session_id})")
        self.interrupt.set_state(PipelineState.PROCESSING)
        await self._broadcast_to_frontend({"type": "status", "status": "processing"})

        self._current_speaker_id = speaker_id
        self._current_speaker_profile = speaker_profile

        # --- Per-speaker isolated memory ---
        speaker_mem = self._get_speaker_memory(speaker_id)
        speaker_mem.add("user", text)

        # Also add to shared memory for global context
        self.memory.add_message("user", text)

        # Build conversation context (include memory context if available)
        memory_context = await self.memory.get_context_for_prompt()
        turn.memory_context = memory_context

        # Use per-speaker conversation context
        conversation = speaker_mem.get_context()
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

        # Get emotion momentum and rhythm hints
        emotion_hint = self.emotion.get_emotion_momentum_hint()
        rhythm_hint = self.emotion.get_conversation_rhythm_hint(
            speaker_mem.turn_count
        )

        # Call LLM
        turn.llm_started_at = time.time()
        response = await self.llm.chat(
            messages, LLMTask.CONVERSATION, speaker_profile,
            emotion_hint=emotion_hint,
            rhythm_hint=rhythm_hint,
        )
        turn.llm_finished_at = time.time()
        turn.llm_model = response.model
        turn.llm_usage = response.usage
        turn.llm_response = response

        # Check for interruption
        if self.interrupt.should_stop():
            logger.info(f"[{request_id}] Interrupted during LLM call")
            self.interrupt.clear_interrupt()
            turn.aborted = True
            turn.finished_at = time.time()
            logger.info(f"Turn: {turn.summary()}")
            return LLMResponse(text="", emotion="neutral", model="interrupted"), turn

        # Update emotion
        turn.emotion = response.emotion
        self.emotion.update_emotion(response.emotion)
        await self._broadcast_to_frontend(self.emotion.get_emotion_for_frontend())

        # Execute tool calls if any (with permission check)
        if response.tool_calls:
            turn.tool_calls = response.tool_calls
            turn.intent = "tool"
            tool_ctx = ToolContext(
                speaker_id=speaker_id,
                speaker_role=speaker_profile.get("role") if speaker_profile else None,
                language=speaker_profile.get("language", "japanese") if speaker_profile else "japanese",
                session_id=request_id,
            )
            tool_results = await self._execute_tools(response.tool_calls, speaker_profile, tool_ctx)
            turn.tool_results = tool_results

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
                turn.llm_started_at = time.time()
                response = await self.llm.chat(messages, LLMTask.CONVERSATION, speaker_profile)
                turn.llm_finished_at = time.time()
                turn.llm_model = response.model
                turn.emotion = response.emotion
                turn.llm_response = response
                self.emotion.update_emotion(response.emotion)

        # Finalize turn
        turn.response_text = response.text
        turn.finished_at = time.time()

        # Add assistant response to per-speaker and global memory
        speaker_mem = self._get_speaker_memory(speaker_id)
        speaker_mem.add("assistant", response.text)
        self.memory.add_message("assistant", response.text)

        # Trigger preference extraction in background (non-blocking)
        await self.memory.maybe_extract_preferences(request_id)

        logger.info(f"[{request_id}] Response: '{response.text[:50]}' (emotion={response.emotion})")
        logger.info(f"Turn: {turn.summary()}")
        return response, turn

    async def _execute_tools(
        self, tool_calls: list[dict], speaker_profile: Optional[dict] = None,
        tool_ctx: Optional[ToolContext] = None,
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
                result = await self.tools.execute(tool_name, arguments, ctx=tool_ctx)
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

    async def process_text_input_stream(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        request_id: Optional[str] = None,
        intent: str = "chat",
        session_id: Optional[str] = None,
    ):
        """Process text input with streaming LLM output.

        Yields partial results as they arrive from the LLM, enabling
        sentence-level TTS synthesis (Pipecat-inspired streaming pipeline).

        Yields:
            dict with keys:
            - "type": "sentence" | "final"
            - "sentence": extracted sentence text (for "sentence" type)
            - "full_text": accumulated text so far
            - "emotion": detected emotion (may be "neutral" until JSON parsed)
            - "turn": TurnContext
            - "tool_calls": list (only in "final")
        """
        from src.utils.sentence_splitter import SentenceSplitter, extract_json_text_and_emotion

        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        # --- Session management ---
        if speaker_id and not session_id:
            session_id = await self.get_or_create_session(speaker_id)

        # --- Build TurnContext ---
        speaker_profile = self._get_speaker_profile(speaker_id)
        turn = TurnContext(
            request_id=request_id,
            intent=intent,
            input_text=text,
            speaker_id=speaker_id,
            session_id=session_id,
            speaker_profile=speaker_profile,
        )

        logger.info(f"[{request_id}] Stream processing: '{text[:50]}' (speaker={speaker_id})")
        self.interrupt.set_state(PipelineState.PROCESSING)
        await self._broadcast_to_frontend({"type": "status", "status": "processing"})

        self._current_speaker_id = speaker_id
        self._current_speaker_profile = speaker_profile

        # --- Per-speaker isolated memory ---
        speaker_mem = self._get_speaker_memory(speaker_id)
        speaker_mem.add("user", text)
        self.memory.add_message("user", text)

        # Build conversation context
        memory_context = await self.memory.get_context_for_prompt()
        turn.memory_context = memory_context

        conversation = speaker_mem.get_context()
        messages = []

        if memory_context:
            messages.append(LLMMessage(
                role="user",
                content=f"[コンテキスト情報]\n{memory_context}\n---\n以上は参考情報です。ユーザーの質問に答えてください。"
            ))
            messages.append(LLMMessage(role="assistant", content="はい、承知しました。"))

        for m in conversation:
            messages.append(LLMMessage(role=m["role"], content=m["content"]))

        # Get emotion hints
        emotion_hint = self.emotion.get_emotion_momentum_hint()
        rhythm_hint = self.emotion.get_conversation_rhythm_hint(
            speaker_mem.turn_count
        )

        # --- Stream LLM tokens ---
        turn.llm_started_at = time.time()
        splitter = SentenceSplitter(min_length=6, max_length=60)
        accumulated_raw = ""  # Raw tokens from LLM (may include JSON wrapper)
        sentences_yielded = 0

        # JSON-aware streaming text extractor:
        # LLM returns {"text": "...", "emotion": "..."} but we receive tokens
        # incrementally. We need to detect and skip the JSON wrapper.
        json_state = "detect"  # detect | inside_text | passthrough
        text_key_found = False
        inside_quotes = False
        escape_next = False

        def extract_text_from_token(token: str, raw_so_far: str) -> str:
            """Strip JSON wrapper and extract only the 'text' value content.

            States:
            - detect: looking for {"text": " pattern → switch to inside_text
            - inside_text: we're inside the text value, pass through
            - passthrough: no JSON detected, pass raw tokens
            """
            nonlocal json_state, text_key_found, inside_quotes, escape_next

            if json_state == "passthrough":
                return token

            if json_state == "detect":
                # Check if the accumulated raw looks like JSON start
                stripped = raw_so_far.lstrip()
                if stripped.startswith("```json"):
                    stripped = stripped[7:].lstrip()
                if stripped.startswith("```"):
                    stripped = stripped[3:].lstrip()

                # Look for the text value opening
                import re
                # Match: {"text": " or {"text":"
                match = re.search(r'["\']text["\']\s*:\s*["\']', stripped)
                if match:
                    # Found the text value — extract everything after the opening quote
                    json_state = "inside_text"
                    after = stripped[match.end():]
                    return after  # Return content after {"text": "
                elif len(stripped) > 50:
                    # Long enough without JSON pattern — treat as plain text
                    json_state = "passthrough"
                    return raw_so_far
                else:
                    return ""  # Still detecting, don't yield anything yet

            if json_state == "inside_text":
                # We're inside the text value — pass through until closing quote
                # Handle JSON escape sequences: \n → newline, \t → tab, etc.
                json_escapes = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', '/': '/'}
                result = []
                for ch in token:
                    if escape_next:
                        # Convert JSON escape to actual character
                        result.append(json_escapes.get(ch, ch))
                        escape_next = False
                        continue
                    if ch == '\\':
                        escape_next = True
                        continue
                    if ch == '"':
                        # End of text value — stop extracting
                        json_state = "done"
                        break
                    result.append(ch)
                return "".join(result)

            return ""  # "done" state — no more text to extract

        try:
            async for token in self.llm.chat_stream(
                messages, LLMTask.CONVERSATION, speaker_profile,
                emotion_hint=emotion_hint,
                rhythm_hint=rhythm_hint,
            ):
                if self.interrupt.should_stop():
                    logger.info(f"[{request_id}] Interrupted during LLM streaming")
                    turn.aborted = True
                    break

                accumulated_raw += token

                # Extract just the text content (strip JSON wrapper in real-time)
                text_part = extract_text_from_token(token, accumulated_raw)
                if not text_part:
                    continue

                complete_sentences = splitter.push(text_part)
                for sentence in complete_sentences:
                    sentences_yielded += 1
                    yield {
                        "type": "sentence",
                        "sentence": sentence,
                        "sentence_index": sentences_yielded - 1,
                        "full_text": accumulated_raw,
                        "emotion": "neutral",
                        "turn": turn,
                    }

        except Exception as e:
            logger.error(f"[{request_id}] LLM stream error: {e}")
            turn.error = str(e)

        turn.llm_finished_at = time.time()

        # Flush remaining buffer
        remaining = splitter.flush()
        if remaining and not turn.aborted:
            sentences_yielded += 1
            yield {
                "type": "sentence",
                "sentence": remaining,
                "sentence_index": sentences_yielded - 1,
                "full_text": accumulated_raw,
                "emotion": "neutral",
                "turn": turn,
            }

        # --- Parse final response (extract text + emotion from JSON) ---
        final_text, emotion = extract_json_text_and_emotion(accumulated_raw)

        # Update emotion
        turn.emotion = emotion
        self.emotion.update_emotion(emotion)
        await self._broadcast_to_frontend(self.emotion.get_emotion_for_frontend())

        # Build LLMResponse for compatibility
        response = LLMResponse(
            text=final_text,
            emotion=emotion,
            model=self.llm._get_model_config(LLMTask.CONVERSATION).get("model", ""),
        )
        turn.llm_response = response
        turn.response_text = final_text

        # Check for tool calls in the full response
        parsed = self.llm._parse_response(accumulated_raw, response.model)
        if parsed.tool_calls:
            response.tool_calls = parsed.tool_calls

        # Handle tool calls (non-streaming, as tools need full execution)
        if response.tool_calls and not turn.aborted:
            turn.tool_calls = response.tool_calls
            turn.intent = "tool"
            tool_ctx = ToolContext(
                speaker_id=speaker_id,
                speaker_role=speaker_profile.get("role") if speaker_profile else None,
                language=speaker_profile.get("language", "japanese") if speaker_profile else "japanese",
                session_id=request_id,
            )
            tool_results = await self._execute_tools(response.tool_calls, speaker_profile, tool_ctx)
            turn.tool_results = tool_results

            if tool_results:
                # Re-run LLM with tool results (non-streaming for simplicity)
                tool_context = "\n".join(
                    f"[ツール結果: {r['name']}] {r['result']}" for r in tool_results
                )
                messages.append(LLMMessage(role="assistant", content=response.text))
                messages.append(LLMMessage(
                    role="user",
                    content=f"ツール実行結果:\n{tool_context}\nこの結果を元に自然に返答してください。"
                ))
                response = await self.llm.chat(messages, LLMTask.CONVERSATION, speaker_profile)
                turn.llm_response = response
                turn.emotion = response.emotion
                turn.response_text = response.text
                self.emotion.update_emotion(response.emotion)

        # --- Finalize ---
        turn.finished_at = time.time()

        # Add to memory
        if final_text and not turn.aborted:
            speaker_mem = self._get_speaker_memory(speaker_id)
            speaker_mem.add("assistant", final_text)
            self.memory.add_message("assistant", final_text)
            await self.memory.maybe_extract_preferences(request_id)

        # Yield final result
        yield {
            "type": "final",
            "response": response,
            "turn": turn,
            "full_text": final_text,
            "emotion": emotion,
            "tool_calls": response.tool_calls if response.tool_calls else [],
        }

        logger.info(f"[{request_id}] Response: '{final_text[:50]}' (emotion={emotion})")
        logger.info(f"Turn: {turn.summary()}")

    async def run(self):
        """Main pipeline loop - processes items from the input queue.

        Routes to streaming or non-streaming based on the item's 'stream' flag.
        """
        self._running = True
        logger.info("Pipeline running - waiting for input...")

        while self._running:
            try:
                # Wait for input (text or audio)
                item = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)

                if item.get("type") == "text":
                    request_id = item.get("request_id", str(uuid.uuid4())[:8])
                    intent = item.get("intent", "chat")
                    use_stream = item.get("stream", True)

                    if use_stream and intent in ("chat", "proactive"):
                        # Streaming path: yield sentences to output queue
                        await self.output_queue.put({
                            "type": "stream_start",
                            "request_id": request_id,
                            "text": item["text"],
                            "speaker_id": item.get("speaker_id"),
                            "session_id": item.get("session_id"),
                            "intent": intent,
                        })
                    else:
                        # Non-streaming path (tools, enrollment, etc.)
                        response, turn = await self.process_text_input(
                            item["text"],
                            item.get("speaker_id"),
                            request_id,
                            intent=intent,
                            session_id=item.get("session_id"),
                        )
                        await self.output_queue.put({
                            "type": "response",
                            "request_id": request_id,
                            "response": response,
                            "turn": turn,
                        })

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                continue
