"""WebSocket server for Live2D frontend communication.

Sends real-time data to the browser-based Live2D frontend:
- Emotion/expression changes
- Lip sync data (volume levels during speech)
- Speech text (for subtitle display)
- Status updates (listening, thinking, speaking)

Also receives text input from the frontend chat box.
Per-client tracking enables access control and session isolation.
"""

import asyncio
import json
from typing import Optional, Set

import websockets
from websockets.server import serve, WebSocketServerProtocol

from src.utils.config_loader import get_main_config
from src.utils.logger import get_logger

logger = get_logger("websocket")


class WebSocketServer:
    """WebSocket server bridging Python backend and Live2D frontend.

    Per-client tracking:
        Each connected client has metadata in _client_info:
        - speaker_id: Authenticated speaker ID (None if not authenticated)
        - session_id: Active conversation session UUID
        - authenticated: Whether this client has been authenticated
    """

    def __init__(self):
        config = get_main_config()
        ws_config = config.get("websocket", {})

        self.host = ws_config.get("host", "0.0.0.0")
        self.port = ws_config.get("port", 8765)

        self._clients: Set[WebSocketServerProtocol] = set()
        self._client_info: dict[WebSocketServerProtocol, dict] = {}
        self._server = None
        self._on_text_input = None   # callback(text, websocket)
        self._on_command = None      # callback(command, data, websocket)
        self._on_audio_input = None  # callback(audio_b64, format, websocket)

    def on_text_input(self, callback):
        """Register callback for text input from frontend.

        Callback signature: async callback(text: str, websocket: WebSocketServerProtocol)
        """
        self._on_text_input = callback

    def on_command(self, callback):
        """Register callback for commands from frontend.

        Callback signature: async callback(command: str, data: dict, websocket: WebSocketServerProtocol)
        """
        self._on_command = callback

    def on_audio_input(self, callback):
        """Register callback for audio input from frontend.

        Frontend sends raw audio (base64-encoded webm/opus) for backend
        ASR and speaker identification.

        Callback signature: async callback(audio_b64: str, format: str, websocket: WebSocketServerProtocol)
        """
        self._on_audio_input = callback

    async def start(self):
        """Start the WebSocket server."""
        self._server = await serve(
            self._handler,
            self.host,
            self.port,
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("WebSocket server stopped")

    # ----------------------------------------------------------------
    # Per-client management
    # ----------------------------------------------------------------

    def set_client_speaker(
        self,
        websocket: WebSocketServerProtocol,
        speaker_id: str,
        session_id: str = None,
    ):
        """Associate a WebSocket client with an authenticated speaker.

        Args:
            websocket: The client connection.
            speaker_id: Authenticated speaker ID.
            session_id: Active conversation session UUID.
        """
        if websocket in self._client_info:
            self._client_info[websocket].update({
                "speaker_id": speaker_id,
                "session_id": session_id,
                "authenticated": True,
            })
            logger.info(f"Client authenticated: speaker={speaker_id}, session={session_id}")

    def get_client_info(self, websocket: WebSocketServerProtocol) -> dict:
        """Get client metadata.

        Args:
            websocket: The client connection.

        Returns:
            Dict with speaker_id, session_id, authenticated.
        """
        return self._client_info.get(websocket, {
            "speaker_id": None,
            "session_id": None,
            "authenticated": False,
        })

    async def send_to_client(self, websocket: WebSocketServerProtocol, message: dict):
        """Send a message to a specific client.

        Args:
            websocket: Target client.
            message: Message dict to send.
        """
        if websocket is None:
            return
        try:
            data = json.dumps(message, ensure_ascii=False)
            await websocket.send(data)
        except websockets.exceptions.ConnectionClosed:
            self._clients.discard(websocket)
            self._client_info.pop(websocket, None)

    # ----------------------------------------------------------------
    # Connection handling
    # ----------------------------------------------------------------

    async def _handler(self, websocket: WebSocketServerProtocol):
        """Handle a WebSocket connection."""
        self._clients.add(websocket)
        self._client_info[websocket] = {
            "speaker_id": None,
            "session_id": None,
            "authenticated": False,
        }
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        # Send initial state
        await websocket.send(json.dumps({
            "type": "init",
            "status": "connected",
            "emotion": "neutral",
            "expression": "normal",
        }))

        try:
            async for message in websocket:
                await self._handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        finally:
            self._clients.discard(websocket)
            self._client_info.pop(websocket, None)

    async def _handle_message(self, websocket: WebSocketServerProtocol, raw_message: str):
        """Process incoming messages from the frontend.

        Message types:
        - text_input: User typed a message in the chat box
        - command: Frontend commands (e.g., mode switch, settings)
        - ping: Keep-alive
        """
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {raw_message[:100]}")
            return

        msg_type = message.get("type", "")

        if msg_type == "text_input":
            text = message.get("text", "")
            if text and self._on_text_input:
                logger.debug(f"Text input from frontend: '{text[:50]}'")
                if asyncio.iscoroutinefunction(self._on_text_input):
                    await self._on_text_input(text, websocket)
                else:
                    self._on_text_input(text, websocket)

        elif msg_type == "command":
            command = message.get("command", "")
            if command and self._on_command:
                if asyncio.iscoroutinefunction(self._on_command):
                    await self._on_command(command, message.get("data"), websocket)
                else:
                    self._on_command(command, message.get("data"), websocket)

        elif msg_type == "audio_input":
            audio_b64 = message.get("audio", "")
            audio_format = message.get("format", "audio/webm")
            if audio_b64 and self._on_audio_input:
                logger.debug(f"Audio input from frontend: {len(audio_b64)} chars base64")
                if asyncio.iscoroutinefunction(self._on_audio_input):
                    await self._on_audio_input(audio_b64, audio_format, websocket)
                else:
                    self._on_audio_input(audio_b64, audio_format, websocket)

        elif msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))

    async def broadcast(self, message: dict):
        """Send a message to all connected clients.

        Args:
            message: Message dict to send (will be JSON-encoded).
        """
        if not self._clients:
            return

        data = json.dumps(message, ensure_ascii=False)
        disconnected = set()

        for client in self._clients:
            try:
                await client.send(data)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        for client in disconnected:
            self._clients.discard(client)
            self._client_info.pop(client, None)

    async def send_emotion(self, emotion: str, expression: str, motion_group: str = "Idle"):
        """Send emotion update to frontend.

        Args:
            emotion: Emotion tag (e.g., "happy", "thinking").
            expression: Live2D expression ID.
            motion_group: Live2D motion group.
        """
        await self.broadcast({
            "type": "emotion",
            "emotion": emotion,
            "expression": expression,
            "motion_group": motion_group,
        })

    async def send_speech_start(self, text: str, emotion: str = "neutral"):
        """Notify frontend that speech is starting.

        Args:
            text: The text being spoken.
            emotion: Current emotion.
        """
        await self.broadcast({
            "type": "speech_start",
            "text": text,
            "emotion": emotion,
        })

    async def send_speech_end(self):
        """Notify frontend that speech has ended."""
        await self.broadcast({
            "type": "speech_end",
        })

    async def send_lip_sync(self, volumes: list[float]):
        """Send lip sync data (volume levels over time).

        Args:
            volumes: List of volume levels (0.0-1.0) per frame.
        """
        await self.broadcast({
            "type": "lip_sync",
            "volumes": volumes,
        })

    async def send_status(self, status: str):
        """Send pipeline status update.

        Args:
            status: "idle" | "listening" | "processing" | "speaking"
        """
        await self.broadcast({
            "type": "status",
            "status": status,
        })

    async def send_subtitle(self, text: str, role: str = "assistant"):
        """Send subtitle text for display.

        Args:
            text: Subtitle text.
            role: "user" or "assistant".
        """
        await self.broadcast({
            "type": "subtitle",
            "text": text,
            "role": role,
        })

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)
