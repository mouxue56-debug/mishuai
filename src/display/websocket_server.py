"""WebSocket server for Live2D frontend communication.

Sends real-time data to the browser-based Live2D frontend:
- Emotion/expression changes
- Lip sync data (volume levels during speech)
- Speech text (for subtitle display)
- Status updates (listening, thinking, speaking)

Also receives text input from the frontend chat box.
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
    """WebSocket server bridging Python backend and Live2D frontend."""

    def __init__(self):
        config = get_main_config()
        ws_config = config.get("websocket", {})

        self.host = ws_config.get("host", "0.0.0.0")
        self.port = ws_config.get("port", 8765)

        self._clients: Set[WebSocketServerProtocol] = set()
        self._server = None
        self._on_text_input = None
        self._on_command = None

    def on_text_input(self, callback):
        """Register callback for text input from frontend."""
        self._on_text_input = callback

    def on_command(self, callback):
        """Register callback for commands from frontend."""
        self._on_command = callback

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

    async def _handler(self, websocket: WebSocketServerProtocol):
        """Handle a WebSocket connection."""
        self._clients.add(websocket)
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
                    await self._on_text_input(text)
                else:
                    self._on_text_input(text)

        elif msg_type == "command":
            command = message.get("command", "")
            if command and self._on_command:
                if asyncio.iscoroutinefunction(self._on_command):
                    await self._on_command(command, message.get("data"))
                else:
                    self._on_command(command, message.get("data"))

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

        self._clients -= disconnected

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
