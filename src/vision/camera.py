"""Vision input module - Camera capture and Gemini Flash scene analysis.

Captures frames from a camera (USB/built-in) and sends them to
Google Gemini Flash for real-time scene understanding.

Use cases:
- Detect visitors arriving at the cattery reception
- Describe what's happening in the room
- Trigger proactive greetings when someone appears

Requires:
    pip install opencv-python google-generativeai
"""

import asyncio
import base64
import io
import time
from typing import Optional, Callable, Awaitable

from src.utils.config_loader import get_main_config, get_api_key
from src.utils.logger import get_logger

logger = get_logger("vision")

# Lazy imports for optional dependencies
cv2 = None
genai = None


def _ensure_imports():
    """Lazy-import opencv and google-generativeai."""
    global cv2, genai
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            raise ImportError("opencv-python is required: pip install opencv-python")
    if genai is None:
        try:
            import google.generativeai as _genai
            genai = _genai
        except ImportError:
            raise ImportError("google-generativeai is required: pip install google-generativeai")


class VisionInput:
    """Captures camera frames and analyzes them with Gemini Flash.

    Architecture:
        Camera (OpenCV) → JPEG frame → Gemini Flash → scene description
                                                         ↓
                                              on_scene_change callback
                                              (feeds into proactive scheduler)
    """

    def __init__(self):
        config = get_main_config()
        vision_config = config.get("vision", {})

        self.enabled = vision_config.get("enabled", False)
        self.camera_index = vision_config.get("camera_index", 0)
        self.capture_interval = vision_config.get("capture_interval_sec", 10)
        self.model_name = vision_config.get("model", "gemini-2.0-flash")
        self.resolution = vision_config.get("resolution", [640, 480])
        self.jpeg_quality = vision_config.get("jpeg_quality", 70)

        self._cap = None
        self._model = None
        self._running = False
        self._last_scene: Optional[str] = None
        self._last_capture_time = 0.0
        self._person_present = False

        # Callbacks
        self._on_scene_change: Optional[Callable] = None
        self._on_person_detected: Optional[Callable] = None

        # Prompt for scene analysis
        self._analysis_prompt = vision_config.get("prompt",
            "この画像は猫カフェ/キャッテリーの受付カメラです。"
            "簡潔に(1-2文で)シーンを説明してください。"
            "人が写っている場合は「[PERSON]」タグを先頭に付けてください。"
            "猫が写っている場合は「[CAT]」タグを付けてください。"
            "特に変化がなく静かなシーンなら「[QUIET]」とだけ返してください。"
        )

    def on_scene_change(self, callback: Callable):
        """Register callback for scene description updates."""
        self._on_scene_change = callback

    def on_person_detected(self, callback: Callable):
        """Register callback for person detection."""
        self._on_person_detected = callback

    async def initialize(self):
        """Initialize camera and Gemini model."""
        if not self.enabled:
            logger.info("Vision input disabled")
            return

        try:
            _ensure_imports()
        except ImportError as e:
            logger.warning(f"Vision disabled: {e}")
            self.enabled = False
            return

        # Initialize Gemini
        try:
            api_key = get_api_key("google")
        except ValueError:
            logger.warning("Vision disabled: no GOOGLE_API_KEY found in .env")
            self.enabled = False
            return

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model_name)
        logger.info(f"Vision model initialized: {self.model_name}")

        # Open camera
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.warning(f"Vision disabled: cannot open camera index {self.camera_index}")
            self.enabled = False
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        logger.info(f"Camera opened: index={self.camera_index}, "
                     f"resolution={self.resolution[0]}x{self.resolution[1]}")

    async def start(self):
        """Start the periodic capture loop."""
        if not self.enabled:
            return

        self._running = True
        logger.info(f"Vision capture started (interval={self.capture_interval}s)")

        while self._running:
            try:
                await self._capture_and_analyze()
            except Exception as e:
                logger.error(f"Vision capture error: {e}")

            await asyncio.sleep(self.capture_interval)

    async def stop(self):
        """Stop capture and release camera."""
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released")

    async def capture_once(self) -> Optional[str]:
        """Capture a single frame and analyze it.

        Returns:
            Scene description string, or None on failure.
        """
        if not self.enabled or not self._cap:
            return None

        frame = await self._grab_frame()
        if frame is None:
            return None

        return await self._analyze_frame(frame)

    @property
    def is_person_present(self) -> bool:
        """Whether a person was detected in the last capture."""
        return self._person_present

    @property
    def last_scene(self) -> Optional[str]:
        """Last scene description."""
        return self._last_scene

    # --- Private ---

    async def _grab_frame(self) -> Optional[bytes]:
        """Capture a frame from the camera as JPEG bytes."""
        if not self._cap or not self._cap.isOpened():
            return None

        # Run in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self._cap.read)
        if not ret or frame is None:
            return None

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return buffer.tobytes()

    async def _analyze_frame(self, jpeg_bytes: bytes) -> Optional[str]:
        """Send a JPEG frame to Gemini Flash for analysis."""
        if not self._model:
            return None

        try:
            image_part = {
                "mime_type": "image/jpeg",
                "data": jpeg_bytes,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(
                    [self._analysis_prompt, image_part],
                    generation_config={"max_output_tokens": 150, "temperature": 0.3},
                )
            )

            description = response.text.strip()
            return description

        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return None

    async def _capture_and_analyze(self):
        """Capture frame, analyze, and fire callbacks if scene changed."""
        frame = await self._grab_frame()
        if frame is None:
            return

        description = await self._analyze_frame(frame)
        if description is None:
            return

        # Detect person presence
        was_person_present = self._person_present
        self._person_present = "[PERSON]" in description

        # Fire person detection callback (on transition: no person → person)
        if self._person_present and not was_person_present:
            logger.info(f"Person detected: {description}")
            if self._on_person_detected:
                cb = self._on_person_detected
                if asyncio.iscoroutinefunction(cb):
                    await cb(description)
                else:
                    cb(description)

        # Fire scene change callback (skip [QUIET] unless person status changed)
        is_quiet = description.strip() == "[QUIET]"
        scene_changed = (not is_quiet) and (description != self._last_scene)

        if scene_changed or (self._person_present != was_person_present):
            self._last_scene = description
            if self._on_scene_change:
                cb = self._on_scene_change
                if asyncio.iscoroutinefunction(cb):
                    await cb(description)
                else:
                    cb(description)

        self._last_capture_time = time.time()
