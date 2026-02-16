"""Speaker management API.

Provides registration, CRUD, access control queries, and audit
for the voiceprint-based speaker system.
"""

import base64
import json
from pathlib import Path
from typing import Any, Optional

from src.utils.config_loader import BASE_DIR
from src.utils.logger import get_logger

logger = get_logger("api.speaker")


class SpeakerAPI:
    """API for speaker registration, access control, and audit."""

    def __init__(self, long_term_memory, speaker_identifier=None):
        """Initialize with a LongTermMemory instance and optional SpeakerIdentifier.

        Args:
            long_term_memory: LongTermMemory instance for DB operations.
            speaker_identifier: SpeakerIdentifier instance for voiceprint
                                operations (None in text-only mode).
        """
        self._ltm = long_term_memory
        self._sid = speaker_identifier

    # ----------------------------------------------------------------
    # Registration
    # ----------------------------------------------------------------

    async def handle_register_speaker(
        self,
        speaker_id: str,
        display_name: str,
        role: str = "visitor",
        language: str = "japanese",
        permissions: list[str] = None,
        style: str = "formal",
        notes: str = "",
        invited_by: str = None,
        relationship: str = "",
        audio_samples_b64: list[str] = None,
    ) -> dict[str, Any]:
        """Register a new speaker with optional voice samples.

        Args:
            speaker_id: Unique identifier.
            display_name: Display name.
            role: 'boss', 'staff', or 'visitor'.
            language: Primary language.
            permissions: List of allowed tool names.
            style: Speech style.
            notes: Freeform notes.
            invited_by: Who invited this speaker.
            relationship: Relationship to inviter.
            audio_samples_b64: Base64 encoded webm audio samples (optional).

        Returns:
            Result dict with success status.
        """
        if not speaker_id or not display_name:
            return {"success": False, "error": "speaker_id and display_name are required"}

        # Validate role
        if role not in ("boss", "staff", "visitor"):
            role = "visitor"

        # Default permissions by role
        if permissions is None:
            if role == "boss":
                permissions = ["all"]
            elif role == "staff":
                permissions = ["cattery_knowledge", "check_schedule", "add_memo"]
            else:
                permissions = ["cattery_knowledge"]

        permissions_json = json.dumps(permissions, ensure_ascii=False)

        # Check if inviter exists (required for non-boss registration)
        if invited_by:
            inviter = await self._ltm.get_speaker(invited_by)
            if not inviter:
                return {"success": False, "error": f"Inviter '{invited_by}' not found"}
        elif role != "boss":
            # For non-boss speakers, check if there's at least one boss
            speakers = await self._ltm.get_all_speakers()
            has_boss = any(s["role"] == "boss" for s in speakers)
            if has_boss:
                return {
                    "success": False,
                    "error": "Non-boss speakers must be invited by an existing user",
                }

        # Process voice samples if provided
        voice_registered = False
        if audio_samples_b64 and self._sid and self._sid.enabled:
            try:
                audio_samples = [base64.b64decode(s) for s in audio_samples_b64]
                voice_registered = await self._sid.register_speaker_from_webm(
                    speaker_id, audio_samples
                )
                if not voice_registered:
                    logger.warning(
                        f"Voice registration failed for {speaker_id}, "
                        "proceeding without voiceprint"
                    )
            except Exception as e:
                logger.error(f"Voice sample processing error: {e}")

        # Save to database
        try:
            await self._ltm.register_speaker(
                speaker_id=speaker_id,
                display_name=display_name,
                role=role,
                language=language,
                permissions=permissions_json,
                style=style,
                notes=notes,
                invited_by=invited_by,
                relationship=relationship,
            )

            speaker = await self._ltm.get_speaker(speaker_id)
            return {
                "success": True,
                "speaker": speaker,
                "voice_registered": voice_registered,
            }
        except Exception as e:
            logger.error(f"Speaker registration failed: {e}")
            return {"success": False, "error": str(e)}

    # ----------------------------------------------------------------
    # CRUD
    # ----------------------------------------------------------------

    async def handle_get_speakers(self) -> dict[str, Any]:
        """List all active registered speakers."""
        try:
            speakers = await self._ltm.get_all_speakers()
            return {"success": True, "speakers": speakers}
        except Exception as e:
            logger.error(f"Failed to get speakers: {e}")
            return {"success": False, "error": str(e)}

    async def handle_update_speaker(
        self, speaker_id: str, updates: dict
    ) -> dict[str, Any]:
        """Update speaker fields.

        Args:
            speaker_id: Speaker to update.
            updates: Dict of fields to update.
        """
        if not speaker_id:
            return {"success": False, "error": "speaker_id is required"}

        # Convert permissions list to JSON if provided
        if "permissions" in updates and isinstance(updates["permissions"], list):
            updates["permissions"] = json.dumps(
                updates["permissions"], ensure_ascii=False
            )

        try:
            await self._ltm.update_speaker(speaker_id, **updates)
            speaker = await self._ltm.get_speaker(speaker_id)
            return {"success": True, "speaker": speaker}
        except Exception as e:
            logger.error(f"Speaker update failed: {e}")
            return {"success": False, "error": str(e)}

    async def handle_delete_speaker(self, speaker_id: str) -> dict[str, Any]:
        """Deactivate a speaker and remove voiceprint.

        Args:
            speaker_id: Speaker to remove.
        """
        if not speaker_id:
            return {"success": False, "error": "speaker_id is required"}

        try:
            # Deactivate in DB
            await self._ltm.deactivate_speaker(speaker_id)

            # Remove voiceprint file
            npy_path = BASE_DIR / "data" / "speaker_embeddings" / f"{speaker_id}.npy"
            if npy_path.exists():
                npy_path.unlink()
                logger.info(f"Removed voiceprint: {npy_path}")

            # Remove from in-memory cache if speaker_id exists
            if self._sid and speaker_id in self._sid._embeddings:
                del self._sid._embeddings[speaker_id]

            return {"success": True, "speaker_id": speaker_id}
        except Exception as e:
            logger.error(f"Speaker deletion failed: {e}")
            return {"success": False, "error": str(e)}

    # ----------------------------------------------------------------
    # Access Control
    # ----------------------------------------------------------------

    async def handle_check_access(self, speaker_id: str) -> Optional[dict]:
        """Check if a speaker is registered and active.

        Args:
            speaker_id: Speaker to check.

        Returns:
            Speaker dict if authorized, None if not.
        """
        if not speaker_id:
            return None

        speaker = await self._ltm.get_speaker(speaker_id)
        if speaker and speaker.get("is_active"):
            return speaker
        return None

    # ----------------------------------------------------------------
    # Audit
    # ----------------------------------------------------------------

    async def handle_get_audit_sessions(
        self, speaker_id: str = None, limit: int = 20
    ) -> dict[str, Any]:
        """Get conversation sessions for audit review.

        Args:
            speaker_id: Filter by speaker (None = all).
            limit: Max sessions.
        """
        try:
            sessions = await self._ltm.get_sessions(
                speaker_id=speaker_id, limit=limit
            )
            return {"success": True, "sessions": sessions}
        except Exception as e:
            logger.error(f"Audit sessions query failed: {e}")
            return {"success": False, "error": str(e)}

    async def handle_get_session_detail(
        self, session_id: str
    ) -> dict[str, Any]:
        """Get all turns in a session for replay.

        Args:
            session_id: Session to query.
        """
        if not session_id:
            return {"success": False, "error": "session_id is required"}

        try:
            turns = await self._ltm.get_session_turns(session_id)
            return {"success": True, "session_id": session_id, "turns": turns}
        except Exception as e:
            logger.error(f"Session detail query failed: {e}")
            return {"success": False, "error": str(e)}
