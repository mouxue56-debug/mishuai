"""Knowledge base management API.

Provides CRUD operations for the knowledge base markdown files
and BM25 index invalidation when content changes.
"""

from pathlib import Path
from typing import Any

from src.utils.config_loader import KNOWLEDGE_DIR
from src.utils.logger import get_logger

logger = get_logger("api.knowledge")

# Category metadata for the frontend
CATEGORIES = {
    "faq": {
        "file": "cattery_faq.md",
        "label": "FAQ",
        "description": "よくある質問と回答",
    },
    "breed": {
        "file": "breed_info.md",
        "label": "品種情報",
        "description": "サイベリアンの品種特性・飼育ガイド",
    },
    "customer": {
        "file": "customer_guide.md",
        "label": "顧客対応",
        "description": "対応フロー・料金・準備ガイド・応答ルール",
    },
}


class KnowledgeAPI:
    """API for knowledge base file management."""

    def __init__(self):
        self._knowledge_tool = None  # Lazily bound to CatteryKnowledgeTool

    def set_knowledge_tool(self, tool):
        """Bind the CatteryKnowledgeTool instance for index invalidation."""
        self._knowledge_tool = tool

    async def handle_get_knowledge_files(self) -> dict[str, Any]:
        """List all knowledge files with metadata."""
        files = []
        for cat_id, meta in CATEGORIES.items():
            filepath = KNOWLEDGE_DIR / meta["file"]
            exists = filepath.exists()
            size = filepath.stat().st_size if exists else 0
            files.append({
                "category": cat_id,
                "filename": meta["file"],
                "label": meta["label"],
                "description": meta["description"],
                "exists": exists,
                "size": size,
            })

        return {"success": True, "files": files}

    async def handle_get_knowledge_content(
        self, category: str
    ) -> dict[str, Any]:
        """Read the content of a knowledge file."""
        meta = CATEGORIES.get(category)
        if not meta:
            return {"success": False, "error": f"Unknown category: {category}"}

        filepath = KNOWLEDGE_DIR / meta["file"]
        if not filepath.exists():
            return {
                "success": True,
                "category": category,
                "filename": meta["file"],
                "content": "",
            }

        try:
            content = filepath.read_text(encoding="utf-8")
            return {
                "success": True,
                "category": category,
                "filename": meta["file"],
                "content": content,
            }
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return {"success": False, "error": str(e)}

    async def handle_save_knowledge_content(
        self, category: str, content: str
    ) -> dict[str, Any]:
        """Save content to a knowledge file and invalidate BM25 index."""
        meta = CATEGORIES.get(category)
        if not meta:
            return {"success": False, "error": f"Unknown category: {category}"}

        filepath = KNOWLEDGE_DIR / meta["file"]

        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            filepath.write_text(content, encoding="utf-8")
            logger.info(
                f"Knowledge file saved: {meta['file']} ({len(content)} bytes)"
            )

            # Invalidate BM25 indexes so they rebuild with new content
            if self._knowledge_tool:
                self._knowledge_tool.invalidate_indexes()

            return {
                "success": True,
                "category": category,
                "filename": meta["file"],
                "size": len(content.encode("utf-8")),
            }
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
            return {"success": False, "error": str(e)}

    async def handle_rebuild_index(self) -> dict[str, Any]:
        """Force rebuild all BM25 indexes."""
        if self._knowledge_tool:
            self._knowledge_tool.invalidate_indexes()
            return {"success": True, "message": "Index invalidated, will rebuild on next search"}
        return {"success": False, "error": "Knowledge tool not available"}
