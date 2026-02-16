"""Cattery knowledge base search tool.

Searches the local knowledge base files using BM25 ranking.
Covers: Siberian cats, Fukuraku Cattery FAQ, customer guides.
"""

import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.mcp.tool_registry import MCPTool
from src.utils.config_loader import KNOWLEDGE_DIR
from src.utils.logger import get_logger

logger = get_logger("tool.knowledge")


# Category → file mapping
CATEGORY_FILES = {
    "breed": ["breed_info.md"],
    "faq": ["cattery_faq.md"],
    "customer": ["customer_guide.md"],
    "pricing": ["pricing.md"],
    "care": ["care_guide.md"],
    "hospital": ["hospital_info.md"],
    "all": [
        "cattery_faq.md", "breed_info.md", "customer_guide.md",
        "pricing.md", "care_guide.md", "hospital_info.md",
    ],
}


def _tokenize_ja(text: str) -> list[str]:
    """Simple Japanese-aware tokenizer.

    Splits on whitespace, punctuation, and particles. Also generates
    character bigrams for kanji/katakana to handle compound words
    that exact split would miss (e.g. "食欲不振" matches "食欲").
    """
    # Normalize
    text = text.lower().strip()
    # Split on whitespace and common punctuation
    tokens = re.split(r'[\s、。！？・（）「」\[\]【】\n\r,.:;!?()]+', text)
    tokens = [t for t in tokens if t]

    # Add character bigrams for CJK characters (helps with compound matching)
    cjk_chars = re.findall(r'[\u3000-\u9fff\uff00-\uffef]+', text)
    for word in cjk_chars:
        if len(word) >= 2:
            for i in range(len(word) - 1):
                tokens.append(word[i:i+2])

    return tokens


class _KnowledgeIndex:
    """BM25 index over markdown knowledge base sections."""

    def __init__(self):
        self.sections: list[dict] = []  # {"file": str, "header": str, "body": str}
        self.bm25: BM25Okapi | None = None

    def build(self, files: list[str]):
        """Parse markdown files into sections and build BM25 index."""
        self.sections = []
        for filename in files:
            filepath = KNOWLEDGE_DIR / filename
            if not filepath.exists():
                logger.warning(f"Knowledge file not found: {filepath}")
                continue
            content = filepath.read_text(encoding="utf-8")
            self.sections.extend(self._split_sections(content, filename))

        if not self.sections:
            logger.warning("No knowledge sections found")
            return

        # Build BM25 index
        corpus = [_tokenize_ja(s["header"] + " " + s["body"]) for s in self.sections]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built: {len(self.sections)} sections from {len(files)} files")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index. Returns top-k sections with scores."""
        if not self.bm25 or not self.sections:
            return []

        tokens = _tokenize_ja(query)
        scores = self.bm25.get_scores(tokens)

        # Pair with sections, sort by score descending
        ranked = sorted(
            zip(scores, self.sections),
            key=lambda x: x[0],
            reverse=True,
        )

        # Filter out zero-score results
        results = []
        for score, section in ranked[:top_k]:
            if score > 0:
                results.append({**section, "score": round(float(score), 3)})

        return results

    @staticmethod
    def _split_sections(content: str, filename: str) -> list[dict]:
        """Split markdown into sections by headers."""
        sections = []
        current_header = filename
        current_lines: list[str] = []

        for line in content.split("\n"):
            if line.startswith("#"):
                # Save previous section
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append({
                            "file": filename,
                            "header": current_header,
                            "body": body,
                        })
                current_header = line.lstrip("#").strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Last section
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append({
                    "file": filename,
                    "header": current_header,
                    "body": body,
                })

        return sections


class CatteryKnowledgeTool(MCPTool):
    """Search the cattery knowledge base using BM25 ranking."""

    name = "cattery_knowledge"
    description = (
        "猫舎のナレッジベースを検索します。"
        "サイベリアンの品種情報、FAQ、顧客対応ガイドを検索できます。"
        "お客様からの質問に答える時に使います。"
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "検索キーワード（例：アレルギー、価格、食欲不振、ワクチン）",
            "required": True,
        },
        "category": {
            "type": "string",
            "enum": ["all", "breed", "faq", "customer", "pricing", "care", "hospital"],
            "description": "検索カテゴリ（デフォルト: all）",
        },
    }

    def __init__(self):
        self._indexes: dict[str, _KnowledgeIndex] = {}

    def invalidate_indexes(self):
        """Clear all cached BM25 indexes so they rebuild on next query."""
        self._indexes.clear()
        logger.info("Knowledge indexes invalidated — will rebuild on next search")

    def _get_index(self, category: str) -> _KnowledgeIndex:
        """Get or build the BM25 index for a category."""
        if category not in self._indexes:
            idx = _KnowledgeIndex()
            files = CATEGORY_FILES.get(category, CATEGORY_FILES["all"])
            idx.build(files)
            self._indexes[category] = idx
        return self._indexes[category]

    async def execute(self, query: str, category: str = "all", **kwargs) -> Any:
        """Search the knowledge base using BM25.

        Args:
            query: Search query (natural language or keywords).
            category: Category to search in.

        Returns:
            Relevant knowledge base content, ranked by relevance.
        """
        logger.info(f"Knowledge search: query='{query}', category={category}")

        index = self._get_index(category)
        results = index.search(query, top_k=5)

        if not results:
            # Fallback: search all categories if specific category found nothing
            if category != "all":
                index = self._get_index("all")
                results = index.search(query, top_k=5)

        if not results:
            return f"「{query}」に関する情報は見つかりませんでした。"

        # Format results
        parts = []
        for r in results:
            parts.append(f"## {r['header']}\n{r['body']}")

        combined = "\n\n".join(parts)
        if len(combined) > 2000:
            combined = combined[:2000] + "...(以下省略)"

        logger.info(f"Knowledge search returned {len(results)} sections (top score: {results[0]['score']})")
        return combined
