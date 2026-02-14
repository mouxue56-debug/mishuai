"""Cattery knowledge base search tool.

Searches the local knowledge base files for information about:
- Siberian cats (breed info, care, health)
- Fukuraku Cattery FAQ
- Customer handling guidelines
- Hospital information
"""

from pathlib import Path
from typing import Any

from src.mcp.tool_registry import MCPTool
from src.utils.config_loader import KNOWLEDGE_DIR
from src.utils.logger import get_logger

logger = get_logger("tool.knowledge")


# Knowledge base index: maps keywords to relevant files and sections
KNOWLEDGE_INDEX = {
    # Breed info keywords
    "サイベリアン": ["breed_info.md", "cattery_faq.md"],
    "品種": ["breed_info.md"],
    "毛色": ["breed_info.md"],
    "体重": ["breed_info.md"],
    "寿命": ["breed_info.md", "cattery_faq.md"],
    "性格": ["breed_info.md", "cattery_faq.md"],
    "アレルギー": ["breed_info.md", "cattery_faq.md"],
    "被毛": ["breed_info.md", "cattery_faq.md"],
    "ブラッシング": ["breed_info.md"],
    "タビー": ["breed_info.md"],
    "ポイント": ["breed_info.md"],
    "ネヴァマスカレード": ["breed_info.md"],

    # FAQ keywords
    "見学": ["cattery_faq.md"],
    "予約": ["cattery_faq.md", "customer_guide.md"],
    "価格": ["cattery_faq.md", "customer_guide.md"],
    "値段": ["cattery_faq.md", "customer_guide.md"],
    "お迎え": ["cattery_faq.md", "customer_guide.md"],
    "ワクチン": ["cattery_faq.md", "breed_info.md"],
    "遺伝子": ["cattery_faq.md"],
    "アフター": ["cattery_faq.md", "customer_guide.md"],
    "営業時間": ["cattery_faq.md"],
    "住所": ["cattery_faq.md"],
    "空輸": ["cattery_faq.md"],

    # Customer guide keywords
    "顧客": ["customer_guide.md"],
    "対応": ["customer_guide.md"],
    "契約": ["customer_guide.md", "cattery_faq.md"],
    "フォロー": ["customer_guide.md"],

    # Hospital keywords
    "慈恵": ["hospital_info.md"],
    "病院": ["hospital_info.md"],
    "再生医療": ["hospital_info.md"],
    "中国語": ["hospital_info.md"],
}


class CatteryKnowledgeTool(MCPTool):
    """Search the cattery knowledge base."""

    name = "cattery_knowledge"
    description = (
        "猫舎のナレッジベースを検索します。"
        "サイベリアンの品種情報、FAQ、顧客対応ガイド、慈恵病院の情報を検索できます。"
        "お客様からの質問に答える時に使います。"
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "検索キーワード（例：アレルギー、価格、見学、ワクチン）",
            "required": True,
        },
        "category": {
            "type": "string",
            "enum": ["all", "breed", "faq", "customer", "hospital"],
            "description": "検索カテゴリ（デフォルト: all）",
        },
    }

    def __init__(self):
        self._knowledge_cache: dict[str, str] = {}

    def _load_file(self, filename: str) -> str:
        """Load a knowledge base file, with caching."""
        if filename in self._knowledge_cache:
            return self._knowledge_cache[filename]

        filepath = KNOWLEDGE_DIR / filename
        if not filepath.exists():
            logger.warning(f"Knowledge file not found: {filepath}")
            return ""

        content = filepath.read_text(encoding="utf-8")
        self._knowledge_cache[filename] = content
        return content

    async def execute(self, query: str, category: str = "all", **kwargs) -> Any:
        """Search the knowledge base.

        Args:
            query: Search keyword.
            category: Category to search in.

        Returns:
            Relevant knowledge base content.
        """
        logger.info(f"Knowledge search: query='{query}', category={category}")

        # Determine which files to search
        files_to_search = self._get_files_for_category(category)

        # Also check keyword index
        for keyword, files in KNOWLEDGE_INDEX.items():
            if keyword in query:
                for f in files:
                    if f not in files_to_search:
                        files_to_search.append(f)

        # Search each file
        results = []
        for filename in files_to_search:
            content = self._load_file(filename)
            if not content:
                continue

            # Find relevant sections
            sections = self._find_relevant_sections(content, query)
            if sections:
                results.extend(sections)

        if not results:
            # Fallback: return full FAQ
            faq_content = self._load_file("cattery_faq.md")
            if faq_content and query:
                sections = self._find_relevant_sections(faq_content, query)
                if sections:
                    results.extend(sections)

        if not results:
            return f"「{query}」に関する情報は見つかりませんでした。"

        # Limit result length
        combined = "\n\n".join(results[:5])
        if len(combined) > 2000:
            combined = combined[:2000] + "...(以下省略)"

        return combined

    def _get_files_for_category(self, category: str) -> list[str]:
        """Map category to knowledge base files."""
        category_map = {
            "breed": ["breed_info.md"],
            "faq": ["cattery_faq.md"],
            "customer": ["customer_guide.md"],
            "hospital": ["hospital_info.md"],
            "all": ["cattery_faq.md", "breed_info.md", "customer_guide.md", "hospital_info.md"],
        }
        return list(category_map.get(category, category_map["all"]))

    def _find_relevant_sections(self, content: str, query: str) -> list[str]:
        """Find sections in markdown content that match the query.

        Splits by headers and returns sections containing the query.
        """
        sections = []
        current_section = []
        current_header = ""

        for line in content.split("\n"):
            if line.startswith("#"):
                # Save previous section if it matches
                if current_section:
                    section_text = "\n".join(current_section)
                    if query.lower() in section_text.lower() or query in current_header:
                        sections.append(section_text)
                current_section = [line]
                current_header = line
            else:
                current_section.append(line)

        # Check last section
        if current_section:
            section_text = "\n".join(current_section)
            if query.lower() in section_text.lower() or query in current_header:
                sections.append(section_text)

        return sections
