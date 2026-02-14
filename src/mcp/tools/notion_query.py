"""Notion customer database search tool.

Searches the Fukuraku Cattery customer database in Notion
for customer records by name, breed preference, status, etc.
"""

import os
from typing import Any

from src.mcp.tool_registry import MCPTool
from src.utils.config_loader import get_api_key, get_main_config
from src.utils.logger import get_logger

logger = get_logger("tool.notion")


class NotionQueryTool(MCPTool):
    """Search customers in the Notion database."""

    name = "search_customers"
    description = (
        "Notionの顧客データベースを検索します。"
        "顧客名、猫種の希望、日付、ステータスなどで絞り込みできます。"
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "検索キーワード（顧客名、猫種名など）",
        },
        "filter_type": {
            "type": "string",
            "enum": ["name", "breed", "date", "status", "all"],
            "description": "フィルター種別。'all'で全文検索",
        },
    }

    def __init__(self):
        self._client = None
        self._database_id = None

    async def _ensure_client(self):
        """Initialize Notion client if not yet initialized."""
        if self._client is not None:
            return

        try:
            from notion_client import AsyncClient

            api_key = get_api_key("notion")
            self._client = AsyncClient(auth=api_key)

            config = get_main_config()
            self._database_id = config.get("notion", {}).get("customer_database_id", "")

            if not self._database_id:
                self._database_id = os.getenv("NOTION_CUSTOMER_DB_ID", "")

            logger.info("Notion client initialized")
        except ImportError:
            logger.error("notion-client not installed. Run: pip install notion-client")
            raise
        except ValueError as e:
            logger.warning(f"Notion API key not configured: {e}")
            raise

    async def execute(self, query: str = "", filter_type: str = "all", **kwargs) -> Any:
        """Search the Notion customer database.

        Args:
            query: Search keyword.
            filter_type: Type of filter to apply.

        Returns:
            Search results as formatted string.
        """
        try:
            await self._ensure_client()
        except Exception as e:
            return f"Notion接続エラー: {e}。Notion APIキーとデータベースIDを設定してください。"

        if not self._database_id:
            return "NotionデータベースIDが設定されていません。config.yamlまたは.envで設定してください。"

        try:
            # Build filter based on filter_type
            notion_filter = self._build_filter(query, filter_type)

            results = await self._client.databases.query(
                database_id=self._database_id,
                filter=notion_filter if notion_filter else undefined,
            )

            # Format results
            if not results.get("results"):
                return f"「{query}」に一致する顧客は見つかりませんでした。"

            formatted = []
            for page in results["results"][:10]:  # Limit to 10 results
                props = page.get("properties", {})
                entry = self._format_page(props)
                formatted.append(entry)

            return f"検索結果 ({len(formatted)}件):\n" + "\n---\n".join(formatted)

        except Exception as e:
            logger.error(f"Notion search error: {e}")
            return f"検索中にエラーが発生しました: {e}"

    def _build_filter(self, query: str, filter_type: str) -> dict | None:
        """Build Notion API filter from search parameters."""
        if not query:
            return None

        if filter_type == "name":
            return {
                "property": "名前",
                "rich_text": {"contains": query},
            }
        elif filter_type == "breed":
            return {
                "property": "希望猫種",
                "rich_text": {"contains": query},
            }
        elif filter_type == "status":
            return {
                "property": "ステータス",
                "select": {"equals": query},
            }
        else:
            # Full-text search: search in name field
            return {
                "property": "名前",
                "rich_text": {"contains": query},
            }

    def _format_page(self, properties: dict) -> str:
        """Format a Notion page's properties into a readable string."""
        parts = []
        for prop_name, prop_value in properties.items():
            value = self._extract_property_value(prop_value)
            if value:
                parts.append(f"{prop_name}: {value}")
        return "\n".join(parts) if parts else "(データなし)"

    def _extract_property_value(self, prop: dict) -> str:
        """Extract the display value from a Notion property."""
        prop_type = prop.get("type", "")

        if prop_type == "title":
            texts = prop.get("title", [])
            return "".join(t.get("plain_text", "") for t in texts)
        elif prop_type == "rich_text":
            texts = prop.get("rich_text", [])
            return "".join(t.get("plain_text", "") for t in texts)
        elif prop_type == "select":
            sel = prop.get("select")
            return sel.get("name", "") if sel else ""
        elif prop_type == "multi_select":
            return ", ".join(s.get("name", "") for s in prop.get("multi_select", []))
        elif prop_type == "date":
            date = prop.get("date")
            return date.get("start", "") if date else ""
        elif prop_type == "phone_number":
            return prop.get("phone_number", "")
        elif prop_type == "email":
            return prop.get("email", "")
        elif prop_type == "number":
            return str(prop.get("number", ""))
        elif prop_type == "checkbox":
            return "はい" if prop.get("checkbox") else "いいえ"
        else:
            return ""
