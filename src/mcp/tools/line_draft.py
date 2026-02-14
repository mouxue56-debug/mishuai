"""LINE message draft tool.

Generates draft replies for LINE messages based on context.
Can also format and prepare messages for sending via LINE API.
"""

from typing import Any

from src.mcp.tool_registry import MCPTool
from src.utils.logger import get_logger

logger = get_logger("tool.line")


class LineDraftTool(MCPTool):
    """Generate LINE message drafts."""

    name = "draft_line_message"
    description = (
        "LINE返信の下書きを生成します。"
        "「〇〇さんにLINEの返事を書いて」「この問い合わせに返信して」等の時に使います。"
        "下書きを生成するだけで、自動送信はしません。"
    )
    parameters = {
        "recipient": {
            "type": "string",
            "description": "送信先の名前",
            "required": True,
        },
        "context": {
            "type": "string",
            "description": "返信の文脈（相手のメッセージ内容や用件）",
            "required": True,
        },
        "tone": {
            "type": "string",
            "enum": ["formal", "polite", "casual"],
            "description": "メッセージのトーン（デフォルト: polite）",
        },
        "language": {
            "type": "string",
            "enum": ["japanese", "chinese"],
            "description": "メッセージの言語（デフォルト: japanese）",
        },
    }

    async def execute(
        self,
        recipient: str,
        context: str,
        tone: str = "polite",
        language: str = "japanese",
        **kwargs,
    ) -> Any:
        """Generate a LINE message draft.

        This tool generates the draft text. The actual sending
        should be confirmed by the user before proceeding.

        Args:
            recipient: Name of the recipient.
            context: Context for the reply.
            tone: Message tone.
            language: Message language.

        Returns:
            Generated draft message.
        """
        logger.info(f"Generating LINE draft for {recipient}")

        # Generate appropriate templates based on context
        draft = self._generate_draft(recipient, context, tone, language)

        return (
            f"📝 LINE下書き（{recipient}さん宛）\n"
            f"---\n"
            f"{draft}\n"
            f"---\n"
            f"※ この下書きを確認して、OKであればLINEアプリから送信してください。"
        )

    def _generate_draft(
        self, recipient: str, context: str, tone: str, language: str
    ) -> str:
        """Generate a draft message based on common patterns.

        For complex drafts, the LLM itself will handle the generation.
        This provides a structured template.
        """
        if language == "chinese":
            return self._generate_chinese_draft(recipient, context, tone)

        # Japanese templates
        if tone == "formal":
            greeting = f"{recipient}様\n\nいつもお世話になっております。\nサイベリアン｜大阪・福楽キャッテリーでございます。\n\n"
            closing = "\n\nご不明な点がございましたら、お気軽にお問い合わせください。\nよろしくお願いいたします。\n\n福楽キャッテリー"
        elif tone == "casual":
            greeting = f"{recipient}さん、こんにちは！\n\n"
            closing = "\n\n何かあったらいつでも聞いてくださいね〜！"
        else:  # polite
            greeting = f"{recipient}様\n\nお問い合わせありがとうございます。\n福楽キャッテリーです。\n\n"
            closing = "\n\nご質問がありましたら、お気軽にどうぞ。\nよろしくお願いいたします。"

        # The actual body will be contextual
        body = f"【用件: {context}】\n（ここにAIが文脈に応じた返信を生成します）"

        return greeting + body + closing

    def _generate_chinese_draft(self, recipient: str, context: str, tone: str) -> str:
        """Generate a Chinese draft message."""
        if tone == "formal":
            greeting = f"{recipient}您好，\n\n感谢您的咨询。\n\n"
            closing = "\n\n如有任何疑问，请随时联系我们。\n祝好！\n\n福楽猫舍"
        else:
            greeting = f"{recipient}你好！\n\n"
            closing = "\n\n有什么问题随时联系哦～"

        body = f"【事项: {context}】\n（AI将根据上下文生成回复）"

        return greeting + body + closing
