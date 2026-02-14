"""SNS caption generation tool.

Generates social media captions for cat photos/videos
for platforms like Instagram, TikTok, Twitter, etc.
"""

import random
from typing import Any

from src.mcp.tool_registry import MCPTool
from src.utils.logger import get_logger

logger = get_logger("tool.sns")


# Hashtag presets
HASHTAGS = {
    "instagram_ja": [
        "#サイベリアン", "#シベリア猫", "#サイベリアンフォレストキャット",
        "#大阪猫舎", "#福楽キャッテリー", "#子猫", "#猫のいる暮らし",
        "#猫好きさんと繋がりたい", "#ねこ部", "#猫スタグラム",
        "#siberian", "#siberiancat", "#kitten",
    ],
    "instagram_zh": [
        "#西伯利亚猫", "#西伯利亚森林猫", "#猫咪", "#大阪",
        "#福楽猫舍", "#小猫咪", "#猫咪日常",
    ],
    "tiktok": [
        "#猫", "#子猫", "#サイベリアン", "#もふもふ",
        "#cat", "#kitten", "#siberian", "#cute",
    ],
}


class SNSCaptionTool(MCPTool):
    """Generate social media captions for cat content."""

    name = "generate_sns_caption"
    description = (
        "SNS投稿用のキャプション（文案）を生成します。"
        "「この写真のキャプション書いて」「インスタ用の文章を作って」等の時に使います。"
    )
    parameters = {
        "description": {
            "type": "string",
            "description": "写真/動画の内容の説明",
            "required": True,
        },
        "platform": {
            "type": "string",
            "enum": ["instagram", "tiktok", "twitter", "line"],
            "description": "投稿先プラットフォーム（デフォルト: instagram）",
        },
        "language": {
            "type": "string",
            "enum": ["japanese", "chinese", "bilingual"],
            "description": "キャプションの言語（デフォルト: japanese）",
        },
        "mood": {
            "type": "string",
            "enum": ["cute", "funny", "informative", "heartwarming"],
            "description": "投稿の雰囲気（デフォルト: cute）",
        },
    }

    async def execute(
        self,
        description: str,
        platform: str = "instagram",
        language: str = "japanese",
        mood: str = "cute",
        **kwargs,
    ) -> Any:
        """Generate an SNS caption.

        Args:
            description: What's in the photo/video.
            platform: Target platform.
            language: Caption language.
            mood: Desired mood/tone.

        Returns:
            Generated caption with hashtags.
        """
        logger.info(f"Generating {platform} caption ({language}, {mood})")

        caption = self._generate_caption(description, platform, language, mood)
        hashtags = self._get_hashtags(platform, language)

        return (
            f"📱 {platform.upper()} 投稿用キャプション:\n"
            f"---\n"
            f"{caption}\n\n"
            f"{hashtags}\n"
            f"---\n"
            f"※ 内容を確認・編集してからご使用ください"
        )

    def _generate_caption(
        self, description: str, platform: str, language: str, mood: str
    ) -> str:
        """Generate caption text based on parameters.

        Note: For production use, this would call the LLM for creative generation.
        These are template-based fallbacks.
        """
        if language == "japanese":
            return self._ja_caption(description, mood, platform)
        elif language == "chinese":
            return self._zh_caption(description, mood)
        else:  # bilingual
            ja = self._ja_caption(description, mood, platform)
            zh = self._zh_caption(description, mood)
            return f"{ja}\n\n{zh}"

    def _ja_caption(self, description: str, mood: str, platform: str) -> str:
        """Generate Japanese caption."""
        templates = {
            "cute": [
                f"🐱 {description}\nもふもふに癒されてください〜✨",
                f"今日のもふもふ〜🐾\n{description}",
                f"この可愛さ、伝わりますか...？🥺\n{description}",
            ],
            "funny": [
                f"😂 {description}\nうちの子たちは毎日こんな感じです笑",
                f"{description}\n...何してるの？笑🐱",
            ],
            "informative": [
                f"📝 サイベリアンのご紹介\n{description}\n\nサイベリアンは低アレルゲンの大型猫で、とても穏やかな性格です。",
            ],
            "heartwarming": [
                f"💕 {description}\n毎日こんな幸せな光景を見られて、ブリーダー冥利に尽きます",
                f"{description}\n家族として迎えていただける日を楽しみにしています🏠",
            ],
        }

        options = templates.get(mood, templates["cute"])
        caption = random.choice(options)

        if platform == "twitter":
            # Twitter has character limits
            if len(caption) > 200:
                caption = caption[:200] + "..."

        return caption

    def _zh_caption(self, description: str, mood: str) -> str:
        """Generate Chinese caption."""
        templates = {
            "cute": f"🐱 {description}\n西伯利亚猫的治愈时刻～✨",
            "funny": f"😂 {description}\n我家的毛孩子们每天都这样～",
            "informative": f"📝 {description}\n西伯利亚森林猫是低过敏原的大型猫咪，性格非常温顺。",
            "heartwarming": f"💕 {description}\n期待它们找到温暖的家～",
        }
        return templates.get(mood, templates["cute"])

    def _get_hashtags(self, platform: str, language: str) -> str:
        """Get relevant hashtags."""
        if platform == "instagram":
            if language == "chinese":
                tags = HASHTAGS.get("instagram_zh", [])
            else:
                tags = HASHTAGS.get("instagram_ja", [])
        elif platform == "tiktok":
            tags = HASHTAGS.get("tiktok", [])
        else:
            tags = HASHTAGS.get("instagram_ja", [])[:5]

        # Select a subset
        selected = random.sample(tags, min(len(tags), 10))
        return " ".join(selected)
