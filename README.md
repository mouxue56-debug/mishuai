# Fukuraku AI Secretary - ミケ

**福楽キャッテリー & 慈恵病院 AI秘書システム**

Live2Dキャラクター「ミケ」による音声対話AI秘書。マルチLLM対応、MCP toolsによる業務自動化、声紋認識による話者識別機能を搭載。

## Features

- **音声対話**: マイク入力 → ASR → LLM → TTS → 音声出力
- **Live2Dキャラクター**: ブラウザ上のキャラクターが表情・口パク連動
- **マルチLLM**: Claude / GPT-4o / Gemini / Kimi を用途別に使い分け
- **MCPツール**: Notion顧客検索、メモ、リマインダー、LINE下書き、ナレッジベース
- **声紋認識**: 3D-Speakerによる話者識別 → 自動口調切替
- **記憶システム**: 短期/中期/長期の3段階メモリ

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Test LLM connection
python -m src.main --test-llm

# 4. Run in text-only mode (no microphone needed)
python -m src.main --text-only

# 5. Run with full voice + Live2D
python -m src.main
```

## Architecture

```
Microphone → VAD → ASR → Speaker ID
                              ↓
                    LLM (Multi-model Router)
                    ├── MCP Tools (Notion, Memo, Schedule, ...)
                    ├── Emotion Analysis
                    └── Memory System (Short/Mid/Long-term)
                              ↓
                    TTS (VOICEVOX / Edge TTS)
                              ↓
                    WebSocket → Live2D Frontend (Browser)
```

## Project Structure

```
src/
├── main.py              # Entry point
├── core/                # Pipeline, LLM router, emotion, interrupt
├── audio/               # Microphone, ASR, TTS, speaker ID, wake word
├── memory/              # Short/mid/long-term memory (SQLite)
├── mcp/tools/           # MCP tool implementations
├── display/             # WebSocket server
└── utils/               # Config loader, logger

frontend/                # Live2D browser UI
config/                  # YAML configuration
knowledge/               # Cattery knowledge base (Markdown)
```

## Configuration

Edit `config/config.yaml` for:
- LLM model selection and modes (quality/balanced/budget)
- TTS engine settings (VOICEVOX/Edge TTS)
- Audio input parameters
- WebSocket server port

Edit `config/persona.yaml` for:
- Character personality
- Emotion mappings
- Speaker profiles

## Requirements

- Python 3.10+
- VOICEVOX (optional, for high-quality Japanese TTS)
- Live2D model (place in `frontend/models/`)
- API keys for LLM providers

## License

Private project for Fukuraku Cattery.
