# 福楽キャッテリー AI秘書 技術仕様書

> プロジェクト名: Fukuraku AI Secretary (Phase 1) → AI Livestream Host (Phase 2)
> 作成日: 2025-02-15
> 対象: Claude Code による開発支援用

この仕様書の完全版は、プロジェクトのIssue/PRに記載されています。
コードの実装は `src/` ディレクトリを参照してください。

## クイックスタート

```bash
# 1. 環境構築
pip install -r requirements.txt

# 2. API Key設定
cp .env.example .env
# .env を編集してAPI Keyを設定

# 3. テスト実行（LLM接続確認）
python -m src.main --test-llm

# 4. テキストモードで起動（マイク不要）
python -m src.main --text-only

# 5. フルモードで起動（マイク + Live2D）
python -m src.main
```

## アーキテクチャ

```
Audio Input → VAD → ASR → Speaker ID → LLM (with MCP tools) → Emotion → TTS → Audio Output
                                                                    ↓
                                                            WebSocket → Live2D Frontend
```

## ディレクトリ構成

```
├── src/
│   ├── main.py              # エントリーポイント
│   ├── core/
│   │   ├── pipeline.py       # 対話パイプライン
│   │   ├── llm_router.py     # マルチLLMルーター
│   │   ├── emotion_analyzer.py  # 感情分析
│   │   └── interrupt_handler.py # 割り込み制御
│   ├── audio/
│   │   ├── microphone.py     # マイク + VAD
│   │   ├── asr.py            # 音声認識
│   │   ├── tts.py            # 音声合成
│   │   ├── speaker_id.py     # 声紋認識
│   │   └── wake_word.py      # ウェイクワード
│   ├── memory/
│   │   ├── memory_manager.py # 記憶管理
│   │   ├── short_term.py     # 短期記憶
│   │   ├── mid_term.py       # 中期記憶
│   │   └── long_term.py      # 長期記憶
│   ├── mcp/
│   │   ├── tool_registry.py  # ツール管理
│   │   └── tools/            # MCPツール群
│   └── display/
│       └── websocket_server.py  # WebSocket
├── frontend/                  # Live2D UI
├── config/                    # 設定ファイル
├── knowledge/                 # ナレッジベース
└── data/                      # SQLite DB
```
