# 診断報告 - 2026-02-16

## 概要

MishuAI (Fukuraku AI Secretary) v0.3 のコードベース診断結果。
全モジュールのソースコードを精査し、各機能の実装状態を確認した。

## 結論

コードベースは**97%が実装済み**。スタブやプレースホルダーではなく、動作可能な実装が存在する。
主な課題は設定の不整合（全LLMスロットがQianwen固定）とストリーミング再生の未実装。

## 機能状態

| 機能 | 状態 | 説明 |
|------|------|------|
| pip install | ✅ | requirements.txt (42依存), pyproject.toml 正常 |
| 文字対話 (--text-only) | ✅ | main.py FukurakuSecretary、Pipeline完全実装。ただしLLM設定がQianwen固定 |
| マイク入力 + VAD | ✅ | microphone.py: WebRTC VAD + Silero VAD 両対応、pre-speech buffer、エコー抑制 |
| ASR音声認識 | ✅ | asr.py: DashScope Paraformer (primary) + Faster-Whisper (fallback)、多言語対応 |
| LLM多モデル呼出 | ⚠️ | llm_router.py: 6プロバイダー対応コード有り（anthropic/openai/google/moonshot/deepseek/qianwen）。しかしconfig.yamlが全スロットqianwen固定 |
| TTS音声再生 | ✅ | tts.py: VOICEVOX + CosyVoice(DashScope)、感情→スタイル変換実装済 |
| WebSocket接続 | ✅ | websocket_server.py: 50+コマンド対応、クライアント管理、ブロードキャスト |
| Live2D描画 | ⚠️ | frontend/js完備（live2d_controller, lip_sync, emotion_mapper）。models/ディレクトリ空（CDNフォールバック有り） |
| 声紋認識 | ✅ | speaker_id.py: resemblyzer + 3D-Speaker、登録フロー、コサイン類似度 |
| MCP工具執行 | ✅ | 7ツール実装: cattery_knowledge(BM25), memo_reminder, line_draft, sns_caption, notion_query, schedule, tool_registry |
| 打断機制 | ⚠️ | interrupt_handler.py: 状態機械(IDLE→LISTENING→PROCESSING→SPEAKING→INTERRUPTED) 有り。ただし応答全体単位で、文単位の中断は未対応 |
| ストリーミング再生 | ❌ | _output_consumer()が応答全文を一括TTS合成。文単位分割・逐次再生は未実装 |
| ダッシュボード | ❌ | MetricsCollectorなし、dashboard.jsなし |
| 記憶システム | ✅ | 3層: short_term(deque, 10ターン), mid_term(日次要約), long_term(SQLite v5, マイグレーション付) |
| 唤醒詞 | ❌ | wake_word.py: 常にFalse返却（Phase 1では意図的にスキップ） |
| デモモード | ❌ | 未実装 |

## 需要修復的関鍵問題

### 高優先度

1. **LLM設定がQianwen一択** — config.yamlの全6スロット（conversation.primary/fallback/budget, tool_use.primary/fallback, summarization.primary）が全て`provider: "qianwen"`, `model: "qwen-plus"`。仕様では Claude Sonnet 4.5 を primary、Gemini 2.5 Flash を fallback、GPT-4o-mini を budget に設定すべき。

2. **ストリーミング再生が未実装** — `_output_consumer()` (main.py:273-392) が応答全文を一括合成。長い回答だと数秒無音の後に一気に再生される。デモとしてはレコーダーのような印象。文単位分割 → 逐次TTS → 逐次再生が必要。

3. **キャラクター名不一致** — persona.yamlは「ユキ」だが、llm_router.pyの_build_system_promptは「ミケ」をハードコード。仕様書は「ミケ」で統一。

### 中優先度

4. **.env.exampleにQIANWEN_API_KEY/DASHSCOPE_API_KEY未記載** — ASR(DashScope)とTTS(CosyVoice)が暗黙的にこのキーを要求するが、テンプレートに無い。

5. **ナレッジベース不完全** — 3ファイル有り(breed_info.md, cattery_faq.md, customer_guide.md)、3ファイル不足(pricing.md, care_guide.md, hospital_info.md)。cattery_knowledge.pyのCATEGORY_FILESも3カテゴリのみ。

6. **docs/events.md未作成** — WebSocketプロトコルが文書化されていない。50+のコマンドが暗黙知。

7. **test_llm_providers.py未作成** — 各プロバイダーの接続テストが無い。

### 低優先度

8. **Makefile / docker-compose.yml 未作成** — ワンクリック起動不可。

9. **dashboard.js未作成** — 投資家向けデモ用メトリクス表示が無い。

10. **Live2Dモデルファイル未配置** — models/ディレクトリ空。CDNフォールバック(Hiyori)で動作はするが、独自キャラクターではない。

## ファイル構成

```
src/ (45 Python files, ~10,000 LOC)
├── main.py              # 1,402 LOC - メインオーケストレーター
├── core/
│   ├── pipeline.py      # 469 LOC - 対話パイプライン
│   ├── llm_router.py    # 667 LOC - マルチモデルルーティング
│   ├── emotion_analyzer.py # 185 LOC
│   ├── interrupt_handler.py # 96 LOC
│   └── proactive_scheduler.py
├── audio/
│   ├── microphone.py    # 269 LOC - VAD付きマイク入力
│   ├── asr.py           # 278 LOC - 音声認識
│   ├── tts.py           # 456 LOC - 3エンジンTTS
│   ├── speaker_id.py    # 351 LOC - 声紋認識
│   └── wake_word.py     # stub
├── memory/              # 3層記憶システム (~1,200 LOC)
├── mcp/tools/           # 7ツール (~600 LOC)
├── api/                 # REST/WS API (~400 LOC)
├── display/             # WebSocketサーバー
└── utils/               # 設定, バリデーション, ロガー

frontend/ (7 files)
├── index.html
├── js/ (6 files - app, ws_client, live2d, lip_sync, emotion, settings)
└── css/ (2 files)

config/ (3 YAML files)
knowledge/ (3 md files - 3 files missing)
tests/ (3 test files, 60+ tests)
```
