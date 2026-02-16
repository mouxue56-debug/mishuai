# WebSocket Event Protocol v1.0

MishuAI (Fukuraku AI Secretary) のフロントエンド-バックエンド間 WebSocket 通信プロトコル。

接続先: `ws://localhost:8765`

## Server → Client (後端→前端)

### 対話イベント

| type | payload | 説明 |
|------|---------|------|
| `init` | `{version, capabilities, persona}` | 接続成功、初期データ |
| `speech_start` | `{text, emotion, request_id}` | 発話開始（字幕表示開始） |
| `audio` | `{data (base64), format, request_id, segment_index, total_segments, is_last}` | 音声データ（セグメント単位） |
| `speech_end` | `{request_id}` | 発話終了 |
| `subtitle` | `{text, speaker, request_id}` | 字幕テキスト |
| `interrupt` | `{request_id}` | 打断された |

### 状態イベント

| type | payload | 説明 |
|------|---------|------|
| `emotion` | `{emotion, intensity}` | 感情変化 → Live2D表情切替 |
| `lip_sync` | `{volumes: number[]}` | リップシンクデータ |
| `status` | `{state: idle|listening|thinking|speaking}` | システム状態変化 |
| `mode_change` | `{mode: quality|balanced|budget}` | LLMモード切替 |
| `speaker_authenticated` | `{speaker_id, name, role, confidence}` | 声紋認識結果 |

### ツールイベント

| type | payload | 説明 |
|------|---------|------|
| `tool_call` | `{tool_name, args, status}` | ツール呼出開始 |
| `tool_result` | `{tool_name, result, success}` | ツール結果 |

### データ応答（設定画面用）

| type | payload | 説明 |
|------|---------|------|
| `config_data` | `{config: object}` | 現在の設定データ |
| `config_update_result` | `{success, key, value}` | 設定更新結果 |
| `api_key_status` | `{keys: {provider: bool}}` | APIキー設定状態 |
| `api_key_update_result` | `{success, provider}` | APIキー更新結果 |
| `persona_data` | `{persona: object}` | ペルソナ設定データ |
| `persona_update_result` | `{success}` | ペルソナ更新結果 |

### メモ・リマインダー

| type | payload | 説明 |
|------|---------|------|
| `memos_data` | `{memos: array}` | メモ一覧 |
| `memo_saved` | `{memo: object}` | メモ保存完了 |
| `memo_updated` | `{memo: object}` | メモ更新完了 |
| `memo_archived` | `{memo_id}` | メモアーカイブ完了 |
| `reminders_data` | `{reminders: array}` | リマインダー一覧 |
| `reminder_saved` | `{reminder: object}` | リマインダー保存完了 |
| `reminder_completed` | `{reminder_id}` | リマインダー完了 |
| `reminder_deleted` | `{reminder_id}` | リマインダー削除完了 |

### 会話履歴

| type | payload | 説明 |
|------|---------|------|
| `history_data` | `{turns: array}` | 会話履歴データ |
| `short_term_cleared` | `{}` | 短期メモリクリア完了 |

### ナレッジベース

| type | payload | 説明 |
|------|---------|------|
| `knowledge_files` | `{files: array}` | ナレッジファイル一覧 |
| `knowledge_content` | `{filename, content}` | ファイル内容 |
| `knowledge_save_result` | `{success, filename}` | ファイル保存結果 |
| `knowledge_rebuild_result` | `{success}` | インデックス再構築結果 |

### スピーカー管理

| type | payload | 説明 |
|------|---------|------|
| `speakers_data` | `{speakers: array}` | 登録スピーカー一覧 |
| `speaker_registered` | `{speaker: object}` | スピーカー登録完了 |
| `speaker_updated` | `{speaker: object}` | スピーカー更新完了 |
| `speaker_deleted` | `{speaker_id}` | スピーカー削除完了 |
| `speaker_enrolled` | `{speaker_id, success}` | 音声登録完了 |

### 監査ログ

| type | payload | 説明 |
|------|---------|------|
| `audit_sessions_data` | `{sessions: array}` | セッション一覧 |
| `session_detail_data` | `{session_id, turns: array}` | セッション詳細 |

### ダッシュボード

| type | payload | 説明 |
|------|---------|------|
| `dashboard_data` | `{total_turns, avg_llm_ms, avg_tts_ms, model_usage, emotion_distribution, abort_rate, error_rate}` | メトリクスデータ |

### システム

| type | payload | 説明 |
|------|---------|------|
| `pong` | `{}` | ping応答 |
| `error` | `{code, message}` | エラー |

---

## Client → Server (前端→後端)

### 入力

| command | payload | 説明 |
|---------|---------|------|
| `text_input` | `{text}` | テキスト入力 |
| `audio_input` | `{data (base64), format}` | 音声入力 |
| `push_to_talk` | `{action: start|stop}` | Push-to-talk |
| `interrupt` | `{}` | ユーザー主動打断 |

### 設定

| command | payload | 説明 |
|---------|---------|------|
| `get_config` | `{}` | 設定取得 |
| `update_config` | `{key, value}` | 設定更新 |
| `get_api_keys` | `{}` | APIキー状態取得 |
| `update_api_key` | `{provider, key}` | APIキー更新 |
| `set_llm_mode` | `{mode}` | LLMモード切替 |
| `set_language` | `{language}` | 言語切替 |
| `set_tts_engine` | `{engine}` | TTSエンジン切替 |
| `mute` | `{muted: bool}` | ミュート切替 |

### ペルソナ

| command | payload | 説明 |
|---------|---------|------|
| `get_persona` | `{}` | ペルソナ取得 |
| `update_persona` | `{field, value}` | ペルソナ更新 |

### メモ・リマインダー

| command | payload | 説明 |
|---------|---------|------|
| `get_memos` | `{filter?}` | メモ一覧取得 |
| `save_memo` | `{content, category?}` | メモ保存 |
| `update_memo` | `{memo_id, content}` | メモ更新 |
| `archive_memo` | `{memo_id}` | メモアーカイブ |
| `get_reminders` | `{}` | リマインダー取得 |
| `save_reminder` | `{content, trigger_time}` | リマインダー保存 |
| `complete_reminder` | `{reminder_id}` | リマインダー完了 |
| `delete_reminder` | `{reminder_id}` | リマインダー削除 |

### 会話履歴

| command | payload | 説明 |
|---------|---------|------|
| `get_history` | `{limit?, offset?}` | 履歴取得 |
| `clear_short_term` | `{}` | 短期メモリクリア |

### ナレッジベース

| command | payload | 説明 |
|---------|---------|------|
| `get_knowledge_files` | `{}` | ファイル一覧取得 |
| `get_knowledge_content` | `{filename}` | ファイル内容取得 |
| `save_knowledge_content` | `{filename, content}` | ファイル保存 |
| `rebuild_knowledge_index` | `{}` | インデックス再構築 |

### スピーカー管理

| command | payload | 説明 |
|---------|---------|------|
| `get_speakers` | `{}` | スピーカー一覧 |
| `register_speaker` | `{speaker_id, name, role}` | スピーカー登録 |
| `update_speaker` | `{speaker_id, fields}` | スピーカー更新 |
| `reenroll_speaker` | `{speaker_id, audio_data}` | 声紋再登録 |
| `delete_speaker` | `{speaker_id}` | スピーカー削除 |
| `authenticate_as` | `{speaker_id}` | 手動認証 |

### 監査ログ

| command | payload | 説明 |
|---------|---------|------|
| `get_audit_sessions` | `{limit?}` | セッション一覧取得 |
| `get_session_detail` | `{session_id}` | セッション詳細取得 |

### ダッシュボード

| command | payload | 説明 |
|---------|---------|------|
| `get_dashboard` | `{}` | メトリクスデータ取得 |

### システム

| command | payload | 説明 |
|---------|---------|------|
| `ping` | `{}` | 生存確認 |

---

## Audio Format

### Server → Client (TTS output)
- **WAV**: RIFF header, 22050Hz/24kHz 16-bit mono (CosyVoice / VOICEVOX)
- Encoding: Base64

### Client → Server (ASR input)
- **PCM**: 16kHz 16-bit mono, base64 encoded
- **WAV**: RIFF header, 16kHz 16-bit mono, base64 encoded

---

## Streaming Playback Protocol

TTS output is sent as segmented audio for low-latency playback:

1. Server sends `speech_start` with full text
2. Server sends multiple `audio` messages with `segment_index` (0, 1, 2, ...)
3. Last `audio` message has `is_last: true`
4. Server sends `speech_end`

If interrupted:
1. Server stops sending further `audio` segments
2. Server sends `interrupt` event
3. Frontend flushes audio queue and stops playback
