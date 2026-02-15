# Handoff: v0.3 Local Testing & Fixes

## Context
Branch: `claude/ai-secretary-livestream-Icj9B`
Latest commit: `feat: add vision input, proactive scheduler, and speaker ID upgrade (v0.3)`

v0.3 added:
1. **Live2D upgrade**: Frequency-based lip sync, natural eye blinking, better expression transitions, angry/sad emotions
2. **Mobile responsive**: Full-width bottom sheet chat on phones, touch targets, safe area insets
3. **Vision input** (`src/vision/camera.py`): OpenCV camera -> Gemini Flash scene analysis
4. **Proactive scheduler** (`src/core/proactive_scheduler.py`): Time/idle/vision-triggered speech
5. **Speaker ID upgrade** (`src/audio/speaker_id.py`): Real resemblyzer embedding extraction

## Task 1: Start & Test Locally

```bash
cd ~/mishuai
python -m src.main --text-only
# Another terminal:
cd ~/mishuai/frontend && python -m http.server 8080
# Open http://localhost:8080
```

Verify:
- [ ] Live2D model loads (may need to download model files to `frontend/models/`)
- [ ] Natural eye blinking works (random 2-6s intervals)
- [ ] Send chat message -> LLM responds
- [ ] During speech: mouth moves with frequency-based lip sync
- [ ] During speech: expression changes based on emotion (happy/sad/angry/neutral)
- [ ] After speech: model returns to idle animation (breathing + subtle sway)
- [ ] Edge TTS audio plays in browser
- [ ] Mobile layout works (open on phone via local IP, e.g. `http://192.168.x.x:8080`)

## Task 2: Known Issues to Fix

### TTS audio might not trigger lip sync
The frontend `playAudioB64()` in `app.js` creates an AnalyserNode and feeds it to `lip_sync.js`.
If lip sync doesn't work, check browser console for:
- AudioContext not resumed (needs user click/tap first)
- AnalyserNode not connected properly
- `live2d.lipSync` reference is null

### Live2D model files
The frontend loads models from CDN. If CDN is blocked/slow in China:
- Download Cubism sample models
- Place in `frontend/models/`
- Update the model URL in `app.js` `tryLoadModel()` function

### Proactive scheduler testing
To test idle trigger quickly, temporarily set in `config/config.yaml`:
```yaml
proactive:
  idle_timeout_min: 1  # Was 15
  cooldown_sec: 30     # Was 300
```

## Task 3: Livestream Readiness (Future)

For OBS streaming setup:
- WebSocket sends audio as base64 -> browser plays it -> OBS captures browser window
- Live2D + lip sync + expressions all render in-browser
- No additional OBS plugin needed, just "Browser Source" or "Window Capture"

## Architecture Reference

```
Text Input (WebSocket) ──> Pipeline (LLM + Tools) ──> Output Queue
Camera (Vision) ───────> Proactive Scheduler ──────────╱     │
Time/Idle triggers ─────────────────────────────────────╱     ▼
                                                     Output Consumer
                                                      ├── TTS (Edge TTS -> MP3)
                                                      ├── WS: speech_start + emotion
                                                      ├── WS: audio (base64 MP3)
                                                      ├── WS: speech_end
                                                      └── State: SPEAKING -> IDLE

Frontend:
  audio msg ──> AudioContext.decodeAudioData()
              ──> source -> AnalyserNode -> destination
              ──> AnalyserNode -> LipSync (frequency bands)
              ──> LipSync -> Live2D ParamMouthOpenY + ParamMouthForm
  speech_start ──> EmotionMapper -> Live2D expression params
```

## Key Files

| File | Role |
|------|------|
| `src/main.py` | Main orchestrator (v0.3) |
| `src/core/pipeline.py` | Dialogue pipeline (LLM + memory + tools) |
| `src/core/proactive_scheduler.py` | Proactive speech triggers |
| `src/vision/camera.py` | Camera -> Gemini Flash |
| `src/audio/tts.py` | Edge TTS synthesis |
| `src/audio/speaker_id.py` | Speaker identification |
| `frontend/js/app.js` | Frontend main (WebSocket + audio playback) |
| `frontend/js/live2d_controller.js` | Live2D model control |
| `frontend/js/lip_sync.js` | Frequency-based lip sync |
| `frontend/js/emotion_mapper.js` | Emotion -> Live2D params |
| `config/config.yaml` | All configuration |
