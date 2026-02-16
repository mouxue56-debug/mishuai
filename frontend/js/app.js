/**
 * Main Application - Fukuraku AI Secretary Frontend (v0.3)
 *
 * Wires up WebSocket, Live2D, lip sync, audio playback, and UI components.
 *
 * v0.3 changes:
 * - Three voice input modes: text, ptt, continuous
 * - Barge-in support (interrupt AI while speaking)
 * - ~15 new WS response handlers for memory, persona, API keys
 */

// --- Global instances ---
let wsClient;
let live2d;
let settingsDrawer;
let subtitleTimer = null;
let isMuted = false;
let audioContext = null;
let lastSentText = null;  // Track last sent text to avoid duplicate chat messages

// --- Voice Input Mode ---
// 'text' = no mic, text only
// 'ptt'  = push-to-talk (tap mic, record one utterance, auto-send)
// 'continuous' = continuous listening with barge-in
let voiceInputMode = 'ptt';
let currentAudioSource = null;  // Track current playing audio for interrupt
let isAISpeaking = false;       // True while AI TTS is playing — suppresses VAD
let vadGainNode = null;         // Gain node to mute mic input during AI speech

// --- Barge-in: user clicks mic while AI is speaking → interrupt ---
// No automatic RMS-based barge-in (too unreliable with browser echo).
// User taps the mic button to interrupt — simple, reliable, natural.

// --- Initialize ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log('[App] Initializing Fukuraku AI Secretary Frontend v0.3...');

    // Initialize Audio Context (needed for audio playback)
    initAudioContext();

    // Initialize WebSocket
    wsClient = new WSClient();
    setupWSHandlers();
    wsClient.connect();

    // Initialize Settings Drawer
    settingsDrawer = new SettingsDrawer(wsClient);
    window._settingsDrawer = settingsDrawer;
    window._wsClient = wsClient;

    // Initialize Live2D
    live2d = new Live2DController();
    await live2d.initialize('live2d-canvas');

    // Load Live2D model
    await tryLoadModel();

    // Handle window resize
    window.addEventListener('resize', () => {
        if (live2d) live2d.onResize();
    });

    // Track mouse/touch for model eye follow
    document.addEventListener('mousemove', (e) => {
        if (live2d && live2d.isLoaded) {
            live2d.lookAt(
                e.clientX / window.innerWidth,
                e.clientY / window.innerHeight
            );
        }
    });

    document.addEventListener('touchmove', (e) => {
        if (live2d && live2d.isLoaded && e.touches.length > 0) {
            const touch = e.touches[0];
            live2d.lookAt(
                touch.clientX / window.innerWidth,
                touch.clientY / window.innerHeight
            );
        }
    }, { passive: true });

    console.log('[App] Initialized');
});

// --- Audio Context Setup ---
function initAudioContext() {
    // Create AudioContext on first user interaction (browser policy)
    const resume = () => {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }
    };
    document.addEventListener('click', resume, { once: true });
    document.addEventListener('touchstart', resume, { once: true });
    document.addEventListener('keydown', resume, { once: true });
}

/**
 * Play base64-encoded audio data received from the backend.
 * @param {string} b64data - Base64-encoded audio (WAV or MP3)
 * @param {string} format - Audio format ('wav' or 'mp3')
 */
async function playAudioB64(b64data, format) {
    if (isMuted) return;

    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    try {
        // Decode base64 to ArrayBuffer
        const binaryStr = atob(b64data);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
        }

        // Decode audio (works for both WAV and MP3)
        const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);

        // Create source
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;

        // Create analyser for frequency-based lip sync
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;

        // Route: source → analyser → speakers
        source.connect(analyser);
        analyser.connect(audioContext.destination);

        // Auto-interrupt: stop any currently playing audio before starting new
        interruptAudio();

        // Track for interrupt + echo suppression
        currentAudioSource = source;
        isAISpeaking = true;

        // Drive lip sync from real-time frequency analysis
        if (live2d) {
            live2d.setSpeaking(true);
            live2d.lipSync.startWithAnalyser(analyser);
        }

        // Stop lip sync + echo suppression when audio finishes
        source.onended = () => {
            currentAudioSource = null;
            isAISpeaking = false;
            if (live2d) {
                live2d.setSpeaking(false);
                live2d.lipSync.stop();
            }
        };

        source.start(0);

        console.log(`[Audio] Playing ${format} audio (${(audioBuffer.duration).toFixed(1)}s)`);
    } catch (e) {
        console.error('[Audio] Playback error:', e);
    }
}

/**
 * Interrupt current audio playback (for barge-in).
 */
function interruptAudio() {
    if (currentAudioSource) {
        try {
            currentAudioSource.stop();
        } catch (e) {}
        currentAudioSource = null;
        isAISpeaking = false;
        if (live2d) {
            live2d.setSpeaking(false);
            live2d.lipSync.stop();
        }
        console.log('[Audio] Interrupted');
    }
}

// --- WebSocket Handlers ---
function setupWSHandlers() {
    wsClient.on('connected', () => {
        updateStatus('idle', '接続済み');
        addSystemMessage('サーバーに接続しました');
    });

    wsClient.on('disconnected', () => {
        updateStatus('idle', '切断');
        addSystemMessage('サーバーとの接続が切断されました。再接続中...');
    });

    wsClient.on('init', (data) => {
        console.log('[App] Server init:', data);
    });

    wsClient.on('emotion', (data) => {
        const emotion = data.emotion || 'neutral';
        const emoji = live2d ? live2d.emotionMapper.getEmoji(emotion) : '😊';

        // Update Live2D expression
        if (live2d && live2d.isLoaded) {
            live2d.setEmotion(emotion);
        }

        // Update UI
        document.getElementById('emotion-badge').textContent = `${emoji} ${emotion}`;
    });

    wsClient.on('speech_start', (data) => {
        isAISpeaking = true;
        updateStatus('speaking', '発話中');
        showSubtitle(data.text);
        addChatMessage(data.text, 'assistant');

        // Apply emotion from speech data
        if (data.emotion && live2d && live2d.isLoaded) {
            live2d.setEmotion(data.emotion);
            const emoji = live2d.emotionMapper.getEmoji(data.emotion);
            document.getElementById('emotion-badge').textContent = `${emoji} ${data.emotion}`;
        }

        // Barge-in: user can tap mic button to interrupt (manual barge-in)
    });

    wsClient.on('speech_end', () => {
        // Stop any playing audio (handles backend-triggered interrupts)
        interruptAudio();
        updateStatus('idle', '待機中');
        hideSubtitle();
        if (live2d) live2d.stopLipSync();
    });

    wsClient.on('lip_sync', (data) => {
        if (live2d && data.volumes) {
            live2d.startLipSync(data.volumes);
        }
    });

    // Handle base64 audio from backend
    wsClient.on('audio', (data) => {
        if (data.data) {
            playAudioB64(data.data, data.format || 'wav');
        }
    });

    wsClient.on('status', (data) => {
        const statusMap = {
            idle: '待機中',
            listening: '聞いています...',
            processing: '考え中...',
            speaking: '発話中',
        };
        updateStatus(data.status, statusMap[data.status] || data.status);
    });

    wsClient.on('subtitle', (data) => {
        if (data.role === 'user') {
            showSubtitle(data.text);
            if (!lastSentText || data.text !== lastSentText) {
                addChatMessage(data.text, 'user');
            }
            lastSentText = null;
        }
    });

    wsClient.on('pong', () => {
        // Keep-alive response
    });

    // --- Settings drawer: config data ---
    wsClient.on('config_data', (data) => {
        if (settingsDrawer) settingsDrawer.onConfigData(data);
    });

    wsClient.on('config_update_result', (data) => {
        if (settingsDrawer) settingsDrawer.onConfigUpdateResult(data);
    });

    // --- API key handlers ---
    wsClient.on('api_key_status', (data) => {
        if (settingsDrawer) settingsDrawer.onApiKeyStatus(data);
    });

    wsClient.on('api_key_update_result', (data) => {
        if (settingsDrawer) settingsDrawer.onApiKeyUpdateResult(data);
    });

    // --- Memory handlers ---
    wsClient.on('memos_data', (data) => {
        if (settingsDrawer) settingsDrawer.onMemosData(data);
    });

    wsClient.on('memo_saved', (data) => {
        if (settingsDrawer) settingsDrawer.onMemoResult(data);
    });

    wsClient.on('memo_updated', (data) => {
        if (settingsDrawer) settingsDrawer.onMemoResult(data);
    });

    wsClient.on('memo_archived', (data) => {
        if (settingsDrawer) settingsDrawer.onMemoResult(data);
    });

    wsClient.on('reminders_data', (data) => {
        if (settingsDrawer) settingsDrawer.onRemindersData(data);
    });

    wsClient.on('reminder_saved', (data) => {
        if (settingsDrawer) settingsDrawer.onReminderResult(data);
    });

    wsClient.on('reminder_completed', (data) => {
        if (settingsDrawer) settingsDrawer.onReminderResult(data);
    });

    wsClient.on('reminder_deleted', (data) => {
        if (settingsDrawer) settingsDrawer.onReminderResult(data);
    });

    // --- History handlers ---
    wsClient.on('history_data', (data) => {
        if (settingsDrawer) settingsDrawer.onHistoryData(data);
    });

    wsClient.on('short_term_cleared', (data) => {
        if (settingsDrawer) settingsDrawer.onShortTermCleared(data);
    });

    // --- Persona handlers ---
    wsClient.on('persona_data', (data) => {
        if (settingsDrawer) settingsDrawer.onPersonaData(data);
    });

    wsClient.on('persona_update_result', (data) => {
        if (settingsDrawer) settingsDrawer.onPersonaUpdateResult(data);
    });

    // --- Knowledge handlers ---
    wsClient.on('knowledge_files', (data) => {
        if (settingsDrawer) settingsDrawer.onKnowledgeFiles(data);
    });

    wsClient.on('knowledge_content', (data) => {
        if (settingsDrawer) settingsDrawer.onKnowledgeContent(data);
    });

    wsClient.on('knowledge_save_result', (data) => {
        if (settingsDrawer) settingsDrawer.onKnowledgeSaveResult(data);
    });

    wsClient.on('knowledge_rebuild_result', (data) => {
        if (settingsDrawer) settingsDrawer.onKnowledgeRebuildResult(data);
    });

    // --- Speaker events ---
    wsClient.on('speakers_data', (data) => {
        if (settingsDrawer) settingsDrawer.onSpeakersData(data);
        // Speaker list used by settings drawer only
    });

    wsClient.on('speaker_registered', (data) => {
        if (settingsDrawer) settingsDrawer.onSpeakerRegistered(data);
    });

    wsClient.on('speaker_updated', (data) => {
        if (settingsDrawer) settingsDrawer.onSpeakerUpdated(data);
    });

    wsClient.on('speaker_deleted', (data) => {
        if (settingsDrawer) settingsDrawer.onSpeakerDeleted(data);
    });

    wsClient.on('audit_sessions_data', (data) => {
        if (settingsDrawer) settingsDrawer.onAuditSessionsData(data);
    });

    wsClient.on('session_detail_data', (data) => {
        if (settingsDrawer) settingsDrawer.onSessionDetailData(data);
    });

    // --- Speaker events (transparent identification) ---
    wsClient.on('speaker_authenticated', (data) => {
        if (data.success) {
            window._currentSpeakerId = data.speaker_id;
            window._currentSessionId = data.session_id;
        }
    });

    wsClient.on('speaker_enrolled', (data) => {
        if (data.success) {
            showNotification(`声紋登録完了: ${data.speaker_id}`);
        }
    });
}

// --- UI Functions ---

function updateStatus(status, text) {
    const dot = document.getElementById('status-dot');
    const textEl = document.getElementById('status-text');

    dot.className = status;
    textEl.textContent = text;
}

function showSubtitle(text) {
    const el = document.getElementById('subtitle-text');
    el.textContent = text;
    el.classList.add('visible');

    clearTimeout(subtitleTimer);
    subtitleTimer = setTimeout(() => {
        hideSubtitle();
    }, Math.max(3000, text.length * 150));
}

function hideSubtitle() {
    const el = document.getElementById('subtitle-text');
    el.classList.remove('visible');
}

function addChatMessage(text, role) {
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = `chat-msg ${role}`;

    const time = new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' });
    msg.innerHTML = `
        <div class="text">${escapeHtml(text)}</div>
        <div class="meta">${time}</div>
    `;

    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

function addSystemMessage(text) {
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = 'chat-msg system';
    msg.style.cssText = 'background: #FFF3E0; font-size: 12px; color: #666; text-align: center; max-width: 100%;';
    msg.textContent = text;
    container.appendChild(msg);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- User Actions ---

function sendChat() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;

    // Display user message
    addChatMessage(text, 'user');
    lastSentText = text;

    // Send to backend
    wsClient.sendTextInput(text);

    // Update status
    updateStatus('processing', '考え中...');

    input.value = '';
}

function toggleChat() {
    const panel = document.getElementById('chat-panel');
    panel.classList.toggle('collapsed');
}

function toggleMute() {
    isMuted = !isMuted;
    const btn = document.getElementById('btn-mute');
    btn.textContent = isMuted ? '🔇' : '🔊';
    wsClient.sendCommand('mute', { muted: isMuted });
}

// --- Voice Input Mode Switching ---

/**
 * Set voice input mode: 'text', 'ptt', or 'continuous'.
 * Called from settings drawer.
 */
function setVoiceInputMode(mode) {
    const prevMode = voiceInputMode;
    voiceInputMode = mode;

    const micBtn = document.getElementById('btn-mic');
    if (!micBtn) return;

    // Stop any current recording
    if (isRecording) {
        stopVoiceInput();
    }

    if (mode === 'text') {
        // Hide mic button
        micBtn.style.display = 'none';
    } else {
        // Show mic button
        micBtn.style.display = '';
    }

    console.log(`[Voice] Mode changed: ${prevMode} → ${mode}`);
}

// --- Voice Input (MediaRecorder → Backend ASR + Speaker ID) ---
// Audio is captured as raw audio, sent via WebSocket to the backend,
// which performs ASR (DashScope Paraformer) and speaker identification
// (resemblyzer) in parallel. No browser-side speech recognition.

let isRecording = false;
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let _savedMimeType = '';

// VAD (Voice Activity Detection) state for continuous mode
let vadAnalyser = null;        // Main analyser (goes through gain node)
let vadRawAnalyser = null;     // Raw analyser (bypasses gain, for barge-in only)
let vadAudioContext = null;
let vadSource = null;
let vadSpeaking = false;
let vadSilenceStart = 0;
const VAD_SILENCE_MS = 800;    // ms of silence to end an utterance
const VAD_THRESHOLD = 0.02;    // RMS threshold for normal speech detection (raised to avoid noise)
const VAD_BARGEIN_THRESHOLD = 0.12; // High threshold for barge-in (human voice near mic)
const VAD_MIN_SPEECH_MS = 600; // Minimum speech duration to trigger barge-in
let vadSpeechStart = 0;        // When current speech started (for min duration check)

function toggleVoiceInput() {
    if (voiceInputMode === 'text') return;

    // If AI is speaking and user clicks mic, interrupt AI first
    if (isAISpeaking) {
        console.log('[Voice] Manual barge-in: interrupting AI');
        interruptAudio();
        wsClient.sendCommand('interrupt', {});
        // In continuous mode, recording is already active, just interrupted AI
        // In PTT mode, fall through to start a new recording
        if (isRecording) return;
    }

    if (isRecording) {
        stopVoiceInput();
    } else {
        startVoiceInput();
    }
}

async function startVoiceInput() {
    const btn = document.getElementById('btn-mic');
    const input = document.getElementById('chat-input');

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });
        console.log('[Voice] Microphone access granted');
    } catch (e) {
        console.error('[Voice] Microphone access denied:', e);
        addSystemMessage('マイクへのアクセスが拒否されました。ブラウザの設定を確認してください。');
        return;
    }

    // Detect supported mime type once
    _savedMimeType = getSupportedMimeType();
    console.log('[Voice] Using MIME type:', _savedMimeType);

    isRecording = true;
    btn.classList.add('recording');
    input.placeholder = '聞いています...';
    updateStatus('listening', '聞いています...');

    if (voiceInputMode === 'continuous') {
        btn.classList.add('continuous-recording');
        startContinuousCapture();
    } else {
        startPTTRecording();
    }

    console.log(`[Voice] Started (mode=${voiceInputMode})`);
}

function stopVoiceInput() {
    console.log('[Voice] stopVoiceInput called, mediaRecorder state:', mediaRecorder ? mediaRecorder.state : 'null');
    const wasRecording = isRecording;
    isRecording = false;

    const btn = document.getElementById('btn-mic');
    btn.classList.remove('recording');
    btn.classList.remove('continuous-recording');
    document.getElementById('chat-input').placeholder = 'テキストで話しかける...';

    // Stop VAD first (continuous mode)
    stopVAD();

    // Stop active recording — onstop fires async and sends audio
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        console.log('[Voice] Stopping active recording...');
        // Wrap onstop to also release media stream after send
        const origOnStop = mediaRecorder.onstop;
        const stream = mediaStream;
        mediaRecorder.onstop = function () {
            console.log('[Voice] onstop fired (wrapped)');
            if (origOnStop) origOnStop.call(this);
            mediaRecorder = null;
            releaseStream(stream);
        };
        mediaStream = null;
        mediaRecorder.stop();
        updateStatus('processing', '考え中...');
    } else {
        // Nothing was actively recording
        console.log('[Voice] No active recording to stop');
        mediaRecorder = null;
        releaseStream(mediaStream);
        mediaStream = null;
        updateStatus('idle', '待機中');
    }

    console.log('[Voice] Stopped');
}

function releaseStream(stream) {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        console.log('[Voice] Media stream released');
    }
}

// ── PTT (Push-to-Talk) Mode ──

function startPTTRecording() {
    if (!mediaStream) {
        console.error('[Voice] PTT: no media stream');
        return;
    }

    audioChunks = [];
    const mimeType = _savedMimeType;

    try {
        mediaRecorder = new MediaRecorder(mediaStream, {
            mimeType: mimeType || undefined,
        });
    } catch (e) {
        console.error('[Voice] PTT: Failed to create MediaRecorder:', e);
        return;
    }

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
            audioChunks.push(e.data);
        }
    };

    mediaRecorder.onstop = () => {
        console.log('[Voice] PTT onstop: chunks=' + audioChunks.length);
        const chunks = audioChunks;
        audioChunks = [];
        if (chunks.length > 0) {
            const blob = new Blob(chunks, { type: mimeType || 'audio/webm' });
            console.log('[Voice] PTT: sending blob, size=' + blob.size);
            sendAudioToBackend(blob);
        } else {
            console.warn('[Voice] PTT: no audio chunks to send');
        }
    };

    mediaRecorder.start(250);
    console.log('[Voice] PTT recording started, state=' + mediaRecorder.state);
}

// ── Continuous Mode (VAD-based) ──

function startContinuousCapture() {
    if (!mediaStream) {
        console.error('[Voice] Continuous: no media stream');
        return;
    }

    // Set up Web Audio API for VAD
    // Route: mic → gainNode → analyser
    // During AI speech, gain=0 → analyser reads silence → no false VAD triggers
    // For barge-in, we periodically sample the raw mic source directly
    try {
        vadAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        vadSource = vadAudioContext.createMediaStreamSource(mediaStream);
        vadGainNode = vadAudioContext.createGain();
        vadGainNode.gain.value = 1.0;
        vadAnalyser = vadAudioContext.createAnalyser();
        vadAnalyser.fftSize = 2048;
        vadSource.connect(vadGainNode);
        vadGainNode.connect(vadAnalyser);
        // Raw analyser for barge-in: reads mic directly, bypasses gain mute
        vadRawAnalyser = vadAudioContext.createAnalyser();
        vadRawAnalyser.fftSize = 2048;
        vadSource.connect(vadRawAnalyser);
    } catch (e) {
        console.error('[Voice] Failed to set up VAD audio context:', e);
        return;
    }

    vadSpeaking = false;
    vadSilenceStart = 0;
    audioChunks = [];

    // In continuous mode, we don't pre-start MediaRecorder.
    // VAD will start it when speech is detected.
    mediaRecorder = null;

    // VAD polling loop
    vadPoll();
    console.log('[Voice] Continuous capture started (VAD active, threshold=' + VAD_THRESHOLD + ')');
}

function vadPoll() {
    if (!isRecording || !vadAnalyser) return;

    const now = Date.now();

    // ── During AI speech: mic gain is 0, main analyser reads silence.
    //    Only check the RAW analyser for barge-in (human voice near mic).
    if (isAISpeaking) {
        // Mute the main VAD path so no false speech detection
        if (vadGainNode) vadGainNode.gain.value = 0;

        // Check raw mic for barge-in
        if (vadRawAnalyser) {
            const rawData = new Float32Array(vadRawAnalyser.fftSize);
            vadRawAnalyser.getFloatTimeDomainData(rawData);
            let rawSum = 0;
            for (let i = 0; i < rawData.length; i++) rawSum += rawData[i] * rawData[i];
            const rawRms = Math.sqrt(rawSum / rawData.length);

            if (rawRms > VAD_BARGEIN_THRESHOLD) {
                if (!vadSpeaking) {
                    vadSpeaking = true;
                    vadSpeechStart = now;
                }
                // Only trigger barge-in after sustained loud speech
                if ((now - vadSpeechStart) > VAD_MIN_SPEECH_MS) {
                    console.log('[Voice] Barge-in! rms=' + rawRms.toFixed(4) +
                                ', duration=' + (now - vadSpeechStart) + 'ms');
                    interruptAudio();
                    wsClient.sendCommand('interrupt', {});
                    // After barge-in, unmute gain so normal VAD resumes
                    if (vadGainNode) vadGainNode.gain.value = 1.0;
                    vadSpeaking = false;
                    vadSilenceStart = 0;
                }
            } else {
                // Reset barge-in speech tracking if silence
                if (vadSpeaking) {
                    vadSpeaking = false;
                    vadSpeechStart = 0;
                }
            }
        }

        requestAnimationFrame(vadPoll);
        return;
    }

    // ── Normal mode: AI is NOT speaking, gain=1, normal VAD ──
    if (vadGainNode) vadGainNode.gain.value = 1.0;

    const dataArray = new Float32Array(vadAnalyser.fftSize);
    vadAnalyser.getFloatTimeDomainData(dataArray);

    let sumSquares = 0;
    for (let i = 0; i < dataArray.length; i++) {
        sumSquares += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sumSquares / dataArray.length);

    const isSpeech = rms > VAD_THRESHOLD;

    if (isSpeech) {
        if (!vadSpeaking) {
            // Speech started
            vadSpeaking = true;
            vadSpeechStart = now;
            console.log('[Voice] VAD: speech started (rms=' + rms.toFixed(4) + ')');
            updateStatus('listening', '聞いています...');

            // Start a new MediaRecorder for this utterance
            audioChunks = [];
            try {
                mediaRecorder = new MediaRecorder(mediaStream, {
                    mimeType: _savedMimeType || undefined,
                });
                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };
                mediaRecorder.start(100);
            } catch (e) {
                console.error('[Voice] VAD: Failed to start MediaRecorder:', e);
                vadSpeaking = false;
            }
        }
        vadSilenceStart = 0;
    } else {
        if (vadSpeaking) {
            if (vadSilenceStart === 0) {
                vadSilenceStart = now;
            } else if (now - vadSilenceStart >= VAD_SILENCE_MS) {
                // Silence threshold reached → end utterance
                const speechDuration = now - vadSpeechStart;
                vadSpeaking = false;
                vadSilenceStart = 0;

                // Discard very short bursts (echo remnants or noise)
                if (speechDuration < 300) {
                    console.log('[Voice] VAD: speech too short (' + speechDuration + 'ms), discarding');
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.onstop = () => { audioChunks = []; mediaRecorder = null; };
                        mediaRecorder.stop();
                    }
                    requestAnimationFrame(vadPoll);
                    return;
                }

                console.log('[Voice] VAD: speech ended (' + speechDuration + 'ms), sending');
                updateStatus('processing', '考え中...');

                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    const mimeType = _savedMimeType;
                    mediaRecorder.onstop = () => {
                        const chunks = audioChunks;
                        audioChunks = [];
                        if (chunks.length > 0) {
                            const blob = new Blob(chunks, { type: mimeType || 'audio/webm' });
                            console.log('[Voice] VAD: sending blob, size=' + blob.size);
                            sendAudioToBackend(blob);
                        }
                        mediaRecorder = null;
                    };
                    mediaRecorder.stop();
                } else {
                    console.warn('[Voice] VAD: no active recorder to stop');
                }
            }
        }
    }

    requestAnimationFrame(vadPoll);
}

function stopVAD() {
    vadSpeaking = false;
    vadSilenceStart = 0;
    if (vadGainNode) { try { vadGainNode.disconnect(); } catch(e) {} vadGainNode = null; }
    if (vadSource) { try { vadSource.disconnect(); } catch(e) {} vadSource = null; }
    if (vadAudioContext) { try { vadAudioContext.close(); } catch(e) {} vadAudioContext = null; }
    vadAnalyser = null;
    vadRawAnalyser = null;
}

// ── Audio Sending ──

function sendAudioToBackend(blob) {
    console.log('[Voice] sendAudioToBackend: size=' + blob.size + ', type=' + blob.type +
                ', ws=' + (wsClient ? 'yes' : 'no') + ', connected=' + (wsClient ? wsClient.isConnected : false));

    if (!wsClient || !wsClient.isConnected) {
        console.warn('[Voice] Not connected, cannot send audio');
        return;
    }

    if (blob.size < 100) {
        console.warn('[Voice] Audio too small (' + blob.size + ' bytes), skipping');
        return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        if (base64) {
            const format = blob.type.split(';')[0] || 'audio/webm';
            wsClient.sendAudio(base64, format);
            console.log('[Voice] Sent ' + Math.round(blob.size / 1024) + 'KB audio (' + format + ') to backend');
            updateStatus('processing', '考え中...');
        } else {
            console.error('[Voice] Failed to extract base64 from blob');
        }
    };
    reader.onerror = (e) => {
        console.error('[Voice] FileReader error:', e);
    };
    reader.readAsDataURL(blob);
}

function getSupportedMimeType() {
    const types = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
    ];
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) {
            console.log('[Voice] Supported MIME type: ' + type);
            return type;
        }
    }
    console.warn('[Voice] No preferred MIME type supported, using browser default');
    return '';
}

// --- Model Loading ---

async function tryLoadModel() {
    // Priority order:
    // 1. Local models in frontend/models/
    // 2. CDN-hosted official free models (Hiyori)
    const localPaths = [
        'models/model.model3.json',
        'models/haru/Haru.model3.json',
        'models/hiyori/Hiyori.model3.json',
        'models/mao/Mao.model3.json',
    ];

    // Try local models first
    for (const path of localPaths) {
        try {
            const response = await fetch(path, { method: 'HEAD' });
            if (response.ok) {
                console.log(`[App] Found local model: ${path}`);
                await live2d.loadModel(path);
                return;
            }
        } catch (e) {
            // File not found, try next
        }
    }

    // Fallback: Use CDN-hosted Hiyori model (free, official Live2D sample)
    const cdnModelUrl = 'https://cdn.jsdelivr.net/gh/Live2D/CubismWebSamples@develop/Samples/Resources/Hiyori/Hiyori.model3.json';
    console.log('[App] No local model found, loading Hiyori from CDN...');
    try {
        await live2d.loadModel(cdnModelUrl);
        console.log('[App] CDN model loaded successfully');
    } catch (e) {
        console.warn('[App] CDN model load failed:', e);
        console.info(
            '[App] To use Live2D offline, download models from:\n' +
            '  https://github.com/Live2D/CubismWebSamples\n' +
            '  Place in frontend/models/ directory'
        );
    }
}

// ================================================================
// Notifications
// ================================================================

function showNotification(message, duration = 3000) {
    let notif = document.getElementById('app-notification');
    if (!notif) {
        notif = document.createElement('div');
        notif.id = 'app-notification';
        notif.style.cssText = `
            position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
            background: var(--accent, #3b82f6); color: white;
            padding: 10px 24px; border-radius: 20px; font-size: 14px;
            z-index: 10000; opacity: 0; transition: opacity 0.3s;
            pointer-events: none;
        `;
        document.body.appendChild(notif);
    }
    notif.textContent = message;
    notif.style.opacity = '1';
    setTimeout(() => { notif.style.opacity = '0'; }, duration);
}

