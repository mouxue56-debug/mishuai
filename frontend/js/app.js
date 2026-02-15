/**
 * Main Application - Fukuraku AI Secretary Frontend (v0.2)
 *
 * Wires up WebSocket, Live2D, lip sync, audio playback, and UI components.
 *
 * v0.2 changes:
 * - Audio playback via WebSocket base64 data
 * - CDN-based Live2D model (Hiyori) as default
 * - Improved audio handling with Web Audio API
 */

// --- Global instances ---
let wsClient;
let live2d;
let subtitleTimer = null;
let isMuted = false;
let audioContext = null;

// --- Initialize ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log('[App] Initializing Fukuraku AI Secretary Frontend v0.2...');

    // Initialize Audio Context (needed for audio playback)
    initAudioContext();

    // Initialize WebSocket
    wsClient = new WSClient();
    setupWSHandlers();
    wsClient.connect();

    // Initialize Live2D
    live2d = new Live2DController();
    await live2d.initialize('live2d-canvas');

    // Load Live2D model
    await tryLoadModel();

    // Handle window resize
    window.addEventListener('resize', () => {
        if (live2d) live2d.onResize();
    });

    // Track mouse for model eye follow
    document.addEventListener('mousemove', (e) => {
        if (live2d && live2d.isLoaded) {
            live2d.lookAt(
                e.clientX / window.innerWidth,
                e.clientY / window.innerHeight
            );
        }
    });

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

        // Create analyser for real-time lip sync
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;

        // Route: source → analyser → speakers
        source.connect(analyser);
        analyser.connect(audioContext.destination);

        // Drive lip sync from real-time audio analysis
        if (live2d) {
            live2d.lipSync.startWithAnalyser(analyser);
        }

        // Stop lip sync when audio finishes
        source.onended = () => {
            if (live2d) {
                live2d.lipSync.stop();
            }
        };

        source.start(0);

        console.log(`[Audio] Playing ${format} audio (${(audioBuffer.duration).toFixed(1)}s)`);
    } catch (e) {
        console.error('[Audio] Playback error:', e);
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
        updateStatus('speaking', '発話中');
        showSubtitle(data.text);
        addChatMessage(data.text, 'assistant');

        // Apply emotion from speech data
        if (data.emotion && live2d && live2d.isLoaded) {
            live2d.setEmotion(data.emotion);
            const emoji = live2d.emotionMapper.getEmoji(data.emotion);
            document.getElementById('emotion-badge').textContent = `${emoji} ${data.emotion}`;
        }
    });

    wsClient.on('speech_end', () => {
        updateStatus('idle', '待機中');
        hideSubtitle();
        if (live2d) live2d.stopLipSync();
    });

    wsClient.on('lip_sync', (data) => {
        if (live2d && data.volumes) {
            live2d.startLipSync(data.volumes);
        }
    });

    // NEW: Handle base64 audio from backend
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
            addChatMessage(data.text, 'user');
        }
    });

    wsClient.on('pong', () => {
        // Keep-alive response
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

    // Auto-hide after some time
    clearTimeout(subtitleTimer);
    subtitleTimer = setTimeout(() => {
        hideSubtitle();
    }, Math.max(3000, text.length * 150));  // Longer text stays longer
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

function toggleSettings() {
    const panel = document.getElementById('settings-panel');
    panel.classList.toggle('hidden');
}

function toggleMute() {
    isMuted = !isMuted;
    const btn = document.getElementById('btn-mute');
    btn.textContent = isMuted ? '🔇' : '🔊';
    wsClient.sendCommand('mute', { muted: isMuted });
}

function setLLMMode(mode) {
    wsClient.sendCommand('set_llm_mode', { mode: mode });
    addSystemMessage(`LLMモード: ${mode}`);
}

function setLanguage(lang) {
    wsClient.sendCommand('set_language', { language: lang });
    addSystemMessage(`言語: ${lang}`);
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
