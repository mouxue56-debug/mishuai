/**
 * Main Application - Fukuraku AI Secretary Frontend
 *
 * Wires up WebSocket, Live2D, lip sync, and UI components.
 */

// --- Global instances ---
let wsClient;
let live2d;
let subtitleTimer = null;
let isMuted = false;

// --- Initialize ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log('[App] Initializing Fukuraku AI Secretary Frontend...');

    // Initialize WebSocket
    wsClient = new WSClient();
    setupWSHandlers();
    wsClient.connect();

    // Initialize Live2D
    live2d = new Live2DController();
    await live2d.initialize('live2d-canvas');

    // Try to load a model (user needs to provide the model file)
    // Default: look for a model in the models/ directory
    tryLoadModel();

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
        showSubtitle(data.text);
        if (data.role === 'user') {
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
    // Try common model paths
    const modelPaths = [
        'models/model.model3.json',
        'models/haru/haru.model3.json',
        'models/mao/mao.model3.json',
    ];

    for (const path of modelPaths) {
        try {
            const response = await fetch(path, { method: 'HEAD' });
            if (response.ok) {
                await live2d.loadModel(path);
                return;
            }
        } catch (e) {
            // File not found, try next
        }
    }

    console.info(
        '[App] No Live2D model found. Place a model in frontend/models/ directory.\n' +
        'You can get free models from: https://booth.pm/\n' +
        'Expected format: .model3.json (Cubism 3/4 format)'
    );
}
