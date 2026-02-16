/**
 * WebSocket Client for Live2D Frontend
 * Handles communication with the Python backend.
 */

class WSClient {
    constructor(url = null) {
        this.url = url || `ws://${location.hostname || 'localhost'}:8765`;
        this.ws = null;
        this.reconnectInterval = 3000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        this.handlers = {};
        this.isConnected = false;
    }

    /**
     * Register a handler for a message type.
     * @param {string} type - Message type (e.g., 'emotion', 'speech_start')
     * @param {function} handler - Callback function(data)
     */
    on(type, handler) {
        if (!this.handlers[type]) {
            this.handlers[type] = [];
        }
        this.handlers[type].push(handler);
    }

    /**
     * Connect to the WebSocket server.
     */
    connect() {
        console.log(`[WS] Connecting to ${this.url}...`);

        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('[WS] Connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this._emit('connected', {});
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this._emit(data.type, data);
            } catch (e) {
                console.error('[WS] Parse error:', e);
            }
        };

        this.ws.onclose = (event) => {
            console.log(`[WS] Disconnected (code=${event.code})`);
            this.isConnected = false;
            this._emit('disconnected', {});
            this._tryReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('[WS] Error:', error);
        };
    }

    /**
     * Send a message to the server.
     * @param {object} data - Message object to send.
     */
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('[WS] Not connected, cannot send');
        }
    }

    /**
     * Send text input from the chat box.
     * @param {string} text - User's text message.
     */
    sendTextInput(text) {
        this.send({
            type: 'text_input',
            text: text,
        });
    }

    /**
     * Send raw audio data to the backend for ASR + speaker identification.
     * @param {string} audioBase64 - Base64-encoded audio data.
     * @param {string} format - Audio MIME type (e.g., 'audio/webm').
     */
    sendAudio(audioBase64, format = 'audio/webm') {
        this.send({
            type: 'audio_input',
            audio: audioBase64,
            format: format,
        });
    }

    /**
     * Send a command to the backend.
     * @param {string} command - Command name.
     * @param {*} data - Optional command data.
     */
    sendCommand(command, data = null) {
        this.send({
            type: 'command',
            command: command,
            data: data,
        });
    }

    /**
     * Disconnect from the server.
     */
    disconnect() {
        this.maxReconnectAttempts = 0; // Prevent reconnection
        if (this.ws) {
            this.ws.close();
        }
    }

    // --- Private ---

    _emit(type, data) {
        const handlers = this.handlers[type] || [];
        handlers.forEach(h => {
            try {
                h(data);
            } catch (e) {
                console.error(`[WS] Handler error for '${type}':`, e);
            }
        });
    }

    _tryReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('[WS] Max reconnect attempts reached');
            return;
        }

        this.reconnectAttempts++;
        console.log(`[WS] Reconnecting in ${this.reconnectInterval}ms (attempt ${this.reconnectAttempts})...`);

        setTimeout(() => {
            this.connect();
        }, this.reconnectInterval);
    }
}
