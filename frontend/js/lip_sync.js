/**
 * Lip Sync Controller
 *
 * Animates the Live2D model's mouth based on audio volume data
 * received from the backend or from real-time audio analysis.
 */

class LipSync {
    constructor() {
        this.isActive = false;
        this.volumes = [];         // Pre-computed volume data from backend
        this.currentIndex = 0;
        this.frameInterval = 23;   // ms per frame (~1024 samples at 44100Hz)
        this.smoothing = 0.3;      // Smoothing factor (0 = no smooth, 1 = max smooth)
        this.currentVolume = 0;
        this.targetVolume = 0;
        this._animationId = null;
        this._startTime = 0;

        // Callback to update Live2D mouth parameter
        this.onVolumeUpdate = null;
    }

    /**
     * Start lip sync with pre-computed volume data.
     * @param {number[]} volumes - Array of volume levels (0.0-1.0).
     */
    startWithData(volumes) {
        this.stop();
        this.volumes = volumes;
        this.currentIndex = 0;
        this.isActive = true;
        this._startTime = performance.now();
        this._animate();
    }

    /**
     * Start lip sync with real-time audio analysis.
     * @param {MediaStream|AudioContext} audioSource - Audio source for analysis.
     */
    startWithAudio(audioSource) {
        this.stop();
        this.isActive = true;

        if (audioSource instanceof AudioContext) {
            this._setupAudioAnalysis(audioSource);
        }
    }

    /**
     * Start lip sync with a connected AnalyserNode (real-time audio).
     * @param {AnalyserNode} analyser - A Web Audio AnalyserNode connected to audio source.
     */
    startWithAnalyser(analyser) {
        this.stop();
        this.isActive = true;
        this._analyser = analyser;
        this._dataArray = new Uint8Array(analyser.frequencyBinCount);
        this._animateRealtime();
    }

    /**
     * Stop lip sync animation.
     */
    stop() {
        this.isActive = false;
        this.currentVolume = 0;
        this.targetVolume = 0;
        this.volumes = [];
        this.currentIndex = 0;
        this._analyser = null;
        this._dataArray = null;

        if (this._animationId) {
            cancelAnimationFrame(this._animationId);
            this._animationId = null;
        }

        // Reset mouth to closed
        if (this.onVolumeUpdate) {
            this.onVolumeUpdate(0);
        }
    }

    /**
     * Get the current mouth open value for Live2D.
     * @returns {number} Mouth open value (0.0-1.0).
     */
    getMouthValue() {
        return this.currentVolume;
    }

    // --- Private ---

    _animate() {
        if (!this.isActive) return;

        const elapsed = performance.now() - this._startTime;
        const frameIndex = Math.floor(elapsed / this.frameInterval);

        if (this.volumes.length > 0) {
            // Pre-computed volume data mode
            if (frameIndex >= this.volumes.length) {
                this.stop();
                return;
            }
            this.targetVolume = this.volumes[frameIndex] || 0;
        }

        // Smooth the volume change
        this.currentVolume += (this.targetVolume - this.currentVolume) * (1 - this.smoothing);

        // Apply minimum threshold (don't open mouth for very quiet sounds)
        const mouthValue = this.currentVolume > 0.05 ? this.currentVolume : 0;

        // Notify callback
        if (this.onVolumeUpdate) {
            this.onVolumeUpdate(mouthValue);
        }

        this._animationId = requestAnimationFrame(() => this._animate());
    }

    _animateRealtime() {
        if (!this.isActive || !this._analyser) return;

        this._analyser.getByteFrequencyData(this._dataArray);

        // Calculate average volume from frequency data
        let sum = 0;
        for (let i = 0; i < this._dataArray.length; i++) {
            sum += this._dataArray[i];
        }
        const average = sum / this._dataArray.length;
        this.targetVolume = Math.min(average / 128, 1.0);

        // Smooth the volume change
        this.currentVolume += (this.targetVolume - this.currentVolume) * (1 - this.smoothing);

        const mouthValue = this.currentVolume > 0.05 ? this.currentVolume : 0;

        if (this.onVolumeUpdate) {
            this.onVolumeUpdate(mouthValue);
        }

        this._animationId = requestAnimationFrame(() => this._animateRealtime());
    }

    _setupAudioAnalysis(audioContext) {
        // For real-time audio analysis (e.g., when playing TTS audio in browser)
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        const analyze = () => {
            if (!this.isActive) return;

            analyser.getByteFrequencyData(dataArray);

            // Calculate average volume from frequency data
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
            }
            const average = sum / dataArray.length;
            this.targetVolume = Math.min(average / 128, 1.0);

            // Smooth
            this.currentVolume += (this.targetVolume - this.currentVolume) * (1 - this.smoothing);

            if (this.onVolumeUpdate) {
                this.onVolumeUpdate(this.currentVolume > 0.05 ? this.currentVolume : 0);
            }

            this._animationId = requestAnimationFrame(analyze);
        };

        analyze();
        return analyser;
    }
}
