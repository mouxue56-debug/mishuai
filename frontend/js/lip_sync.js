/**
 * Lip Sync Controller (v2 - Frequency-based)
 *
 * Analyzes audio frequency bands to drive multiple mouth parameters:
 * - ParamMouthOpenY: How wide the mouth opens (driven by overall volume)
 * - ParamMouthForm:  Mouth shape (-1 = "o/a", +1 = "i/e" smile)
 *
 * References DenchiSoft/Live2DFrequencyLipSync approach.
 */

class LipSync {
    constructor() {
        this.isActive = false;
        this.volumes = [];
        this.currentIndex = 0;
        this.frameInterval = 23;
        this.smoothing = 0.35;
        this._animationId = null;
        this._startTime = 0;
        this._analyser = null;
        this._dataArray = null;

        // Multi-parameter state (frequency-based)
        this.currentMouthOpen = 0;
        this.targetMouthOpen = 0;
        this.currentMouthForm = 0;
        this.targetMouthForm = 0;

        // Callback: receives { mouthOpen, mouthForm } object
        this.onMouthUpdate = null;

        // Legacy single-value callback (kept for backward compat)
        this.onVolumeUpdate = null;
    }

    /**
     * Start lip sync with a connected AnalyserNode (real-time, frequency-based).
     * @param {AnalyserNode} analyser - A Web Audio AnalyserNode connected to audio source.
     */
    startWithAnalyser(analyser) {
        this.stop();
        this.isActive = true;
        this._analyser = analyser;
        this._dataArray = new Uint8Array(analyser.frequencyBinCount);
        this._animateFrequency();
    }

    /**
     * Start lip sync with pre-computed volume data (legacy/fallback).
     * @param {number[]} volumes - Array of volume levels (0.0-1.0).
     */
    startWithData(volumes) {
        this.stop();
        this.volumes = volumes;
        this.currentIndex = 0;
        this.isActive = true;
        this._startTime = performance.now();
        this._animateLegacy();
    }

    /**
     * Stop lip sync animation.
     */
    stop() {
        this.isActive = false;
        this.currentMouthOpen = 0;
        this.targetMouthOpen = 0;
        this.currentMouthForm = 0;
        this.targetMouthForm = 0;
        this.volumes = [];
        this.currentIndex = 0;
        this._analyser = null;
        this._dataArray = null;

        if (this._animationId) {
            cancelAnimationFrame(this._animationId);
            this._animationId = null;
        }

        // Reset mouth to closed
        this._notifyUpdate(0, 0);
    }

    // --- Private: Frequency-based analysis ---

    _animateFrequency() {
        if (!this.isActive || !this._analyser) return;

        this._analyser.getByteFrequencyData(this._dataArray);
        const binCount = this._dataArray.length; // 128 bins for fftSize=256

        // Split into 3 frequency bands
        // At 44100Hz, each bin ~= 172Hz. At 48000Hz, ~= 187Hz.
        // Low:  bins 0-3   (0~690Hz)   - fundamental, "a/o" vowels
        // Mid:  bins 3-12  (690~2070Hz) - formants, "e/i" vowels
        // High: bins 12-40 (2070~6900Hz)- consonants, sibilants
        const lowEnd = Math.min(4, binCount);
        const midEnd = Math.min(13, binCount);
        const highEnd = Math.min(41, binCount);

        let lowSum = 0, midSum = 0, highSum = 0;
        for (let i = 0; i < lowEnd; i++) lowSum += this._dataArray[i];
        for (let i = lowEnd; i < midEnd; i++) midSum += this._dataArray[i];
        for (let i = midEnd; i < highEnd; i++) highSum += this._dataArray[i];

        const lowAvg = lowSum / lowEnd;
        const midAvg = midSum / (midEnd - lowEnd);
        const highAvg = highSum / (highEnd - midEnd);

        // Mouth open: driven by overall energy (weighted toward low+mid)
        const overallEnergy = (lowAvg * 0.5 + midAvg * 0.35 + highAvg * 0.15);
        this.targetMouthOpen = Math.min(overallEnergy / 100, 1.0);

        // Mouth form: frequency balance determines shape
        // More low energy → "a/o" (negative form, round mouth)
        // More mid/high energy → "i/e" (positive form, smile-like)
        const totalEnergy = lowAvg + midAvg + highAvg;
        if (totalEnergy > 15) {
            const balance = (midAvg + highAvg * 0.5) / totalEnergy;
            // balance ~0.33 = even → 0, high balance → positive, low → negative
            this.targetMouthForm = (balance - 0.35) * 3.0;
            this.targetMouthForm = Math.max(-1.0, Math.min(1.0, this.targetMouthForm));
        } else {
            this.targetMouthForm = 0;
            this.targetMouthOpen = 0;
        }

        // Smooth transitions
        const openSmooth = 1 - this.smoothing;
        const formSmooth = 1 - this.smoothing * 1.5; // Form changes slower

        this.currentMouthOpen += (this.targetMouthOpen - this.currentMouthOpen) * openSmooth;
        this.currentMouthForm += (this.targetMouthForm - this.currentMouthForm) * formSmooth;

        // Apply threshold
        const mouthOpen = this.currentMouthOpen > 0.04 ? this.currentMouthOpen : 0;
        const mouthForm = mouthOpen > 0 ? this.currentMouthForm : 0;

        this._notifyUpdate(mouthOpen, mouthForm);

        this._animationId = requestAnimationFrame(() => this._animateFrequency());
    }

    // --- Private: Legacy volume-only animation ---

    _animateLegacy() {
        if (!this.isActive) return;

        const elapsed = performance.now() - this._startTime;
        const frameIndex = Math.floor(elapsed / this.frameInterval);

        if (this.volumes.length > 0) {
            if (frameIndex >= this.volumes.length) {
                this.stop();
                return;
            }
            this.targetMouthOpen = this.volumes[frameIndex] || 0;
        }

        this.currentMouthOpen += (this.targetMouthOpen - this.currentMouthOpen) * (1 - this.smoothing);
        const mouthOpen = this.currentMouthOpen > 0.05 ? this.currentMouthOpen : 0;

        this._notifyUpdate(mouthOpen, 0);

        this._animationId = requestAnimationFrame(() => this._animateLegacy());
    }

    // --- Notification ---

    _notifyUpdate(mouthOpen, mouthForm) {
        if (this.onMouthUpdate) {
            this.onMouthUpdate({ mouthOpen, mouthForm });
        }
        // Legacy callback compatibility
        if (this.onVolumeUpdate) {
            this.onVolumeUpdate(mouthOpen);
        }
    }
}
