/**
 * Live2D Controller (v2)
 *
 * Manages the Live2D model rendering, expression changes,
 * and parameter updates using pixi-live2d-display.
 *
 * v2 upgrades:
 * - Frequency-based lip sync (multi-parameter mouth control)
 * - Natural eye blinking with random intervals
 * - Expression transitions coexist with idle animation
 * - References N.E.K.O and Open-LLM-VTuber approaches
 */

class Live2DController {
    constructor() {
        this.app = null;
        this.model = null;
        this.isLoaded = false;
        this.emotionMapper = new EmotionMapper();
        this.lipSync = new LipSync();

        // Transition animation state
        this._transitionAnimation = null;
        this._activeEmotionParams = {};  // Currently applied emotion params

        // Idle animation
        this._idleTimer = null;
        this._lookAtTarget = { x: 0, y: 0 };

        // Eye blink state
        this._blinkTimer = null;
        this._blinkPhase = 'open';  // 'open' | 'closing' | 'opening'
        this._blinkProgress = 0;
        this._isSpeaking = false;

        // Zoom & drag state
        this._baseScale = 1;       // Scale computed from loadModel
        this._userZoom = 1;        // User zoom multiplier (0.3 - 3.0)
        this._dragOffset = { x: 0, y: 0 };
        this._isDragging = false;
        this._dragStart = { x: 0, y: 0 };
        this._pinchStartDist = 0;
        this._pinchStartZoom = 1;
    }

    /**
     * Initialize the Live2D renderer.
     * @param {string} canvasId - Canvas element ID.
     */
    async initialize(canvasId = 'live2d-canvas') {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error('[Live2D] Canvas not found:', canvasId);
            return;
        }

        // Create PIXI application
        this.app = new PIXI.Application({
            view: canvas,
            autoStart: true,
            resizeTo: window,
            backgroundAlpha: 0,
        });

        // Set up lip sync callbacks (both new and legacy)
        this.lipSync.onMouthUpdate = (params) => {
            this._setMouthParams(params);
        };
        this.lipSync.onVolumeUpdate = null; // Use onMouthUpdate instead

        // Set up zoom & drag interactions
        this._setupZoomDrag(canvas);

        console.log('[Live2D] Renderer initialized (v2)');
    }

    /**
     * Load a Live2D model.
     * @param {string} modelPath - Path to the model3.json file.
     */
    async loadModel(modelPath) {
        if (!this.app) {
            console.error('[Live2D] Not initialized');
            return;
        }

        try {
            if (this.model) {
                this.app.stage.removeChild(this.model);
            }

            this.model = await PIXI.live2d.Live2DModel.from(modelPath);

            // Use internalModel dimensions for correct scaling
            // (model.width/height is unreliable right after loading)
            const modelW = this.model.internalModel.originalWidth || this.model.width;
            const modelH = this.model.internalModel.originalHeight || this.model.height;
            const scale = Math.min(
                this.app.screen.width / modelW,
                this.app.screen.height / modelH
            ) * 0.8;

            this._baseScale = scale;
            this._userZoom = 1;
            this._dragOffset = { x: 0, y: 0 };
            this.model.scale.set(scale * this._userZoom);
            this.model.x = this.app.screen.width / 2;
            this.model.y = this.app.screen.height;
            this.model.anchor.set(0.5, 1.0);

            this.model.interactive = true;
            this.model.on('pointertap', () => this._onModelTap());

            this.app.stage.addChild(this.model);

            this.isLoaded = true;
            console.log('[Live2D] Model loaded:', modelPath);

            this._startIdleAnimation();
            this._startBlinking();

        } catch (error) {
            console.error('[Live2D] Model load error:', error);
            console.info(
                '[Live2D] To use Live2D, place a model in frontend/models/ ' +
                'and update the model path. You can get models from BOOTH.'
            );
        }
    }

    /**
     * Set a Live2D expression/emotion.
     * @param {string} emotion - Emotion tag.
     */
    setEmotion(emotion) {
        if (!this.model || !this.isLoaded) return;

        const transition = this.emotionMapper.setEmotion(emotion);
        this._animateTransition(transition);

        try {
            const motionGroup = transition.data.motionGroup;
            this.model.motion(motionGroup);
        } catch (e) {
            // Motion group might not exist in the model
        }
    }

    /**
     * Notify controller that speech is starting (affects blink behavior).
     */
    setSpeaking(speaking) {
        this._isSpeaking = speaking;
    }

    /**
     * Start lip sync animation.
     * @param {number[]} volumes - Volume data from backend.
     */
    startLipSync(volumes) {
        this._isSpeaking = true;
        this.lipSync.startWithData(volumes);
    }

    /**
     * Stop lip sync animation.
     */
    stopLipSync() {
        this._isSpeaking = false;
        this.lipSync.stop();
    }

    /**
     * Make the model look at a screen position.
     */
    lookAt(x, y) {
        if (!this.model) return;

        const paramX = (x - 0.5) * 2;
        const paramY = -(y - 0.5) * 2;

        this._lookAtTarget = { x: paramX, y: paramY };

        try {
            const cm = this.model.internalModel.coreModel;
            cm.setParameterValueById('ParamAngleX', paramX * 30);
            cm.setParameterValueById('ParamAngleY', paramY * 30);
            cm.setParameterValueById('ParamBodyAngleX', paramX * 10);
            cm.setParameterValueById('ParamEyeBallX', paramX);
            cm.setParameterValueById('ParamEyeBallY', paramY);
        } catch (e) {}
    }

    // --- Private: Mouth control ---

    _setMouthParams({ mouthOpen, mouthForm }) {
        if (!this.model || !this.isLoaded) return;

        try {
            const cm = this.model.internalModel.coreModel;
            cm.setParameterValueById('ParamMouthOpenY', mouthOpen);

            // Only drive form from lip sync if not overridden by emotion
            // Blend: lip sync form + emotion form
            const emotionForm = this._activeEmotionParams.ParamMouthForm || 0;
            const blendedForm = mouthOpen > 0.05
                ? emotionForm * 0.4 + mouthForm * 0.6
                : emotionForm;
            cm.setParameterValueById('ParamMouthForm', blendedForm);
        } catch (e) {}
    }

    // --- Private: Expression transition ---

    _animateTransition(transition) {
        if (!this.model) return;

        if (this._transitionAnimation) {
            cancelAnimationFrame(this._transitionAnimation);
        }

        const startTime = performance.now();
        const duration = transition.duration;
        const fromEmotion = transition.from;
        const toEmotion = transition.to;

        const animate = () => {
            const elapsed = performance.now() - startTime;
            const progress = Math.min(elapsed / duration, 1.0);

            const params = this.emotionMapper.interpolateParams(fromEmotion, toEmotion, progress);

            try {
                const cm = this.model.internalModel.coreModel;
                for (const [key, value] of Object.entries(params)) {
                    // Skip mouth open during speech (lip sync controls it)
                    if (key === 'ParamMouthOpenY' && this._isSpeaking) continue;
                    cm.setParameterValueById(key, value);
                }
            } catch (e) {}

            // Store current emotion params for blending
            this._activeEmotionParams = params;

            if (progress < 1.0) {
                this._transitionAnimation = requestAnimationFrame(animate);
            } else {
                this._transitionAnimation = null;
            }
        };

        animate();
    }

    // --- Private: Idle animation ---

    _startIdleAnimation() {
        if (this._idleTimer) cancelAnimationFrame(this._idleTimer);

        const idleLoop = () => {
            if (!this.model || !this.isLoaded) return;

            const time = performance.now() / 1000;

            try {
                const cm = this.model.internalModel.coreModel;

                // Breathing
                const breathValue = Math.sin(time * 0.8) * 0.05 + 0.5;
                cm.setParameterValueById('ParamBreath', breathValue);

                // Subtle body sway (only when not overridden by emotion transition)
                if (!this._transitionAnimation) {
                    const swayX = Math.sin(time * 0.3) * 2;
                    const swayY = Math.cos(time * 0.5) * 1;
                    cm.setParameterValueById('ParamBodyAngleX', swayX);
                    cm.setParameterValueById('ParamBodyAngleZ', swayY);
                }
            } catch (e) {}

            this._idleTimer = requestAnimationFrame(idleLoop);
        };

        idleLoop();
    }

    // --- Private: Natural eye blinking ---

    _startBlinking() {
        if (this._blinkTimer) clearTimeout(this._blinkTimer);

        const scheduleBlink = () => {
            // Random interval: 2-6 seconds between blinks
            const interval = 2000 + Math.random() * 4000;
            this._blinkTimer = setTimeout(() => {
                this._doBlink();
                scheduleBlink();
            }, interval);
        };

        scheduleBlink();
    }

    _doBlink() {
        if (!this.model || !this.isLoaded) return;

        const blinkDuration = 120; // ms for full close
        const startTime = performance.now();

        const animateBlink = () => {
            const elapsed = performance.now() - startTime;
            const totalDuration = blinkDuration * 2; // close + open
            let eyeOpen;

            if (elapsed < blinkDuration) {
                // Closing
                eyeOpen = 1.0 - (elapsed / blinkDuration);
            } else if (elapsed < totalDuration) {
                // Opening
                eyeOpen = (elapsed - blinkDuration) / blinkDuration;
            } else {
                // Done
                eyeOpen = 1.0;
            }

            // Respect emotion's eye settings as the baseline
            const emotionEyeL = this._activeEmotionParams.ParamEyeLOpen;
            const emotionEyeR = this._activeEmotionParams.ParamEyeROpen;
            const baseL = emotionEyeL !== undefined ? emotionEyeL : 1.0;
            const baseR = emotionEyeR !== undefined ? emotionEyeR : 1.0;

            try {
                const cm = this.model.internalModel.coreModel;
                cm.setParameterValueById('ParamEyeLOpen', baseL * eyeOpen);
                cm.setParameterValueById('ParamEyeROpen', baseR * eyeOpen);
            } catch (e) {}

            if (elapsed < totalDuration) {
                requestAnimationFrame(animateBlink);
            }
        };

        animateBlink();
    }

    // --- Private: Interaction ---

    _onModelTap() {
        console.log('[Live2D] Model tapped!');
        this.setEmotion('surprised');

        setTimeout(() => {
            this.setEmotion(this.emotionMapper.currentEmotion || 'neutral');
        }, 2000);
    }

    /**
     * Handle window resize.
     */
    onResize() {
        if (!this.model || !this.app) return;

        const modelW = this.model.internalModel.originalWidth || this.model.width;
        const modelH = this.model.internalModel.originalHeight || this.model.height;
        const scale = Math.min(
            this.app.screen.width / modelW,
            this.app.screen.height / modelH
        ) * 0.8;

        this._baseScale = scale;
        this.model.scale.set(scale * this._userZoom);
        this.model.x = this.app.screen.width / 2 + this._dragOffset.x;
        this.model.y = this.app.screen.height + this._dragOffset.y;
    }

    // --- Zoom & Drag ---

    _applyTransform() {
        if (!this.model) return;
        this.model.scale.set(this._baseScale * this._userZoom);
        this.model.x = this.app.screen.width / 2 + this._dragOffset.x;
        this.model.y = this.app.screen.height + this._dragOffset.y;
    }

    _setupZoomDrag(canvas) {
        // Mouse wheel zoom
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeed = 0.001;
            this._userZoom *= 1 - e.deltaY * zoomSpeed;
            this._userZoom = Math.max(0.3, Math.min(3.0, this._userZoom));
            this._applyTransform();
        }, { passive: false });

        // Mouse drag
        canvas.addEventListener('mousedown', (e) => {
            // Middle mouse or left + alt for drag
            if (e.button === 1 || (e.button === 0 && e.altKey)) {
                this._isDragging = true;
                this._dragStart = { x: e.clientX - this._dragOffset.x, y: e.clientY - this._dragOffset.y };
                e.preventDefault();
            }
        });
        canvas.addEventListener('mousemove', (e) => {
            if (this._isDragging) {
                this._dragOffset.x = e.clientX - this._dragStart.x;
                this._dragOffset.y = e.clientY - this._dragStart.y;
                this._applyTransform();
            }
        });
        canvas.addEventListener('mouseup', () => { this._isDragging = false; });
        canvas.addEventListener('mouseleave', () => { this._isDragging = false; });

        // Touch: pinch-to-zoom + two-finger drag
        canvas.addEventListener('touchstart', (e) => {
            if (e.touches.length === 2) {
                e.preventDefault();
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                this._pinchStartDist = Math.hypot(dx, dy);
                this._pinchStartZoom = this._userZoom;
                // Center of two fingers for drag
                this._isDragging = true;
                const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
                this._dragStart = { x: cx - this._dragOffset.x, y: cy - this._dragOffset.y };
            }
        }, { passive: false });

        canvas.addEventListener('touchmove', (e) => {
            if (e.touches.length === 2) {
                e.preventDefault();
                // Pinch zoom
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                const dist = Math.hypot(dx, dy);
                if (this._pinchStartDist > 0) {
                    this._userZoom = this._pinchStartZoom * (dist / this._pinchStartDist);
                    this._userZoom = Math.max(0.3, Math.min(3.0, this._userZoom));
                }
                // Two-finger drag
                const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
                this._dragOffset.x = cx - this._dragStart.x;
                this._dragOffset.y = cy - this._dragStart.y;
                this._applyTransform();
            }
        }, { passive: false });

        canvas.addEventListener('touchend', () => {
            this._isDragging = false;
            this._pinchStartDist = 0;
        });

        // Double-click/tap to reset
        canvas.addEventListener('dblclick', (e) => {
            if (e.altKey) {
                this._userZoom = 1;
                this._dragOffset = { x: 0, y: 0 };
                this._applyTransform();
            }
        });
    }
}
