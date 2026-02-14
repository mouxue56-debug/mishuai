/**
 * Live2D Controller
 *
 * Manages the Live2D model rendering, expression changes,
 * and parameter updates using pixi-live2d-display.
 *
 * References N.E.K.O's Live2D frontend approach.
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

        // Idle animation
        this._idleTimer = null;
        this._lookAtTarget = { x: 0, y: 0 };
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
            backgroundAlpha: 0,  // Transparent for OBS
        });

        // Set up lip sync callback
        this.lipSync.onVolumeUpdate = (value) => {
            this._setMouthOpen(value);
        };

        console.log('[Live2D] Renderer initialized');
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
            // Remove existing model
            if (this.model) {
                this.app.stage.removeChild(this.model);
            }

            // Load model using pixi-live2d-display
            this.model = await PIXI.live2d.Live2DModel.from(modelPath);

            // Scale and position
            const scale = Math.min(
                this.app.screen.width / this.model.width,
                this.app.screen.height / this.model.height
            ) * 0.8;

            this.model.scale.set(scale);
            this.model.x = this.app.screen.width / 2;
            this.model.y = this.app.screen.height;
            this.model.anchor.set(0.5, 1.0);

            // Make interactive (click reactions)
            this.model.interactive = true;
            this.model.on('pointertap', () => this._onModelTap());

            // Add to stage
            this.app.stage.addChild(this.model);

            this.isLoaded = true;
            console.log('[Live2D] Model loaded:', modelPath);

            // Start idle animation
            this._startIdleAnimation();

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

        // Also try to play a motion
        try {
            const motionGroup = transition.data.motionGroup;
            this.model.motion(motionGroup);
        } catch (e) {
            // Motion group might not exist in the model
        }
    }

    /**
     * Start lip sync animation.
     * @param {number[]} volumes - Volume data from backend.
     */
    startLipSync(volumes) {
        this.lipSync.startWithData(volumes);
    }

    /**
     * Stop lip sync animation.
     */
    stopLipSync() {
        this.lipSync.stop();
    }

    /**
     * Make the model look at a screen position.
     * @param {number} x - Screen X (0.0 = left, 1.0 = right).
     * @param {number} y - Screen Y (0.0 = top, 1.0 = bottom).
     */
    lookAt(x, y) {
        if (!this.model) return;

        // Convert to Live2D parameter range (-1 to 1)
        const paramX = (x - 0.5) * 2;
        const paramY = -(y - 0.5) * 2;

        this._lookAtTarget = { x: paramX, y: paramY };

        try {
            this.model.internalModel.coreModel.setParameterValueById('ParamAngleX', paramX * 30);
            this.model.internalModel.coreModel.setParameterValueById('ParamAngleY', paramY * 30);
            this.model.internalModel.coreModel.setParameterValueById('ParamBodyAngleX', paramX * 10);
            this.model.internalModel.coreModel.setParameterValueById('ParamEyeBallX', paramX);
            this.model.internalModel.coreModel.setParameterValueById('ParamEyeBallY', paramY);
        } catch (e) {
            // Parameters might not exist
        }
    }

    // --- Private ---

    _setMouthOpen(value) {
        if (!this.model || !this.isLoaded) return;

        try {
            const coreModel = this.model.internalModel.coreModel;
            coreModel.setParameterValueById('ParamMouthOpenY', value);
        } catch (e) {
            // Parameter might not exist
        }
    }

    _animateTransition(transition) {
        if (!this.model) return;

        // Cancel existing transition
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

            // Interpolate parameters
            const params = this.emotionMapper.interpolateParams(fromEmotion, toEmotion, progress);

            // Apply to model
            try {
                const coreModel = this.model.internalModel.coreModel;
                for (const [key, value] of Object.entries(params)) {
                    coreModel.setParameterValueById(key, value);
                }
            } catch (e) {
                // Silently handle missing parameters
            }

            if (progress < 1.0) {
                this._transitionAnimation = requestAnimationFrame(animate);
            } else {
                this._transitionAnimation = null;
            }
        };

        animate();
    }

    _startIdleAnimation() {
        // Subtle idle movements (breathing, eye blinks)
        const idleLoop = () => {
            if (!this.model || !this.isLoaded) return;

            const time = performance.now() / 1000;

            try {
                const coreModel = this.model.internalModel.coreModel;

                // Breathing
                const breathValue = Math.sin(time * 0.8) * 0.05;
                coreModel.setParameterValueById('ParamBreath', breathValue + 0.5);

                // Subtle body sway
                const swayX = Math.sin(time * 0.3) * 2;
                const swayY = Math.cos(time * 0.5) * 1;
                coreModel.setParameterValueById('ParamBodyAngleX', swayX);
                coreModel.setParameterValueById('ParamBodyAngleZ', swayY);

            } catch (e) {
                // Parameters might not exist
            }

            this._idleTimer = requestAnimationFrame(idleLoop);
        };

        idleLoop();
    }

    _onModelTap() {
        // React to being tapped
        console.log('[Live2D] Model tapped!');
        this.setEmotion('surprised');

        // Return to neutral after a moment
        setTimeout(() => {
            this.setEmotion(this.emotionMapper.currentEmotion || 'neutral');
        }, 2000);
    }

    /**
     * Handle window resize.
     */
    onResize() {
        if (!this.model || !this.app) return;

        const scale = Math.min(
            this.app.screen.width / this.model.width,
            this.app.screen.height / this.model.height
        ) * 0.8;

        this.model.scale.set(scale);
        this.model.x = this.app.screen.width / 2;
        this.model.y = this.app.screen.height;
    }
}
