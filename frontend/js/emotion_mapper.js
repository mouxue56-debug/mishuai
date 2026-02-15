/**
 * Emotion Mapper - Maps emotion tags to Live2D expressions.
 *
 * Handles the mapping between the backend's emotion tags
 * and the Live2D model's expression/motion parameters.
 */

class EmotionMapper {
    constructor() {
        // Default emotion → Live2D parameter mapping
        // These should be adjusted based on the actual Live2D model being used
        this.emotionMap = {
            neutral: {
                expression: 'normal',
                motionGroup: 'Idle',
                emoji: '😊',
                params: {
                    ParamEyeLOpen: 1.0,
                    ParamEyeROpen: 1.0,
                    ParamMouthForm: 0.0,
                    ParamBrowLY: 0.0,
                    ParamBrowRY: 0.0,
                }
            },
            happy: {
                expression: 'smile',
                motionGroup: 'Happy',
                emoji: '😄',
                params: {
                    ParamEyeLOpen: 0.8,
                    ParamEyeROpen: 0.8,
                    ParamMouthForm: 1.0,  // Smile
                    ParamBrowLY: 0.3,
                    ParamBrowRY: 0.3,
                }
            },
            excited: {
                expression: 'excited',
                motionGroup: 'Happy',
                emoji: '🤩',
                params: {
                    ParamEyeLOpen: 1.2,
                    ParamEyeROpen: 1.2,
                    ParamMouthForm: 1.0,
                    ParamBrowLY: 0.5,
                    ParamBrowRY: 0.5,
                }
            },
            thinking: {
                expression: 'thinking',
                motionGroup: 'Thinking',
                emoji: '🤔',
                params: {
                    ParamEyeLOpen: 0.7,
                    ParamEyeROpen: 0.9,
                    ParamMouthForm: -0.3,
                    ParamBrowLY: 0.3,
                    ParamBrowRY: -0.2,
                }
            },
            surprised: {
                expression: 'surprised',
                motionGroup: 'Surprised',
                emoji: '😲',
                params: {
                    ParamEyeLOpen: 1.3,
                    ParamEyeROpen: 1.3,
                    ParamMouthForm: -0.5,
                    ParamMouthOpenY: 0.8,
                    ParamBrowLY: 0.8,
                    ParamBrowRY: 0.8,
                }
            },
            sleepy: {
                expression: 'sleepy',
                motionGroup: 'Sleepy',
                emoji: '😴',
                params: {
                    ParamEyeLOpen: 0.3,
                    ParamEyeROpen: 0.3,
                    ParamMouthForm: -0.1,
                    ParamBrowLY: -0.3,
                    ParamBrowRY: -0.3,
                }
            },
            worried: {
                expression: 'worried',
                motionGroup: 'Worried',
                emoji: '😟',
                params: {
                    ParamEyeLOpen: 0.9,
                    ParamEyeROpen: 0.9,
                    ParamMouthForm: -0.4,
                    ParamBrowLY: -0.4,
                    ParamBrowRY: 0.2,
                }
            },
            shy: {
                expression: 'shy',
                motionGroup: 'Shy',
                emoji: '😊',
                params: {
                    ParamEyeLOpen: 0.6,
                    ParamEyeROpen: 0.6,
                    ParamMouthForm: 0.3,
                    ParamBrowLY: -0.1,
                    ParamBrowRY: -0.1,
                    ParamCheek: 1.0,  // Blush
                }
            },
            angry: {
                expression: 'angry',
                motionGroup: 'Angry',
                emoji: '😠',
                params: {
                    ParamEyeLOpen: 0.9,
                    ParamEyeROpen: 0.9,
                    ParamMouthForm: -0.6,
                    ParamBrowLY: -0.6,
                    ParamBrowRY: -0.6,
                }
            },
            sad: {
                expression: 'sad',
                motionGroup: 'Sad',
                emoji: '😢',
                params: {
                    ParamEyeLOpen: 0.7,
                    ParamEyeROpen: 0.7,
                    ParamMouthForm: -0.3,
                    ParamBrowLY: -0.5,
                    ParamBrowRY: -0.3,
                }
            },
        };

        this.currentEmotion = 'neutral';
        this.transitionDuration = 500; // ms (smoother blend)
    }

    /**
     * Get the Live2D parameters for an emotion.
     * @param {string} emotion - Emotion tag.
     * @returns {object} Emotion data with params, expression, etc.
     */
    getEmotionData(emotion) {
        return this.emotionMap[emotion] || this.emotionMap.neutral;
    }

    /**
     * Get the emoji for an emotion.
     * @param {string} emotion - Emotion tag.
     * @returns {string} Emoji character.
     */
    getEmoji(emotion) {
        const data = this.getEmotionData(emotion);
        return data.emoji;
    }

    /**
     * Calculate interpolated parameters for smooth transitions.
     * @param {string} fromEmotion - Starting emotion.
     * @param {string} toEmotion - Target emotion.
     * @param {number} progress - Interpolation progress (0.0 to 1.0).
     * @returns {object} Interpolated Live2D parameters.
     */
    interpolateParams(fromEmotion, toEmotion, progress) {
        const fromData = this.getEmotionData(fromEmotion);
        const toData = this.getEmotionData(toEmotion);
        const result = {};

        // Get all parameter keys from both emotions
        const allKeys = new Set([
            ...Object.keys(fromData.params),
            ...Object.keys(toData.params),
        ]);

        for (const key of allKeys) {
            const fromVal = fromData.params[key] || 0;
            const toVal = toData.params[key] || 0;
            result[key] = fromVal + (toVal - fromVal) * this._easeInOut(progress);
        }

        return result;
    }

    /**
     * Set the current emotion with smooth transition.
     * @param {string} emotion - New emotion tag.
     * @returns {object} Target emotion data.
     */
    setEmotion(emotion) {
        const prevEmotion = this.currentEmotion;
        this.currentEmotion = emotion;

        return {
            from: prevEmotion,
            to: emotion,
            data: this.getEmotionData(emotion),
            duration: this.transitionDuration,
        };
    }

    /**
     * Ease in-out function for smooth transitions.
     * @param {number} t - Progress (0.0 to 1.0).
     * @returns {number} Eased value.
     */
    _easeInOut(t) {
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }
}
