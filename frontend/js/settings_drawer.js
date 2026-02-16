/**
 * Settings Drawer - Mobile-First Configuration UI (Phase 2)
 *
 * Mobile (≤768px): Bottom Sheet sliding up from bottom
 * Desktop (>768px): Right-side Drawer sliding in from right
 *
 * Tabs:
 *   Voice   — Input mode, TTS engine, VOICEVOX params
 *   Model   — Provider/model per tier, API keys, cache
 *   Memory  — Memos, reminders, history (sub-tabs)
 *   Character — Persona editing (personality, speech, trivia)
 *   System  — Log level, debug, version
 */

class SettingsDrawer {
    constructor(wsClient) {
        this.ws = wsClient;
        this.isOpen = false;
        this.activeTab = 'voice';
        this.config = {};
        this.cacheStats = null;

        // Lazy-load flags (reset on close)
        this._modelLoaded = false;
        this._memoryLoaded = false;
        this._characterLoaded = false;
        this._knowledgeLoaded = false;

        // Memory sub-tab data
        this._memos = [];
        this._reminders = {};
        this._history = {};
        this._persona = {};
        this._apiKeys = {};
        this._activeMemorySubTab = 'memos';

        // Knowledge data
        this._knowledgeFiles = [];
        this._knowledgeActiveCategory = null;
        this._knowledgeEditDirty = false;

        // Speaker data
        this._speakerLoaded = false;
        this._speakers = [];
        this._auditSessions = [];
        this._activeSpeakerSubTab = 'list';  // 'list' | 'register' | 'audit'

        // Gesture tracking
        this._touchStartY = 0;
        this._contentScrollTop = 0;
        this._isDragging = false;

        // Debounce timer for sliders
        this._debounceTimers = {};

        this._buildDOM();
        this._bindEvents();
    }

    // ================================================================
    // DOM Construction
    // ================================================================

    _buildDOM() {
        // Backdrop
        this.backdropEl = document.createElement('div');
        this.backdropEl.className = 'settings-backdrop';
        this.backdropEl.addEventListener('click', () => this.close());

        // Drawer container
        this.drawerEl = document.createElement('div');
        this.drawerEl.className = 'settings-drawer';

        // Drag handle (mobile)
        const handle = document.createElement('div');
        handle.className = 'settings-handle';
        handle.innerHTML = '<div class="settings-handle-bar"></div>';
        this.drawerEl.appendChild(handle);

        // Header
        const header = document.createElement('div');
        header.className = 'settings-header';
        header.innerHTML = `
            <span class="settings-title">設定</span>
            <button class="settings-close-btn" aria-label="Close">✕</button>
        `;
        header.querySelector('.settings-close-btn').addEventListener('click', () => this.close());
        this.drawerEl.appendChild(header);

        // Tab bar
        this._buildTabBar();

        // Content area
        this.contentEl = document.createElement('div');
        this.contentEl.className = 'settings-content';
        this.drawerEl.appendChild(this.contentEl);

        // Build tab panels
        this._buildVoicePanel();
        this._buildModelPanel();
        this._buildMemoryPanel();
        this._buildKnowledgePanel();
        this._buildSpeakerPanel();
        this._buildCharacterPanel();
        this._buildSystemPanel();

        // Add to DOM
        document.body.appendChild(this.backdropEl);
        document.body.appendChild(this.drawerEl);

        // Bind gestures
        this._bindGestures();
    }

    _buildTabBar() {
        const tabs = [
            { id: 'voice',     icon: '🎙️', label: '语音' },
            { id: 'model',     icon: '🧠', label: '模型' },
            { id: 'memory',    icon: '💾', label: '记忆' },
            { id: 'knowledge', icon: '📚', label: '知识库' },
            { id: 'speaker',   icon: '🔐', label: '声纹' },
            { id: 'character', icon: '🐱', label: '角色' },
            { id: 'system',    icon: '⚙️', label: '系统' },
        ];

        this.tabBarEl = document.createElement('div');
        this.tabBarEl.className = 'settings-tabs';

        tabs.forEach(tab => {
            const btn = document.createElement('button');
            btn.className = 'settings-tab' + (tab.id === this.activeTab ? ' active' : '');
            btn.dataset.tab = tab.id;
            btn.innerHTML = `
                <span class="settings-tab-icon">${tab.icon}</span>
                <span class="settings-tab-label">${tab.label}</span>
            `;
            btn.addEventListener('click', () => this._switchTab(tab.id));
            this.tabBarEl.appendChild(btn);
        });

        this.drawerEl.appendChild(this.tabBarEl);
    }

    // ----------------------------------------------------------------
    // Voice Tab
    // ----------------------------------------------------------------

    _buildVoicePanel() {
        const panel = this._createPanel('voice');

        // Input Mode
        panel.appendChild(this._createSectionTitle('输入模式'));
        panel.appendChild(this._createSelectFull('input-mode', '语音输入方式', [
            { value: 'ptt', label: '按键说话（点击录音）' },
            { value: 'continuous', label: '连续语音（可随时打断）' },
            { value: 'text', label: '仅文字输入' },
        ], (val) => {
            if (typeof setVoiceInputMode === 'function') {
                setVoiceInputMode(val);
            }
        }));

        // TTS Engine
        panel.appendChild(this._createSectionTitle('语音合成引擎'));
        panel.appendChild(this._createSelectFull('tts-engine', '引擎', [
            { value: 'voicevox', label: 'VOICEVOX（情感风格）' },
            { value: 'cosyvoice_dashscope', label: 'CosyVoice v3（自动情感）' },
            { value: 'edge_tts', label: 'Edge TTS（轻量）' },
        ], (val) => {
            this._applyChange('tts.japanese', 'engine', val);
            this._updateVoicevoxVisibility(val);
        }));

        // Language
        panel.appendChild(this._createSelectFull('tts-language', '语言', [
            { value: 'japanese', label: '日语' },
            { value: 'chinese', label: '中文' },
        ], (val) => this._applyChange('tts', 'active_language', val)));

        // VOICEVOX params (conditional)
        this.voicevoxSection = document.createElement('div');
        this.voicevoxSection.className = 'settings-conditional visible';
        this.voicevoxSection.id = 'voicevox-settings';

        this.voicevoxSection.appendChild(this._createSectionTitle('VOICEVOX 参数'));

        // Speaker select
        this.voicevoxSection.appendChild(this._createSelectFull('vv-character', '角色', [
            { value: 'zundamon', label: 'ずんだもん' },
            { value: 'metan', label: '四国めたん' },
        ], (val) => {
            this._applyChange('tts.japanese.voicevox', 'character', val);
            const speakerMap = { 'zundamon': 3, 'metan': 2 };
            if (speakerMap[val] !== undefined) {
                this._applyChange('tts.japanese.voicevox', 'speaker_id', speakerMap[val]);
            }
        }));

        // Speed slider
        this.voicevoxSection.appendChild(this._createSlider('vv-speed', '语速', {
            min: 0.5, max: 2.0, step: 0.1, defaultVal: 1.1,
            format: v => v.toFixed(1),
            onChange: (val) => this._applyChangeDebounced('tts.japanese.voicevox', 'speed_scale', val),
        }));

        // Pitch slider
        this.voicevoxSection.appendChild(this._createSlider('vv-pitch', '音高', {
            min: -0.15, max: 0.15, step: 0.01, defaultVal: 0.0,
            format: v => (v >= 0 ? '+' : '') + v.toFixed(2),
            onChange: (val) => this._applyChangeDebounced('tts.japanese.voicevox', 'pitch_scale', val),
        }));

        // Intonation slider
        this.voicevoxSection.appendChild(this._createSlider('vv-intonation', '语调', {
            min: 0.0, max: 2.0, step: 0.1, defaultVal: 1.2,
            format: v => v.toFixed(1),
            onChange: (val) => this._applyChangeDebounced('tts.japanese.voicevox', 'intonation_scale', val),
        }));

        panel.appendChild(this.voicevoxSection);

        // Test voice button
        panel.appendChild(this._createBtnRow([
            { label: '🔊 测试播放', variant: 'primary', onClick: () => this._testVoice() },
        ]));

        this.contentEl.appendChild(panel);
    }

    // ----------------------------------------------------------------
    // Model Tab
    // ----------------------------------------------------------------

    _buildModelPanel() {
        const panel = this._createPanel('model');

        const providerOptions = [
            { value: 'qianwen', label: '通义千问' },
            { value: 'openai', label: 'OpenAI' },
            { value: 'anthropic', label: 'Anthropic' },
            { value: 'google', label: 'Google' },
            { value: 'deepseek', label: 'DeepSeek' },
            { value: 'moonshot', label: 'Moonshot' },
        ];

        // --- Primary Model ---
        panel.appendChild(this._createSectionTitle('💬 主力模型'));

        panel.appendChild(this._createSelectFull('llm-primary-provider', '提供商', providerOptions,
            (val) => this._applyChange('llm.conversation.primary', 'provider', val)));

        panel.appendChild(this._createTextInput('llm-primary-model', '模型名称', '', (val) => {
            this._applyChange('llm.conversation.primary', 'model', val);
        }));

        panel.appendChild(this._createSlider('llm-primary-temp', '温度（越高越有创意）', {
            min: 0.0, max: 2.0, step: 0.1, defaultVal: 1.0,
            format: v => v.toFixed(1),
            onChange: (val) => this._applyChangeDebounced('llm.conversation.primary', 'temperature', val),
        }));

        panel.appendChild(this._createSlider('llm-primary-tokens', '最大 Tokens', {
            min: 256, max: 4096, step: 128, defaultVal: 1024,
            format: v => Math.round(v).toString(),
            onChange: (val) => this._applyChangeDebounced('llm.conversation.primary', 'max_tokens', Math.round(val)),
        }));

        // --- Fallback Model ---
        panel.appendChild(this._createSectionTitle('🔄 备用模型'));

        panel.appendChild(this._createSelectFull('llm-fallback-provider', '提供商', providerOptions,
            (val) => this._applyChange('llm.conversation.fallback', 'provider', val)));

        panel.appendChild(this._createTextInput('llm-fallback-model', '模型名称', '', (val) => {
            this._applyChange('llm.conversation.fallback', 'model', val);
        }));

        // --- Tool / Summary info ---
        panel.appendChild(this._createSectionTitle('其他模型'));
        this.modelInfoContainer = document.createElement('div');
        this.modelInfoContainer.id = 'model-info-container';
        panel.appendChild(this.modelInfoContainer);

        // API Keys section
        panel.appendChild(this._createSectionTitle('API 密钥'));
        this.apiKeysContainer = document.createElement('div');
        this.apiKeysContainer.id = 'api-keys-container';
        this.apiKeysContainer.innerHTML = '<div class="memory-empty">加载中...</div>';
        panel.appendChild(this.apiKeysContainer);

        // Cache section
        panel.appendChild(this._createSectionTitle('缓存'));

        this.cacheStatsEl = document.createElement('div');
        this.cacheStatsEl.className = 'setting-info';
        this.cacheStatsEl.innerHTML = `
            <div class="setting-info-label">缓存统计</div>
            <div class="setting-info-value" id="cache-stats-value">加载中...</div>
        `;
        panel.appendChild(this.cacheStatsEl);

        panel.appendChild(this._createBtnRow([
            { label: '🗑️ 清除缓存', variant: 'danger', onClick: () => this._clearCache() },
        ]));

        this.contentEl.appendChild(panel);
    }

    // ----------------------------------------------------------------
    // Memory Tab
    // ----------------------------------------------------------------

    _buildMemoryPanel() {
        const panel = this._createPanel('memory');

        // Sub-tabs
        const subTabs = document.createElement('div');
        subTabs.className = 'settings-sub-tabs';
        ['memos', 'reminders', 'history'].forEach(id => {
            const labels = { memos: '备忘录', reminders: '提醒', history: '历史' };
            const btn = document.createElement('button');
            btn.className = 'settings-sub-tab' + (id === 'memos' ? ' active' : '');
            btn.dataset.subtab = id;
            btn.textContent = labels[id];
            btn.addEventListener('click', () => this._switchMemorySubTab(id));
            subTabs.appendChild(btn);
        });
        panel.appendChild(subTabs);

        // Sub-tab content containers
        this.memosPanel = document.createElement('div');
        this.memosPanel.dataset.subtabPanel = 'memos';
        this.memosPanel.style.display = 'block';
        panel.appendChild(this.memosPanel);

        this.remindersPanel = document.createElement('div');
        this.remindersPanel.dataset.subtabPanel = 'reminders';
        this.remindersPanel.style.display = 'none';
        panel.appendChild(this.remindersPanel);

        this.historyPanel = document.createElement('div');
        this.historyPanel.dataset.subtabPanel = 'history';
        this.historyPanel.style.display = 'none';
        panel.appendChild(this.historyPanel);

        this.contentEl.appendChild(panel);
    }

    _switchMemorySubTab(id) {
        this._activeMemorySubTab = id;
        const panel = this.contentEl.querySelector('[data-panel="memory"]');
        panel.querySelectorAll('.settings-sub-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.subtab === id);
        });
        panel.querySelectorAll('[data-subtab-panel]').forEach(p => {
            p.style.display = p.dataset.subtabPanel === id ? 'block' : 'none';
        });

        // Lazy-load data for sub-tab
        if (id === 'memos') this.ws.sendCommand('get_memos', {});
        if (id === 'reminders') this.ws.sendCommand('get_reminders', { include_completed: true });
        if (id === 'history') this.ws.sendCommand('get_history', { limit: 10 });
    }

    _renderMemos(memos) {
        this._memos = memos || [];
        const container = this.memosPanel;
        container.innerHTML = '';

        // Add new memo button
        const addRow = document.createElement('div');
        addRow.className = 'setting-btn-row';
        const addBtn = document.createElement('button');
        addBtn.className = 'setting-btn setting-btn-primary';
        addBtn.textContent = '+ 新建备忘录';
        addBtn.addEventListener('click', () => this._showNewMemoForm());
        addRow.appendChild(addBtn);
        container.appendChild(addRow);

        // New memo form placeholder
        this.newMemoFormEl = document.createElement('div');
        this.newMemoFormEl.id = 'new-memo-form';
        container.appendChild(this.newMemoFormEl);

        if (memos.length === 0) {
            container.appendChild(this._createEmpty('暂无备忘录'));
            return;
        }

        memos.forEach(memo => {
            container.appendChild(this._createMemoCard(memo));
        });
    }

    _createMemoCard(memo) {
        const card = document.createElement('div');
        card.className = 'memory-card';
        card.dataset.memoId = memo.id;

        const category = document.createElement('span');
        category.className = 'memory-card-category';
        category.textContent = memo.category || 'general';
        card.appendChild(category);

        const content = document.createElement('div');
        content.className = 'memory-card-content';
        content.textContent = memo.content || '';
        card.appendChild(content);

        const meta = document.createElement('div');
        meta.className = 'memory-card-meta';
        meta.textContent = memo.created_at ? new Date(memo.created_at).toLocaleString('ja-JP') : '';
        card.appendChild(meta);

        const actions = document.createElement('div');
        actions.className = 'memory-card-actions';

        const editBtn = document.createElement('button');
        editBtn.textContent = '✏️ 编辑';
        editBtn.addEventListener('click', () => this._editMemo(card, memo));
        actions.appendChild(editBtn);

        const archiveBtn = document.createElement('button');
        archiveBtn.textContent = '🗑️ 归档';
        archiveBtn.className = 'btn-delete';
        archiveBtn.addEventListener('click', () => {
            this.ws.sendCommand('archive_memo', { memo_id: memo.id });
            card.style.opacity = '0.3';
        });
        actions.appendChild(archiveBtn);

        card.appendChild(actions);
        return card;
    }

    _showNewMemoForm() {
        const el = this.newMemoFormEl;
        if (el.children.length > 0) { el.innerHTML = ''; return; }

        const form = document.createElement('div');
        form.className = 'memory-inline-form';

        const textarea = document.createElement('textarea');
        textarea.placeholder = '输入备忘录内容...';
        form.appendChild(textarea);

        const catInput = document.createElement('input');
        catInput.placeholder = '分类（例：general）';
        catInput.value = 'general';
        form.appendChild(catInput);

        const btns = document.createElement('div');
        btns.className = 'memory-inline-form-actions';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'setting-btn setting-btn-secondary';
        cancelBtn.textContent = '取消';
        cancelBtn.addEventListener('click', () => { el.innerHTML = ''; });
        btns.appendChild(cancelBtn);

        const saveBtn = document.createElement('button');
        saveBtn.className = 'setting-btn setting-btn-primary';
        saveBtn.textContent = '保存';
        saveBtn.addEventListener('click', () => {
            const txt = textarea.value.trim();
            if (!txt) return;
            this.ws.sendCommand('save_memo', { content: txt, category: catInput.value.trim() || 'general' });
            el.innerHTML = '';
        });
        btns.appendChild(saveBtn);

        form.appendChild(btns);
        el.appendChild(form);
    }

    _editMemo(card, memo) {
        card.classList.add('editing');
        const contentEl = card.querySelector('.memory-card-content');
        const oldText = contentEl.textContent;
        const textarea = document.createElement('textarea');
        textarea.className = 'setting-textarea';
        textarea.value = oldText;
        contentEl.replaceWith(textarea);

        const actionsEl = card.querySelector('.memory-card-actions');
        actionsEl.innerHTML = '';

        const saveBtn = document.createElement('button');
        saveBtn.textContent = '💾 保存';
        saveBtn.addEventListener('click', () => {
            this.ws.sendCommand('update_memo', {
                memo_id: memo.id,
                content: textarea.value.trim(),
            });
            card.classList.remove('editing');
            const newContent = document.createElement('div');
            newContent.className = 'memory-card-content';
            newContent.textContent = textarea.value.trim();
            textarea.replaceWith(newContent);
            // Refresh
            this.ws.sendCommand('get_memos', {});
        });
        actionsEl.appendChild(saveBtn);

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = '取消';
        cancelBtn.addEventListener('click', () => {
            // Refresh to reset state
            this.ws.sendCommand('get_memos', {});
        });
        actionsEl.appendChild(cancelBtn);
    }

    _renderReminders(data) {
        this._reminders = data;
        const container = this.remindersPanel;
        container.innerHTML = '';

        // Add new reminder button
        const addRow = document.createElement('div');
        addRow.className = 'setting-btn-row';
        const addBtn = document.createElement('button');
        addBtn.className = 'setting-btn setting-btn-primary';
        addBtn.textContent = '+ 新建提醒';
        addBtn.addEventListener('click', () => this._showNewReminderForm());
        addRow.appendChild(addBtn);
        container.appendChild(addRow);

        this.newReminderFormEl = document.createElement('div');
        this.newReminderFormEl.id = 'new-reminder-form';
        container.appendChild(this.newReminderFormEl);

        const pending = data.pending || [];
        const upcoming = data.upcoming || [];
        const completed = data.completed || [];

        if (pending.length > 0) {
            container.appendChild(this._createSectionTitle('⚠️ 已过期'));
            pending.forEach(r => container.appendChild(this._createReminderCard(r, 'pending')));
        }

        if (upcoming.length > 0) {
            container.appendChild(this._createSectionTitle('📅 即将到来'));
            upcoming.forEach(r => container.appendChild(this._createReminderCard(r, 'upcoming')));
        }

        if (completed.length > 0) {
            container.appendChild(this._createSectionTitle('✅ 已完成'));
            completed.forEach(r => container.appendChild(this._createReminderCard(r, 'completed')));
        }

        if (pending.length === 0 && upcoming.length === 0 && completed.length === 0) {
            container.appendChild(this._createEmpty('暂无提醒'));
        }
    }

    _createReminderCard(reminder, status) {
        const card = document.createElement('div');
        card.className = 'memory-card';
        card.dataset.reminderId = reminder.id;

        const content = document.createElement('div');
        content.className = 'memory-card-content';
        content.textContent = reminder.content || '';
        card.appendChild(content);

        const meta = document.createElement('div');
        meta.className = 'memory-card-meta';
        const remindAt = reminder.remind_at ? new Date(reminder.remind_at).toLocaleString('ja-JP') : '';
        const repeat = reminder.repeat && reminder.repeat !== 'none' ? ` (${reminder.repeat})` : '';
        meta.textContent = `${remindAt}${repeat}`;
        card.appendChild(meta);

        const actions = document.createElement('div');
        actions.className = 'memory-card-actions';

        if (status !== 'completed') {
            const completeBtn = document.createElement('button');
            completeBtn.textContent = '✅ 完成';
            completeBtn.className = 'btn-complete';
            completeBtn.addEventListener('click', () => {
                this.ws.sendCommand('complete_reminder', { reminder_id: reminder.id });
                card.style.opacity = '0.3';
            });
            actions.appendChild(completeBtn);
        }

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = '🗑️ 删除';
        deleteBtn.className = 'btn-delete';
        deleteBtn.addEventListener('click', () => {
            this.ws.sendCommand('delete_reminder', { reminder_id: reminder.id });
            card.style.opacity = '0.3';
        });
        actions.appendChild(deleteBtn);

        card.appendChild(actions);
        return card;
    }

    _showNewReminderForm() {
        const el = this.newReminderFormEl;
        if (el.children.length > 0) { el.innerHTML = ''; return; }

        const form = document.createElement('div');
        form.className = 'memory-inline-form';

        const textarea = document.createElement('textarea');
        textarea.placeholder = '提醒内容...';
        form.appendChild(textarea);

        const dateInput = document.createElement('input');
        dateInput.type = 'datetime-local';
        form.appendChild(dateInput);

        const repeatSelect = document.createElement('select');
        [
            { value: 'none', label: '不重复' },
            { value: 'daily', label: '每天' },
            { value: 'weekly', label: '每周' },
        ].forEach(opt => {
            const o = document.createElement('option');
            o.value = opt.value; o.textContent = opt.label;
            repeatSelect.appendChild(o);
        });
        form.appendChild(repeatSelect);

        const btns = document.createElement('div');
        btns.className = 'memory-inline-form-actions';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'setting-btn setting-btn-secondary';
        cancelBtn.textContent = '取消';
        cancelBtn.addEventListener('click', () => { el.innerHTML = ''; });
        btns.appendChild(cancelBtn);

        const saveBtn = document.createElement('button');
        saveBtn.className = 'setting-btn setting-btn-primary';
        saveBtn.textContent = '保存';
        saveBtn.addEventListener('click', () => {
            const txt = textarea.value.trim();
            const dt = dateInput.value;
            if (!txt || !dt) { this._showToast('请输入内容和时间'); return; }
            this.ws.sendCommand('save_reminder', {
                content: txt,
                remind_at: new Date(dt).toISOString(),
                repeat: repeatSelect.value,
            });
            el.innerHTML = '';
        });
        btns.appendChild(saveBtn);

        form.appendChild(btns);
        el.appendChild(form);
    }

    _renderHistory(data) {
        this._history = data;
        const container = this.historyPanel;
        container.innerHTML = '';

        // Current session
        const current = data.current_session || [];
        if (current.length > 0) {
            container.appendChild(this._createSectionTitle(`当前会话（${current.length}轮）`));
            current.forEach(msg => {
                const el = document.createElement('div');
                el.className = 'history-message';
                const role = msg.role === 'user' ? 'USER' : 'ASST';
                const time = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' }) : '';
                el.innerHTML = `<span class="role">${role}</span><span class="content">${this._escapeHtml(msg.content || '')}</span>${time ? `<span class="time">${time}</span>` : ''}`;
                container.appendChild(el);
            });
        }

        // Past sessions
        const past = data.past_sessions || [];
        if (past.length > 0) {
            container.appendChild(this._createSectionTitle('历史会话'));
            past.forEach(session => {
                const el = document.createElement('div');
                el.className = 'history-session';
                const date = session.date || session.session_date || '';
                const count = session.message_count || session.turn_count || '?';
                el.innerHTML = `<span class="date">${date}</span> — ${count} 条消息`;
                container.appendChild(el);
            });
        }

        if (current.length === 0 && past.length === 0) {
            container.appendChild(this._createEmpty('暂无历史记录'));
        }

        // Clear button
        container.appendChild(this._createBtnRow([
            { label: '🗑️ 清除当前会话', variant: 'danger', onClick: () => {
                this.ws.sendCommand('clear_short_term', {});
                this._showToast('已清除会话历史');
            }},
        ]));
    }

    // ----------------------------------------------------------------
    // Knowledge Tab
    // ----------------------------------------------------------------

    _buildKnowledgePanel() {
        const panel = this._createPanel('knowledge');

        // File list area
        this.knowledgeListEl = document.createElement('div');
        this.knowledgeListEl.id = 'knowledge-file-list';
        this.knowledgeListEl.innerHTML = '<div class="memory-empty">加载中...</div>';
        panel.appendChild(this.knowledgeListEl);

        // Editor area (hidden by default)
        this.knowledgeEditorEl = document.createElement('div');
        this.knowledgeEditorEl.id = 'knowledge-editor';
        this.knowledgeEditorEl.style.display = 'none';
        panel.appendChild(this.knowledgeEditorEl);

        this.contentEl.appendChild(panel);
    }

    _renderKnowledgeFiles(files) {
        this._knowledgeFiles = files || [];
        const container = this.knowledgeListEl;
        container.innerHTML = '';

        container.appendChild(this._createSectionTitle('知识库文件'));

        const desc = document.createElement('div');
        desc.className = 'knowledge-description';
        desc.textContent = 'AI客服使用这些文件回答客户问题。编辑后自动重建搜索索引。';
        container.appendChild(desc);

        if (files.length === 0) {
            container.appendChild(this._createEmpty('没有知识库文件'));
            return;
        }

        files.forEach(file => {
            const card = document.createElement('div');
            card.className = 'knowledge-card';
            card.addEventListener('click', () => this._openKnowledgeEditor(file.category));

            const icon = document.createElement('span');
            icon.className = 'knowledge-card-icon';
            icon.textContent = file.category === 'faq' ? '❓' :
                               file.category === 'breed' ? '🐱' :
                               file.category === 'customer' ? '📋' : '🏥';
            card.appendChild(icon);

            const info = document.createElement('div');
            info.className = 'knowledge-card-info';

            const title = document.createElement('div');
            title.className = 'knowledge-card-title';
            title.textContent = file.label;
            info.appendChild(title);

            const meta = document.createElement('div');
            meta.className = 'knowledge-card-meta';
            meta.textContent = `${file.description} · ${this._formatBytes(file.size)}`;
            info.appendChild(meta);

            card.appendChild(info);

            const arrow = document.createElement('span');
            arrow.className = 'knowledge-card-arrow';
            arrow.textContent = '▶';
            card.appendChild(arrow);

            container.appendChild(card);
        });

        // Rebuild index button
        container.appendChild(this._createBtnRow([
            { label: '🔄 重建搜索索引', variant: 'secondary', onClick: () => {
                this.ws.sendCommand('rebuild_knowledge_index', {});
                this._showToast('索引重建中...');
            }},
        ]));
    }

    _openKnowledgeEditor(category) {
        this._knowledgeActiveCategory = category;
        this._knowledgeEditDirty = false;

        // Hide file list, show editor
        this.knowledgeListEl.style.display = 'none';
        this.knowledgeEditorEl.style.display = 'block';

        // Show loading
        this.knowledgeEditorEl.innerHTML = '<div class="memory-empty">加载中...</div>';

        // Request content
        this.ws.sendCommand('get_knowledge_content', { category });
    }

    _renderKnowledgeEditor(data) {
        const container = this.knowledgeEditorEl;
        container.innerHTML = '';

        const file = this._knowledgeFiles.find(f => f.category === data.category);
        const label = file ? file.label : data.category;

        // Header with back button
        const header = document.createElement('div');
        header.className = 'knowledge-editor-header';

        const backBtn = document.createElement('button');
        backBtn.className = 'knowledge-back-btn';
        backBtn.textContent = '← 返回';
        backBtn.addEventListener('click', () => this._closeKnowledgeEditor());
        header.appendChild(backBtn);

        const titleEl = document.createElement('span');
        titleEl.className = 'knowledge-editor-title';
        titleEl.textContent = `编辑: ${label}`;
        header.appendChild(titleEl);

        container.appendChild(header);

        // Filename info
        const filenameEl = document.createElement('div');
        filenameEl.className = 'knowledge-editor-filename';
        filenameEl.textContent = `📄 ${data.filename}`;
        container.appendChild(filenameEl);

        // Textarea
        const textarea = document.createElement('textarea');
        textarea.className = 'knowledge-textarea';
        textarea.value = data.content || '';
        textarea.spellcheck = false;
        textarea.addEventListener('input', () => {
            this._knowledgeEditDirty = true;
            saveBtn.classList.add('dirty');
            saveBtn.textContent = '💾 保存 *';
        });
        container.appendChild(textarea);

        // Status bar
        const statusBar = document.createElement('div');
        statusBar.className = 'knowledge-status-bar';

        const charCount = document.createElement('span');
        charCount.className = 'knowledge-char-count';
        charCount.textContent = `${(data.content || '').length} 字符`;
        textarea.addEventListener('input', () => {
            charCount.textContent = `${textarea.value.length} 字符`;
        });
        statusBar.appendChild(charCount);

        container.appendChild(statusBar);

        // Action buttons
        const btnRow = document.createElement('div');
        btnRow.className = 'setting-btn-row';

        const saveBtn = document.createElement('button');
        saveBtn.className = 'setting-btn setting-btn-primary';
        saveBtn.textContent = '💾 保存';
        saveBtn.addEventListener('click', () => {
            this.ws.sendCommand('save_knowledge_content', {
                category: data.category,
                content: textarea.value,
            });
            saveBtn.classList.remove('dirty');
            saveBtn.textContent = '💾 保存';
            this._knowledgeEditDirty = false;
        });
        btnRow.appendChild(saveBtn);

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'setting-btn setting-btn-secondary';
        cancelBtn.textContent = '取消';
        cancelBtn.addEventListener('click', () => this._closeKnowledgeEditor());
        btnRow.appendChild(cancelBtn);

        container.appendChild(btnRow);

        // Focus textarea
        setTimeout(() => textarea.focus(), 100);
    }

    _closeKnowledgeEditor() {
        if (this._knowledgeEditDirty) {
            if (!confirm('有未保存的修改，确定离开吗？')) return;
        }
        this.knowledgeEditorEl.style.display = 'none';
        this.knowledgeListEl.style.display = 'block';
        this._knowledgeActiveCategory = null;
        this._knowledgeEditDirty = false;
    }

    _formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    // ----------------------------------------------------------------
    // Character Tab
    // ----------------------------------------------------------------

    _buildCharacterPanel() {
        const panel = this._createPanel('character');
        this.characterContent = document.createElement('div');
        this.characterContent.id = 'character-content';
        this.characterContent.innerHTML = '<div class="memory-empty">加载中...</div>';
        panel.appendChild(this.characterContent);
        this.contentEl.appendChild(panel);
    }

    _renderPersona(persona) {
        this._persona = persona;
        const container = this.characterContent;
        container.innerHTML = '';

        const char = persona.character || {};

        // Basic info
        container.appendChild(this._createSectionTitle('基本信息'));

        container.appendChild(this._createTextInput('persona-name', '名称', char.name || '', (val) => {
            this.ws.sendCommand('set_persona', { updates: [{ path: 'character.name', value: val }] });
        }));

        container.appendChild(this._createTextInput('persona-role', '角色', char.role || '', (val) => {
            this.ws.sendCommand('set_persona', { updates: [{ path: 'character.role', value: val }] });
        }));

        container.appendChild(this._createSelectFull('persona-gender', '性别', [
            { value: 'female', label: '女性' },
            { value: 'male', label: '男性' },
            { value: 'neutral', label: '中性' },
        ], (val) => {
            this.ws.sendCommand('set_persona', { updates: [{ path: 'character.gender', value: val }] });
        }));

        // Personality traits (editable list)
        container.appendChild(this._createSectionTitle('性格特征'));
        container.appendChild(this._createEditableList(
            'persona-personality',
            char.personality || [],
            (newList) => {
                this.ws.sendCommand('set_persona', { updates: [{ path: 'character.personality', value: newList }] });
            }
        ));

        // Speech patterns (grouped by emotion)
        container.appendChild(this._createSectionTitle('口头禅'));
        const speechPatterns = char.speech_patterns || {};
        Object.entries(speechPatterns).forEach(([emotion, phrases]) => {
            const section = this._createCollapsible(emotion, () => {});
            const body = section.querySelector('.setting-collapsible-body');
            body.appendChild(this._createEditableList(
                `speech-${emotion}`,
                Array.isArray(phrases) ? phrases : [],
                (newList) => {
                    const updated = { ...speechPatterns, [emotion]: newList };
                    this.ws.sendCommand('set_persona', { updates: [{ path: 'character.speech_patterns', value: updated }] });
                }
            ));
            container.appendChild(section);
        });

        // Cat trivia
        container.appendChild(this._createSectionTitle('猫咪冷知识'));
        container.appendChild(this._createEditableList(
            'persona-trivia',
            char.cat_trivia || [],
            (newList) => {
                this.ws.sendCommand('set_persona', { updates: [{ path: 'character.cat_trivia', value: newList }] });
            }
        ));
    }

    // ----------------------------------------------------------------
    // System Tab
    // ----------------------------------------------------------------

    _buildSystemPanel() {
        const panel = this._createPanel('system');

        panel.appendChild(this._createSectionTitle('通用'));

        // Log level
        panel.appendChild(this._createSelectFull('log-level', '日志级别', [
            { value: 'DEBUG', label: 'DEBUG（详细）' },
            { value: 'INFO', label: 'INFO（正常）' },
            { value: 'WARNING', label: 'WARNING（仅警告）' },
            { value: 'ERROR', label: 'ERROR（仅错误）' },
        ], (val) => this._applyChange('logging', 'level', val)));

        // Debug toggle
        panel.appendChild(this._createToggleRow('debug-mode', '调试模式',
            '启用详细日志输出', false,
            (val) => this._applyChange('app', 'debug', val)));

        // Version info
        panel.appendChild(this._createSectionTitle('信息'));

        this.versionInfoEl = document.createElement('div');
        this.versionInfoEl.className = 'setting-info';
        this.versionInfoEl.innerHTML = `
            <div class="setting-info-label">版本</div>
            <div class="setting-info-value" id="version-info">v0.3.0</div>
        `;
        panel.appendChild(this.versionInfoEl);

        this.contentEl.appendChild(panel);
    }

    // ================================================================
    // Control Builders (reusable)
    // ================================================================

    _createPanel(id) {
        const panel = document.createElement('div');
        panel.className = 'settings-panel' + (id === this.activeTab ? ' active' : '');
        panel.dataset.panel = id;
        return panel;
    }

    _createSectionTitle(text) {
        const el = document.createElement('div');
        el.className = 'settings-section-title';
        el.textContent = text;
        return el;
    }

    _createSelectFull(id, label, options, onChange) {
        const container = document.createElement('div');
        container.className = 'setting-select-full';

        const labelEl = document.createElement('label');
        labelEl.className = 'setting-label';
        labelEl.textContent = label;
        labelEl.setAttribute('for', `setting-${id}`);
        container.appendChild(labelEl);

        const select = document.createElement('select');
        select.className = 'setting-select';
        select.id = `setting-${id}`;
        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            select.appendChild(option);
        });
        select.addEventListener('change', () => onChange(select.value));
        container.appendChild(select);

        return container;
    }

    _createSlider(id, label, opts) {
        const group = document.createElement('div');
        group.className = 'setting-slider-group';

        const header = document.createElement('div');
        header.className = 'setting-slider-header';

        const labelEl = document.createElement('span');
        labelEl.className = 'setting-slider-label';
        labelEl.textContent = label;

        const valueEl = document.createElement('span');
        valueEl.className = 'setting-slider-value';
        valueEl.id = `setting-${id}-value`;
        valueEl.textContent = opts.format(opts.defaultVal);

        header.appendChild(labelEl);
        header.appendChild(valueEl);
        group.appendChild(header);

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'setting-slider';
        slider.id = `setting-${id}`;
        slider.min = opts.min;
        slider.max = opts.max;
        slider.step = opts.step;
        slider.value = opts.defaultVal;

        slider.addEventListener('input', () => {
            valueEl.textContent = opts.format(parseFloat(slider.value));
        });
        slider.addEventListener('change', () => {
            opts.onChange(parseFloat(slider.value));
        });

        group.appendChild(slider);
        return group;
    }

    _createToggleRow(id, label, sublabel, defaultVal, onChange) {
        const row = document.createElement('div');
        row.className = 'setting-row';

        const labelGroup = document.createElement('div');
        labelGroup.className = 'setting-label-group';
        labelGroup.innerHTML = `
            <div class="setting-label">${label}</div>
            ${sublabel ? `<div class="setting-sublabel">${sublabel}</div>` : ''}
        `;

        const toggle = document.createElement('label');
        toggle.className = 'setting-toggle';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = `setting-${id}`;
        input.checked = defaultVal;
        input.addEventListener('change', () => onChange(input.checked));

        const slider = document.createElement('span');
        slider.className = 'setting-toggle-slider';

        toggle.appendChild(input);
        toggle.appendChild(slider);

        row.appendChild(labelGroup);
        row.appendChild(toggle);
        return row;
    }

    _createBtnRow(buttons) {
        const row = document.createElement('div');
        row.className = 'setting-btn-row';
        buttons.forEach(btn => {
            const el = document.createElement('button');
            el.className = `setting-btn setting-btn-${btn.variant || 'secondary'}`;
            el.textContent = btn.label;
            el.addEventListener('click', btn.onClick);
            row.appendChild(el);
        });
        return row;
    }

    _createTextInput(id, label, defaultVal, onChange) {
        const container = document.createElement('div');
        container.className = 'setting-text-input';

        const labelEl = document.createElement('label');
        labelEl.className = 'setting-label';
        labelEl.textContent = label;
        labelEl.setAttribute('for', `setting-${id}`);
        container.appendChild(labelEl);

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'setting-input';
        input.id = `setting-${id}`;
        input.value = defaultVal;

        let timer;
        input.addEventListener('input', () => {
            clearTimeout(timer);
            timer = setTimeout(() => onChange(input.value), 500);
        });

        container.appendChild(input);
        return container;
    }

    _createEditableList(id, items, onUpdate) {
        const wrapper = document.createElement('div');
        wrapper.dataset.listId = id;

        const renderItems = () => {
            wrapper.innerHTML = '';
            items.forEach((item, i) => {
                const row = document.createElement('div');
                row.className = 'setting-list-item';

                const input = document.createElement('input');
                input.type = 'text';
                input.value = item;
                let timer;
                input.addEventListener('input', () => {
                    items[i] = input.value;
                    clearTimeout(timer);
                    timer = setTimeout(() => onUpdate([...items]), 500);
                });
                row.appendChild(input);

                const delBtn = document.createElement('button');
                delBtn.className = 'setting-list-delete';
                delBtn.textContent = '✕';
                delBtn.addEventListener('click', () => {
                    items.splice(i, 1);
                    onUpdate([...items]);
                    renderItems();
                });
                row.appendChild(delBtn);

                wrapper.appendChild(row);
            });

            const addBtn = document.createElement('button');
            addBtn.className = 'setting-list-add';
            addBtn.textContent = '+ 添加';
            addBtn.addEventListener('click', () => {
                items.push('');
                renderItems();
                // Focus the new input
                const inputs = wrapper.querySelectorAll('input');
                if (inputs.length > 0) inputs[inputs.length - 1].focus();
            });
            wrapper.appendChild(addBtn);
        };

        renderItems();
        return wrapper;
    }

    _createCollapsible(title, onExpand) {
        const section = document.createElement('div');

        const header = document.createElement('div');
        header.className = 'setting-collapsible-header';
        header.innerHTML = `
            <span class="setting-label">${title}</span>
            <span class="setting-collapsible-arrow">▶</span>
        `;

        const body = document.createElement('div');
        body.className = 'setting-collapsible-body';

        header.addEventListener('click', () => {
            header.classList.toggle('open');
            body.classList.toggle('open');
            if (body.classList.contains('open')) onExpand();
        });

        section.appendChild(header);
        section.appendChild(body);
        return section;
    }

    _createEmpty(text) {
        const el = document.createElement('div');
        el.className = 'memory-empty';
        el.textContent = text;
        return el;
    }

    // ================================================================
    // API Keys Rendering
    // ================================================================

    _renderApiKeys(keys) {
        this._apiKeys = keys;
        const container = this.apiKeysContainer;
        container.innerHTML = '';

        Object.entries(keys).forEach(([provider, info]) => {
            const row = document.createElement('div');
            row.className = 'api-key-row';

            // Status dot
            const dot = document.createElement('span');
            dot.className = `api-key-status ${info.is_set ? 'set' : 'unset'}`;
            row.appendChild(dot);

            // Provider label
            const label = document.createElement('span');
            label.className = 'api-key-provider';
            label.textContent = info.label || provider;
            row.appendChild(label);

            // Masked key
            const masked = document.createElement('span');
            masked.className = 'api-key-masked';
            masked.textContent = info.is_set ? info.masked : '（未设置）';
            row.appendChild(masked);

            // Toggle button
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'api-key-btn';
            toggleBtn.textContent = '设置';
            row.appendChild(toggleBtn);

            // Input row (hidden by default)
            const inputWrap = document.createElement('div');
            inputWrap.className = 'api-key-input-wrap';

            const input = document.createElement('input');
            input.type = 'password';
            input.className = 'api-key-input';
            input.placeholder = `${info.env_var || provider}`;
            inputWrap.appendChild(input);

            const saveBtn = document.createElement('button');
            saveBtn.className = 'api-key-btn';
            saveBtn.textContent = '保存';
            saveBtn.addEventListener('click', () => {
                const key = input.value.trim();
                if (!key) return;
                this.ws.sendCommand('set_api_key', { provider, key });
                input.value = '';
                inputWrap.classList.remove('visible');
            });
            inputWrap.appendChild(saveBtn);

            toggleBtn.addEventListener('click', () => {
                inputWrap.classList.toggle('visible');
                if (inputWrap.classList.contains('visible')) input.focus();
            });

            row.appendChild(inputWrap);
            container.appendChild(row);
        });
    }

    // ================================================================
    // Model Info Rendering
    // ================================================================

    _renderModelInfo(config) {
        const container = this.modelInfoContainer;
        container.innerHTML = '';

        const llm = config.llm || {};

        const providerNames = {
            anthropic: 'Anthropic', openai: 'OpenAI', google: 'Google',
            moonshot: 'Moonshot', deepseek: 'DeepSeek', qianwen: '通义千问',
        };

        const tool = llm.tool_use?.primary || {};
        const summ = llm.summarization?.primary || {};

        const fmtModel = (cfg) => {
            if (!cfg.provider && !cfg.model) return '未设置';
            const p = providerNames[cfg.provider] || cfg.provider || '?';
            return `${p} / ${cfg.model || '?'}`;
        };

        const card = document.createElement('div');
        card.className = 'model-info-card';
        card.innerHTML = `
            <div style="margin-bottom:4px;">🔧 <strong>工具调用</strong>：${fmtModel(tool)}</div>
            <div>📝 <strong>摘要</strong>：${fmtModel(summ)}</div>
        `;
        container.appendChild(card);
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    open() {
        this.isOpen = true;
        this.backdropEl.classList.add('visible');
        this.drawerEl.classList.add('open');

        const btn = document.getElementById('btn-settings');
        if (btn) btn.classList.add('active');

        // Fetch current config from backend
        this._fetchConfig();
    }

    close() {
        this.isOpen = false;
        this.backdropEl.classList.remove('visible');
        this.drawerEl.classList.remove('open');
        this.drawerEl.style.transform = '';

        const btn = document.getElementById('btn-settings');
        if (btn) btn.classList.remove('active');

        // Reset lazy-load flags
        this._modelLoaded = false;
        this._memoryLoaded = false;
        this._characterLoaded = false;
        this._knowledgeLoaded = false;
    }

    toggle() {
        if (this.isOpen) this.close();
        else this.open();
    }

    // ================================================================
    // Tab Navigation
    // ================================================================

    _switchTab(tabId) {
        this.activeTab = tabId;

        // Update tab buttons
        this.tabBarEl.querySelectorAll('.settings-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });

        // Update panels
        this.contentEl.querySelectorAll('.settings-panel').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.panel === tabId);
        });

        // Scroll content to top
        this.contentEl.scrollTop = 0;

        // Lazy-load data for tabs
        if (tabId === 'model' && !this._modelLoaded) {
            this._modelLoaded = true;
            this.ws.sendCommand('get_api_key_status', {});
        }
        if (tabId === 'memory' && !this._memoryLoaded) {
            this._memoryLoaded = true;
            this._switchMemorySubTab(this._activeMemorySubTab);
        }
        if (tabId === 'knowledge' && !this._knowledgeLoaded) {
            this._knowledgeLoaded = true;
            this.ws.sendCommand('get_knowledge_files', {});
        }
        if (tabId === 'character' && !this._characterLoaded) {
            this._characterLoaded = true;
            this.ws.sendCommand('get_persona', {});
        }
        if (tabId === 'speaker' && !this._speakerLoaded) {
            this._speakerLoaded = true;
            this.ws.sendCommand('get_speakers', {});
            this.ws.sendCommand('get_audit_sessions', {});
        }

        try { localStorage.setItem('settings_tab', tabId); } catch (e) {}
    }

    // ================================================================
    // Config Sync
    // ================================================================

    _fetchConfig() {
        this.ws.sendCommand('get_config', { sections: null });
    }

    /** Called when backend returns config_data. */
    onConfigData(data) {
        this.config = data.config || {};
        this.cacheStats = data.cache_stats || null;
        this._populateControls();
    }

    /** Called when backend returns config_update_result. */
    onConfigUpdateResult(data) {
        if (data.success) {
            console.log('[Settings] Config updated:', data.applied);
        } else {
            console.warn('[Settings] Config update errors:', data.errors);
            if (data.errors && data.errors.length > 0) {
                const errMsg = data.errors.map(e => `${e.key}: ${e.error}`).join(', ');
                this._showToast(`设置错误: ${errMsg}`);
            }
        }
    }

    /** Called when backend returns api_key_status. */
    onApiKeyStatus(data) {
        if (data.success && data.keys) {
            this._renderApiKeys(data.keys);
        }
    }

    /** Called when backend returns api_key_update_result. */
    onApiKeyUpdateResult(data) {
        if (data.success) {
            this._showToast('API密钥已保存');
            // Refresh API key status
            this.ws.sendCommand('get_api_key_status', {});
        } else {
            this._showToast(`错误: ${data.error || 'unknown'}`);
        }
    }

    /** Called when backend returns memos_data. */
    onMemosData(data) {
        if (data.success) {
            this._renderMemos(data.memos || []);
        }
    }

    /** Called on memo_saved / memo_updated / memo_archived. */
    onMemoResult(data) {
        if (data.success) {
            this._showToast('备忘录已更新');
            this.ws.sendCommand('get_memos', {});
        } else {
            this._showToast(`错误: ${data.error || 'unknown'}`);
        }
    }

    /** Called when backend returns reminders_data. */
    onRemindersData(data) {
        if (data.success) {
            this._renderReminders(data);
        }
    }

    /** Called on reminder_saved / reminder_completed / reminder_deleted. */
    onReminderResult(data) {
        if (data.success) {
            this._showToast('提醒已更新');
            this.ws.sendCommand('get_reminders', { include_completed: true });
        } else {
            this._showToast(`错误: ${data.error || 'unknown'}`);
        }
    }

    /** Called when backend returns history_data. */
    onHistoryData(data) {
        if (data.success) {
            this._renderHistory(data);
        }
    }

    /** Called when backend returns short_term_cleared. */
    onShortTermCleared(data) {
        if (data.success) {
            this._showToast('已清除会话历史');
            this.ws.sendCommand('get_history', { limit: 10 });
        }
    }

    /** Called when backend returns knowledge_files. */
    onKnowledgeFiles(data) {
        if (data.success) {
            this._renderKnowledgeFiles(data.files || []);
        }
    }

    /** Called when backend returns knowledge_content. */
    onKnowledgeContent(data) {
        if (data.success) {
            this._renderKnowledgeEditor(data);
        } else {
            this._showToast(`错误: ${data.error || 'unknown'}`);
        }
    }

    /** Called when backend returns knowledge_save_result. */
    onKnowledgeSaveResult(data) {
        if (data.success) {
            this._showToast('知识库已保存，索引将自动重建');
            // Refresh file list to update sizes
            this.ws.sendCommand('get_knowledge_files', {});
        } else {
            this._showToast(`保存失败: ${data.error || 'unknown'}`);
        }
    }

    /** Called when backend returns knowledge_rebuild_result. */
    onKnowledgeRebuildResult(data) {
        if (data.success) {
            this._showToast('索引已重建');
        } else {
            this._showToast(`重建失败: ${data.error || 'unknown'}`);
        }
    }

    /** Called when backend returns persona_data. */
    onPersonaData(data) {
        if (data.success && data.persona) {
            this._renderPersona(data.persona);
        }
    }

    /** Called when backend returns persona_update_result. */
    onPersonaUpdateResult(data) {
        if (data.success) {
            this._showToast('角色设定已更新');
        } else {
            this._showToast(`错误: ${data.error || 'unknown'}`);
        }
    }

    _populateControls() {
        const c = this.config;

        // --- Voice tab ---
        const ttsJp = c.tts?.japanese || {};
        this._setSelect('tts-engine', ttsJp.engine || 'voicevox');
        this._setSelect('tts-language', c.tts?.active_language || 'japanese');
        this._updateVoicevoxVisibility(ttsJp.engine || 'voicevox');

        const vv = ttsJp.voicevox || {};
        this._setSelect('vv-character', vv.character || 'zundamon');
        this._setSlider('vv-speed', vv.speed_scale ?? 1.1);
        this._setSlider('vv-pitch', vv.pitch_scale ?? 0.0);
        this._setSlider('vv-intonation', vv.intonation_scale ?? 1.2);

        // Input mode (if the global is set)
        if (typeof voiceInputMode !== 'undefined') {
            this._setSelect('input-mode', voiceInputMode);
        }

        // --- Model tab ---
        const convPrimary = c.llm?.conversation?.primary || {};
        const convFallback = c.llm?.conversation?.fallback || {};
        this._setSelect('llm-primary-provider', convPrimary.provider || 'qianwen');
        this._setTextInput('llm-primary-model', convPrimary.model || '');
        this._setSlider('llm-primary-temp', convPrimary.temperature ?? 1.0);
        this._setSlider('llm-primary-tokens', convPrimary.max_tokens ?? 1024);
        this._setSelect('llm-fallback-provider', convFallback.provider || 'qianwen');
        this._setTextInput('llm-fallback-model', convFallback.model || '');
        this._renderModelInfo(c);

        // Cache stats
        if (this.cacheStats) {
            const stats = this.cacheStats;
            const statsEl = document.getElementById('cache-stats-value');
            if (statsEl) {
                statsEl.textContent =
                    `命中: ${stats.hits || 0} / 未命中: ${stats.misses || 0} ` +
                    `(${stats.hit_rate || '0%'}) · 容量: ${stats.size || 0}/${stats.maxsize || 128}`;
            }
        }

        // --- System tab ---
        this._setSelect('log-level', c.logging?.level || 'INFO');
        this._setToggle('debug-mode', c.app?.debug === true);

        const versionEl = document.getElementById('version-info');
        if (versionEl && c.app?.version) {
            versionEl.textContent = `v${c.app.version}`;
        }

        // Restore last active tab
        try {
            const savedTab = localStorage.getItem('settings_tab');
            if (savedTab) this._switchTab(savedTab);
        } catch (e) {}
    }

    // ================================================================
    // Config Change Helpers
    // ================================================================

    _applyChange(section, key, value) {
        this.ws.sendCommand('set_config', {
            updates: [{ section, key, value }],
        });
    }

    _applyChangeDebounced(section, key, value, delay = 300) {
        const timerId = `${section}.${key}`;
        clearTimeout(this._debounceTimers[timerId]);
        this._debounceTimers[timerId] = setTimeout(() => {
            this._applyChange(section, key, value);
        }, delay);
    }

    // ================================================================
    // Actions
    // ================================================================

    _testVoice() {
        const testPhrases = [
            'テスト音声です。今の設定はこんな感じですよ〜',
            'こんにちは！音声テストです。聞こえますか？',
            'にゃ〜ん、テストですよ〜',
        ];
        const phrase = testPhrases[Math.floor(Math.random() * testPhrases.length)];
        this.ws.sendTextInput(phrase);
        this._showToast('已发送测试语音');
    }

    _clearCache() {
        this.ws.sendCommand('clear_cache');
        const statsEl = document.getElementById('cache-stats-value');
        if (statsEl) {
            statsEl.textContent = '命中: 0 / 未命中: 0 (0%) · 容量: 0/128';
        }
        this._showToast('缓存已清除');
    }

    // ================================================================
    // UI Helpers
    // ================================================================

    _setSelect(id, value) {
        const el = document.getElementById(`setting-${id}`);
        if (el) el.value = value;
    }

    _setSlider(id, value) {
        const el = document.getElementById(`setting-${id}`);
        const valueEl = document.getElementById(`setting-${id}-value`);
        if (el) {
            el.value = value;
            if (valueEl) {
                const step = parseFloat(el.step) || 0.1;
                if (step < 0.1) {
                    valueEl.textContent = (value >= 0 ? '+' : '') + value.toFixed(2);
                } else if (step >= 1) {
                    valueEl.textContent = Math.round(value).toString();
                } else {
                    valueEl.textContent = value.toFixed(1);
                }
            }
        }
    }

    _setToggle(id, value) {
        const el = document.getElementById(`setting-${id}`);
        if (el) el.checked = value;
    }

    _setTextInput(id, value) {
        const el = document.getElementById(`setting-${id}`);
        if (el) el.value = value;
    }

    _updateVoicevoxVisibility(engine) {
        if (this.voicevoxSection) {
            if (engine === 'voicevox') {
                this.voicevoxSection.classList.add('visible');
            } else {
                this.voicevoxSection.classList.remove('visible');
            }
        }
    }

    _showToast(message) {
        let toast = document.getElementById('settings-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'settings-toast';
            toast.style.cssText = `
                position: fixed;
                bottom: 100px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 10px 20px;
                border-radius: 20px;
                font-size: 14px;
                z-index: 600;
                opacity: 0;
                transition: opacity 0.3s;
                pointer-events: none;
            `;
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.style.opacity = '1';
        setTimeout(() => { toast.style.opacity = '0'; }, 2000);
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ================================================================
    // Gesture Handling (swipe-down to close on mobile)
    // ================================================================

    _bindGestures() {
        const drawer = this.drawerEl;
        const handle = drawer.querySelector('.settings-handle');
        const dragTarget = handle || drawer;

        dragTarget.addEventListener('touchstart', (e) => {
            this._touchStartY = e.touches[0].clientY;
            this._contentScrollTop = this.contentEl.scrollTop;
            this._isDragging = false;
        }, { passive: true });

        dragTarget.addEventListener('touchmove', (e) => {
            const dy = e.touches[0].clientY - this._touchStartY;
            if (this._contentScrollTop <= 0 && dy > 0) {
                this._isDragging = true;
                const dampedDy = Math.min(dy * 0.6, 200);
                drawer.style.transform = `translateY(${dampedDy}px)`;
                drawer.style.transition = 'none';
                e.preventDefault();
            }
        }, { passive: false });

        dragTarget.addEventListener('touchend', (e) => {
            if (!this._isDragging) return;
            const dy = e.changedTouches[0].clientY - this._touchStartY;
            drawer.style.transition = '';
            drawer.style.transform = '';
            if (dy > 100) this.close();
            this._isDragging = false;
        });

        this.contentEl.addEventListener('touchstart', (e) => {
            this._touchStartY = e.touches[0].clientY;
            this._contentScrollTop = this.contentEl.scrollTop;
            this._isDragging = false;
        }, { passive: true });

        this.contentEl.addEventListener('touchmove', (e) => {
            const dy = e.touches[0].clientY - this._touchStartY;
            if (this._contentScrollTop <= 0 && dy > 10) {
                this._isDragging = true;
                const dampedDy = Math.min(dy * 0.6, 200);
                drawer.style.transform = `translateY(${dampedDy}px)`;
                drawer.style.transition = 'none';
            }
        }, { passive: true });

        this.contentEl.addEventListener('touchend', (e) => {
            if (!this._isDragging) return;
            const dy = e.changedTouches[0].clientY - this._touchStartY;
            drawer.style.transition = '';
            drawer.style.transform = '';
            if (dy > 100) this.close();
            this._isDragging = false;
        });
    }

    // ================================================================
    // Speaker Tab
    // ================================================================

    _buildSpeakerPanel() {
        const panel = document.createElement('div');
        panel.className = 'settings-panel';
        panel.dataset.panel = 'speaker';

        panel.innerHTML = `
            <div class="settings-section">
                <div class="settings-sub-tabs">
                    <button class="settings-sub-tab active" data-subtab="list">已注册</button>
                    <button class="settings-sub-tab" data-subtab="register">注册</button>
                    <button class="settings-sub-tab" data-subtab="audit">审计</button>
                </div>

                <div class="speaker-subtab-content" data-content="list">
                    <div class="speaker-list-container">
                        <div class="setting-label" style="text-align:center;color:var(--text-secondary);padding:20px">
                            加载中...
                        </div>
                    </div>
                </div>

                <div class="speaker-subtab-content" data-content="register" style="display:none">
                    <div class="speaker-register-form">
                        <div class="setting-row">
                            <label class="setting-label">名前 / Name</label>
                            <input type="text" class="setting-text-input" id="sp-display-name" placeholder="例: 田中太郎">
                        </div>
                        <div class="setting-row">
                            <label class="setting-label">Speaker ID</label>
                            <input type="text" class="setting-text-input" id="sp-id" placeholder="例: tanaka (英字小文字)">
                        </div>
                        <div class="setting-row">
                            <label class="setting-label">角色</label>
                            <select class="setting-select" id="sp-role">
                                <option value="visitor">来客 (visitor)</option>
                                <option value="staff">スタッフ (staff)</option>
                                <option value="boss">ボス (boss)</option>
                            </select>
                        </div>
                        <div class="setting-row">
                            <label class="setting-label">言語</label>
                            <select class="setting-select" id="sp-language">
                                <option value="japanese">日本語</option>
                                <option value="chinese">中文</option>
                                <option value="english">English</option>
                                <option value="mixed">混合</option>
                            </select>
                        </div>
                        <div class="setting-row">
                            <label class="setting-label">备注</label>
                            <input type="text" class="setting-text-input" id="sp-notes" placeholder="自由备注">
                        </div>
                        <div class="setting-row">
                            <label class="setting-label">邀请人</label>
                            <select class="setting-select" id="sp-invited-by">
                                <option value="">（无需邀请 / 首位注册）</option>
                            </select>
                        </div>
                        <div class="setting-row">
                            <label class="setting-label">关系</label>
                            <input type="text" class="setting-text-input" id="sp-relationship" placeholder="例: スタッフ、友人">
                        </div>

                        <div class="setting-section-title" style="margin-top:16px">🎤 录音采集（可选）</div>
                        <div class="setting-label" style="color:var(--text-secondary);font-size:12px;margin-bottom:8px">
                            录制至少3段自然语音，每段2秒以上。用于声纹识别。
                        </div>
                        <div class="voice-samples-container">
                            <div class="voice-sample" data-index="0">
                                <span class="voice-sample-label">采样 1</span>
                                <span class="voice-sample-status">未录制</span>
                                <button class="voice-sample-btn">🎤 录制</button>
                            </div>
                            <div class="voice-sample" data-index="1">
                                <span class="voice-sample-label">采样 2</span>
                                <span class="voice-sample-status">未录制</span>
                                <button class="voice-sample-btn">🎤 录制</button>
                            </div>
                            <div class="voice-sample" data-index="2">
                                <span class="voice-sample-label">采样 3</span>
                                <span class="voice-sample-status">未录制</span>
                                <button class="voice-sample-btn">🎤 录制</button>
                            </div>
                        </div>

                        <div style="display:flex;gap:8px;margin-top:16px">
                            <button class="setting-btn" id="sp-register-btn" style="flex:1">注册</button>
                            <button class="setting-btn" id="sp-cancel-btn" style="flex:1;background:var(--bg-tertiary)">取消</button>
                        </div>
                    </div>
                </div>

                <div class="speaker-subtab-content" data-content="audit" style="display:none">
                    <div class="audit-sessions-container">
                        <div class="setting-label" style="text-align:center;color:var(--text-secondary);padding:20px">
                            加载中...
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.contentEl.appendChild(panel);

        // Sub-tab switching
        panel.querySelectorAll('.settings-sub-tab').forEach(btn => {
            btn.addEventListener('click', () => {
                this._switchSpeakerSubTab(btn.dataset.subtab);
            });
        });

        // Voice recording buttons
        this._voiceSamples = [null, null, null]; // base64 audio data
        this._mediaRecorder = null;
        this._recordingChunks = [];

        panel.querySelectorAll('.voice-sample-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.closest('.voice-sample').dataset.index);
                this._toggleVoiceRecording(idx);
            });
        });

        // Register button
        panel.querySelector('#sp-register-btn').addEventListener('click', () => {
            this._submitSpeakerRegistration();
        });

        // Cancel button
        panel.querySelector('#sp-cancel-btn').addEventListener('click', () => {
            this._switchSpeakerSubTab('list');
        });
    }

    _switchSpeakerSubTab(subtab) {
        this._activeSpeakerSubTab = subtab;

        const panel = this.contentEl.querySelector('[data-panel="speaker"]');
        if (!panel) return;

        // Update sub-tab buttons
        panel.querySelectorAll('.settings-sub-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.subtab === subtab);
        });

        // Show/hide content
        panel.querySelectorAll('.speaker-subtab-content').forEach(el => {
            el.style.display = el.dataset.content === subtab ? '' : 'none';
        });

        // Populate inviter dropdown when switching to register
        if (subtab === 'register') {
            this._populateInviterDropdown();
        }
    }

    _populateInviterDropdown() {
        const select = document.getElementById('sp-invited-by');
        if (!select) return;

        // Keep first empty option, add speakers
        select.innerHTML = '<option value="">（无需邀请 / 首位注册）</option>';
        this._speakers.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = `${s.display_name} (${s.role})`;
            select.appendChild(opt);
        });
    }

    _renderSpeakerList(speakers) {
        this._speakers = speakers;
        const container = this.contentEl.querySelector('.speaker-list-container');
        if (!container) return;

        if (!speakers.length) {
            container.innerHTML = `
                <div style="text-align:center;color:var(--text-secondary);padding:30px">
                    <div style="font-size:40px;margin-bottom:12px">🔇</div>
                    <div>尚无注册声纹</div>
                    <div style="font-size:12px;margin-top:4px">点击「注册」添加第一位用户</div>
                </div>
            `;
            return;
        }

        const roleIcons = { boss: '👑', staff: '🧑‍💼', visitor: '👤' };
        const roleColors = { boss: '#f59e0b', staff: '#3b82f6', visitor: '#6b7280' };

        container.innerHTML = speakers.map(s => `
            <div class="speaker-card" data-id="${s.id}">
                <div class="speaker-card-header">
                    <span class="speaker-card-name">${s.display_name}</span>
                    <span class="role-badge" style="background:${roleColors[s.role] || '#6b7280'}">
                        ${roleIcons[s.role] || '👤'} ${s.role}
                    </span>
                </div>
                <div class="speaker-card-meta">
                    <span>ID: ${s.id}</span>
                    <span>言語: ${s.language}</span>
                    ${s.invited_by ? `<span>邀请: ${s.invited_by}</span>` : ''}
                    ${s.relationship ? `<span>关系: ${s.relationship}</span>` : ''}
                    ${s.notes ? `<span>备注: ${s.notes}</span>` : ''}
                </div>
                <div class="speaker-card-footer">
                    <span class="speaker-card-date">${s.registered_at || ''}</span>
                    <div class="speaker-card-actions">
                        <button class="speaker-action-btn" data-action="reenroll" data-id="${s.id}" title="声紋再登録">🔄</button>
                        <button class="speaker-action-btn" data-action="delete" data-id="${s.id}" title="削除">🗑️</button>
                    </div>
                </div>
            </div>
        `).join('');

        // Bind reenroll buttons
        container.querySelectorAll('[data-action="reenroll"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.dataset.id;
                this._startReenrollRecording(id, btn);
            });
        });

        // Bind delete buttons
        container.querySelectorAll('[data-action="delete"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.dataset.id;
                if (confirm(`确定删除 ${id} 的声纹？`)) {
                    this.ws.sendCommand('delete_speaker', { speaker_id: id });
                }
            });
        });
    }

    async _startReenrollRecording(speakerId, btn) {
        const prompts = [
            '今日はいい天気ですね。猫たちも元気です。',
            'サイベリアンは大きくてふわふわで、とても可愛い猫です。',
            '福楽キャッテリーへようこそ。素敵な猫ちゃんがお待ちしています。',
        ];
        const origText = btn.textContent;
        btn.disabled = true;
        btn.textContent = '🎙️ 录音中...';

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1 }
            });
            const samples = [];

            for (let i = 0; i < 3; i++) {
                btn.textContent = `🎙️ (${i+1}/3) 请读...`;
                // Show prompt to user
                const promptDiv = document.createElement('div');
                promptDiv.style.cssText = 'position:fixed;top:20%;left:50%;transform:translateX(-50%);background:#1a1a2e;color:#fff;padding:20px 30px;border-radius:12px;z-index:99999;font-size:16px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.4);';
                promptDiv.innerHTML = `<div style="font-size:12px;color:#aaa;margin-bottom:8px">声纹采样 ${i+1}/3</div><div>${prompts[i]}</div><div style="font-size:12px;color:#aaa;margin-top:8px">请大声朗读上面的句子</div>`;
                document.body.appendChild(promptDiv);

                // Record 4 seconds
                const chunks = [];
                const rec = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                rec.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
                rec.start();
                await new Promise(r => setTimeout(r, 4000));
                await new Promise(r => { rec.onstop = r; rec.stop(); });
                document.body.removeChild(promptDiv);

                const blob = new Blob(chunks, { type: 'audio/webm' });
                // Convert to base64
                const b64 = await new Promise(r => {
                    const reader = new FileReader();
                    reader.onloadend = () => r(reader.result.split(',')[1]);
                    reader.readAsDataURL(blob);
                });
                samples.push(b64);

                // Brief pause between samples
                if (i < 2) await new Promise(r => setTimeout(r, 500));
            }

            stream.getTracks().forEach(t => t.stop());

            // Send to backend
            btn.textContent = '⏳ 处理中...';
            this.ws.sendCommand('reenroll_speaker', {
                speaker_id: speakerId,
                audio_samples: samples,
            });

            // Wait a bit and show result
            setTimeout(() => {
                btn.disabled = false;
                btn.textContent = origText;
                // Refresh speaker list
                this.ws.sendCommand('get_speakers', {});
            }, 3000);

        } catch (e) {
            console.error('[Settings] Reenroll recording error:', e);
            btn.disabled = false;
            btn.textContent = origText;
            alert('录音失败: ' + e.message);
        }
    }

    _renderAuditSessions(sessions) {
        this._auditSessions = sessions;
        const container = this.contentEl.querySelector('.audit-sessions-container');
        if (!container) return;

        if (!sessions.length) {
            container.innerHTML = `
                <div style="text-align:center;color:var(--text-secondary);padding:30px">
                    <div style="font-size:40px;margin-bottom:12px">📝</div>
                    <div>暂无对话记录</div>
                </div>
            `;
            return;
        }

        container.innerHTML = sessions.map(s => `
            <div class="audit-session-card" data-session-id="${s.id}">
                <div class="audit-session-header">
                    <span class="audit-session-speaker">${s.speaker_name}</span>
                    <span class="audit-session-count">${s.turn_count} turns</span>
                </div>
                <div class="audit-session-meta">
                    ${s.started_at || 'N/A'}
                    ${s.ended_at ? ` → ${s.ended_at}` : ' (进行中)'}
                </div>
                ${s.summary ? `<div class="audit-session-summary">${s.summary}</div>` : ''}
                <button class="setting-btn" style="margin-top:8px;font-size:12px;padding:4px 8px"
                        onclick="window._settingsDrawer && window._settingsDrawer._loadSessionDetail('${s.id}')">
                    查看详情
                </button>
            </div>
        `).join('');
    }

    _loadSessionDetail(sessionId) {
        this.ws.sendCommand('get_session_detail', { session_id: sessionId });
    }

    _renderSessionDetail(data) {
        if (!data.success) return;
        const turns = data.turns || [];
        const container = this.contentEl.querySelector('.audit-sessions-container');
        if (!container) return;

        let html = `
            <button class="knowledge-back-btn" id="audit-back-btn">← 返回列表</button>
            <div class="setting-section-title">Session: ${data.session_id.substring(0, 8)}...</div>
        `;

        turns.forEach(t => {
            const role = t.intent === 'chat' ? '💬' : '🔧';
            html += `
                <div class="audit-turn">
                    <div class="audit-turn-header">
                        <span>${role} ${t.created_at || ''}</span>
                        <span class="audit-turn-emotion">${t.emotion || 'neutral'}</span>
                    </div>
                    <div class="audit-turn-input">🗣️ ${this._escapeHtml(t.input_text || '')}</div>
                    <div class="audit-turn-response">🤖 ${this._escapeHtml(t.response_text || '')}</div>
                </div>
            `;
        });

        container.innerHTML = html;

        // Back button
        const backBtn = container.querySelector('#audit-back-btn');
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                this._renderAuditSessions(this._auditSessions);
            });
        }
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // --- Voice Recording ---
    async _toggleVoiceRecording(sampleIndex) {
        const sampleEl = this.contentEl.querySelector(`.voice-sample[data-index="${sampleIndex}"]`);
        if (!sampleEl) return;
        const btn = sampleEl.querySelector('.voice-sample-btn');
        const status = sampleEl.querySelector('.voice-sample-status');

        if (this._mediaRecorder && this._mediaRecorder.state === 'recording') {
            // Stop recording
            this._mediaRecorder.stop();
            btn.textContent = '🎤 录制';
            sampleEl.classList.remove('recording');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1 }
            });

            // Determine mime type
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            this._mediaRecorder = new MediaRecorder(stream, { mimeType });
            this._recordingChunks = [];
            const startTime = Date.now();

            this._mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this._recordingChunks.push(e.data);
            };

            this._mediaRecorder.onstop = async () => {
                const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                const blob = new Blob(this._recordingChunks, { type: mimeType });

                // Convert to base64
                const buffer = await blob.arrayBuffer();
                const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
                this._voiceSamples[sampleIndex] = base64;

                status.textContent = `✅ ${duration}s`;
                sampleEl.classList.add('done');
                sampleEl.classList.remove('recording');

                // Stop tracks
                stream.getTracks().forEach(t => t.stop());
            };

            this._mediaRecorder.start();
            btn.textContent = '⏹️ 停止';
            status.textContent = '录制中...';
            sampleEl.classList.add('recording');

            // Auto-stop after 10 seconds
            setTimeout(() => {
                if (this._mediaRecorder && this._mediaRecorder.state === 'recording') {
                    this._mediaRecorder.stop();
                    btn.textContent = '🎤 录制';
                }
            }, 10000);

        } catch (err) {
            console.error('Microphone access failed:', err);
            status.textContent = '❌ 麦克风无法访问';
        }
    }

    _submitSpeakerRegistration() {
        const displayName = document.getElementById('sp-display-name')?.value?.trim();
        const speakerId = document.getElementById('sp-id')?.value?.trim()?.toLowerCase();
        const role = document.getElementById('sp-role')?.value || 'visitor';
        const language = document.getElementById('sp-language')?.value || 'japanese';
        const notes = document.getElementById('sp-notes')?.value?.trim() || '';
        const invitedBy = document.getElementById('sp-invited-by')?.value || '';
        const relationship = document.getElementById('sp-relationship')?.value?.trim() || '';

        if (!displayName || !speakerId) {
            this._showToast('名前とSpeaker IDは必須です');
            return;
        }

        // Validate speaker ID format
        if (!/^[a-z0-9_-]+$/.test(speakerId)) {
            this._showToast('Speaker IDは英字小文字・数字・ハイフンのみ');
            return;
        }

        // Collect valid audio samples
        const audioSamples = this._voiceSamples.filter(s => s !== null);

        const data = {
            speaker_id: speakerId,
            display_name: displayName,
            role,
            language,
            notes,
            invited_by: invitedBy || undefined,
            relationship,
            audio_samples: audioSamples.length > 0 ? audioSamples : undefined,
        };

        this.ws.sendCommand('register_speaker', data);
        this._showToast('注册中...');
    }

    // --- Speaker WS Handlers ---

    onSpeakersData(data) {
        if (data.success && data.speakers) {
            this._renderSpeakerList(data.speakers);
        }
    }

    onSpeakerRegistered(data) {
        if (data.success) {
            this._showToast(`注册成功: ${data.speaker?.display_name || ''}`);
            // Reset form
            ['sp-display-name', 'sp-id', 'sp-notes', 'sp-relationship'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = '';
            });
            this._voiceSamples = [null, null, null];
            this.contentEl.querySelectorAll('.voice-sample').forEach(el => {
                el.classList.remove('done', 'recording');
                el.querySelector('.voice-sample-status').textContent = '未录制';
            });
            this._switchSpeakerSubTab('list');
        } else {
            this._showToast(`注册失败: ${data.error || 'unknown'}`);
        }
    }

    onSpeakerUpdated(data) {
        if (data.success) {
            this._showToast('更新成功');
        } else {
            this._showToast(`更新失败: ${data.error || 'unknown'}`);
        }
    }

    onSpeakerDeleted(data) {
        if (data.success) {
            this._showToast('已删除');
        } else {
            this._showToast(`删除失败: ${data.error || 'unknown'}`);
        }
    }

    onAuditSessionsData(data) {
        if (data.success && data.sessions) {
            this._renderAuditSessions(data.sessions);
        }
    }

    onSessionDetailData(data) {
        this._renderSessionDetail(data);
    }

    // ================================================================
    // Event Binding
    // ================================================================

    _bindEvents() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
        });
    }
}
