/**
 * DocuMind — Production Frontend Script
 * Handles: PDF upload/drag-drop, resizer, tabs, chat, notes, theme toggle, math rendering
 */

// ── KaTeX math rendering helper ──────────────────────────────────────────────
// Renders LaTeX inside a given DOM element using KaTeX auto-render.
// Supports: $inline$, $$block$$, \(inline\), \[block\]
function renderMath(element) {
    if (typeof renderMathInElement !== 'function') return;
    try {
        renderMathInElement(element, {
            delimiters: [
                { left: '$$',  right: '$$',  display: true  },
                { left: '$',   right: '$',   display: false },
                { left: '\\[', right: '\\]', display: true  },
                { left: '\\(', right: '\\)', display: false },
            ],
            throwOnError: false,   // degrade gracefully on bad LaTeX
            output: 'html',
        });
    } catch (e) {
        console.warn('KaTeX render error:', e);
    }
}

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM References ──────────────────────────────────────────────────────
    const el = {
        // Header
        headerStatus:       document.getElementById('headerStatus'),
        statusDot:          document.querySelector('.status-dot'),
        statusText:         document.querySelector('.status-text'),
        themeToggle:        document.getElementById('themeToggle'),
        themeIconSun:       document.getElementById('themeIconSun'),
        themeIconMoon:      document.getElementById('themeIconMoon'),
        settingsBtn:        document.getElementById('settingsBtn'),
        apiKeyModal:        document.getElementById('apiKeyModal'),
        apiKeyValue:        document.getElementById('apiKeyValue'),
        apiModalCancel:     document.getElementById('apiModalCancel'),
        apiModalSave:       document.getElementById('apiModalSave'),

        // Left Pane / PDF
        leftPane:           document.getElementById('leftPane'),
        uploadOverlay:      document.getElementById('uploadOverlay'),
        dropZone:           document.getElementById('dropZone'),
        pdfUploadOverlay:   document.getElementById('pdfUploadOverlay'),
        pdfViewerWrapper:   document.getElementById('pdfViewerWrapper'),
        pdfViewer:          document.getElementById('pdfViewer'),
        pdfDocName:         document.getElementById('pdfDocName'),
        pdfUploadMain:      document.getElementById('pdfUploadMain'),

        // Right Pane
        rightPane:          document.getElementById('rightPane'),
        resizer:            document.getElementById('dragResizer'),

        // Upload status
        uploadProgressBar:  document.getElementById('uploadProgressBar'),
        progressFill:       document.getElementById('progressFill'),
        uploadStatus:       document.getElementById('uploadStatus'),

        // Chat
        chatHistory:        document.getElementById('chatHistory'),
        chatEmptyState:     document.getElementById('chatEmptyState'),
        userInput:          document.getElementById('userInput'),
        askButton:          document.getElementById('askButton'),

        // Notes
        savedNotesList:     document.getElementById('savedNotesList'),
        notesEmptyState:    document.getElementById('notesEmptyState'),
        downloadAllNotes:   document.getElementById('downloadAllNotes'),
        notesBadge:         document.getElementById('notesBadge'),

        // Modal
        noteModal:          document.getElementById('noteModal'),
        noteNameInput:      document.getElementById('noteNameInput'),
        modalCancel:        document.getElementById('modalCancel'),
        modalSave:          document.getElementById('modalSave'),
    };

    // ── API Key Management ────────────────────────────────────────────────────
    function initApiKey() {
        const savedKey = localStorage.getItem('docuMindApiKey');
        if (savedKey) {
            el.apiKeyValue.value = savedKey;
        } else {
            // Prompt user on first load
            el.apiKeyModal.style.display = 'flex';
        }

        el.settingsBtn.addEventListener('click', () => {
            const currentKey = localStorage.getItem('docuMindApiKey');
            el.apiKeyValue.value = currentKey || '';
            el.apiKeyModal.style.display = 'flex';
        });

        el.apiModalCancel.addEventListener('click', () => {
            el.apiKeyModal.style.display = 'none';
        });

        el.apiModalSave.addEventListener('click', () => {
            const key = el.apiKeyValue.value.trim();
            if (key) {
                localStorage.setItem('docuMindApiKey', key);
                el.apiKeyModal.style.display = 'none';
            } else {
                alert('Please enter a valid API Key.');
            }
        });
    }

    // ── State ───────────────────────────────────────────────────────────────
    let isResizing = false;
    let pendingSaveNote = null;       // { query, response } — set when modal opens

    // ── Markdown / Sanitise Utilities ───────────────────────────────────────
    window.marked = window.marked || { parse: t => t };

    const md = {
        render:  (text) => DOMPurify.sanitize(window.marked.parse(text || '')),
        strip:   (text) => text
            .replace(/(^#+\s+)/gm, '')
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/\*(.*?)\*/g, '$1')
            .replace(/`{3}.*\n/g, '')
            .replace(/\[(.*?)\]\(.*?\)/g, '$1')
            .replace(/~~(.*?)~~/g, '$1'),
        restore: (text) => text
            .replace(/(\r\n|\n|\r)/g, '\n\n')
            .replace(/^\s*-\s+/gm, '- '),
    };

    // ── Theme ───────────────────────────────────────────────────────────────
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('docuMindTheme', theme);
        const isDark = theme === 'dark';
        el.themeIconSun.style.display  = isDark ? '' : 'none';
        el.themeIconMoon.style.display = isDark ? 'none' : '';
    }

    el.themeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        applyTheme(current === 'dark' ? 'light' : 'dark');
    });

    // Init theme
    applyTheme(localStorage.getItem('docuMindTheme') || 'dark');

    // ── Header Status ────────────────────────────────────────────────────────
    function setStatus(state, text) {
        el.statusDot.className = `status-dot ${state}`;
        el.statusText.textContent = text;
    }

    // ── Upload Status Bar ────────────────────────────────────────────────────
    function setUploadMsg(text, type = '') {
        el.uploadStatus.textContent = text;
        el.uploadStatus.className = `upload-status-msg ${type}`;
    }

    function showProgress(pct) {
        el.uploadProgressBar.style.display = 'block';
        el.progressFill.style.width = `${pct}%`;
    }

    function hideProgress() {
        setTimeout(() => {
            el.uploadProgressBar.style.display = 'none';
            el.progressFill.style.width = '0%';
        }, 600);
    }

    // ── PDF Handling ─────────────────────────────────────────────────────────
    async function uploadPDF(file) {
        if (!file || file.type !== 'application/pdf') {
            setUploadMsg('Please select a valid PDF file.', 'error');
            return;
        }
        if (file.size > 50 * 1024 * 1024) {
            setUploadMsg('File too large. Max size is 50 MB.', 'error');
            return;
        }

        // Show progress UI
        showProgress(20);
        setStatus('loading', 'Processing…');
        setUploadMsg(`Uploading "${file.name}"…`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            showProgress(55);
            const response = await fetch('/upload', { method: 'POST', body: formData });
            showProgress(90);
            const result = await response.json();

            if (response.ok) {
                showProgress(100);
                hideProgress();
                setUploadMsg(`"${file.name}" ready ✓`, 'success');
                setStatus('ready', file.name.length > 28 ? file.name.substring(0, 25) + '…' : file.name);
                loadPDFViewer(result.pdf_url, file.name);
                // Clear chat on new document
                el.chatHistory.innerHTML = '';
                el.chatHistory.appendChild(buildEmptyState());
            } else {
                throw new Error(result.error || 'Unknown error');
            }
        } catch (err) {
            showProgress(0);
            hideProgress();
            setUploadMsg(`Upload failed: ${err.message}`, 'error');
            setStatus('error', 'Upload failed');
        }
    }

    function loadPDFViewer(url, filename) {
        el.pdfViewer.src = url;
        el.pdfDocName.textContent = filename || 'document.pdf';
        el.uploadOverlay.style.display = 'none';
        el.pdfViewerWrapper.style.display = 'flex';
    }

    // File input handlers
    el.pdfUploadOverlay.addEventListener('change', e => { if (e.target.files[0]) uploadPDF(e.target.files[0]); });
    el.pdfUploadMain.addEventListener('change',    e => { if (e.target.files[0]) uploadPDF(e.target.files[0]); });

    // Drag-and-drop on the overlay
    el.dropZone.addEventListener('dragover',  e => { e.preventDefault(); el.dropZone.classList.add('drag-over'); });
    el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('drag-over'));
    el.dropZone.addEventListener('drop', e => {
        e.preventDefault();
        el.dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) uploadPDF(file);
    });

    // Also allow drop anywhere on the left pane when overlay hidden
    el.leftPane.addEventListener('dragover',  e => { e.preventDefault(); });
    el.leftPane.addEventListener('drop', e => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) uploadPDF(file);
    });

    // ── Resizer ──────────────────────────────────────────────────────────────
    function initResizer() {
        const container = document.querySelector('.app-container');

        function applyWidths(leftPx) {
            const total = container.offsetWidth;
            const rW    = el.resizer.offsetWidth;
            const minW  = total * 0.2;
            const maxW  = total * 0.8;
            const left  = Math.min(Math.max(leftPx, minW), maxW);
            const right = total - left - rW;

            el.leftPane.style.width  = `${left}px`;
            el.rightPane.style.width = `${right}px`;
            el.resizer.style.left    = `${left}px`;
        }

        // Load saved or default
        const savedL = localStorage.getItem('docuMindLeftPx');
        if (savedL) {
            applyWidths(parseFloat(savedL));
        } else {
            applyWidths(container.offsetWidth * 0.62);
        }

        let startX = 0;
        let startLeft = 0;

        el.resizer.addEventListener('mousedown', e => {
            isResizing = true;
            startX    = e.clientX;
            startLeft = el.leftPane.offsetWidth;
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            el.resizer.classList.add('active');
        });

        document.addEventListener('mousemove', e => {
            if (!isResizing) return;
            applyWidths(startLeft + (e.clientX - startX));
        });

        document.addEventListener('mouseup', () => {
            if (!isResizing) return;
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            el.resizer.classList.remove('active');
            localStorage.setItem('docuMindLeftPx', el.leftPane.offsetWidth);
        });
    }

    // ── Tabs ─────────────────────────────────────────────────────────────────
    function initTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                document.querySelectorAll('.tab-btn, .tab-content').forEach(el => el.classList.remove('active'));
                btn.classList.add('active');
                btn.setAttribute('aria-selected', 'true');
                document.querySelectorAll('.tab-btn:not(.active)').forEach(b => b.setAttribute('aria-selected', 'false'));
                document.getElementById(`${tab}Tab`).classList.add('active');
                if (tab === 'notes') renderNotes();
            });
        });
    }

    // ── Auto-grow Textarea ────────────────────────────────────────────────────
    el.userInput.addEventListener('input', () => {
        el.userInput.style.height = 'auto';
        el.userInput.style.height = Math.min(el.userInput.scrollHeight, 160) + 'px';
    });

    el.userInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuestion();
        }
    });

    el.askButton.addEventListener('click', submitQuestion);

    // ── Suggestion Chips ──────────────────────────────────────────────────────
    window.setQuestion = function(btn) {
        el.userInput.value = btn.textContent.trim();
        el.userInput.dispatchEvent(new Event('input'));
        el.userInput.focus();
    };

    // ── Chat ─────────────────────────────────────────────────────────────────
    function buildEmptyState() {
        const div = document.createElement('div');
        div.className = 'chat-empty-state';
        div.id = 'chatEmptyState';
        div.innerHTML = `
            <div class="empty-icon">
                <svg viewBox="0 0 80 80" fill="none"><circle cx="40" cy="40" r="38" stroke="var(--accent)" stroke-width="2" opacity="0.2"/><circle cx="40" cy="40" r="28" stroke="var(--accent)" stroke-width="1.5" opacity="0.15"/><path d="M27 40c0-7.18 5.82-13 13-13s13 5.82 13 13c0 5.2-3.06 9.7-7.5 11.8" stroke="var(--accent)" stroke-width="2" stroke-linecap="round"/><path d="M40 53v2M40 57v2" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round"/></svg>
            </div>
            <h3>Ready to assist</h3>
            <p>Upload a PDF on the left, then ask anything about it here.</p>
            <div class="suggestion-chips">
                <button class="chip" onclick="setQuestion(this)">Summarize this document</button>
                <button class="chip" onclick="setQuestion(this)">What are the key findings?</button>
                <button class="chip" onclick="setQuestion(this)">Explain the main concepts</button>
                <button class="chip" onclick="setQuestion(this)">List all important terms</button>
            </div>
        `;
        return div;
    }

    function removeEmptyState() {
        const empty = document.getElementById('chatEmptyState');
        if (empty) empty.remove();
    }

    function appendTypingIndicator() {
        const group = document.createElement('div');
        group.className = 'message-group';
        group.id = 'typingGroup';
        group.innerHTML = `
            <div class="msg-label ai-label">
                <div class="avatar">AI</div> DocuMind
            </div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        el.chatHistory.appendChild(group);
        el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
        return group;
    }

    function removeTypingIndicator() {
        const t = document.getElementById('typingGroup');
        if (t) t.remove();
    }

    function appendUserMessage(text) {
        removeEmptyState();
        const group = document.createElement('div');
        group.className = 'message-group';
        group.innerHTML = `
            <div class="msg-label user-label">
                <div class="avatar">U</div> You
            </div>
            <div class="user-bubble">${DOMPurify.sanitize(text)}</div>
        `;
        el.chatHistory.appendChild(group);
        el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
    }

    function appendAIMessage(question, responseText) {
        removeTypingIndicator();
        const group = document.createElement('div');
        group.className = 'message-group';
        const htmlContent = md.render(responseText);

        group.innerHTML = `
            <div class="msg-label ai-label">
                <div class="avatar">AI</div> DocuMind
            </div>
            <div class="ai-bubble">${htmlContent}</div>
            <div class="msg-actions">
                <button class="msg-action-btn save-btn">
                    <svg viewBox="0 0 24 24" fill="none"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" stroke="currentColor" stroke-width="1.8"/><polyline points="17 21 17 13 7 13 7 21" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><polyline points="7 3 7 8 15 8" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>
                    Save to Notes
                </button>
                <button class="msg-action-btn copy-btn">
                    <svg viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" stroke-width="1.8"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>
                    Copy
                </button>
            </div>
        `;

        // Save to notes
        group.querySelector('.save-btn').addEventListener('click', () => openNoteModal(question, responseText));

        // Copy to clipboard
        const copyBtn = group.querySelector('.copy-btn');
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(responseText).then(() => {
                copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none"><polyline points="20 6 9 17 4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg> Copied!`;
                setTimeout(() => {
                    copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" stroke-width="1.8"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg> Copy`;
                }, 2000);
            });
        });

        el.chatHistory.appendChild(group);
        el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
        // Render any LaTeX math in the new AI bubble
        renderMath(group.querySelector('.ai-bubble'));
    }

    function appendErrorMessage(text) {
        removeTypingIndicator();
        const group = document.createElement('div');
        group.className = 'message-group';
        group.innerHTML = `
            <div class="error-bubble">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.8"/><line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><line x1="12" y1="16" x2="12.01" y2="16" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/></svg>
                ${DOMPurify.sanitize(text)}
            </div>
        `;
        el.chatHistory.appendChild(group);
        el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
    }

    async function submitQuestion() {
        const question = el.userInput.value.trim();
        if (!question) return;

        el.userInput.value = '';
        el.userInput.style.height = 'auto';
        el.askButton.disabled = true;

        const apiKey = localStorage.getItem('docuMindApiKey');
        if (!apiKey) {
            el.apiKeyModal.style.display = 'flex';
            el.askButton.disabled = false;
            return;
        }

        appendUserMessage(question);
        appendTypingIndicator();

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: question, api_key: apiKey }),
            });

            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Query failed.');

            appendAIMessage(question, result.response || 'No response received.');
        } catch (err) {
            appendErrorMessage(`Error: ${err.message}`);
        } finally {
            el.askButton.disabled = false;
            el.userInput.focus();
        }
    }

    // ── Note Modal ────────────────────────────────────────────────────────────
    function openNoteModal(query, response) {
        pendingSaveNote = { query, response };
        el.noteNameInput.value = query.substring(0, 60);
        el.noteModal.style.display = 'flex';
        setTimeout(() => el.noteNameInput.select(), 50);
    }

    function closeNoteModal() {
        el.noteModal.style.display = 'none';
        pendingSaveNote = null;
    }

    el.modalCancel.addEventListener('click', closeNoteModal);

    el.noteModal.addEventListener('click', e => {
        if (e.target === el.noteModal) closeNoteModal();
    });

    el.noteNameInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') el.modalSave.click();
        if (e.key === 'Escape') closeNoteModal();
    });

    el.modalSave.addEventListener('click', () => {
        if (!pendingSaveNote) return;
        const name = el.noteNameInput.value.trim() || `Note ${loadNotes().length + 1}`;
        saveNote(name, pendingSaveNote.query, pendingSaveNote.response);
        closeNoteModal();
        updateNotesBadge();
        // Brief flash on the notes tab
        const notesTab = document.querySelector('[data-tab="notes"]');
        notesTab.style.color = 'var(--success)';
        setTimeout(() => notesTab.style.color = '', 800);
    });

    // ── Notes Storage ─────────────────────────────────────────────────────────
    function loadNotes() { return JSON.parse(localStorage.getItem('docuMindNotes') || '[]'); }
    function saveNotes(notes) { localStorage.setItem('docuMindNotes', JSON.stringify(notes)); }

    function saveNote(name, query, response) {
        const notes = loadNotes();
        notes.push({ name, query, response, timestamp: new Date().toISOString() });
        saveNotes(notes);
    }

    function updateNotesBadge() {
        const notes = loadNotes();
        if (notes.length > 0) {
            el.notesBadge.textContent = notes.length;
            el.notesBadge.style.display = '';
        } else {
            el.notesBadge.style.display = 'none';
        }
    }

    function formatTimestamp(iso) {
        const d = new Date(iso);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
            + ' · ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    }

    // ── Notes Rendering ───────────────────────────────────────────────────────
    function renderNotes() {
        const notes = loadNotes();
        updateNotesBadge();

        if (notes.length === 0) {
            el.notesEmptyState.style.display = '';
            el.savedNotesList.innerHTML = '';
            return;
        }

        el.notesEmptyState.style.display = 'none';
        el.savedNotesList.innerHTML = notes.map((note, i) => `
            <div class="note-card" data-index="${i}">
                <div class="note-header">
                    <h4 class="note-title">${DOMPurify.sanitize(note.name)}</h4>
                    <div class="note-actions">
                        <button class="note-icon-btn" onclick="event.stopPropagation(); noteEdit(${i})" title="Edit">✏️</button>
                        <button class="note-icon-btn" onclick="event.stopPropagation(); noteDownload(${i})" title="Download">⬇️</button>
                        <button class="note-icon-btn danger" onclick="event.stopPropagation(); noteDelete(${i})" title="Delete">🗑</button>
                    </div>
                </div>
                <div class="note-preview">
                    <strong>Q:</strong> ${DOMPurify.sanitize(note.query.substring(0, 80))}${note.query.length > 80 ? '…' : ''}
                </div>
                <div class="note-timestamp">${formatTimestamp(note.timestamp)}</div>
                <div class="note-expanded-content">
                    <div class="note-section-label">Question</div>
                    <div class="note-query-text">${DOMPurify.sanitize(note.query)}</div>
                    <div class="note-section-label">Response</div>
                    <div class="note-response-text">${md.render(note.response)}</div>
                </div>
            </div>
        `).join('');

        // Toggle expand/collapse
        document.querySelectorAll('.note-card').forEach(card => {
            card.addEventListener('click', e => {
                if (e.target.closest('button') || card.classList.contains('edit-mode')) return;
                card.classList.toggle('expanded');
                // Render math in expanded note responses
                if (card.classList.contains('expanded')) {
                    renderMath(card.querySelector('.note-response-text'));
                }
            });
        });
    }

    // ── Note Actions (global so inline onclick works) ─────────────────────────
    window.noteDelete = function(index) {
        const notes = loadNotes();
        notes.splice(index, 1);
        saveNotes(notes);
        renderNotes();
    };

    window.noteDownload = function(index) {
        const note = loadNotes()[index];
        const content = `NOTE: ${note.name}\nDate: ${formatTimestamp(note.timestamp)}\n\nQUESTION:\n${note.query}\n\nRESPONSE:\n${md.strip(note.response)}`;
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `${note.name.replace(/[^a-z0-9]/gi,'_')}.txt` });
        a.click();
        URL.revokeObjectURL(a.href);
    };

    window.noteEdit = function(index) {
        const notes = loadNotes();
        const note  = notes[index];
        const card  = document.querySelector(`.note-card[data-index="${index}"]`);
        card.classList.add('edit-mode', 'expanded');

        card.innerHTML = `
            <input type="text" class="edit-name-input" value="${DOMPurify.sanitize(note.name)}" placeholder="Note title">
            <div style="font-size:0.72rem;color:var(--text-muted);margin:4px 0 6px;font-weight:600;text-transform:uppercase;letter-spacing:.5px">Question</div>
            <textarea class="edit-query-input">${DOMPurify.sanitize(note.query)}</textarea>
            <div style="font-size:0.72rem;color:var(--text-muted);margin:4px 0 6px;font-weight:600;text-transform:uppercase;letter-spacing:.5px">Response (Markdown)</div>
            <textarea class="edit-response-input">${DOMPurify.sanitize(md.strip(note.response))}</textarea>
            <div class="edit-actions">
                <button class="edit-save-btn" onclick="event.stopPropagation(); noteSave(${index})">💾 Save</button>
                <button class="edit-cancel-btn" onclick="event.stopPropagation(); renderNotes()">Cancel</button>
            </div>
        `;
    };

    window.noteSave = function(index) {
        const card  = document.querySelector(`.note-card[data-index="${index}"]`);
        const notes = loadNotes();
        notes[index].name     = DOMPurify.sanitize(card.querySelector('.edit-name-input').value.trim()) || notes[index].name;
        notes[index].query    = DOMPurify.sanitize(card.querySelector('.edit-query-input').value);
        notes[index].response = md.restore(card.querySelector('.edit-response-input').value);
        saveNotes(notes);
        renderNotes();
    };

    // Download all notes
    el.downloadAllNotes.addEventListener('click', () => {
        const notes = loadNotes();
        if (!notes.length) return;
        const content = notes.map((n, i) =>
            `═══ NOTE ${i + 1}: ${n.name} ═══\nDate: ${formatTimestamp(n.timestamp)}\n\nQUESTION:\n${n.query}\n\nRESPONSE:\n${md.strip(n.response)}`
        ).join('\n\n' + '─'.repeat(60) + '\n\n');
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'documind_notes.txt' });
        a.click();
        URL.revokeObjectURL(a.href);
    });

    // ── Initialise ────────────────────────────────────────────────────────────
    initResizer();
    initTabs();
    updateNotesBadge();
    renderNotes();
    initApiKey();

});