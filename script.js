document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        pdfUpload: document.getElementById('pdfUpload'),
        uploadButton: document.getElementById('uploadButton'),
        uploadStatus: document.getElementById('uploadStatus'),
        chatHistory: document.getElementById('chatHistory'),
        userInput: document.getElementById('userInput'),
        askButton: document.getElementById('askButton'),
        pdfViewer: document.getElementById('pdfViewer'),
        savedNotesList: document.getElementById('savedNotesList'),
        downloadAllNotes: document.getElementById('downloadAllNotes')
    };

    // Initialize Marked.js
    window.marked = window.marked || { parse: text => text };

    // Tab Management
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            document.querySelectorAll('.tab-button, .tab-content').forEach(el => {
                el.classList.remove('active');
            });
            button.classList.add('active');
            document.getElementById(`${tabName}Tab`).classList.add('active');
            if (tabName === 'notes') displaySavedNotes();
        });
    });

    // PDF Upload Handler
    async function uploadPDF() {
        const file = elements.pdfUpload.files?.[0];
        if (!file) {
            elements.uploadStatus.textContent = "Please select a PDF file first!";
            return;
        }

        try {
            elements.uploadStatus.textContent = "Uploading and processing document...";
            elements.chatHistory.innerHTML = '';

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (response.ok) {
                elements.uploadStatus.textContent = "Document processed successfully!";
                elements.pdfViewer.src = result.pdf_url || '';
            } else {
                elements.uploadStatus.textContent = `Error: ${result.error || 'Unknown error'}`;
            }
        } catch (error) {
            elements.uploadStatus.textContent = `Upload failed: ${error.message}`;
        }
    }

    // Notes Functions
    function saveNoteWithName(query, response) {
        const noteName = prompt('Name your note (optional):', query.substring(0, 50));
        const savedNotes = JSON.parse(localStorage.getItem('llm_notes') || '[]');
        
        savedNotes.push({
            name: noteName || `Note ${savedNotes.length + 1}`,
            query,
            response,
            timestamp: new Date().toISOString()
        });
        
        localStorage.setItem('llm_notes', JSON.stringify(savedNotes));
        displaySavedNotes();
    }

    function loadNotesFromStorage() {
        return JSON.parse(localStorage.getItem('llm_notes')) || [];
    }

    // Edit Functions (now properly exposed to global scope)
    window.enterEditMode = function(index) {
        const notes = loadNotesFromStorage();
        const note = notes[index];
        const noteCard = document.querySelector(`.note-card[data-index="${index}"]`);
        
        noteCard.classList.add('edit-mode');
        noteCard.innerHTML = `
            <div class="note-header">
                <input type="text" class="edit-name-input" value="${DOMPurify.sanitize(note.name)}">
                <div class="note-actions">
                    <button onclick="event.stopPropagation(); window.saveEditedNote(${index})">💾 Save</button>
                    <button onclick="event.stopPropagation(); window.cancelEditNote(${index})">❌ Cancel</button>
                </div>
            </div>
            <div class="edit-content">
                <div class="edit-query">
                    <strong>Question:</strong>
                    <textarea class="edit-query-input">${DOMPurify.sanitize(note.query)}</textarea>
                </div>
                <div class="edit-response">
                    <strong>Response:</strong>
                    <textarea class="edit-response-input">${DOMPurify.sanitize(note.response)}</textarea>
                </div>
            </div>
        `;
    };

    window.saveEditedNote = function(index) {
        const notes = loadNotesFromStorage();
        const note = notes[index];
        const noteCard = document.querySelector(`.note-card[data-index="${index}"]`);
        
        note.name = DOMPurify.sanitize(noteCard.querySelector('.edit-name-input').value);
        note.query = DOMPurify.sanitize(noteCard.querySelector('.edit-query-input').value);
        note.response = DOMPurify.sanitize(noteCard.querySelector('.edit-response-input').value);
        
        localStorage.setItem('llm_notes', JSON.stringify(notes));
        displaySavedNotes();
    };

    window.cancelEditNote = function(index) {
        displaySavedNotes();
    };

    function displaySavedNotes() {
        const notes = loadNotesFromStorage();
        elements.savedNotesList.innerHTML = notes.map((note, index) => `
            <div class="note-card" data-index="${index}">
                <div class="note-header">
                    <h4 class="note-title">${DOMPurify.sanitize(note.name)}</h4>
                    <div class="note-actions">
                        <button onclick="event.stopPropagation(); window.downloadNote(${index})">📥</button>
                        <button onclick="event.stopPropagation(); window.deleteNote(${index})">🗑️</button>
                        <button onclick="event.stopPropagation(); window.enterEditMode(${index})">✏️</button>
                    </div>
                </div>
                <div class="note-content preview-text">
                    <strong>Q:</strong> ${DOMPurify.sanitize(note.query.substring(0, 70))}${note.query.length > 70 ? '...' : ''}
                </div>
                <div class="note-content preview-text">
                    ${DOMPurify.sanitize(window.marked.parse(note.response.substring(0, 100) + (note.response.length > 100 ? '...' : '')))}
                </div>
                <div class="full-content">
                    <div class="full-query"><strong>Full Question:</strong> ${DOMPurify.sanitize(note.query)}</div>
                    <div class="full-response"><strong>Response:</strong> ${DOMPurify.sanitize(window.marked.parse(note.response))}</div>
                </div>
            </div>
        `).join('');

        document.querySelectorAll('.note-card').forEach(card => {
            card.addEventListener('click', (e) => {
                if (!e.target.closest('button') && !card.classList.contains('edit-mode')) {
                    card.classList.toggle('expanded');
                }
            });
        });
    }

    window.downloadNote = function(index) {
        const notes = loadNotesFromStorage();
        const note = notes[index];
        const blob = new Blob([`NOTE NAME: ${note.name}\n\nQUESTION: ${note.query}\n\nRESPONSE: ${note.response}`], 
            { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `note_${index + 1}.txt`;
        a.click();
    };

    window.deleteNote = function(index) {
        const notes = loadNotesFromStorage();
        notes.splice(index, 1);
        localStorage.setItem('llm_notes', JSON.stringify(notes));
        displaySavedNotes();
    };

    // Download All Notes
    elements.downloadAllNotes.addEventListener('click', () => {
        const notes = loadNotesFromStorage();
        const allNotesText = notes.map((note, i) => 
            `NOTE ${i + 1}:\nNAME: ${note.name}\nQ: ${note.query}\nA: ${note.response}\n\n`
        ).join('\n');
        const blob = new Blob([allNotesText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'all_notes.txt';
        a.click();
    });

    // Chat Functions
    async function submitQuestion() {
        const question = elements.userInput.value.trim();
        if (!question) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-container';
        messageDiv.innerHTML = `
            <div class="user-message">
                <strong>You:</strong> ${DOMPurify.sanitize(question)}
            </div>
            <div class="loading-message">Generating response...</div>
        `;
        elements.chatHistory.appendChild(messageDiv);

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: question })
            });

            const result = await response.json();
            if (!response.ok) throw new Error(result.error || "Query failed.");

            const htmlResponse = window.marked.parse(result.response || "No response");
            messageDiv.innerHTML = `
                <div class="user-message">
                    <strong>You:</strong> ${DOMPurify.sanitize(question)}
                </div>
                <div class="assistant-message">
                    <strong>Assistant:</strong>
                    <div class="assistant-content">${DOMPurify.sanitize(htmlResponse)}</div>
                    <button class="save-note-btn" 
                            data-query="${encodeURIComponent(question)}" 
                            data-response="${encodeURIComponent(result.response)}">
                        💾 Save to Notes
                    </button>
                </div>
            `;

            messageDiv.querySelector('.save-note-btn').addEventListener('click', (e) => {
                const query = decodeURIComponent(e.target.dataset.query);
                const response = decodeURIComponent(e.target.dataset.response);
                saveNoteWithName(query, response);
                document.querySelector('[data-tab="notes"]').click();
            });

        } catch (error) {
            messageDiv.innerHTML = `
                <div class="user-message">
                    <strong>You:</strong> ${DOMPurify.sanitize(question)}
                </div>
                <div class="error-message">Error: ${error.message}</div>
            `;
        }

        elements.userInput.value = '';
        elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
    }

    // Event Listeners
    elements.uploadButton.addEventListener('click', uploadPDF);
    elements.askButton.addEventListener('click', submitQuestion);
    elements.userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitQuestion();
    });

    // Initial Setup
    displaySavedNotes();
});