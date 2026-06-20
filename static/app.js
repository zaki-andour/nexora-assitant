const API = "";
let sessionId = null;
let currentUser = null;
let pendingFeedback = false;
let currentChatId = null;
let currentMessages = [];

// ── STORAGE ───────────────────────────────────────────
async function loadChats() {
    try {
        const res = await fetch(`${API}/api/history?session_id=${sessionId}`);
        const data = await res.json();
        return data.chats || [];
    } catch { return []; }
}

async function saveChat(chat) {
    try {
        await fetch(`${API}/api/history/save`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, chat })
        });
    } catch(e) { console.error("Failed to save chat"); }
}

async function saveCurrentChat() {
    if (!currentChatId || currentMessages.length === 0) return;
    const firstQ = currentMessages.find(m => m.role === "user");
    const title = firstQ ? firstQ.text.substring(0, 40) + (firstQ.text.length > 40 ? "..." : "") : "New Chat";
    const chat = {
        id: currentChatId,
        title,
        messages: currentMessages,
        date: new Date().toLocaleDateString()
    };
    await saveChat(chat);
    renderChatHistory();
}

async function renderChatHistory() {
    const chats = await loadChats();
    const container = document.getElementById("chat-history");
    container.innerHTML = "";
    if (chats.length === 0) {
        container.innerHTML = "<div style='color:#666;font-size:12px;padding:8px'>No conversations yet</div>";
        return;
    }
    chats.forEach(chat => {
        const div = document.createElement("div");
        div.className = "history-item" + (chat.id === currentChatId ? " active" : "");
        div.innerHTML = `
            <div style="font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${chat.title}</div>
            <div style="font-size:11px;color:#666;margin-top:2px">${chat.date}</div>
        `;
        div.onclick = () => loadChat(chat.id);
        container.appendChild(div);
    });
}

async function loadChat(chatId) {
    const chats = await loadChats();
    const chat = chats.find(c => c.id === chatId);
    if (!chat) return;
    currentChatId = chatId;
    currentMessages = chat.messages;
    // Render messages
    const msgs = document.getElementById("messages");
    msgs.innerHTML = "";
    document.getElementById("welcome-screen").classList.add("hidden");
    chat.messages.forEach(m => {
        if (m.role === "user") addMessage("user", m.text, null, false);
        else if (m.role === "ai") addMessage("ai", m.text, m.meta, false);
        else if (m.role === "blocked") addBlockedMessage(m.text, false);
    });
    document.getElementById("feedback-bar").classList.add("hidden");
    document.getElementById("sources-display").textContent = "";
    renderChatHistory();
}

// ── LOGIN ──────────────────────────────────────────────
document.getElementById("login-btn").addEventListener("click", login);
document.getElementById("password").addEventListener("keydown", e => { if (e.key === "Enter") login(); });

async function login() {
    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value.trim();
    const errorEl  = document.getElementById("login-error");
    if (!username || !password) { errorEl.textContent = "Please enter username and password."; return; }
    try {
        const res = await fetch(`${API}/api/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password })
        });
        if (!res.ok) { errorEl.textContent = "Invalid username or password."; return; }
        const data = await res.json();
        sessionId   = data.session_id;
        currentUser = data;
        document.getElementById("user-name").textContent    = data.username;
        document.getElementById("user-role").textContent    = data.role.toUpperCase();
        document.getElementById("user-avatar").textContent  = data.username[0].toUpperCase();
        document.getElementById("topbar-dept").textContent  = `${data.department} · ${data.role.toUpperCase()}`;
        await loadModels();
        document.getElementById("login-page").classList.add("hidden");
        document.getElementById("app").classList.remove("hidden");
        errorEl.textContent = "";
        newChat();
        await renderChatHistory();
    } catch(e) {
        errorEl.textContent = "Connection error. Please try again.";
    }
}

// ── MODELS ────────────────────────────────────────────
async function loadModels() {
    try {
        const res    = await fetch(`${API}/api/models`);
        const models = await res.json();
        const select = document.getElementById("model-select");
        select.innerHTML = "";
        for (const [key, label] of Object.entries(models)) {
            const opt   = document.createElement("option");
            opt.value   = key;
            opt.textContent = label;
            select.appendChild(opt);
        }
    } catch(e) { console.error("Failed to load models"); }
}

async function switchModel() {
    const model = document.getElementById("model-select").value;
    if (!sessionId) return;
    await fetch(`${API}/api/switch_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, model })
    });
    addSystemMessage(`Model switched to ${model}`);
}

// ── CHAT ──────────────────────────────────────────────
function newChat() {
    currentChatId = Date.now();
    currentMessages = [];
    document.getElementById("messages").innerHTML = "";
    document.getElementById("welcome-screen").classList.remove("hidden");
    document.getElementById("feedback-bar").classList.add("hidden");
    document.getElementById("sources-display").textContent = "";
    pendingFeedback = false;
    renderChatHistory();
}

function useSuggestion(el) {
    document.getElementById("question-input").value = el.textContent;
    sendQuestion();
}

function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

async function sendQuestion() {
    const input    = document.getElementById("question-input");
    const question = input.value.trim();
    if (!question || !sessionId) return;
    document.getElementById("welcome-screen").classList.add("hidden");
    document.getElementById("feedback-bar").classList.add("hidden");
    addMessage("user", question);
    input.value = "";
    input.style.height = "auto";
    const typingId = showTyping();
    document.getElementById("send-btn").disabled = true;
    try {
        const res = await fetch(`${API}/api/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, session_id: sessionId })
        });
        const data = await res.json();
        removeTyping(typingId);
        if (data.blocked) {
            addBlockedMessage(data.answer);
        } else {
            addMessage("ai", data.answer, data);
            document.getElementById("feedback-bar").classList.remove("hidden");
            document.getElementById("fb-comment").value = "";
            pendingFeedback = true;
            if (data.sources && data.sources.length > 0) {
                document.getElementById("sources-display").textContent =
                    `Sources: ${data.sources.join(" · ")} · ${data.latency}s`;
            }
        }
        saveCurrentChat();
    } catch(e) {
        removeTyping(typingId);
        addMessage("ai", "Sorry, an error occurred. Please try again.");
    }
    document.getElementById("send-btn").disabled = false;
}

// ── FEEDBACK ──────────────────────────────────────────
let feedbackScore = null;

function sendFeedback(score) {
    feedbackScore = score;
    document.querySelectorAll(".fb-btn").forEach(b => b.style.borderColor = "");
    event.target.style.borderColor = score === 1 ? "#10a37f" : "#ef4444";
}

async function submitFeedback() {
    if (!pendingFeedback) return;
    const comment = document.getElementById("fb-comment").value;
    const score   = feedbackScore || 1;
    await fetch(`${API}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, score, comment })
    });
    document.getElementById("feedback-bar").classList.add("hidden");
    feedbackScore   = null;
    pendingFeedback = false;
    addSystemMessage("Feedback submitted. Thank you!");
}

// ── UI HELPERS ────────────────────────────────────────
function addMessage(role, text, meta = null, save = true) {
    const msgs   = document.getElementById("messages");
    const div    = document.createElement("div");
    div.className = `message ${role}`;
    const avatar  = role === "user" ? currentUser.username[0].toUpperCase() : "N";
    const metaStr = meta ? `${meta.category} · ${meta.model} · ${meta.latency}s` : "";
    div.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-content">
            <div class="msg-bubble">${formatText(text)}</div>
            ${meta ? `<div class="msg-meta">${metaStr}</div>` : ""}
        </div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    if (save) {
        currentMessages.push({ role, text, meta });
    }
}

function addBlockedMessage(text, save = true) {
    const msgs = document.getElementById("messages");
    const div  = document.createElement("div");
    div.className = "message ai";
    div.innerHTML = `
        <div class="msg-avatar">N</div>
        <div class="msg-content">
            <div class="blocked-msg">🔒 ${text}</div>
        </div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    if (save) {
        currentMessages.push({ role: "blocked", text });
    }
}

function addSystemMessage(text) {
    const msgs = document.getElementById("messages");
    const div  = document.createElement("div");
    div.style.cssText = "text-align:center;font-size:12px;color:#8e8ea0;padding:8px;";
    div.textContent = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}

function showTyping() {
    const msgs = document.getElementById("messages");
    const div  = document.createElement("div");
    const id   = `typing-${Date.now()}`;
    div.id     = id;
    div.className = "message ai";
    div.innerHTML = `
        <div class="msg-avatar">N</div>
        <div class="msg-content">
            <div class="msg-bubble">
                <div class="typing"><span></span><span></span><span></span></div>
            </div>
        </div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function formatText(text) {
    return text
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>");
}

// ── LOGOUT ────────────────────────────────────────────
async function logout() {
    await fetch(`${API}/api/logout`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
    });
    sessionId     = null;
    currentUser   = null;
    currentChatId = null;
    currentMessages = [];
    document.getElementById("app").classList.add("hidden");
    document.getElementById("login-page").classList.remove("hidden");
    document.getElementById("username").value = "";
    document.getElementById("password").value = "";
    document.getElementById("messages").innerHTML = "";
    document.getElementById("chat-history").innerHTML = "";
}

// Markdown rendering (uses marked if available, fallback otherwise)
function formatText(text) {
    if (window.marked && typeof marked.parse === "function") return marked.parse(text);
    return text.replace(/\n/g, "<br>").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\*(.*?)\*/g, "<em>$1</em>");
}

// Category badge + latency under each answer (overrides earlier addMessage)
function addMessage(role, text, meta = null, save = true) {
    const msgs   = document.getElementById("messages");
    const div    = document.createElement("div");
    div.className = `message ${role}`;
    const avatar = role === "user" ? currentUser.username[0].toUpperCase() : "N";
    const catColors = { TEXT:"#3b82f6", STRUCTURED:"#10b981", GRAPH:"#8b5cf6", HYBRID:"#f59e0b", BLOCKED:"#ef4444" };
    const metaHtml = meta ? `
        <div class="msg-meta">
            <span class="cat-badge" style="background:${catColors[meta.category] || '#6b7280'}">${meta.category || ""}</span>
            <span class="meta-pill">${meta.latency}s</span>
            ${meta.model ? `<span class="meta-pill">${meta.model}</span>` : ""}
        </div>` : "";
    div.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-content">
            <div class="msg-bubble">${formatText(text)}</div>
            ${metaHtml}
        </div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    if (save) { currentMessages.push({ role, text, meta }); }
}

// System status panel (calls /api/health)
async function checkHealth() {
    const panel = document.getElementById("status-panel");
    panel.style.display = (panel.style.display === "none") ? "block" : "none";
    if (panel.style.display === "none") return;
    panel.innerHTML = "Checking…";
    try {
        const res = await fetch(`${API}/api/health`);
        const s   = await res.json();
        const dot = ok => `<span class="dot ${ok ? 'up' : 'down'}"></span>`;
        panel.innerHTML =
            `<div>${dot(s.postgres)} PostgreSQL</div>` +
            `<div>${dot(s.ollama)} Ollama <small>(${s.model})</small></div>` +
            `<div>${dot(s.milvus)} Milvus</div>`;
    } catch (e) {
        panel.innerHTML = "Cannot reach the server.";
    }
}

// Voice input: record -> /api/transcribe -> fill question box -> send
let mediaRecorder = null, audioChunks = [];
async function toggleMic() {
    const btn = document.getElementById("mic-btn");
    if (mediaRecorder && mediaRecorder.state === "recording") { mediaRecorder.stop(); return; }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            btn.classList.remove("recording");
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            const fd = new FormData();
            fd.append("audio", blob, "speech.webm");
            const input = document.getElementById("question-input");
            input.value = "Transcribing…";
            try {
                const res  = await fetch(`${API}/api/transcribe`, { method: "POST", body: fd });
                const data = await res.json();
                input.value = data.text || "";
                if (input.value.trim()) sendQuestion();
            } catch (e) {
                input.value = "";
                alert("Transcription failed.");
            }
        };
        mediaRecorder.start();
        btn.classList.add("recording");
    } catch (e) {
        alert("Microphone access denied or unavailable.");
    }
}

// Improved mic with a visible "Recording…" cue (overrides earlier toggleMic)
async function toggleMic() {
    const btn = document.getElementById("mic-btn");
    const input = document.getElementById("question-input");
    if (mediaRecorder && mediaRecorder.state === "recording") { mediaRecorder.stop(); return; }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        const savedPlaceholder = input.placeholder;
        input.placeholder = "🔴 Recording… click the mic again to stop";
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            btn.classList.remove("recording");
            input.placeholder = savedPlaceholder;
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            const fd = new FormData();
            fd.append("audio", blob, "speech.webm");
            input.value = "Transcribing…"; input.disabled = true;
            try {
                const res  = await fetch(`${API}/api/transcribe`, { method: "POST", body: fd });
                const data = await res.json();
                input.disabled = false; input.value = data.text || "";
                if (input.value.trim()) sendQuestion();
            } catch (e) { input.disabled = false; input.value = ""; alert("Transcription failed."); }
        };
        mediaRecorder.start();
        btn.classList.add("recording");
    } catch (e) { alert("Microphone access denied or unavailable."); }
}

// Fix: "Transcribing…" must never be sent (placeholder, not value) + disable send while transcribing
async function toggleMic() {
    const btn = document.getElementById("mic-btn");
    const input = document.getElementById("question-input");
    const sendBtn = document.getElementById("send-btn");
    if (mediaRecorder && mediaRecorder.state === "recording") { mediaRecorder.stop(); return; }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        const savedPlaceholder = input.placeholder;
        input.placeholder = "🔴 Recording… click the mic again to stop";
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            btn.classList.remove("recording");
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            const fd = new FormData(); fd.append("audio", blob, "speech.webm");
            input.placeholder = "Transcribing…"; input.disabled = true;
            if (sendBtn) sendBtn.disabled = true;
            try {
                const res  = await fetch(`${API}/api/transcribe`, { method: "POST", body: fd });
                const data = await res.json();
                input.disabled = false; if (sendBtn) sendBtn.disabled = false;
                input.placeholder = savedPlaceholder;
                input.value = (data.text || "").trim();
                if (input.value) sendQuestion();
            } catch (e) {
                input.disabled = false; if (sendBtn) sendBtn.disabled = false;
                input.placeholder = savedPlaceholder; input.value = "";
                alert("Transcription failed.");
            }
        };
        mediaRecorder.start();
        btn.classList.add("recording");
    } catch (e) { alert("Microphone access denied or unavailable."); }
}
