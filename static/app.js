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
