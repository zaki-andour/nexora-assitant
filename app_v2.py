from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import time
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.routing.pipeline import run_pipeline
from src.evaluation.feedback_store import save_feedback, get_feedback_stats
from src.auth.auth import authenticate
from src.auth.rbac import apply_rbac_filter
from src.auth.audit import log_action, get_audit_stats
import src.config as config
from src.utils.logger import get_logger

logger  = get_logger("app_v2")
app     = FastAPI(title="Nexora HR Platform")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── SESSION STORE ─────────────────────────────────────
sessions      = {}
last_results  = {}
# Chat history stored in PostgreSQL — persists across restarts
import psycopg2, json as _json

def get_db():
    return psycopg2.connect(host="localhost", dbname="nexora", user="raguser", password="ragpass123")

def db_get_chats(username):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT chat_id, title, messages, updated_at FROM chat_sessions WHERE username=%s ORDER BY updated_at DESC LIMIT 20", (username,))
        rows = cur.fetchall()
        conn.close()
        return [{"id": r[0], "title": r[1], "messages": r[2], "date": r[3].strftime("%Y-%m-%d")} for r in rows]
    except Exception as e:
        return []

def db_save_chat(username, chat):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO chat_sessions (username, chat_id, title, messages, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (username, chat_id) DO UPDATE
            SET title=EXCLUDED.title, messages=EXCLUDED.messages, updated_at=NOW()
        """, (username, chat["id"], chat["title"], _json.dumps(chat["messages"])))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

# ── MODELS ────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class QuestionRequest(BaseModel):
    question: str
    session_id: str

class FeedbackRequest(BaseModel):
    session_id: str
    score: int
    comment: str = ""

class ModelRequest(BaseModel):
    session_id: str
    model: str

# ── ROUTES ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/login")
async def login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    session_id = f"{req.username}_{int(time.time())}"
    sessions[session_id] = user
    logger.info(f"Login: {req.username}")
    return {
        "session_id": session_id,
        "username":   user["username"],
        "role":       user["role"],
        "department": user["department"]
    }

@app.post("/api/ask")
async def ask(req: QuestionRequest):
    user = sessions.get(req.session_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    logger.info(f"Question from {user['username']}: {req.question}")

    rbac = apply_rbac_filter(user, req.question)
    if not rbac["allowed"]:
        log_action(user, req.question, "BLOCKED", False)
        return {"answer": rbac["reason"], "sources": [], "blocked": True}

    t0     = time.time()
    result = run_pipeline(req.question, user=user)
    elapsed = round(time.time() - t0, 2)

    last_results[req.session_id] = {
        "question": req.question,
        "answer":   result["answer"],
        "category": ("HYBRID" if result.get("is_complex") and len(set(result["categories"])) > 1 else (result["categories"][0] if result["categories"] else "TEXT")),
        "sources":  result["sources"],
        "latency":  elapsed,
        "model":    config.MODEL,
    }

    log_action(user, req.question, last_results[req.session_id]["category"], True)

    return {
        "answer":   result["answer"],
        "sources":  result["sources"],
        "category": last_results[req.session_id]["category"],
        "latency":  elapsed,
        "model":    config.MODEL,
        "blocked":  False
    }

@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    user   = sessions.get(req.session_id)
    result = last_results.get(req.session_id)
    if not user or not result:
        raise HTTPException(status_code=400, detail="No result to rate")

    save_feedback(
        question        = result["question"],
        category        = result["category"],
        answer          = result["answer"],
        sources         = result["sources"],
        score           = req.score,
        model_used      = result["model"],
        latency_sec     = result["latency"],
        top_chunk_score = 0.0,
        chunks_count    = len(result["sources"]),
        comment         = req.comment
    )
    return {"status": "ok"}

@app.post("/api/switch_model")
async def switch_model(req: ModelRequest):
    user = sessions.get(req.session_id)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    config.MODEL = req.model
    logger.info(f"Model switched to {req.model} by {user['username']}")
    return {"status": "ok", "model": req.model}

@app.get("/api/models")
async def get_models():
    return config.AVAILABLE_MODELS

@app.get("/api/stats")
async def get_stats(session_id: str):
    user = sessions.get(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    stats  = get_feedback_stats()
    astats = get_audit_stats()
    return {"feedback": stats, "audit": astats}

@app.get("/api/history")
async def get_history(session_id: str):
    user = sessions.get(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"chats": db_get_chats(user["username"])}

@app.post("/api/history/save")
async def save_history(req: dict):
    session_id = req.get("session_id")
    user = sessions.get(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    chat = req.get("chat")
    if not chat:
        return {"status": "ok"}
    db_save_chat(user["username"], chat)
    return {"status": "ok"}

@app.post("/api/logout")
async def logout(req: dict):
    session_id = req.get("session_id")
    if session_id in sessions:
        del sessions[session_id]
    if session_id in last_results:
        del last_results[session_id]
    return {"status": "ok"}

@app.get("/api/health")
def health():
    import socket, requests
    status = {}
    try:
        conn = get_db(); conn.close()
        status["postgres"] = True
    except Exception:
        status["postgres"] = False
    try:
        base = config.OLLAMA_URL.split("/api/")[0]
        r = requests.get(base + "/api/tags", timeout=2)
        status["ollama"] = (r.status_code == 200)
    except Exception:
        status["ollama"] = False
    try:
        sk = socket.create_connection((config.MILVUS_HOST, int(config.MILVUS_PORT)), timeout=2)
        sk.close()
        status["milvus"] = True
    except Exception:
        status["milvus"] = False
    status["model"] = config.MODEL
    status["all_ok"] = all([status["postgres"], status["ollama"], status["milvus"]])
    return status


# ---- Voice input (Whisper speech-to-text) ----
from fastapi import UploadFile, File
from faster_whisper import WhisperModel
try:
    WHISPER_MODEL = WhisperModel("medium", device="cuda", compute_type="int8_float16")
except Exception:
    WHISPER_MODEL = WhisperModel("medium", device="cpu", compute_type="int8")

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    import tempfile, os
    data = await audio.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(data); path = tmp.name
    try:
        segments, info = WHISPER_MODEL.transcribe(path, beam_size=5, vad_filter=True)
        text = " ".join(seg.text for seg in segments).strip()
        return {"text": text, "language": info.language}
    finally:
        os.remove(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861,
                ssl_certfile="cert.pem", ssl_keyfile="key.pem")
