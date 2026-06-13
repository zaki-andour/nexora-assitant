# ══════════════════════════════════════════════════════
# NEXORA RAG PIPELINE — CENTRAL CONFIGURATION
# ══════════════════════════════════════════════════════

# ── LLM ───────────────────────────────────────────────
MODEL = "qwen2.5-q4:7b"
# ── AVAILABLE MODELS ──────────────────────────────────
AVAILABLE_MODELS = {
    "qwen2.5-q4:7b":    "Qwen2.5 Q4 (Recommended)",
    "qwen2.5:7b":       "Qwen2.5 Q3 (Faster)",
    "deepseek-r1:7b":   "DeepSeek R1 (Reasoning)",
    "qwen2.5:1.5b":     "Qwen2.5 1.5B (Lightweight)",
}
OLLAMA_URL  = "http://localhost:11434/api/generate"
TIMEOUT     = 900
NUM_PREDICT = 2048
TEMPERATURE = 0.1

# ── MILVUS ────────────────────────────────────────────
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
TOP_K       = 5

# ── POSTGRESQL ────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "database": "nexora",
    "user":     "raguser",
    "password": "ragpass123"
}

# ── FILES ─────────────────────────────────────────────
CHUNKS_FILE        = "hr_dataset/processed/text/chunks.json"
EMPLOYEES_CHUNKS   = "hr_dataset/processed/structured/employees_chunks.json"
EMBEDDED_CHUNKS    = "hr_dataset/processed/embedded/chunks.json"
EMBEDDED_EMPLOYEES = "hr_dataset/processed/embedded/employees_chunks.json"

# ── LOGGING ───────────────────────────────────────────
LOG_FILE  = "logs/pipeline.log"
LOG_LEVEL = "INFO"
