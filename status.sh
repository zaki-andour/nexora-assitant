#!/bin/bash
echo ""
echo "████████████████████████████████████████████████████"
echo "  NEXORA RAG PIPELINE — SYSTEM STATUS"
echo "████████████████████████████████████████████████████"

# ── MILVUS ────────────────────────────────────────────
echo ""
echo "── MILVUS (Vector Database)"
MILVUS=$(docker ps --filter "name=milvus-standalone" --format "{{.Status}}" 2>/dev/null)
if [[ $MILVUS == *"Up"* ]]; then
    echo "   Status  : ✅ Running"
    echo "   Port    : 19530"
else
    echo "   Status  : ❌ Stopped"
    echo "   Fix     : cd ~ && docker network prune -f && docker-compose down && docker-compose up -d"
fi

# ── MILVUS CHUNKS ─────────────────────────────────────
echo ""
echo "── MILVUS CHUNKS (Policy Documents)"
CHUNKS_FILE="/root/project/hr_dataset/processed/text/chunks.json"
if [[ -f "$CHUNKS_FILE" ]]; then
    CHUNKS=$(python3 -c "import json; d=json.load(open('$CHUNKS_FILE')); print(len(d))" 2>/dev/null)
    echo "   Chunks  : ✅ $CHUNKS chunks indexed"
    echo "   File    : $CHUNKS_FILE"
else
    echo "   Chunks  : ❌ Not ingested"
    echo "   Fix     : cd ~/project && python3 src/ingestion/ingest.py"
fi

# ── OLLAMA ────────────────────────────────────────────
echo ""
echo "── OLLAMA (LLM Runtime)"
OLLAMA=$(ps aux | grep "ollama serve" | grep -v grep)
if [[ -n $OLLAMA ]]; then
    echo "   Status  : ✅ Running"
    echo "   Port    : 11434"
    LAST_LOG=$(cat /tmp/ollama.log 2>/dev/null | tail -1)
    echo "   Last log: $LAST_LOG"
else
    echo "   Status  : ❌ Stopped"
    echo "   Fix     : systemctl start ollama"
fi

# ── MODELS ────────────────────────────────────────────
echo ""
echo "── OLLAMA MODELS"
ollama list 2>/dev/null | while read line; do
    echo "   $line"
done

# ── ACTIVE MODEL ──────────────────────────────────────
echo ""
echo "── ACTIVE MODEL (in config.py)"
MODEL=$(grep "^MODEL" /root/project/src/config.py 2>/dev/null | head -1)
echo "   $MODEL"

# ── POSTGRESQL ────────────────────────────────────────
echo ""
echo "── POSTGRESQL (Relational Database)"
PG=$(pg_isready -h localhost -p 5432 2>/dev/null)
if [[ $PG == *"accepting"* ]]; then
    echo "   Status  : ✅ Running"
    echo "   Port    : 5432"
    ROWS=$(PGPASSWORD=ragpass123 psql -h localhost -U raguser -d nexora -c "SELECT COUNT(*) FROM employees;" -t 2>/dev/null | tr -d ' ')
    USERS=$(PGPASSWORD=ragpass123 psql -h localhost -U raguser -d nexora -c "SELECT COUNT(*) FROM users;" -t 2>/dev/null | tr -d ' ')
    AUDITS=$(PGPASSWORD=ragpass123 psql -h localhost -U raguser -d nexora -c "SELECT COUNT(*) FROM audit_logs;" -t 2>/dev/null | tr -d ' ')
    echo "   Employees: $ROWS rows"
    echo "   Users    : $USERS rows"
    echo "   Audits   : $AUDITS rows"
else
    echo "   Status  : ❌ Stopped"
    echo "   Fix     : systemctl start postgresql"
fi

# ── APP ───────────────────────────────────────────────
echo ""
echo "── GRADIO APP"
APP=$(ps aux | grep "python3 app.py" | grep -v grep)
if [[ -n $APP ]]; then
    echo "   Status  : ✅ Running"
    echo "   Port    : 7861"
    echo "   URL     : http://localhost:7861"
else
    echo "   Status  : ❌ Stopped"
    echo "   Fix     : cd ~/project && python3 app.py"
fi

# ── GPU ───────────────────────────────────────────────
echo ""
echo "── GPU (Tesla T4)"
GPU=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [[ -n $GPU ]]; then
    TEMP=$(echo $GPU | awk -F', ' '{print $1}')
    POWER=$(echo $GPU | awk -F', ' '{print $2}')
    MEM_USED=$(echo $GPU | awk -F', ' '{print $3}')
    MEM_TOTAL=$(echo $GPU | awk -F', ' '{print $4}')
    UTIL=$(echo $GPU | awk -F', ' '{print $5}')
    echo "   Temp    : ${TEMP} °C"
    echo "   Power   : ${POWER} W"
    echo "   VRAM    : ${MEM_USED} MB / ${MEM_TOTAL} MB"
    echo "   Util    : ${UTIL} %"
else
    echo "   Status  : ❌ Not available"
fi

# ── DISK SPACE ────────────────────────────────────────
echo ""
echo "── DISK SPACE"
df -h / | tail -1 | awk '{print "   Used: " $3 " / " $2 " (" $5 " full)"}'

# ── MEMORY ────────────────────────────────────────────
echo ""
echo "── MEMORY"
free -h | grep Mem | awk '{print "   Used: " $3 " / " $2}'

echo ""
echo "████████████████████████████████████████████████████"
echo "  START SEQUENCE:"
echo "  1. cd ~ && docker-compose up -d          # Milvus"
echo "  2. sleep 30                               # Wait"
echo "  3. cd ~/project && python3 src/ingestion/ingest.py  # if needed"
echo "  4. python3 app.py                         # App"
echo "████████████████████████████████████████████████████"
