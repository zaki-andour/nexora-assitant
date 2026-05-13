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

# ── POSTGRESQL ────────────────────────────────────────
echo ""
echo "── POSTGRESQL (Relational Database)"
PG=$(pg_isready -h localhost -p 5432 2>/dev/null)
if [[ $PG == *"accepting"* ]]; then
    echo "   Status  : ✅ Running"
    echo "   Port    : 5432"
    ROWS=$(PGPASSWORD=ragpass123 psql -h localhost -U raguser -d nexora -c "SELECT COUNT(*) FROM employees;" -t 2>/dev/null | tr -d ' ')
    echo "   Employees: $ROWS rows"
else
    echo "   Status  : ❌ Stopped"
    echo "   Fix     : systemctl start postgresql"
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
