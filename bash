
████████████████████████████████████████████████████
  NEXORA RAG PIPELINE — SYSTEM STATUS
████████████████████████████████████████████████████

── MILVUS (Vector Database)
   Status  : ✅ Running
   Port    : 19530

── MILVUS CHUNKS (Policy Documents)
   Chunks  : ✅ 117 chunks indexed
   File    : /root/project/hr_dataset/processed/text/chunks.json

── OLLAMA (LLM Runtime)
   Status  : ✅ Running
   Port    : 11434
   Last log: Error: listen tcp 127.0.0.1:11434: bind: address already in use

── OLLAMA MODELS
   NAME              ID              SIZE      MODIFIED
   deepseek-r1:7b    2875ba301f5b    3.8 GB    22 minutes ago
   qwen2.5-q4:7b     781392918eca    4.7 GB    2 days ago
   qwen2.5:1.5b      65ec06548149    986 MB    2 weeks ago
   qwen2.5:7b        0b3a5b58ab72    3.8 GB    2 weeks ago

── ACTIVE MODEL (in config.py)
   MODEL = "qwen2.5-q4:7b"

── POSTGRESQL (Relational Database)
   Status  : ✅ Running
   Port    : 5432
   Employees: 420 rows
   Users    : 5 rows
   Audits   : 235 rows

── GRADIO APP
   Status  : ❌ Stopped
   Fix     : cd ~/project && python3 app.py

── GPU (Tesla T4)
   Temp    : 30 °C
   Power   : 9.51 W
   VRAM    : 1 MB / 15360 MB
   Util    : 0 %

── DISK SPACE
   Used: 103G / 197G (55% full)

── MEMORY
   Used: 3.0Gi / 31Gi

████████████████████████████████████████████████████
  START SEQUENCE:
  1. cd ~ && docker-compose up -d          # Milvus
  2. sleep 30                               # Wait
  3. cd ~/project && python3 src/ingestion/ingest.py  # if needed
  4. python3 app.py                         # App
████████████████████████████████████████████████████
