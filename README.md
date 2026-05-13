# Nexora HR Assistant — RAG Pipeline

Enterprise Sovereign AI Assistant built on Huawei Cloud ECS.

## Stack
- **LLMs**: Qwen2.5:7b + DeepSeek-R1:7b (via Ollama)
- **Embeddings**: BGE-M3 (1024-dim)
- **Vector DB**: Milvus
- **Relational DB**: PostgreSQL
- **Interface**: Gradio
- **GPU**: NVIDIA Tesla T4

## Features
- Multilingual support (EN, FR, AR, ES, DE, ZH)
- Hybrid retrieval (BM25 + Vector)
- Query decomposition for complex questions
- RLHF feedback store
- Fuzzy name matching (pg_trgm)

## Setup
```bash
cd ~ && docker-compose up -d
cd ~/project && python3 app.py
```
