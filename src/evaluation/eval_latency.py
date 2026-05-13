import time
import json
import re
import requests
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── SETUP ─────────────────────────────────────────────────────
connections.connect(host="localhost", port="19530")
model      = SentenceTransformer("BAAI/bge-m3")
collection = Collection("policy_chunks")
collection.load()

with open("hr_dataset/processed/text/chunks.json") as f:
    all_chunks = json.load(f)

def tokenize(text):
    return re.findall(r'\w+', text.lower())

bm25 = BM25Okapi([tokenize(c["text"]) for c in all_chunks])

question = "How many annual leave days are full-time employees entitled to?"

# ── 1. Embedding latency ───────────────────────────────────────
t0           = time.time()
query_vector = model.encode(question, normalize_embeddings=True).tolist()
t_embed      = time.time() - t0

# ── 2. Milvus search latency ───────────────────────────────────
t0 = time.time()
collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["text", "source_file"]
)
t_milvus = time.time() - t0

# ── 3. BM25 latency ────────────────────────────────────────────
t0           = time.time()
bm25_temp    = BM25Okapi([tokenize(c["text"]) for c in all_chunks])
scores       = bm25_temp.get_scores(tokenize(question))
t_bm25       = time.time() - t0

# ── 4. LLM latency ────────────────────────────────────────────
t0 = time.time()
requests.post("http://localhost:11434/api/generate", json={
    "model":   "qwen2.5:1.5b",
    "prompt":  f"Answer briefly: {question}",
    "stream":  False,
    "options": {"num_predict": 256, "temperature": 0.1}
})
t_llm = time.time() - t0

# ── 5. Total ──────────────────────────────────────────────────
t_total = t_embed + t_milvus + t_bm25 + t_llm

# ── RESULTS ───────────────────────────────────────────────────
print("="*55)
print("LATENCY EVALUATION")
print("="*55)
print(f"  Embedding (bge-m3)   : {t_embed*1000:.0f} ms")
print(f"  Milvus search        : {t_milvus*1000:.0f} ms")
print(f"  BM25 search          : {t_bm25*1000:.0f} ms")
print(f"  LLM generation       : {t_llm:.1f} sec")
print(f"  Total end-to-end     : {t_total:.1f} sec")
print("="*55)
print(f"\nCibles :")
print(f"  Embedding  < 1000ms  : {'✅' if t_embed < 1 else '❌'}")
print(f"  Milvus     < 500ms   : {'✅' if t_milvus < 0.5 else '❌'}")
print(f"  BM25       < 100ms   : {'✅' if t_bm25 < 0.1 else '❌'}")
print(f"  LLM        < 30sec   : {'✅' if t_llm < 30 else '❌'}")
