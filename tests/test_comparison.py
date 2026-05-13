import sys
sys.path.insert(0, '.')
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import psycopg2
import requests
from src.config import MILVUS_HOST, MILVUS_PORT, DB_CONFIG

# ── SETUP ─────────────────────────────────────────────────────
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
model  = SentenceTransformer("BAAI/bge-m3")
conn   = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

question = "How many contractors work in the Finance department?"
print(f"Question: {question}")
print("="*60)

# ── APPROCHE 1 : bge-m3 + Milvus ──────────────────────────────
print("\n[APPROCHE 1 — bge-m3 + Milvus]")
query_vector = model.encode(question, normalize_embeddings=True).tolist()
collection   = Collection("employee_chunks")
collection.load()

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["text"]
)

context_milvus = ""
for hit in results[0]:
    print(f"  Retrieved [{hit.score:.3f}]: {hit.entity.get('text')}")
    context_milvus += hit.entity.get("text") + "\n"

response1 = requests.post("http://localhost:11434/api/generate", json={
    "model":   "qwen2.5:1.5b",
    "prompt":  f"Answer based only on this context:\n{context_milvus}\nQuestion: {question}\nAnswer:",
    "stream":  False,
    "options": {"num_predict": 256, "temperature": 0.1}
})
print(f"\n  LLM Answer: {response1.json()['response']}")

# ── APPROCHE 2 : SQL direct ────────────────────────────────────
print("\n[APPROCHE 2 — SQL direct]")
cursor.execute("""
    SELECT name, role FROM employees
    WHERE contract_type = 'Contractor'
    AND department = 'Finance'
    ORDER BY name
""")
rows        = cursor.fetchall()
exact_count = len(rows)
names_list  = "\n".join([f"{i+1}. {r[0]} ({r[1]})" for i, r in enumerate(rows)])
print(f"  SQL result: {exact_count} contractors in Finance")
print(f"{names_list}")

response2 = requests.post("http://localhost:11434/api/generate", json={
    "model":   "qwen2.5:1.5b",
    "prompt":  f"Answer this question precisely:\nThere are {exact_count} contractors in the Finance department:\n{names_list}\nQuestion: {question}\nAnswer:",
    "stream":  False,
    "options": {"num_predict": 128, "temperature": 0.1}
})
print(f"  LLM Answer: {response2.json()['response']}")


cursor.close()
conn.close()
