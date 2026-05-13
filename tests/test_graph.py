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

question = "Who reports directly to Paul Davis?"
print("="*60)
print(f"Question: {question}")

# ── APPROCHE 1 : bge-m3 + Milvus ──────────────────────────────
print("\n[APPROCHE 1 — bge-m3 + Milvus]")
query_vector = model.encode(question, normalize_embeddings=True).tolist()
collection   = Collection("employee_chunks")
collection.load()

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text"]
)
context = ""
for hit in results[0]:
    print(f"  Retrieved [{hit.score:.3f}]: {hit.entity.get('text')}")
    context += hit.entity.get("text") + "\n"

response1 = requests.post("http://localhost:11434/api/generate", json={
    "model":   "qwen2.5:1.5b",
    "prompt":  f"Answer based only on this context:\n{context}\nQuestion: {question}\nAnswer:",
    "stream":  False,
    "options": {"num_predict": 128, "temperature": 0.1}
})
print(f"\n  LLM Answer: {response1.json()['response']}")

# ── APPROCHE 2 : Graph SQL ─────────────────────────────────────
print("\n[APPROCHE 2 — Graph SQL]")
cursor.execute("""
    SELECT n.name, e2.role
    FROM graph_edges e
    JOIN graph_nodes n ON e.from_node = n.node_id
    JOIN employees e2 ON n.employee_id = e2.employee_id
    WHERE e.to_node = 'emp_2'
    AND e.relationship = 'REPORTS_TO'
    ORDER BY n.name
""")
reports    = cursor.fetchall()
names_list = "\n".join([f"{i+1}. {r[0]} ({r[1]})" for i, r in enumerate(reports)])
print(f"  Graph result: {len(reports)} people report to Paul Davis:")
print(f"{names_list}")

response2 = requests.post("http://localhost:11434/api/generate", json={
    "model":   "qwen2.5:1.5b",
    "prompt":  f"Answer precisely. Do not confuse the direction of reporting.\nThe following {len(reports)} people report directly TO Paul Davis (Paul Davis is their manager):\n{names_list}\nQuestion: {question}\nAnswer:",
    "stream":  False,
    "options": {"num_predict": 256, "temperature": 0.1}
})
print(f"\n  LLM Answer: {response2.json()['response']}")


cursor.close()
conn.close()
