from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import requests
import json

# ── CONNECT ───────────────────────────────────────────────────
connections.connect(host="localhost", port="19530")
model = SentenceTransformer("BAAI/bge-m3")

# ── QUESTION ──────────────────────────────────────────────────
question = "What is the expense reimbursement policy for business travel?"
print(f"Question: {question}\n")

# ── EMBED QUESTION ────────────────────────────────────────────
query_vector = model.encode(question, normalize_embeddings=True).tolist()

# ── SEARCH MILVUS ─────────────────────────────────────────────
collection = Collection("policy_chunks")
collection.load()

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text", "source_file", "section_title", "document_title"]
)

context = ""
sources = []
for hit in results[0]:
    text         = hit.entity.get("text")
    source_file  = hit.entity.get("source_file")
    section      = hit.entity.get("section_title")
    doc_title    = hit.entity.get("document_title")
    print(f"Retrieved [{hit.score:.3f}]: {text[:150]}...")
    context += f"[Source: {doc_title} — {section}]\n{text}\n\n"
    sources.append(f"{doc_title} ({source_file}) — {section}")

# ── ASK LLM ───────────────────────────────────────────────────
prompt = f"""You are an HR assistant. Answer the question based ONLY on the context below.
At the end of your answer, always cite the source document.
Give a complete and detailed answer — do not stop mid-sentence.

Context:
{context}

Question: {question}

Answer (be complete, do not truncate):"""

response = requests.post("http://localhost:11434/api/generate", json={
    "model":      "qwen2.5:1.5b",
    "prompt":     prompt,
    "stream":     False,
    "options": {
        "num_predict": 512,   # max tokens — évite la troncature
        "temperature": 0.1,   # réponse plus précise et factuelle
    }
})

answer = response.json()["response"]

print(f"\nLLM Answer:\n{answer}")
print(f"\nSources utilisées:")
for s in set(sources):
    print(f"  - {s}")
