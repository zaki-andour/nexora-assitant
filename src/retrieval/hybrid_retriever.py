from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json
import re

# ── CONFIGURATION ─────────────────────────────────────────────
MILVUS_HOST   = "localhost"
MILVUS_PORT   = "19530"
MODEL_NAME    = "BAAI/bge-m3"
TOP_K         = 5
CHUNKS_FILE   = "hr_dataset/processed/text/chunks.json"

# ── LOAD MODEL ────────────────────────────────────────────────
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
model = SentenceTransformer(MODEL_NAME)

# ── LOAD CHUNKS FOR BM25 ──────────────────────────────────────
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

# Tokenise chaque chunk pour BM25
def tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())

corpus        = [c["text"] for c in all_chunks]
tokenized     = [tokenize(t) for t in corpus]
bm25          = BM25Okapi(tokenized)

# ── HYBRID RETRIEVAL ──────────────────────────────────────────

def hybrid_search(question: str, top_k: int = TOP_K) -> list:

    # ── 1. Vector search via Milvus ──
    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    collection   = Collection("policy_chunks")
    collection.load()

    vector_results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text", "source_file", "section_title", "document_title", "chunk_id"]
    )

    vector_scores = {}
    for hit in vector_results[0]:
        chunk_id = hit.entity.get("chunk_id")
        vector_scores[chunk_id] = {
            "score":         hit.score,
            "text":          hit.entity.get("text"),
            "source_file":   hit.entity.get("source_file"),
            "section_title": hit.entity.get("section_title"),
            "document_title":hit.entity.get("document_title"),
            "chunk_id":      chunk_id,
        }

    # ── 2. BM25 search ──
    tokenized_query = tokenize(question)
    bm25_scores     = bm25.get_scores(tokenized_query)

    # Get top_k BM25 results
    bm25_top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    bm25_results = {}
    max_bm25     = max(bm25_scores) if max(bm25_scores) > 0 else 1
    for idx in bm25_top_indices:
        chunk    = all_chunks[idx]
        chunk_id = chunk["chunk_id"]
        bm25_results[chunk_id] = {
            "score":         bm25_scores[idx] / max_bm25,  # normalize 0-1
            "text":          chunk["text"],
            "source_file":   chunk["source_file"],
            "section_title": chunk["section_title"],
            "document_title":chunk["document_title"],
            "chunk_id":      chunk_id,
        }

    # ── 3. Fusion (Reciprocal Rank Fusion) ──
    all_chunk_ids = set(vector_scores.keys()) | set(bm25_results.keys())
    fused = []

    for chunk_id in all_chunk_ids:
        v_score = vector_scores.get(chunk_id, {}).get("score", 0)
        b_score = bm25_results.get(chunk_id, {}).get("score", 0)

        # Combined score — equal weight
        combined = 0.5 * v_score + 0.5 * b_score

        chunk_data = vector_scores.get(chunk_id) or bm25_results.get(chunk_id)
        chunk_data["combined_score"] = combined
        fused.append(chunk_data)

    # Sort by combined score
    fused = sorted(fused, key=lambda x: x["combined_score"], reverse=True)[:top_k]
    return fused


# ── TEST ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import requests

    question = "How many contractors work in the Finance department?"
    print(f"Question: {question}\n")

    results = hybrid_search(question)

    context = ""
    sources = []
    for r in results:
        print(f"Retrieved [{r['combined_score']:.3f}] vector+BM25: {r['text'][:150]}...")
        context += f"[Source: {r['document_title']} — {r['section_title']}]\n{r['text']}\n\n"
        sources.append(f"{r['document_title']} ({r['source_file']}) — {r['section_title']}")

    prompt = f"""You are an HR assistant. Answer the question based ONLY on the context below.
At the end of your answer, always cite the source document.
Give a complete and detailed answer.

Context:
{context}

Question: {question}

Answer:"""

    response = requests.post("http://localhost:11434/api/generate", json={
        "model":   "qwen2.5:1.5b",
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "num_predict": 512,
            "temperature": 0.1,
        }
    })

    answer = response.json()["response"]
    print(f"\nLLM Answer:\n{answer}")
    print(f"\nSources:")
    for s in set(sources):
        print(f"  - {s}")
