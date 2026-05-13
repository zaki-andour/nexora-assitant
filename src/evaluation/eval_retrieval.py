import json
import re
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from collections import defaultdict

# ── CONFIGURATION ─────────────────────────────────────────────
QA_FILE      = "hr_dataset/structured/qa_test_set.json"
CHUNKS_FILE  = "hr_dataset/processed/text/chunks.json"
TOP_K        = 5

# ── SETUP ─────────────────────────────────────────────────────
connections.connect(host="localhost", port="19530")
model      = SentenceTransformer("BAAI/bge-m3")
collection = Collection("policy_chunks")
collection.load()

# Load chunks for BM25
with open(CHUNKS_FILE, "r") as f:
    all_chunks = json.load(f)

def tokenize(text):
    return re.findall(r'\w+', text.lower())

corpus    = [c["text"] for c in all_chunks]
tokenized = [tokenize(t) for t in corpus]
bm25      = BM25Okapi(tokenized)

# Load QA test set — only text modality
with open(QA_FILE, "r") as f:
    qa_data = json.load(f)

text_questions = [q for q in qa_data if q["modality"] == "text"]
print(f"Evaluating {len(text_questions)} text questions...\n")

# ── METRICS ───────────────────────────────────────────────────
hit_at_3      = 0
hit_at_5      = 0
mrr_scores    = []
results_log   = []

for qa in text_questions:
    query       = qa["query"]
    answer      = qa["answer"].lower()
    source_file = qa["source"]

    # ── Hybrid search ──
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    vector_results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=TOP_K,
        output_fields=["text", "source_file", "chunk_id"]
    )

    vector_scores = {}
    for hit in vector_results[0]:
        chunk_id = hit.entity.get("chunk_id")
        vector_scores[chunk_id] = {
            "score":       hit.score,
            "text":        hit.entity.get("text"),
            "source_file": hit.entity.get("source_file"),
            "chunk_id":    chunk_id,
        }

    bm25_scores      = bm25.get_scores(tokenize(query))
    bm25_top_indices = sorted(range(len(bm25_scores)),
                              key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
    max_bm25         = max(bm25_scores) if max(bm25_scores) > 0 else 1

    bm25_results = {}
    for idx in bm25_top_indices:
        chunk    = all_chunks[idx]
        chunk_id = chunk["chunk_id"]
        bm25_results[chunk_id] = {
            "score":       bm25_scores[idx] / max_bm25,
            "text":        chunk["text"],
            "source_file": chunk["source_file"],
            "chunk_id":    chunk_id,
        }

    # Fusion
    all_ids = set(vector_scores.keys()) | set(bm25_results.keys())
    fused   = []
    for chunk_id in all_ids:
        v = vector_scores.get(chunk_id, {}).get("score", 0)
        b = bm25_results.get(chunk_id, {}).get("score", 0)
        chunk_data = vector_scores.get(chunk_id) or bm25_results.get(chunk_id)
        chunk_data["combined_score"] = 0.5 * v + 0.5 * b
        fused.append(chunk_data)

    fused = sorted(fused, key=lambda x: x["combined_score"], reverse=True)[:TOP_K]

    # ── Evaluate ──
    # A chunk is relevant if it comes from the correct source file
    correct_rank = None
    for rank, chunk in enumerate(fused, start=1):
        if chunk["source_file"] == source_file:
            correct_rank = rank
            break

    # Hit@3
    if correct_rank and correct_rank <= 3:
        hit_at_3 += 1

    # Hit@5
    if correct_rank and correct_rank <= 5:
        hit_at_5 += 1

    # MRR
    mrr_scores.append(1 / correct_rank if correct_rank else 0)

    results_log.append({
        "id":           qa["id"],
        "query":        query,
        "source":       source_file,
        "correct_rank": correct_rank,
        "hit@3":        correct_rank <= 3 if correct_rank else False,
        "hit@5":        correct_rank <= 5 if correct_rank else False,
        "mrr":          1 / correct_rank if correct_rank else 0,
    })

# ── RESULTS ───────────────────────────────────────────────────
total = len(text_questions)
print("="*60)
print("RETRIEVAL EVALUATION RESULTS")
print("="*60)
print(f"Total text questions evaluated : {total}")
print(f"Hit Rate@3                     : {hit_at_3}/{total} = {hit_at_3/total*100:.1f}%")
print(f"Hit Rate@5                     : {hit_at_5}/{total} = {hit_at_5/total*100:.1f}%")
print(f"MRR                            : {sum(mrr_scores)/len(mrr_scores):.3f}")

print("\nDetailed results:")
print(f"{'ID':<10} {'Hit@3':<8} {'Hit@5':<8} {'Rank':<6} {'Query'}")
print("-"*70)
for r in results_log:
    rank = r['correct_rank'] if r['correct_rank'] else 'MISS'
    print(f"{r['id']:<10} {'✅' if r['hit@3'] else '❌':<8} {'✅' if r['hit@5'] else '❌':<8} {str(rank):<6} {r['query'][:50]}")

print("\nFailed questions (not found in top 5):")
failed = [r for r in results_log if not r['hit@5']]
if failed:
    for r in failed:
        print(f"  - {r['id']}: {r['query']}")
else:
    print("  None — all questions found in top 5" )
