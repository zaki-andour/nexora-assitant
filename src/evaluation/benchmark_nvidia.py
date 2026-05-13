import subprocess
import time
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.logger import get_logger
logger = get_logger("benchmark")

# ── GPU INFO ──────────────────────────────────────────
def get_gpu_info():
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,driver_version,count",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    values = result.stdout.strip().split(", ")
    return {
        "gpu_name":        values[0].strip(),
        "vram_total_mb":   values[1].strip(),
        "vram_free_mb":    values[2].strip(),
        "vram_used_mb":    values[3].strip(),
        "temperature_c":   values[4].strip(),
        "power_draw_w":    values[5].strip(),
        "power_limit_w":   values[6].strip(),
        "gpu_utilization": values[7].strip() + "%",
        "mem_utilization": values[8].strip() + "%",
        "driver":          values[9].strip(),
        "gpu_count":       values[10].strip(),
    }

# ── LATENCY BENCHMARK ─────────────────────────────────
def benchmark_latency():
    import requests
    from pymilvus import connections, Collection
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import re

    from src.config import MILVUS_HOST, MILVUS_PORT, OLLAMA_URL, MODEL, CHUNKS_FILE, TIMEOUT

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    embed_model = SentenceTransformer("BAAI/bge-m3")
    collection  = Collection("policy_chunks")
    collection.load()

    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    bm25     = BM25Okapi([tokenize(c["text"]) for c in chunks])
    question = "How many annual leave days are full-time employees entitled to?"

    # Embedding
    t0      = time.time()
    vector  = embed_model.encode(question, normalize_embeddings=True).tolist()
    t_embed = time.time() - t0

    # Milvus
    t0 = time.time()
    collection.search(
        data=[vector], anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=5, output_fields=["text"]
    )
    t_milvus = time.time() - t0

    # BM25
    t0        = time.time()
    bm25_temp = BM25Okapi([tokenize(c["text"]) for c in chunks])
    bm25_temp.get_scores(tokenize(question))
    t_bm25    = time.time() - t0

    # LLM
    t0 = time.time()
    requests.post(OLLAMA_URL, json={
        "model":   MODEL,
        "prompt":  f"Answer briefly: {question}",
        "stream":  False,
        "options": {"num_predict": 256}
    }, timeout=TIMEOUT)
    t_llm = time.time() - t0

    # GPU metrics during LLM inference
    gpu_during = get_gpu_info()

    return {
        "embedding_ms":       round(t_embed * 1000),
        "milvus_ms":          round(t_milvus * 1000),
        "bm25_ms":            round(t_bm25 * 1000),
        "llm_sec":            round(t_llm, 1),
        "total_sec":          round(t_embed + t_milvus + t_bm25 + t_llm, 1),
        "gpu_during_llm": {
            "power_draw_w":    gpu_during["power_draw_w"],
            "gpu_utilization": gpu_during["gpu_utilization"],
            "vram_used_mb":    gpu_during["vram_used_mb"],
            "temperature_c":   gpu_during["temperature_c"],
        }
    }

# ── RETRIEVAL BENCHMARK ───────────────────────────────
def benchmark_retrieval():
    import re
    from pymilvus import connections, Collection
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from src.config import MILVUS_HOST, MILVUS_PORT, CHUNKS_FILE

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    embed_model = SentenceTransformer("BAAI/bge-m3")
    collection  = Collection("policy_chunks")
    collection.load()

    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    with open("hr_dataset/structured/qa_test_set.json") as f:
        qa_data = [q for q in json.load(f) if q["modality"] == "text"]

    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    hit3, hit5, mrr = 0, 0, []

    for qa in qa_data:
        query  = qa["query"]
        source = qa["source"]
        vector = embed_model.encode(query, normalize_embeddings=True).tolist()

        results = collection.search(
            data=[vector], anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=5, output_fields=["source_file", "chunk_id"]
        )

        vector_scores = {}
        for hit in results[0]:
            vector_scores[hit.entity.get("chunk_id")] = {
                "score":       hit.score,
                "source_file": hit.entity.get("source_file"),
                "chunk_id":    hit.entity.get("chunk_id"),
            }

        bm25_scores = bm25.get_scores(tokenize(query))
        max_bm25    = max(bm25_scores) if max(bm25_scores) > 0 else 1
        top_indices = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)[:5]

        bm25_results = {}
        for idx in top_indices:
            c = chunks[idx]
            bm25_results[c["chunk_id"]] = {
                "score":       bm25_scores[idx] / max_bm25,
                "source_file": c["source_file"],
                "chunk_id":    c["chunk_id"],
            }

        all_ids = set(vector_scores.keys()) | set(bm25_results.keys())
        fused   = []
        for cid in all_ids:
            v  = vector_scores.get(cid, {}).get("score", 0)
            b  = bm25_results.get(cid, {}).get("score", 0)
            cd = vector_scores.get(cid) or bm25_results.get(cid)
            cd["combined_score"] = 0.5 * v + 0.5 * b
            fused.append(cd)

        fused = sorted(fused, key=lambda x: x["combined_score"], reverse=True)[:5]

        rank = None
        for i, c in enumerate(fused, 1):
            if c["source_file"] == source:
                rank = i
                break

        if rank and rank <= 3: hit3 += 1
        if rank and rank <= 5: hit5 += 1
        mrr.append(1/rank if rank else 0)

    total = len(qa_data)
    return {
        "total_questions": total,
        "hit_rate_at_3":   round(hit3/total*100, 1),
        "hit_rate_at_5":   round(hit5/total*100, 1),
        "mrr":             round(sum(mrr)/len(mrr), 3),
    }

# ── MAIN ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("NVIDIA TESLA T4 — BASELINE BENCHMARK")
    print("="*60)

    print("\n[1/3] GPU Info...")
    gpu = get_gpu_info()
    print(f"  GPU Name         : {gpu['gpu_name']}")
    print(f"  GPU Count        : {gpu['gpu_count']}")
    print(f"  VRAM Total       : {gpu['vram_total_mb']} MB")
    print(f"  VRAM Used        : {gpu['vram_used_mb']} MB")
    print(f"  VRAM Free        : {gpu['vram_free_mb']} MB")
    print(f"  Temperature      : {gpu['temperature_c']} C")
    print(f"  Power Draw       : {gpu['power_draw_w']} W")
    print(f"  Power Limit      : {gpu['power_limit_w']} W")
    print(f"  GPU Utilization  : {gpu['gpu_utilization']}")
    print(f"  MEM Utilization  : {gpu['mem_utilization']}")
    print(f"  Driver           : {gpu['driver']}")

    print("\n[2/3] Latency Benchmark...")
    latency = benchmark_latency()
    print(f"  Embedding        : {latency['embedding_ms']} ms")
    print(f"  Milvus search    : {latency['milvus_ms']} ms")
    print(f"  BM25 search      : {latency['bm25_ms']} ms")
    print(f"  LLM generation   : {latency['llm_sec']} sec")
    print(f"  Total            : {latency['total_sec']} sec")
    print(f"  --- GPU during LLM ---")
    print(f"  Power draw       : {latency['gpu_during_llm']['power_draw_w']} W")
    print(f"  GPU utilization  : {latency['gpu_during_llm']['gpu_utilization']}")
    print(f"  VRAM used        : {latency['gpu_during_llm']['vram_used_mb']} MB")
    print(f"  Temperature      : {latency['gpu_during_llm']['temperature_c']} C")

    print("\n[3/3] Retrieval Quality...")
    retrieval = benchmark_retrieval()
    print(f"  Total questions  : {retrieval['total_questions']}")
    print(f"  Hit Rate@3       : {retrieval['hit_rate_at_3']}%")
    print(f"  Hit Rate@5       : {retrieval['hit_rate_at_5']}%")
    print(f"  MRR              : {retrieval['mrr']}")

    # Sauvegarde JSON
    results = {
        "hardware":  "NVIDIA Tesla T4",
        "driver":    gpu["driver"],
        "cuda":      "12.7",
        "model":     "qwen2.5:7b q3_k_m",
        "gpu":       gpu,
        "latency":   latency,
        "retrieval": retrieval,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/nvidia_t4_baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Benchmark saved to benchmarks/nvidia_t4_baseline.json")
    print("="*60)
