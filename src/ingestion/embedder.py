import os
import json
from sentence_transformers import SentenceTransformer

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODEL_NAME   = "BAAI/bge-m3"
CHUNK_FILES  = [
    "hr_dataset/processed/text/chunks.json",
    "hr_dataset/processed/structured/employees_chunks.json",
]
OUTPUT_DIR   = "hr_dataset/processed/embedded"
BATCH_SIZE   = 32

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")
    return model

# ── EMBED CHUNKS ──────────────────────────────────────────────────────────────

def embed_chunks(chunks: list, model: SentenceTransformer) -> list:
    texts = [c["text"] for c in chunks]

    print(f"  Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # normalise for cosine similarity
    )

    # Attach embedding to each chunk
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    return chunks

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()

    for chunk_file in CHUNK_FILES:
        if not os.path.exists(chunk_file):
            print(f"\nSkipping {chunk_file} — file not found")
            continue

        print(f"\nProcessing: {chunk_file}")

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"  Chunks loaded: {len(chunks)}")

        chunks = embed_chunks(chunks, model)

        # Save output — same filename in embedded/ folder
        filename    = os.path.basename(chunk_file)
        output_path = os.path.join(OUTPUT_DIR, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        print(f"  Saved to: {output_path}")
        print(f"  Embedding dimension: {len(chunks[0]['embedding'])}")

    print(f"\n{'='*55}")
    print("Embedding complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*55}")
