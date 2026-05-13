import json
import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MILVUS_HOST    = "localhost"
MILVUS_PORT    = "19530"
EMBEDDED_DIR   = "hr_dataset/processed/embedded"

COLLECTIONS = {
    "policy_chunks":   "hr_dataset/processed/embedded/chunks.json",
    "employee_chunks": "hr_dataset/processed/embedded/employees_chunks.json",
}

EMBEDDING_DIM = 1024   # bge-m3 output dimension

# ── CONNECT ───────────────────────────────────────────────────────────────────

def connect():
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    print("Connected.")

# ── SCHEMA ────────────────────────────────────────────────────────────────────

def create_policy_collection() -> Collection:
    """Schema for policy document chunks."""
    fields = [
        FieldSchema(name="chunk_id",       dtype=DataType.VARCHAR,      max_length=64,  is_primary=True),
        FieldSchema(name="embedding",      dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="text",           dtype=DataType.VARCHAR,      max_length=4096),
        FieldSchema(name="source_file",    dtype=DataType.VARCHAR,      max_length=256),
        FieldSchema(name="document_title", dtype=DataType.VARCHAR,      max_length=256),
        FieldSchema(name="section_number", dtype=DataType.VARCHAR,      max_length=32),
        FieldSchema(name="section_title",  dtype=DataType.VARCHAR,      max_length=256),
        FieldSchema(name="policy_topic",   dtype=DataType.VARCHAR,      max_length=64),
        FieldSchema(name="access_level",   dtype=DataType.VARCHAR,      max_length=32),
        FieldSchema(name="word_count",     dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="HR policy document chunks")
    collection = Collection(name="policy_chunks", schema=schema)
    print("  Collection 'policy_chunks' created.")
    return collection


def create_employee_collection() -> Collection:
    """Schema for employee record chunks."""
    fields = [
        FieldSchema(name="chunk_id",      dtype=DataType.VARCHAR,      max_length=64,  is_primary=True),
        FieldSchema(name="embedding",     dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="text",          dtype=DataType.VARCHAR,      max_length=2048),
        FieldSchema(name="employee_id",   dtype=DataType.INT64),
        FieldSchema(name="manager_id",    dtype=DataType.INT64),
        FieldSchema(name="manager_name",  dtype=DataType.VARCHAR,      max_length=128),
        FieldSchema(name="department",    dtype=DataType.VARCHAR,      max_length=64),
        FieldSchema(name="role",          dtype=DataType.VARCHAR,      max_length=128),
        FieldSchema(name="location",      dtype=DataType.VARCHAR,      max_length=64),
        FieldSchema(name="contract_type", dtype=DataType.VARCHAR,      max_length=32),
        FieldSchema(name="access_level",  dtype=DataType.VARCHAR,      max_length=32),
        # sensitive fields stored as metadata but never embedded
        FieldSchema(name="salary_band",   dtype=DataType.VARCHAR,      max_length=16),
        FieldSchema(name="email",         dtype=DataType.VARCHAR,      max_length=128),
        FieldSchema(name="start_date",    dtype=DataType.VARCHAR,      max_length=32),
    ]
    schema = CollectionSchema(fields, description="HR employee record chunks")
    collection = Collection(name="employee_chunks", schema=schema)
    print("  Collection 'employee_chunks' created.")
    return collection

# ── INGEST ────────────────────────────────────────────────────────────────────

def ingest_policy_chunks(collection: Collection, chunks: list):
    data = [
        [c["chunk_id"]                    for c in chunks],  # chunk_id
        [c["embedding"]                   for c in chunks],  # embedding
        [c["text"][:4000]                 for c in chunks],  # text
        [c.get("source_file", "")         for c in chunks],  # source_file
        [c.get("document_title", "")      for c in chunks],  # document_title
        [c.get("section_number", "")      for c in chunks],  # section_number
        [c.get("section_title", "")       for c in chunks],  # section_title
        [c.get("policy_topic", "")        for c in chunks],  # policy_topic
        [c.get("access_level", "")        for c in chunks],  # access_level
        [c.get("word_count", 0)           for c in chunks],  # word_count
    ]
    collection.insert(data)
    print(f"  Inserted {len(chunks)} policy chunks.")


def ingest_employee_chunks(collection: Collection, chunks: list):
    data = [
        [c["chunk_id"]                    for c in chunks],  # chunk_id
        [c["embedding"]                   for c in chunks],  # embedding
        [c["text"][:2000]                 for c in chunks],  # text
        [c.get("employee_id", 0)          for c in chunks],  # employee_id
        [c.get("manager_id", 0)           for c in chunks],  # manager_id
        [c.get("manager_name", "")        for c in chunks],  # manager_name
        [c.get("department", "")          for c in chunks],  # department
        [c.get("role", "")                for c in chunks],  # role
        [c.get("location", "")            for c in chunks],  # location
        [c.get("contract_type", "")       for c in chunks],  # contract_type
        [c.get("access_level", "")        for c in chunks],  # access_level
        [c.get("salary_band", "")         for c in chunks],  # salary_band
        [c.get("email", "")               for c in chunks],  # email
        [c.get("start_date", "")          for c in chunks],  # start_date
    ]
    collection.insert(data)
    print(f"  Inserted {len(chunks)} employee chunks.")

# ── INDEX ─────────────────────────────────────────────────────────────────────

def create_index(collection: Collection):
    index_params = {
        "metric_type": "COSINE",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"  Index created on '{collection.name}'.")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    connect()

    # ── Policy chunks ──
    print("\nIngesting policy chunks...")
    if utility.has_collection("policy_chunks"):
        utility.drop_collection("policy_chunks")
        print("  Dropped existing 'policy_chunks' collection.")

    with open("hr_dataset/processed/embedded/chunks.json", "r", encoding="utf-8") as f:
        policy_chunks = json.load(f)

    pol_collection = create_policy_collection()
    ingest_policy_chunks(pol_collection, policy_chunks)
    pol_collection.flush()
    print(f"  Flushed policy collection.")
    create_index(pol_collection)
    pol_collection.load()
    print(f"  Total in Milvus: {pol_collection.num_entities} policy chunks")

    # ── Employee chunks ──
    print("\nIngesting employee chunks...")
    if utility.has_collection("employee_chunks"):
        utility.drop_collection("employee_chunks")
        print("  Dropped existing 'employee_chunks' collection.")

    with open("hr_dataset/processed/embedded/employees_chunks.json", "r", encoding="utf-8") as f:
        employee_chunks = json.load(f)

    emp_collection = create_employee_collection()
    ingest_employee_chunks(emp_collection, employee_chunks)
    emp_collection.flush()
    print(f"  Flushed employee collection.")
    create_index(emp_collection)
    emp_collection.load()
    print(f"  Total in Milvus: {emp_collection.num_entities} employee chunks")

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"Ingestion complete.")
    print(f"  policy_chunks   : {pol_collection.num_entities} vectors")
    print(f"  employee_chunks : {emp_collection.num_entities} vectors")
    print(f"{'='*55}")
