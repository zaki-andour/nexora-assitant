import requests
import json
import psycopg2
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.config import (
    MODEL, OLLAMA_URL, TIMEOUT, NUM_PREDICT, TEMPERATURE,
    MILVUS_HOST, MILVUS_PORT, TOP_K, DB_CONFIG, CHUNKS_FILE
)
from src.routing.query_classifier import classify_query
from src.utils.logger import get_logger

logger = get_logger("router")

# ── SETUP ─────────────────────────────────────────────
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
model      = SentenceTransformer("BAAI/bge-m3")
collection = Collection("policy_chunks")
collection.load()

with open(CHUNKS_FILE, "r") as f:
    all_chunks = json.load(f)

def tokenize(text):
    return re.findall(r'\w+', text.lower())

bm25 = BM25Okapi([tokenize(c["text"]) for c in all_chunks])

# ── RETRIEVE TEXT ─────────────────────────────────────
def retrieve_text(question, top_k=TOP_K):
    logger.info(f"Vector + BM25 search: {question[:50]}")
    query_vector = model.encode(question, normalize_embeddings=True).tolist()

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
            "score":          hit.score,
            "text":           hit.entity.get("text"),
            "source_file":    hit.entity.get("source_file"),
            "section_title":  hit.entity.get("section_title"),
            "document_title": hit.entity.get("document_title"),
            "chunk_id":       chunk_id,
        }

    bm25_scores      = bm25.get_scores(tokenize(question))
    bm25_top_indices = sorted(range(len(bm25_scores)),
                              key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

    bm25_results = {}
    for idx in bm25_top_indices:
        chunk    = all_chunks[idx]
        chunk_id = chunk["chunk_id"]
        bm25_results[chunk_id] = {
            "score":          bm25_scores[idx] / max_bm25,
            "text":           chunk["text"],
            "source_file":    chunk["source_file"],
            "section_title":  chunk["section_title"],
            "document_title": chunk["document_title"],
            "chunk_id":       chunk_id,
        }

    all_ids = set(vector_scores.keys()) | set(bm25_results.keys())
    fused   = []
    for chunk_id in all_ids:
        v = vector_scores.get(chunk_id, {}).get("score", 0)
        b = bm25_results.get(chunk_id, {}).get("score", 0)
        chunk_data = vector_scores.get(chunk_id) or bm25_results.get(chunk_id)
        chunk_data["combined_score"] = 0.5 * v + 0.5 * b
        fused.append(chunk_data)

    results = sorted(fused, key=lambda x: x["combined_score"], reverse=True)[:top_k]
    logger.info(f"Retrieved {len(results)} chunks")
    return results

# ── RETRIEVE STRUCTURED ───────────────────────────────
def retrieve_structured(question, rbac_clause=""):
    logger.info(f"SQL retrieval: {question[:50]}")
    sql_prompt = f"""You are a PostgreSQL expert. Generate a query for the employees table.
The question may be in any language — understand the intent first.
Common translations: "chef/patron/directeur general" = CEO, "responsable" = manager.

Table: employees(employee_id, name, department, role, manager_id, salary_band, location, contract_type, start_date, email, access_level)

Exact values:
  contract_type: 'Full-Time', 'Contractor', 'Part-Time', 'Intern'
  department: 'Engineering', 'HR', 'Finance', 'Sales', 'Marketing', 'Operations', 'Product', 'Legal', 'Data', 'Security', 'Executive'
  salary_band: 'Band1', 'Band2', 'Band3', 'Band4', 'Band5'

Rules:
- Return ONLY the SQL query — no explanation, no markdown, no comments
- ONLY ONE table exists: employees — NEVER reference or JOIN any other table
- department is already a column in employees — never join to get it
- Use similarity() ONLY for person name searches — NEVER for contract_type, department, role, salary_band
- Interns are identified by contract_type = 'Intern' — NEVER use role = 'Intern'
- If question contains [Current user: X, employee_id: N], use WHERE employee_id = N for personal queries like 'my contract', 'my role', 'my department'
- For filtering by category: use exact WHERE contract_type = 'X' or WHERE department = 'X'
- For person name search: WHERE similarity(LOWER(name), LOWER('X')) > 0.35 ORDER BY similarity(LOWER(name), LOWER('X')) DESC
- For manager lookup: LEFT JOIN employees m ON e.manager_id = m.employee_id
- For year counts: SELECT EXTRACT(YEAR FROM start_date) AS year, COUNT(*) AS count FROM employees WHERE EXTRACT(YEAR FROM start_date) IN (X,Y) GROUP BY year ORDER BY year
- For role queries like CEO/CTO/CFO: SELECT name, role, department, email FROM employees WHERE role = 'CEO' LIMIT 1
- NEVER add manager_id conditions unless explicitly asked
- access_level values are text: 'employee', 'hr_staff', 'hr_leadership' — ordered low to high
- For highest access_level: WHERE access_level = 'hr_leadership'
- NEVER use MAX() on access_level — it is text not numeric
- For subqueries with MAX/MIN: SELECT name, col FROM employees WHERE col = (SELECT MAX(col) FROM employees) LIMIT 20
- Select only relevant columns — max 4 columns, always include name unless COUNT or MAX/MIN query
- Limit results to 20 rows

Question: {question}

SQL:"""

    response = requests.post(OLLAMA_URL, json={
        "model":   MODEL,
        "prompt":  sql_prompt,
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 400}
    }, timeout=TIMEOUT)

    sql = response.json()["response"].strip()
    sql = re.sub(r'```sql|```', '', sql).strip()
    # Force include name column in SELECT if not present
    import re as _re
    select_cols = _re.search(r'SELECT\s+(.*?)\s+FROM', sql, _re.IGNORECASE)
    has_aggregate = any(kw in sql.upper() for kw in ['COUNT(', 'MAX(', 'MIN(', 'AVG(', 'SUM('])
    if select_cols and 'name' not in select_cols.group(1).lower() and not has_aggregate:
        sql = sql.replace('SELECT ', 'SELECT name, ', 1)
    # Fix subquery LIMIT issue
    sql = re.sub(r'\)\s*LIMIT\s+\d+\s*\)', ')', sql, flags=re.IGNORECASE)

    # Fix CEO/CTO queries that incorrectly add manager_id IS NULL
    if 'manager_id IS NULL' in sql and any(r in sql.upper() for r in ["'CEO'", "'CTO'", "'CFO'"]):
        sql = re.sub(r'AND\s+manager_id\s+IS\s+NULL', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'manager_id\s+IS\s+NULL\s+AND', '', sql, flags=re.IGNORECASE)
        sql = sql.strip()

    logger.info(f"Generated SQL: {sql[:100]}")


    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows    = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        logger.info(f"SQL returned {len(rows)} rows")
        return {"sql": sql, "columns": columns, "rows": rows, "error": None}
    except Exception as e:
        logger.error(f"SQL error: {e}")
        return {"sql": sql, "columns": [], "rows": [], "error": str(e)}

# ── RETRIEVE GRAPH ────────────────────────────────────
def retrieve_graph(question, rbac_clause=""):
    logger.info(f"Graph SQL retrieval: {question[:50]}")

    # Use LLM to extract intent and entities from question in any language
    extract_prompt = f"""Extract information from this HR question (any language).
Return ONLY this JSON:
{{
  "intent": "manages" or "reports_to",
  "department": "department name in English or null",
  "person_name": "full name or null"
}}

Examples:
- "Who manages Engineering?" → {{"intent": "manages", "department": "Engineering", "person_name": null}}
- "Qui gère le département Finance ?" → {{"intent": "manages", "department": "Finance", "person_name": null}}
- "Who reports to Paul Davis?" → {{"intent": "reports_to", "department": null, "person_name": "Paul Davis"}}
- "Qui rapporte à Julia Jackson ?" → {{"intent": "reports_to", "department": null, "person_name": "Julia Jackson"}}

Question: {question}"""

    response = requests.post(OLLAMA_URL, json={
        "model":   MODEL,
        "prompt":  extract_prompt,
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 100}
    }, timeout=TIMEOUT)

    raw = response.json()["response"].strip()
    try:
        extracted = json.loads(raw)
    except:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        extracted = json.loads(match.group()) if match else {}

    intent     = extracted.get("intent", "")
    department = extracted.get("department")
    person     = extracted.get("person_name")

    conn    = psycopg2.connect(**DB_CONFIG)
    cursor  = conn.cursor()
    results = []

    if intent == "manages" and department:
        dept = department.capitalize()
        cursor.execute("""
            SELECT n.name, e.role, e.email, e.location
            FROM graph_edges ge
            JOIN graph_nodes n ON ge.from_node = n.node_id
            JOIN employees e ON n.employee_id = e.employee_id
            WHERE ge.to_node = %s AND ge.relationship = 'MANAGES'
        """, (f"dept_{dept}",))
        results = cursor.fetchall()

    elif intent == "reports_to" and person:
        cursor.execute("SELECT employee_id FROM employees WHERE name = %s", (person,))
        row = cursor.fetchone()
        if row:
            emp_node_id = f"emp_{row[0]}"
            cursor.execute("""
                SELECT n.name, e.role, e.location, e.email
                FROM graph_edges ge
                JOIN graph_nodes n ON ge.from_node = n.node_id
                JOIN employees e ON n.employee_id = e.employee_id
                WHERE ge.to_node = %s AND ge.relationship = 'REPORTS_TO'
                ORDER BY n.name
            """, (emp_node_id,))
            results = cursor.fetchall()

    cursor.close()
    conn.close()
    logger.info(f"Graph returned {len(results)} rows — intent: {intent}")
    return results

# ── BUILD CONTEXT ─────────────────────────────────────
def build_context(category, question, rbac_clause=""):
    """Route to correct source and build context string for LLM."""
    context = ""
    sources = []

    if category == "TEXT":
        logger.info("Routing to Milvus (policy chunks)")
        chunks = retrieve_text(question)
        for c in chunks:
            context += f"[{c['document_title']} — {c['section_title']}]\n{c['text']}\n\n"
            sources.append(f"{c['document_title']} ({c['source_file']})")
            logger.debug(f"  [{c['combined_score']:.3f}] {c['document_title']} — {c['section_title']}")

    elif category == "STRUCTURED":
        logger.info("Routing to PostgreSQL (employees table)")
        result = retrieve_structured(question, rbac_clause=rbac_clause)
        if result["error"]:
            context = f"SQL Error: {result['error']}"
            logger.error(f"SQL failed: {result['error']}")
        else:
            rows = result["rows"]
            cols = result["columns"]
            context = f"Query results ({len(rows)} rows):\n"
            for row in rows:
                context += "  " + " | ".join(f"{cols[i]}: {row[i]}" for i in range(len(cols))) + "\n"
            if rows:
                context += f"\nNOTE: Use EXACTLY these names from the database, not the names from the question.\n"
            sources.append("PostgreSQL employees table")

    elif category == "GRAPH":
        logger.info("Routing to PostgreSQL (graph tables)")
        rows = retrieve_graph(question, rbac_clause=rbac_clause)
        if rows:
            q = question.lower()
            if "who manages" in q or "head of" in q or "qui gère" in q or "qui dirige" in q or "qui manage" in q:
                dept_match = re.search(
                    r'(engineering|hr|finance|sales|marketing|operations|product|legal|data|security)',
                    q
                )
                dept = dept_match.group(1).capitalize() if dept_match else "the department"
                context = f"The following person manages the {dept} department:\n"
                for row in rows:
                    context += f"  Name: {row[0]}, Role: {row[1]}, Email: {row[2]}, Location: {row[3]}\n"
                context += f"\nThis means {rows[0][0]} ({rows[0][1]}) is the manager of the {dept} department.\n"
            elif "who reports" in q or "reports to" in q:
                name_match = re.search(r'to\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', question)
                manager = name_match.group(1) if name_match else "this manager"
                context = f"The following people report directly to {manager}:\n"
                for row in rows:
                    context += f"  Name: {row[0]}, Role: {row[1]}, Location: {row[2]}, Email: {row[3]}\n"
            else:
                context = "Organizational data:\n"
                for row in rows:
                    context += "  " + " | ".join(str(v) for v in row) + "\n"

            # Enrich with more employee details
            try:
                conn2   = psycopg2.connect(**DB_CONFIG)
                cursor2 = conn2.cursor()
                cursor2.execute("""
                    SELECT name, role, department, location, email, contract_type, start_date
                    FROM employees WHERE name = %s
                """, (rows[0][0],))
                emp = cursor2.fetchone()
                if emp:
                    context += f"\nAdditional details about {emp[0]}:\n"
                    context += f"  Department  : {emp[2]}\n"
                    context += f"  Location    : {emp[3]}\n"
                    context += f"  Email       : {emp[4]}\n"
                    context += f"  Contract    : {emp[5]}\n"
                    context += f"  Start Date  : {emp[6]}\n"
                cursor2.close()
                conn2.close()
            except Exception as e:
                logger.warning(f"Could not enrich graph context: {e}")

            sources.append("PostgreSQL graph tables")
        else:
            context = "No graph data found for this query."
            logger.warning("Graph query returned no results") 

    elif category == "HYBRID":
        logger.info("Routing to Milvus + PostgreSQL (hybrid)")
        chunks = retrieve_text(question, top_k=2)
        for c in chunks:
            context += f"[Policy: {c['document_title']} — {c['section_title']}]\n{c['text']}\n\n"
            sources.append(f"{c['document_title']}")
        result = retrieve_structured(question, rbac_clause=rbac_clause)
        if not result["error"] and result["rows"]:
            context += "Employee data:\n"
            cols = result["columns"]
            for row in result["rows"]:
                context += "  " + " | ".join(f"{cols[i]}: {row[i]}" for i in range(len(cols))) + "\n"
            sources.append("PostgreSQL employees table")

    return context, list(set(sources))
