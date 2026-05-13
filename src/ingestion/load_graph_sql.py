import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import psycopg2
from src.config import DB_CONFIG

NODES_FILE = "hr_dataset/graph/graph_nodes.csv"
EDGES_FILE = "hr_dataset/graph/graph_edges.csv"

# ── CONNECT ───────────────────────────────────────────────────
conn   = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# ── CREATE TABLES ─────────────────────────────────────────────
cursor.execute("""
    DROP TABLE IF EXISTS graph_edges;
    DROP TABLE IF EXISTS graph_nodes;

    CREATE TABLE graph_nodes (
        node_id       VARCHAR(64) PRIMARY KEY,
        label         VARCHAR(32),
        name          VARCHAR(128),
        employee_id   INTEGER,
        salary_band   VARCHAR(16),
        contract_type VARCHAR(32),
        start_date    VARCHAR(32),
        email         VARCHAR(128)
    );

    CREATE TABLE graph_edges (
        from_node     VARCHAR(64),
        to_node       VARCHAR(64),
        relationship  VARCHAR(32)
    );
""")

# ── LOAD NODES ────────────────────────────────────────────────
nodes_df = pd.read_csv(NODES_FILE)
for _, row in nodes_df.iterrows():
    cursor.execute("""
        INSERT INTO graph_nodes VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        row["node_id"],
        row["label"],
        row["name"],
        None if pd.isna(row["employee_id"]) else int(row["employee_id"]),
        None if pd.isna(row["salary_band"]) else row["salary_band"],
        None if pd.isna(row["contract_type"]) else row["contract_type"],
        None if pd.isna(row["start_date"]) else row["start_date"],
        None if pd.isna(row["email"]) else row["email"],
    ))

print(f"  {len(nodes_df)} nodes loaded")

# ── LOAD EDGES ────────────────────────────────────────────────
edges_df = pd.read_csv(EDGES_FILE)
for _, row in edges_df.iterrows():
    cursor.execute("""
        INSERT INTO graph_edges VALUES (%s,%s,%s)
    """, (
        row["from"],
        row["to"],
        row["relationship"],
    ))

print(f"  {len(edges_df)} edges loaded")

# ── TEST QUERIES ──────────────────────────────────────────────

# Test 1 — Who manages Engineering?
cursor.execute("""
    SELECT n.name 
    FROM graph_edges e
    JOIN graph_nodes n ON e.from_node = n.node_id
    WHERE e.to_node = 'dept_Engineering'
    AND e.relationship = 'MANAGES'
""")
manager = cursor.fetchone()
print(f"\n  Who manages Engineering? → {manager[0]}")

# Test 2 — Who reports directly to Paul Davis?
cursor.execute("""
    SELECT n.name
    FROM graph_edges e
    JOIN graph_nodes n ON e.from_node = n.node_id
    WHERE e.to_node = 'emp_2'
    AND e.relationship = 'REPORTS_TO'
""")
reports = cursor.fetchall()
print(f" Who reports to Paul Davis?")
for r in reports:
    print(f"   - {r[0]}")

conn.commit()
cursor.close()
conn.close()
