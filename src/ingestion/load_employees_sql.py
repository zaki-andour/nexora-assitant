import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import psycopg2
from src.config import DB_CONFIG

CSV_FILE = "hr_dataset/structured/employees.csv"

# ── CONNECT ───────────────────────────────────────────────────
conn   = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# ── CREATE TABLE ──────────────────────────────────────────────
cursor.execute("""
    DROP TABLE IF EXISTS employees;
    CREATE TABLE employees (
        employee_id   INTEGER PRIMARY KEY,
        name          VARCHAR(128),
        department    VARCHAR(64),
        role          VARCHAR(128),
        manager_id    INTEGER,
        salary_band   VARCHAR(16),
        location      VARCHAR(64),
        contract_type VARCHAR(32),
        start_date    DATE,
        email         VARCHAR(128),
        access_level  VARCHAR(32)
    );
""")

# ── LOAD CSV ──────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO employees VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        int(row["employee_id"]),
        row["name"],
        row["department"],
        row["role"],
        int(row["manager_id"]),
        row["salary_band"],
        row["location"],
        row["contract_type"],
        row["start_date"],
        row["email"],
        row["access_level"]
    ))

conn.commit()
print(f" {len(df)} employees loaded into PostgreSQL")

# ── TEST QUERY ────────────────────────────────────────────────
cursor.execute("""
    SELECT COUNT(*) FROM employees 
    WHERE contract_type = 'Contractor' AND department = 'Finance'
""")
count = cursor.fetchone()[0]
print(f" Contractors in Finance: {count}")

cursor.close()
conn.close()
