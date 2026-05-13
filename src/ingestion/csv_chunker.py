import os
import json
import uuid
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
INPUT_FILE  = "hr_dataset/structured/employees.csv"
OUTPUT_FILE = "hr_dataset/processed/structured/employees_chunks.json"

# ── DATA CLASS ────────────────────────────────────────────────────────────────
@dataclass
class EmployeeChunk:
    chunk_id:       str
    text:           str
    document_type:  str = "structured"
    source_file:    str = "employees.csv"
    employee_id:    int  = 0
    manager_id:     int  = 0
    manager_name:   str  = ""
    salary_band:    str  = ""
    email:          str  = ""
    start_date:     str  = ""
    access_level:   str  = "employee"
    department:     str  = ""
    location:       str  = ""
    contract_type:  str  = ""
    role:           str  = ""

# ── SERIALISATION ─────────────────────────────────────────────────────────────

def serialise_row(row: pd.Series, manager_name: str) -> str:
    if manager_name:
        manager_line = f"reporting to {manager_name}"
    else:
        manager_line = "with no direct manager"

    contract = row["contract_type"].lower()

    text = (
        f"{row['name']} is a {row['role']} in the {row['department']} department, "
        f"based in {row['location']}, on a {contract} contract, "
        f"{manager_line}."
    )
    return text

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def chunk_employee_csv(input_file: str) -> list:
    df = pd.read_csv(input_file)

    id_to_name = dict(zip(df["employee_id"], df["name"]))

    chunks = []
    for _, row in df.iterrows():
        manager_name = id_to_name.get(row["manager_id"], "")
        text = serialise_row(row, manager_name)

        chunk = EmployeeChunk(
            chunk_id      = str(uuid.uuid4()),
            text          = text,
            employee_id   = int(row["employee_id"]),
            manager_id    = int(row["manager_id"]),
            manager_name  = manager_name,
            salary_band   = row["salary_band"],
            email         = row["email"],
            start_date    = row["start_date"],
            access_level  = row["access_level"],
            department    = row["department"],
            location      = row["location"],
            contract_type = row["contract_type"],
            role          = row["role"],
        )
        chunks.append(chunk)

    return chunks

# ── SERIALIZATION ─────────────────────────────────────────────────────────────

def chunks_to_dict(chunks: list) -> list:
    return [
        {
            "chunk_id":      c.chunk_id,
            "text":          c.text,
            "document_type": c.document_type,
            "source_file":   c.source_file,
            "employee_id":   c.employee_id,
            "manager_id":    c.manager_id,
            "manager_name":  c.manager_name,
            "salary_band":   c.salary_band,
            "email":         c.email,
            "start_date":    c.start_date,
            "access_level":  c.access_level,
            "department":    c.department,
            "location":      c.location,
            "contract_type": c.contract_type,
            "role":          c.role,
        }
        for c in chunks
    ]

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("hr_dataset/processed/structured", exist_ok=True)

    chunks = chunk_employee_csv(INPUT_FILE)

    output = chunks_to_dict(chunks)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    from collections import Counter
    print(f"\n{'='*55}")
    print(f"Total chunks produced : {len(chunks)}")
    print(f"Output saved to       : {OUTPUT_FILE}")
    print(f"{'='*55}")
    print(f"\nAccess level distribution:")
    for level, count in Counter(c.access_level for c in chunks).items():
        print(f"  {level:<20} {count} chunks")
    print(f"\nContract type distribution:")
    for ct, count in Counter(c.contract_type for c in chunks).items():
        print(f"  {ct:<20} {count} chunks")
    print(f"\nSample chunks:")
    for chunk in chunks[:3]:
        print(f"\n  [{chunk.access_level}] {chunk.text}")
        print(f"   → salary_band: {chunk.salary_band} | email: {chunk.email} | manager: {chunk.manager_name}")
