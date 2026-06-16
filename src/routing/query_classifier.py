import requests
import json
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import MODEL, OLLAMA_URL, TIMEOUT
from src.utils.logger import get_logger

logger = get_logger("classifier")

LLM_PROMPT = """You are an intelligent multilingual HR query classifier.
The question can be in ANY language. Classify it into EXACTLY ONE category.

TEXT — a company policy, rule, entitlement or procedure (answer = policy documents):
- "What is the remote work policy?" → TEXT
- "How many annual leave days do employees get?" → TEXT
- "What are the sick leave entitlements?" → TEXT
- "Quelle est la politique de télétravail ?" → TEXT
- "ما هي سياسة الإجازات؟" → TEXT

STRUCTURED — an ATTRIBUTE or a COUNT about an employee / employees (answer = employee database):
- One person's field: "What is Julia Jackson's role / department / salary band / contract / location / start date?" → STRUCTURED
- Personal data: "what is my role", "quel est mon contrat", "what is my salary band" → STRUCTURED
- Counts & lists: "How many employees are in London?", "How many contractors are there?", "List all Band5 employees" → STRUCTURED
- Key signal: you look up a FIELD or a NUMBER about a person or about employees.

GRAPH — a RELATIONSHIP or a POSITION in the org hierarchy (answer = org chart):
- Head / leader of a department: "Who is the head of the Engineering department?", "Who leads Finance?", "Who manages Sales?" → GRAPH
- Direct reports of a person: "Who reports to Paul Davis?", "Who are the members of the Engineering team?" → GRAPH
- The manager OF a person: "Who is the manager of Julia Jackson?", "Who does the CTO report to?" → GRAPH
- "من يدير قسم الهندسة؟" → GRAPH ; "Qui dirige le département Finance ?" → GRAPH
- Key signal: the answer is a LINK between a person and a department, or between two people
  (who-leads-a-department, who-reports-to-whom, who-is-whose-manager).

HYBRID — the question needs BOTH a policy AND employee/org data at the same time:
- A count/person AND a policy: "How many people work in Engineering, and what is the remote work policy?" → HYBRID
- A person's attribute AND a policy: "What is Julia Jackson's contract type, and what is her sick leave entitlement?" → HYBRID
- "How many contractors are there, and what maternity leave applies to them?" → HYBRID
- Key signal: the question has TWO parts — one needs the database/org chart, the other needs a policy document.

DECISION GUIDE (apply in this order):
1. TWO things at once (employee/org data AND a policy)             → HYBRID
2. A hierarchy LINK (head of a dept, manager of a person, reports) → GRAPH
3. A single RULE / ENTITLEMENT / POLICY                            → TEXT
4. A single FIELD or NUMBER about employees                        → STRUCTURED

Question: {question}

Reply ONLY with this JSON:
{{"category": "TEXT|STRUCTURED|GRAPH|HYBRID", "confidence": 0.0-1.0, "reasoning": "one sentence explaining your reasoning"}}"""


def classify_query(question: str) -> dict:
    """
    Fully LLM-based classifier — no keyword rules.
    Works in any language. Returns category, confidence, reasoning.
    """
    logger.info(f"Classifying: {question[:60]}")

    response = requests.post(OLLAMA_URL, json={
        "model":   MODEL,
        "prompt":  LLM_PROMPT.format(question=question),
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 150}
    }, timeout=TIMEOUT)

    raw = response.json()["response"].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except:
                result = {}
        else:
            result = {}

    # Validate category
    valid = {"TEXT", "STRUCTURED", "GRAPH", "HYBRID"}
    if result.get("category") not in valid:
        logger.warning(f"Invalid category from LLM: {result} — defaulting to TEXT")
        result["category"] = "TEXT"

    result["question"] = question
    result["method"]   = "llm"

    logger.info(f"Classified as {result['category']} ({result.get('confidence', 0)}) — {result.get('reasoning', '')[:60]}")
    return result
