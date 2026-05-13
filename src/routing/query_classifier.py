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
The question can be in ANY language.

STEP 1 — Look at these examples to understand the patterns:

TEXT examples (HR policies, rules, entitlements):
- "What is the remote work policy?" → TEXT
- "How many annual leave days do employees get?" → TEXT
- "What are the sick leave entitlements?" → TEXT
- "Can I work from home every day?" → TEXT
- "Quelle est la politique de teletravail ?" → TEXT
- "ما هي سياسة الإجازات؟" → TEXT

STRUCTURED — question about a specific person or employee statistics:
- Any question asking ABOUT A PERSON by name or by role/title
- Titles include: CEO, CTO, CFO, boss, chef, patron, director, manager, head, 
  رئيس, مدير, chef, Geschäftsführer, jefe, 老板 — ALL mean a person with a role
- Employee counts, locations, departments, contracts
- Key signal: the answer requires looking up a person in the employee database

GRAPH examples (org hierarchy, who manages who, team members):
- "Who manages the Engineering department?" → GRAPH
- "Who reports directly to Paul Davis?" → GRAPH
- "Who are the members of the Engineering team?" → GRAPH
- "Qui sont les membres de l equipe Engineering ?" → GRAPH
- "من يدير قسم الهندسة؟" → GRAPH

HYBRID examples (needs both policy AND employee/org data):
- "What is the maternity leave policy and who in HR approves it?" → HYBRID
- "Tell me about the HR department and their policies" → HYBRID
- "What is Paul Davis role and his team?" → HYBRID

STEP 2 — Now reason about the question:
- What is the user really trying to find out?
- Does it need policy documents, employee data, org chart, or both?
- Which example above is most similar to this question?

STEP 3 — Classify based on your reasoning.

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
