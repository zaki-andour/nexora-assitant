import requests
import json
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import MODEL, OLLAMA_URL, TIMEOUT
from src.utils.logger import get_logger

logger = get_logger("decomposer")

DECOMPOSE_PROMPT = """You are an expert at breaking down complex HR questions.

Analyze this question and decide:
1. Is it a SIMPLE question (one topic, one source needed)?
2. Is it a COMPLEX question (multiple topics or sources needed)?

If SIMPLE: return it as-is with 1 sub-question.
If COMPLEX: break it into 2-3 independent sub-questions, each answerable alone.

Rules:
- Each sub-question must be self-contained
- Keep sub-questions short and clear
- Maximum 3 sub-questions

Question: {question}

Reply with ONLY this JSON:
{{
  "is_complex": true/false,
  "sub_questions": ["sub-question 1", "sub-question 2"]
}}"""

def decompose_query(question: str) -> dict:
    q_lower = question.lower()

    # ── DETECT MULTIPLE QUESTIONS ─────────────────────
    # If question contains newline or multiple ? → complex
    has_newline   = "\n" in question or "\r" in question
    has_multi_q   = q_lower.count("?") > 1
    has_complex_kw = any(s in q_lower for s in [" and ", " also ", " as well as ", " plus ", " both "])

    if has_newline or has_multi_q:
        # Split by newline first
        parts = [p.strip() for p in re.split(r'[\n\r]+', question) if p.strip()]
        if len(parts) > 1:
            logger.info(f"Decomposed into {len(parts)} sub-questions")
            return {
                "is_complex":    True,
                "sub_questions": parts,
                "method":        "newline"
            }

    # ── SIMPLE PATTERNS ───────────────────────────────
    simple_patterns = ["who is ", "qui est ", "wer ist ", "quien es ", "من هو", "谁是"]
    if any(q_lower.startswith(p) for p in simple_patterns) and not has_complex_kw:
        return {"is_complex": False, "sub_questions": [question], "method": "simple"}

    # ── NO COMPLEX SIGNALS → SIMPLE ───────────────────
    if not has_complex_kw:
        logger.info(f"Simple question detected: {question[:50]}")
        return {
            "is_complex":    False,
            "sub_questions": [question],
            "method":        "keyword"
        }

    # ── LLM DECOMPOSITION ─────────────────────────────
    logger.info(f"Complex question — calling LLM decomposer: {question[:50]}")
    response = requests.post(OLLAMA_URL, json={
        "model":   MODEL,
        "prompt":  DECOMPOSE_PROMPT.format(question=question),
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 200}
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

    sub_questions = result.get("sub_questions", [])
    if not sub_questions:
        parts = re.split(r'\s+and\s+', question, maxsplit=2)
        sub_questions = [p.strip() for p in parts if p.strip()] if len(parts) > 1 else [question]

    is_complex = len(sub_questions) > 1
    logger.info(f"Decomposed into {len(sub_questions)} sub-questions")

    return {
        "is_complex":    is_complex,
        "sub_questions": sub_questions,
        "method":        "llm"
    }
