import requests
import re
import sys
import os
from difflib import SequenceMatcher
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import MODEL, OLLAMA_URL, TIMEOUT, NUM_PREDICT, TEMPERATURE
from src.routing.query_classifier   import classify_query
from src.routing.query_decomposer   import decompose_query
from src.routing.query_router       import build_context
from src.routing.query_preprocessor import preprocess_query
from src.auth.rbac import apply_rbac_filter, get_rbac_sql_clause
from src.utils.logger import get_logger

logger = get_logger("pipeline")

def run_pipeline(question: str, user: dict = None) -> dict:
    logger.info("=" * 50)
    logger.info(f"New question: {question}")

    # ── STEP 0 : PREPROCESS ───────────────────────────
    logger.info("STEP 0 — Preprocessing question")
    preprocessed       = preprocess_query(question)
    query_for_pipeline = preprocessed["reformulated"]
    detected_language  = preprocessed["language"]
    logger.info(f"Reformulated: {query_for_pipeline}")
    logger.info(f"Language detected: {detected_language}")

    # ── STEP 0.5 : RBAC CHECK ─────────────────────────
    rbac_filter = {"filter": "none"}
    if user:
        rbac_result = apply_rbac_filter(user, query_for_pipeline)
        if not rbac_result["allowed"]:
            logger.warning(f"RBAC blocked: {rbac_result.get('reason')}")
            return {
                "question":      question,
                "is_complex":    False,
                "sub_questions": [question],
                "categories":    ["BLOCKED"],
                "answer":        rbac_result["reason"],
                "sources":       [],
                "language":      detected_language,
            }
        rbac_filter = rbac_result

    # ── STEP 1 : DECOMPOSE ────────────────────────────
    logger.info("STEP 1 — Decomposing question")
    decomposition = decompose_query(query_for_pipeline)
    sub_questions = decomposition["sub_questions"]
    is_complex    = decomposition["is_complex"]
    logger.info(f"Complex: {is_complex} | Sub-questions: {len(sub_questions)}")

    # ── STEP 2 & 3 : CLASSIFY + ROUTE ────────────────
    logger.info("STEP 2-3 — Classifying and routing")
    all_contexts   = []
    all_sources    = []
    all_categories = []

    for sq in sub_questions:
        classification = classify_query(sq)
        category       = classification["category"]
        all_categories.append(category)
        logger.info(f"Sub-Q [{category}]: {sq[:50]}")

        rbac_clause = get_rbac_sql_clause(rbac_filter)
        # For personal queries, enrich question with user identity
        enriched_sq = sq
        if user and any(kw in sq.lower() for kw in ["my ", "i am", "i have", "mon ", "je "]):
            emp_id = user.get("employee_id", "")
            uname  = user.get("username", "")
            enriched_sq = sq + f" [Current user: {uname}, employee_id: {emp_id}]"
        context, sources = build_context(category, enriched_sq, rbac_clause=rbac_clause)
        all_contexts.append(f"[{category} — {sq}]\n{context}")
        all_sources.extend(sources)

    # ── STEP 4 : CONTEXT FUSION ───────────────────────
    logger.info("STEP 4 — Fusing contexts")
    fused_context  = "\n\n".join(all_contexts)
    unique_sources = list(set(all_sources))
    logger.info(f"Sources: {unique_sources}")

    # ── STEP 5 : LLM GENERATION ───────────────────────
    logger.info("STEP 5 — Generating answer")

    # Build user context
    if user:
        emp_id   = user.get("employee_id", "unknown")
        username = user.get("username", "unknown")
        dept     = user.get("department", "unknown")
        role_u   = user.get("role", "unknown")
        user_context = f"Current user: {username} | employee_id: {emp_id} | department: {dept} | role: {role_u}"
    else:
        user_context = ""

    prompt = f""""You are an HR assistant for Nexora Solutions.
Answer the question based ONLY on the context provided below.
IMPORTANT: The user asked in {detected_language}. Always respond in {detected_language}.
CRITICAL: Respond ONLY and ENTIRELY in {detected_language}. NEVER use any other language. NEVER mix languages. NEVER add self-correction text. Respond ONLY in {detected_language}.
Every single word in your response must be in {detected_language}.
Do NOT use markdown formatting, asterisks, bullet symbols or bold text.
Do NOT include sources or references in your answer.
Write in plain text with numbered lists if needed.
Be direct, specific and detailed.
If the context contains MULTIPLE results for a person search, list ALL of them and ask the user to specify which one they mean.
If the information is not in the context, say clearly in {detected_language} that you don't have this information.
Do NOT invent sources, websites, or information not in the context.

Context:
{fused_context}

Original question: {question}
{user_context}
Answer in {detected_language}:"""

    response = requests.post(OLLAMA_URL, json={
        "model":   MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT}
    }, timeout=TIMEOUT)

    answer = response.json()["response"]

    # Fix wrong names from DB context
    for match in re.finditer(r"correct name is '([^']+)'", fused_context):
        correct_name = match.group(1)
        for correct_part in correct_name.split():
            words = re.findall(r'[A-Za-z]+', answer)
            for word in set(words):
                if word.lower() != correct_part.lower() and len(word) > 2:
                    ratio = SequenceMatcher(None, word.lower(), correct_part.lower()).ratio()
                    if ratio > 0.75:
                        answer = answer.replace(word, correct_part)

    logger.info("Answer generated successfully")
    logger.info(f"Answer: {answer[:200]}")

    return {
        "question":      question,
        "is_complex":    is_complex,
        "sub_questions": sub_questions,
        "categories":    all_categories,
        "answer":        answer,
        "sources":       unique_sources,
        "language":      detected_language,
    }

def display_result(result: dict):
    print(f"\n{'█'*65}")
    print(f"  QUESTION   : {result['question']}")
    print(f"  LANGUAGE   : {result['language']}")
    print(f"  COMPLEX    : {result['is_complex']}")
    print(f"  CATEGORIES : {result['categories']}")
    print(f"  SOURCES    : {result['sources']}")
    print(f"\n  ANSWER:")
    for line in result['answer'].strip().split('\n'):
        print(f"    {line}")
    print(f"{'█'*65}")
