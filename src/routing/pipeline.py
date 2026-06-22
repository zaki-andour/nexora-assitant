import requests
import re
import sys
import os
from difflib import SequenceMatcher
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.config as _config
from src.routing.query_classifier   import classify_query
from src.routing.query_decomposer   import decompose_query
from src.routing.query_router       import build_context
from src.routing.query_preprocessor import preprocess_query
from src.auth.rbac import apply_rbac_filter, get_rbac_sql_clause
from src.utils.logger import get_logger

try:
    from langdetect import detect as langdetect_detect
    LANGDETECT_AVAILABLE = True
except:
    LANGDETECT_AVAILABLE = False

logger = get_logger("pipeline")

LANG_MAP = {
    "fr": "French", "ar": "Arabic", "es": "Spanish",
    "de": "German", "zh-cn": "Chinese", "zh-tw": "Chinese",
    "en": "English", "it": "Italian", "pt": "Portuguese"
}

def detect_language(text):
    # Script-based detection first — reliable for CJK and Arabic, which langdetect
    # frequently misclassifies on short text or when Latin names are present
    # (e.g. "Nexora Solutions 有多少个部门？" was being detected as English/Korean).
    if re.search(r'[\u4e00-\u9fff]', text):          # Han (Chinese) characters
        return "Chinese"
    if re.search(r'[\u0600-\u06ff\u0750-\u077f]', text):  # Arabic script
        return "Arabic"
    # Latin-script languages (English, French, Spanish, German) -> langdetect
    if LANGDETECT_AVAILABLE:
        try:
            return LANG_MAP.get(langdetect_detect(text), "English")
        except:
            return "English"
    return "English"


def ensure_language(answer, target_language):
    """Language-consistency fallback.

    The quantised model sometimes answers in English even when another language
    was requested (mostly Chinese). If the produced answer is in English while a
    different language was asked, translate it into the target language with a
    single extra LLM call. Names, emails, numbers, dates and symbols are kept as
    is. This only triggers on the English-fallback failure, so answers already in
    the correct language (or any non-English language) are never altered.
    """
    if target_language == "English" or not answer or not answer.strip():
        return answer
    try:
        detected = LANG_MAP.get(langdetect_detect(answer), "English")
    except Exception:
        return answer
    if detected != "English":
        return answer  # already non-English -> leave untouched

    logger.info(f"Language fallback: answer came out English, target={target_language} — translating")
    translate_prompt = f"""Translate the following text into {target_language}.
Keep person names (e.g. Paul Davis, Stefan Bell), email addresses, numbers, dates and currency symbols (£, $) EXACTLY as they are — do NOT translate or modify them.
Output ONLY the translation in {target_language}, with no preamble and no extra comment.

Text:
{answer}

Translation in {target_language}:"""
    try:
        r = requests.post(_config.OLLAMA_URL, json={
            "model":   _config.MODEL,
            "prompt":  translate_prompt,
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": _config.NUM_PREDICT}
        }, timeout=_config.TIMEOUT)
        translated = r.json().get("response", "").strip()
        return translated if translated else answer
    except Exception as e:
        logger.warning(f"Translation fallback failed: {e}")
        return answer

def run_pipeline(question: str, user: dict = None) -> dict:
    logger.info("=" * 50)
    logger.info(f"New question: {question}")

    # ── STEP 0 : DETECT LANGUAGE ──────────────────────
    detected_language = detect_language(question)
    logger.info(f"Language detected: {detected_language}")

    # ── STEP 0.1 : DECOMPOSE ORIGINAL ─────────────────
    pre_decomposition = decompose_query(question)

    # ── STEP 0.2 : PREPROCESS ─────────────────────────
    preprocessed       = preprocess_query(question)
    query_for_pipeline = preprocessed["reformulated"]
    logger.info(f"Reformulated: {query_for_pipeline}")

    # ── STEP 0.5 : RBAC CHECK ─────────────────────────
    rbac_filter = {"filter": "none"}
    if user:
        rbac_result = apply_rbac_filter(user, query_for_pipeline, original=question)
        if not rbac_result["allowed"]:
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
    if pre_decomposition["is_complex"]:
        processed_subs = []
        for sq in pre_decomposition["sub_questions"]:
            p = preprocess_query(sq)
            processed_subs.append(p["reformulated"])
        decomposition = {"is_complex": True, "sub_questions": processed_subs}
    else:
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
        enriched_sq = sq
        if user:
            emp_id = user.get("employee_id", "")
            uname  = user.get("username", "")
            dept   = user.get("department", "")
            # Always enrich personal queries with user identity
            personal_kw = ["my ", "i am", "i have", "mon ", "je ", "my role", "my contract",
                          "my department", "my start", "my job", "my title", "your current",
                          "about me", "employment", "tell me about my", "everything about"]
            if any(kw in sq.lower() for kw in personal_kw):
                enriched_sq = sq + f" [Current user: {uname}, employee_id: {emp_id}, department: {dept}]"
            # Also enrich if question has "my" but was reformulated to "your"
            elif "your " in sq.lower() and emp_id:
                enriched_sq = sq + f" [Current user: {uname}, employee_id: {emp_id}, department: {dept}]"
        context, sources = build_context(category, enriched_sq, rbac_clause=rbac_clause)
        if context.startswith("__RBAC_DENIED__"):
            return {"question": question, "is_complex": False, "sub_questions": [question],
                    "categories": ["BLOCKED"], "answer": context.replace("__RBAC_DENIED__", "", 1),
                    "sources": [], "language": detected_language}
        all_contexts.append(f"[{category} — {sq}]\n{context}")
        all_sources.extend(sources)

    # ── STEP 4 : CONTEXT FUSION ───────────────────────
    logger.info("STEP 4 — Fusing contexts")
    fused_context  = "\n\n".join(all_contexts)
    unique_sources = list(set(all_sources))
    logger.info(f"Sources: {unique_sources}")

    # ── STEP 5 : LLM GENERATION ───────────────────────
    logger.info("STEP 5 — Generating answer")

    if user:
        emp_id   = user.get("employee_id", "unknown")
        username = user.get("username", "unknown")
        dept     = user.get("department", "unknown")
        role_u   = user.get("role", "unknown")
        user_context = f"Current user: {username} | employee_id: {emp_id} | department: {dept} | role: {role_u}"
    else:
        user_context = ""

    # ── Language-aware script guard ───────────────────
    # Forbid a script ONLY when it is NOT the target language, so we never
    # block the language we actually want (this fixes Chinese answers coming
    # out in English) while still preventing script-mixing in other languages.
    if detected_language == "Chinese":
        script_rule = "Write the entire answer in Simplified Chinese characters. Do NOT use Cyrillic/Russian characters."
    elif detected_language == "Arabic":
        script_rule = ("Write the entire answer in Arabic script. "
                       "Do NOT use Chinese characters. Do NOT use Cyrillic/Russian characters.")
    else:
        script_rule = "Do NOT use Chinese characters. Do NOT use Cyrillic/Russian characters."

    prompt = f"""You are an HR assistant for Nexora Solutions.
{user_context}
Answer the question based ONLY on the context provided below.
IMPORTANT: Always respond in {detected_language}.
CRITICAL: Respond ONLY and ENTIRELY in {detected_language}. NEVER use any other language. NEVER mix languages.
Every single word in your response must be in {detected_language}.
{script_rule}
Person names (like Alexandra Chen, Paul Davis) must NEVER be translated — keep them exactly as written.
When presenting data, translate English department/role names to {detected_language} naturally.
For example in French: HR → Ressources Humaines, Engineering → Ingénierie.
In Arabic: HR → الموارد البشرية, Engineering → الهندسة.
In Spanish: HR → Recursos Humanos, Engineering → Ingeniería.

Do NOT use markdown formatting or asterisks.
Do NOT include sources in your answer.
Be direct, specific and detailed.
When the context contains a person's details (department, email, location, start date), include ALL of those fields in your answer, not just the name and role.
The SQL query already filtered the correct data — ALWAYS trust and list ALL rows from the context.
NEVER say you don't have information if the context contains rows.
If the context shows employees with their salary_band filtered, those ARE the Band5 employees — list them all.
If the information is not in the context, say clearly in {detected_language} that you don't have this information.

Context:
{fused_context}

Question: {question}

Answer in {detected_language}:"""

    # Dynamic temperature — lower for structured (data accuracy), higher for text (natural style)
    primary_category = all_categories[0] if all_categories else "TEXT"
    dynamic_temp = 0.1 if primary_category in ["STRUCTURED", "GRAPH"] else 0.4

    response = requests.post(_config.OLLAMA_URL, json={
        "model":   _config.MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": dynamic_temp, "num_predict": _config.NUM_PREDICT}
    }, timeout=_config.TIMEOUT)

    answer = response.json().get("response", "").strip()

    # ── STEP 5.1 : LANGUAGE-CONSISTENCY FALLBACK ──────
    answer = ensure_language(answer, detected_language)

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
