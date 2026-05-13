import requests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import MODEL, OLLAMA_URL, TIMEOUT, NUM_PREDICT, TEMPERATURE
from src.routing.query_classifier   import classify_query
from src.routing.query_decomposer   import decompose_query
from src.routing.query_router       import build_context
from src.routing.query_preprocessor import preprocess_query
from src.utils.logger import get_logger

logger = get_logger("pipeline")

def run_pipeline(question: str) -> dict:
    logger.info("=" * 50)
    logger.info(f"New question: {question}")

    # ── STEP 0 : PREPROCESS ───────────────────────────
    logger.info("STEP 0 — Preprocessing question")
    preprocessed       = preprocess_query(question)
    query_for_pipeline = preprocessed["reformulated"]
    detected_language  = preprocessed["language"]
    logger.info(f"Reformulated: {query_for_pipeline}")
    logger.info(f"Language detected: {detected_language}")

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

        context, sources = build_context(category, sq)
        all_contexts.append(f"[{category} — {sq}]\n{context}")
        all_sources.extend(sources)

    # ── STEP 4 : CONTEXT FUSION ───────────────────────
    logger.info("STEP 4 — Fusing contexts")
    fused_context  = "\n\n".join(all_contexts)
    unique_sources = list(set(all_sources))
    logger.info(f"Sources: {unique_sources}")

    # ── STEP 5 : LLM GENERATION ───────────────────────
    logger.info("STEP 5 — Generating answer")
    prompt = f"""You are an HR assistant for Nexora Solutions.
Answer the question based ONLY on the context provided below.
IMPORTANT: The user asked in {detected_language}. Always respond in {detected_language}.
CRITICAL: Never mix languages. Respond ONLY in {detected_language}.
Every single word in your response must be in {detected_language}.
Do NOT use markdown formatting, asterisks, bullet symbols or bold text.
Do NOT include sources or references in your answer.
Write in plain text with numbered lists if needed.
Be direct, specific and detailed.
Use EXACTLY the names as they appear in the context — never use the name from the question if it differs.
If the context contains MULTIPLE results for a person search, list ALL of them and ask the user to specify which one they mean.
If the information is not in the context, say clearly in {detected_language} that you don't have this information.
Do NOT invent sources, websites, or information not in the context.

Context:
{fused_context}

Original question: {question}

Answer in {detected_language}:"""

    response = requests.post(OLLAMA_URL, json={
        "model":   MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT}
    }, timeout=TIMEOUT)

    answer = response.json()["response"]
    logger.info("Answer generated successfully")

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
