import requests
import json
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import MODEL, OLLAMA_URL, TIMEOUT
from src.utils.logger import get_logger

logger = get_logger("preprocessor")

PREPROCESS_PROMPT = """You are an intelligent multilingual query preprocessor.

The user may write in any language with typos, abbreviations, incomplete sentences, or grammatical errors.
Your job is to understand the TRUE INTENT regardless of how poorly the question is written, and reformulate it as a clear precise English question.

This is for an HR assistant at Nexora Solutions. All questions are related to HR topics: leave policies, remote work, sick days, maternity, employee information, organizational structure, company conduct rules, and similar workplace topics. Never interpret questions as being about non-HR topics.

Question: {question}

Reply ONLY with this JSON:
{{"reformulated": "clear English question", "language": "detected language name", "intent": "one sentence"}}"""


def preprocess_query(question: str) -> dict:
    logger.info(f"Preprocessing: {question[:60]}")

    response = requests.post(OLLAMA_URL, json={
        "model":   MODEL,
        "prompt":  PREPROCESS_PROMPT.format(question=question),
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

    reformulated = result.get("reformulated", question)
    language     = result.get("language", "English")
    intent       = result.get("intent", "")

    logger.info(f"Reformulated: {reformulated[:60]}")
    logger.info(f"Language: {language}")

    return {
        "original":     question,
        "reformulated": reformulated,
        "language":     language,
        "intent":       intent
    }
