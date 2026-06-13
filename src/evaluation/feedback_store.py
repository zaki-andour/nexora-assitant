import psycopg2
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import DB_CONFIG
from src.utils.logger import get_logger

logger = get_logger("feedback")

def save_feedback(question, category, answer, sources, score,
                  model_used, latency_sec, top_chunk_score, chunks_count, comment="",
                  hallucination_flag=False, hallucination_reason=""):
    """Save user feedback to PostgreSQL."""
    # Auto-detect hallucination — score -1 = potential hallucination
    if score == -1 and not hallucination_flag:
        hallucination_flag = True
        hallucination_reason = comment if comment else "User marked answer as incorrect"
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback
            (question, category, answer, sources, score, comment,
             model_used, latency_sec, top_chunk_score, chunks_count,
             hallucination_flag, hallucination_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            question, category, answer,
            ", ".join(sources) if isinstance(sources, list) else sources,
            score, comment, model_used, latency_sec, top_chunk_score, chunks_count,
            hallucination_flag, hallucination_reason
        ))
        conn.commit()
        cursor.close()
        conn.close()
        if hallucination_flag:
            logger.warning(f"Hallucination flagged | question: {question[:50]}")
        logger.info(f"Feedback saved — score: {score} | question: {question[:50]}")
        return True
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return False

def get_feedback_stats():
    """Get feedback statistics."""
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN score = 1  THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN score = -1 THEN 1 ELSE 0 END) as negative,
                AVG(latency_sec) as avg_latency,
                AVG(top_chunk_score) as avg_score,
                SUM(CASE WHEN hallucination_flag = true THEN 1 ELSE 0 END) as hallucinations
            FROM feedback
        """)
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row and row[0] > 0:
            return {
                "total":          row[0],
                "positive":       row[1],
                "negative":       row[2],
                "avg_latency":    round(row[3], 2) if row[3] else 0,
                "avg_score":      round(row[4], 3) if row[4] else 0,
                "hallucinations": row[5] or 0,
            }
        return {"total": 0, "positive": 0, "negative": 0, "avg_latency": 0, "avg_score": 0}
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {}
