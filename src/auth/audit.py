import psycopg2
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import DB_CONFIG
from src.utils.logger import get_logger

logger = get_logger("audit")

def log_action(user: dict, question: str, category: str, allowed: bool):
    """Save every user action to audit_logs table."""
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (user_id, username, role, question, category, allowed)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            user.get("id"),
            user.get("username"),
            user.get("role"),
            question,
            category,
            allowed
        ))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Audit log saved: {user.get('username')} | {category} | allowed={allowed}")
    except Exception as e:
        logger.error(f"Audit log error: {e}")

def get_audit_stats() -> dict:
    """Get audit statistics for admin dashboard."""
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN allowed = TRUE  THEN 1 ELSE 0 END) as allowed,
                SUM(CASE WHEN allowed = FALSE THEN 1 ELSE 0 END) as denied
            FROM audit_logs
        """)
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return {
            "total":   row[0] or 0,
            "allowed": row[1] or 0,
            "denied":  row[2] or 0
        }
    except Exception as e:
        logger.error(f"Audit stats error: {e}")
        return {"total": 0, "allowed": 0, "denied": 0}
