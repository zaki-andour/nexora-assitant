import bcrypt
import psycopg2
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import DB_CONFIG
from src.utils.logger import get_logger

logger = get_logger("auth")

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def authenticate(username: str, password: str) -> dict:
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, password_hash, role, department, employee_id
            FROM users WHERE username = %s
        """, (username,))
        user = cursor.fetchone()

        if not user:
            logger.warning(f"Login failed — user not found: {username}")
            cursor.close()
            conn.close()
            return None

        if not verify_password(password, user[2]):
            logger.warning(f"Login failed — wrong password: {username}")
            cursor.close()
            conn.close()
            return None

        cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user[0],))
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Login success: {username} ({user[3]})")
        return {
            "id":          user[0],
            "username":    user[1],
            "role":        user[3],
            "department":  user[4],
            "employee_id": user[5]
        }

    except Exception as e:
        logger.error(f"Auth error: {e}")
        return None

def get_access_filter(user: dict) -> dict:
    role       = user["role"]
    department = user["department"]
    emp_id     = user["employee_id"]

    if role in ["admin", "hr"]:
        return {"filter": "none", "department": None, "employee_id": None}
    elif role == "manager":
        return {"filter": "department", "department": department, "employee_id": None}
    else:
        return {"filter": "self", "department": department, "employee_id": emp_id}
