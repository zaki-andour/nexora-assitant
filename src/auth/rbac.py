import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import get_logger
logger = get_logger("rbac")
ROLE_PERMISSIONS = {
    "admin":    {"can_see_all_employees": True,  "can_see_all_salaries": True,  "can_see_audit_logs": True},
    "hr":       {"can_see_all_employees": True,  "can_see_all_salaries": True,  "can_see_audit_logs": False},
    "manager":  {"can_see_all_employees": False, "can_see_all_salaries": False, "can_see_audit_logs": False},
    "employee": {"can_see_all_employees": False, "can_see_all_salaries": False, "can_see_audit_logs": False},
}
SALARY_KEYWORDS = ["salary","salaire","pay","compensation","band","راتب","sueldo","gehalt"]
def has_permission(user, permission):
    return ROLE_PERMISSIONS.get(user.get("role","employee"), {}).get(permission, False)
def apply_rbac_filter(user, question, original=None):
    role=user["role"]; department=user["department"]; emp_id=user["employee_id"]
    q_orig=(original or question).lower()
    if role in ("admin","hr"):
        logger.info(f"RBAC: {role} full access granted"); return {"allowed":True,"filter":"none"}
    if any(re.search(r'\b'+re.escape(kw)+r'\b', q_orig) for kw in SALARY_KEYWORDS):
        if role=="manager":
            logger.info("RBAC: manager team salaries"); return {"allowed":True,"filter":"department","department":department}
        logger.warning("RBAC: employee denied salary")
        return {"allowed":False,"reason":"You don't have permission to view salary information."}
    if role=="manager":
        logger.info(f"RBAC: manager dept {department}"); return {"allowed":True,"filter":"department","department":department}
    logger.info("RBAC: employee own record only"); return {"allowed":True,"filter":"self","employee_id":emp_id}
def get_rbac_sql_clause(r):
    if r["filter"]=="department": return f"AND department = '{r.get('department','')}'"
    if r["filter"]=="self":       return f"AND employee_id = {r.get('employee_id')}"
    return ""
