import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.logger import get_logger
logger = get_logger("rbac")

# ── ACCESS RULES ──────────────────────────────────────
ROLE_PERMISSIONS = {
    "admin": {
        "can_see_all_employees":  True,
        "can_see_all_salaries":   True,
        "can_see_all_policies":   True,
        "can_see_audit_logs":     True,
        "can_manage_users":       True,
        "can_see_other_dept":     True,
    },
    "hr": {
        "can_see_all_employees":  True,
        "can_see_all_salaries":   True,
        "can_see_all_policies":   True,
        "can_see_audit_logs":     False,
        "can_manage_users":       False,
        "can_see_other_dept":     True,
    },
    "manager": {
        "can_see_all_employees":  False,
        "can_see_all_salaries":   False,
        "can_see_all_policies":   True,
        "can_see_audit_logs":     False,
        "can_manage_users":       False,
        "can_see_other_dept":     False,
    },
    "employee": {
        "can_see_all_employees":  False,
        "can_see_all_salaries":   False,
        "can_see_all_policies":   True,
        "can_see_audit_logs":     False,
        "can_manage_users":       False,
        "can_see_other_dept":     False,
    },
}

# Public figures always visible to everyone
PUBLIC_ROLES = ["CEO", "CTO", "CFO", "COO", "Chairman"]

def has_permission(user: dict, permission: str) -> bool:
    role = user.get("role", "employee")
    return ROLE_PERMISSIONS.get(role, {}).get(permission, False)

def apply_rbac_filter(user: dict, question: str) -> dict:
    """
    Analyze the query and apply RBAC filters based on user role.
    Returns filter instructions.
    """
    role       = user["role"]
    department = user["department"]
    emp_id     = user["employee_id"]
    q          = question.lower()

    # Admin and HR have full access
    if role in ["admin", "hr"]:
        logger.info(f"RBAC: {role} full access granted")
        return {"allowed": True, "filter": "none"}

    # ── Salary check ──────────────────────────────────
    salary_keywords = ["salary", "salaire", "pay", "compensation", "band",
                       "راتب", "sueldo", "gehalt"]
    if any(kw in q for kw in salary_keywords):
        if role == "manager":
            logger.info(f"RBAC: manager can see team salaries only")
            return {"allowed": True, "filter": "department", "department": department}
        else:
            logger.warning(f"RBAC: employee denied salary access")
            return {
                "allowed": False,
                "reason": "You don't have permission to view salary information."
            }

    # ── Other department employee check ───────────────
    other_dept_keywords = [
        "all employees", "tous les employes", "list all employees",
        "employees in", "employes dans", "staff in",
        "how many employees", "combien d employes",
        "كم عدد الموظفين", "cuantos empleados",
        "all managers", "list all managers", "tous les managers",
        "who works in", "qui travaille",
        "who is in", "members of", "staff of",
        "list all", "everyone in", "people in",
        "contact info", "contact information", "email address",
        "give me the contact", "contact of everyone"
    ]

    # Check if asking about specific person in other department
    person_keywords = ["who is", "qui est", "tell me about", "give me info",
                       "من هو", "quien es", "wer ist"]
    asks_person = any(kw in q for kw in person_keywords)
    asks_all    = any(kw in q for kw in other_dept_keywords)

    if asks_all and not has_permission(user, "can_see_all_employees"):
        if role in ["manager", "employee"]:
            logger.info(f"RBAC: {role} restricted to department {department}")
            return {"allowed": True, "filter": "department", "department": department}

    # Manager asking COUNT about other department — restrict
    count_keywords = ["how many", "combien", "count", "number of", "كم عدد"]
    asks_count = any(kw in q for kw in count_keywords)
    if asks_count and role == "manager" and not has_permission(user, "can_see_all_employees"):
        # Check if asking about a different department
        dept_names = ["hr", "finance", "sales", "marketing", "operations",
                      "product", "legal", "data", "security", "executive"]
        asking_other_dept = any(d in q for d in dept_names if d != department.lower())
        if asking_other_dept:
            logger.info(f"RBAC: manager restricted from cross-dept count")
            return {"allowed": True, "filter": "department", "department": department}
        else:
            logger.info(f"RBAC: manager access granted for own department")
            return {"allowed": True, "filter": "department", "department": department}

    # Employee asking about a person
    if asks_person and role == "employee":
        # Allow if asking about public figures (CEO, CTO...)
        public_check = any(pub.lower() in q for pub in [p.lower() for p in PUBLIC_ROLES])
        if public_check:
            logger.info(f"RBAC: employee allowed to see public figure")
            return {"allowed": True, "filter": "none"}

        # Allow only people in same department (their manager)
        logger.info(f"RBAC: employee restricted to own department")
        return {"allowed": True, "filter": "department", "department": department}

    # Manager asking about person outside department
    if asks_person and role == "manager":
        # Allow if asking about public figures by role (CEO, CTO...)
        public_role_check = any(pub.lower() in q for pub in [p.lower() for p in PUBLIC_ROLES])
        # Allow if asking about known public figure names
        public_names = ["alexandra chen", "paul davis"]
        public_name_check = any(name in q for name in public_names)
        if public_role_check or public_name_check:
            logger.info(f"RBAC: manager allowed to see public figure")
            return {"allowed": True, "filter": "none"}
        # Otherwise restrict to department
        logger.info(f"RBAC: manager restricted to department for person query")
        return {"allowed": True, "filter": "department", "department": department}

    logger.info(f"RBAC: {role} access granted")
    return {"allowed": True, "filter": "none"}

def get_rbac_sql_clause(rbac_result: dict) -> str:
    """Returns SQL WHERE clause based on RBAC filter."""
    if rbac_result["filter"] == "department":
        dept = rbac_result.get("department", "")
        return f"AND department = '{dept}'"
    elif rbac_result["filter"] == "self":
        emp_id = rbac_result.get("employee_id")
        return f"AND employee_id = {emp_id}"
    else:
        return ""
