import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")
from src.routing.query_classifier import classify_query

tests = [
    # English basic
    ("What is the remote work policy?", "TEXT"),
    ("How many contractors work in the Finance department?", "STRUCTURED"),
    ("Who manages the Engineering department?", "GRAPH"),
    ("Who reports directly to Paul Davis?", "GRAPH"),
    ("What is the maternity leave policy and who in HR approves it?", "HYBRID"),
    ("How many annual leave days do full-time employees get?", "TEXT"),
    ("What department does Julia Jackson work in?", "STRUCTURED"),
    ("Who is Paul Davis?", "STRUCTURED"),
    ("What are the sick leave entitlements for contractors?", "TEXT"),
    ("Who is the CEO of Nexora?", "STRUCTURED"),
    ("Tell me about the HR department and their policies", "HYBRID"),
    # English tricky
    ("Give me details about Alexandra Chen", "STRUCTURED"),
    ("I need information on Julia Jackson", "STRUCTURED"),
    ("Show me employees based in London", "STRUCTURED"),
    ("What is the notice period for resignation?", "TEXT"),
    ("Can I work from home every day?", "TEXT"),
    ("Is there a dress code policy?", "TEXT"),
    ("Tell me the org chart of the Sales department", "GRAPH"),
    ("What is Paul Davis role and his team?", "HYBRID"),
    ("How does the grievance procedure work?", "TEXT"),
    ("List all employees in the HR department", "STRUCTURED"),
    ("What are the travel reimbursement rules?", "TEXT"),
    # French
    ("Quelle est la politique de teletravail ?", "TEXT"),
    ("Qui gere le departement Engineering ?", "GRAPH"),
    ("Combien de contractuels dans le departement Finance ?", "STRUCTURED"),
    ("Qui est Paul Davis ?", "STRUCTURED"),
    ("Quelle est la politique de maladie ?", "TEXT"),
    ("Qui est Julia Jackson ?", "STRUCTURED"),
    ("Combien d employes travaillent a Londres ?", "STRUCTURED"),
    ("Donnez moi des informations sur Alexandra Chen", "STRUCTURED"),
    ("Quelle est la procedure de reclamation ?", "TEXT"),
    ("Qui sont les membres de l equipe Engineering ?", "GRAPH"),
    # Arabic
    ("من يدير قسم الهندسة؟", "GRAPH"),
    ("ما هي سياسة الإجازات؟", "TEXT"),
    ("كم عدد المتعاقدين في قسم المالية؟", "STRUCTURED"),
    ("من هو بول ديفيس؟", "STRUCTURED"),
    ("ما هي سياسة العمل عن بعد؟", "TEXT"),
    ("من يرفع تقاريره إلى بول ديفيس؟", "GRAPH"),
    # Spanish
    ("Cuantos contratistas trabajan en Finanzas?", "STRUCTURED"),
    ("Quien gestiona el departamento de Ingenieria?", "GRAPH"),
    ("Cual es la politica de vacaciones?", "TEXT"),
    ("Dame informacion sobre Julia Jackson", "STRUCTURED"),
    ("Cuantos empleados hay en Londres?", "STRUCTURED"),
    # German
    ("Wer leitet die Ingenieurswesen-Abteilung?", "GRAPH"),
    ("Was ist die Fernarbeitsrichtlinie?", "TEXT"),
    ("Wer ist Paul Davis?", "STRUCTURED"),
    ("Wie viele Mitarbeiter arbeiten in London?", "STRUCTURED"),
    # Chinese
    ("谁管理工程部门？", "GRAPH"),
    ("休假政策是什么？", "TEXT"),
    ("保罗·戴维斯是谁？", "STRUCTURED"),
    ("远程办公政策是什么？", "TEXT"),
]

print("=" * 70)
print("MULTILINGUAL QUERY CLASSIFIER TEST — LLM ONLY")
print("=" * 70)

correct = 0
results_log = []

for question, expected in tests:
    result = classify_query(question)
    got = result["category"]
    is_ok = got == expected
    if is_ok:
        correct += 1
    results_log.append((question, expected, got, is_ok, result.get("reasoning", ""), result.get("method", "")))

print("")
print(f"{'#':<4} {'ST':<5} {'EXP':<12} {'GOT':<12} {'METHOD':<8}")
print("-" * 50)

for i, (question, expected, got, is_ok, reasoning, method) in enumerate(results_log, 1):
    status = "OK" if is_ok else "FAIL"
    print(f"{i:<4} {status:<5} {expected:<12} {got:<12} {method:<8}")
    print(f"     Q: {question}")
    if not is_ok:
        print(f"     -> {reasoning}")

print("-" * 50)
print(f"Score: {correct}/{len(tests)} correct  ({round(correct/len(tests)*100)}%)")

failures = [(q, e, g) for q, e, g, ok, _, m in results_log if not ok]
if failures:
    print(f"\nFailed ({len(failures)}):")
    for q, e, g in failures:
        print(f"  [{e}->{g}] {q}")
print("=" * 70)

