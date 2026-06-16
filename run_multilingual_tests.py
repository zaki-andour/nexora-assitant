#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Multilingual test runner for Nexora app v2 (self-contained, compact).
# RUN:  python3 run_multilingual_tests.py   (app v2 must be running on :7861)
import json, time, requests, urllib3
urllib3.disable_warnings()

BASE="https://localhost:7861"; USER="admin"; PWD="admin123"; OUT="multilingual_results.json"

# (category, language, question)
Q = [
    ("GRAPH","English","Who is the head of the Engineering department?"),
    ("GRAPH","French","Qui est le responsable du département Ingénierie ?"),
    ("GRAPH","Spanish","¿Quién es el responsable del departamento de Ingeniería?"),
    ("GRAPH","German","Wer leitet die Engineering-Abteilung?"),
    ("GRAPH","Arabic","من هو رئيس قسم الهندسة؟"),
    ("GRAPH","Chinese","谁是工程部门的负责人？"),
    ("GRAPH","English","Who reports to Fatima Nguyen?"),
    ("GRAPH","French","Qui relève de Fatima Nguyen ?"),
    ("GRAPH","Spanish","¿Quién depende de Fatima Nguyen?"),
    ("GRAPH","German","Wer berichtet an Fatima Nguyen?"),
    ("GRAPH","Arabic","من يتبع فاطمة نغوين في العمل؟"),
    ("GRAPH","Chinese","谁向 Fatima Nguyen 汇报？"),
    ("GRAPH","English","Who is the manager of Julia Jackson?"),
    ("GRAPH","French","Qui est le manager de Julia Jackson ?"),
    ("GRAPH","Spanish","¿Quién es el gerente de Julia Jackson?"),
    ("GRAPH","German","Wer ist der Vorgesetzte von Julia Jackson?"),
    ("GRAPH","Arabic","من هو مدير جوليا جاكسون؟"),
    ("GRAPH","Chinese","Julia Jackson 的经理是谁？"),
    ("GRAPH","English","How many departments does Nexora Solutions have?"),
    ("GRAPH","French","Combien de départements compte Nexora Solutions ?"),
    ("GRAPH","Spanish","¿Cuántos departamentos tiene Nexora Solutions?"),
    ("GRAPH","German","Wie viele Abteilungen hat Nexora Solutions?"),
    ("GRAPH","Arabic","كم عدد الأقسام في شركة Nexora Solutions؟"),
    ("GRAPH","Chinese","Nexora Solutions 有多少个部门？"),
    ("GRAPH","English","Who is the head of the Finance department?"),
    ("GRAPH","French","Qui dirige le département Finance ?"),
    ("GRAPH","Spanish","¿Quién dirige el departamento de Finanzas?"),
    ("GRAPH","German","Wer leitet die Finanzabteilung?"),
    ("GRAPH","Arabic","من هو رئيس قسم المالية؟"),
    ("GRAPH","Chinese","谁是财务部门的负责人？"),
    ("HYBRID","English","How many people work in Engineering, and what is the remote work policy?"),
    ("HYBRID","French","Combien de personnes travaillent dans l'Ingénierie, et quelle est la politique de télétravail ?"),
    ("HYBRID","Spanish","¿Cuántas personas trabajan en Ingeniería y cuál es la política de teletrabajo?"),
    ("HYBRID","German","Wie viele Personen arbeiten im Engineering, und wie lautet die Homeoffice-Richtlinie?"),
    ("HYBRID","Arabic","كم عدد الأشخاص العاملين في قسم الهندسة، وما هي سياسة العمل عن بُعد؟"),
    ("HYBRID","Chinese","工程部门有多少人，远程办公政策是什么？"),
    ("HYBRID","English","What is Julia Jackson's contract type, and what is her sick leave entitlement?"),
    ("HYBRID","French","Quel est le type de contrat de Julia Jackson, et à combien de jours de congé maladie a-t-elle droit ?"),
    ("HYBRID","Spanish","¿Cuál es el tipo de contrato de Julia Jackson y a cuántos días de baja por enfermedad tiene derecho?"),
    ("HYBRID","German","Welchen Vertragstyp hat Julia Jackson, und wie viele Krankheitstage stehen ihr zu?"),
    ("HYBRID","Arabic","ما نوع عقد جوليا جاكسون، وما هو رصيدها من الإجازات المرضية؟"),
    ("HYBRID","Chinese","Julia Jackson 的合同类型是什么，她有多少病假天数？"),
    ("HYBRID","English","How many contractors are there, and what is the maternity leave policy for contractors?"),
    ("HYBRID","French","Combien y a-t-il de prestataires (contractors), et quelle est la politique de congé maternité pour eux ?"),
    ("HYBRID","Spanish","¿Cuántos contratistas hay y cuál es la política de licencia de maternidad para los contratistas?"),
    ("HYBRID","German","Wie viele Contractors gibt es, und wie lautet die Mutterschaftsurlaubsregelung für sie?"),
    ("HYBRID","Arabic","كم عدد المتعاقدين، وما هي سياسة إجازة الأمومة الخاصة بهم؟"),
    ("HYBRID","Chinese","有多少名合同工，针对合同工的产假政策是什么？"),
    ("HYBRID","English","Who leads the Finance department, and what is the expense claim submission deadline?"),
    ("HYBRID","French","Qui dirige le département Finance, et quel est le délai de soumission des notes de frais ?"),
    ("HYBRID","Spanish","¿Quién dirige el departamento de Finanzas y cuál es el plazo para presentar los gastos?"),
    ("HYBRID","German","Wer leitet die Finanzabteilung, und bis wann müssen Spesenabrechnungen eingereicht werden?"),
    ("HYBRID","Arabic","من يرأس قسم المالية، وما هو الموعد النهائي لتقديم مطالبات المصاريف؟"),
    ("HYBRID","Chinese","谁领导财务部门，报销申请的提交截止时间是什么？"),
    ("HYBRID","English","How many employees are based in London, and what is the maximum London accommodation rate for business travel?"),
    ("HYBRID","French","Combien d'employés sont basés à Londres, et quel est le plafond d'hébergement à Londres pour les voyages d'affaires ?"),
    ("HYBRID","Spanish","¿Cuántos empleados están en Londres y cuál es la tarifa máxima de alojamiento en Londres para viajes de negocios?"),
    ("HYBRID","German","Wie viele Mitarbeiter sind in London tätig, und wie hoch ist der maximale Übernachtungssatz in London für Geschäftsreisen?"),
    ("HYBRID","Arabic","كم عدد الموظفين في لندن، وما هو الحد الأقصى لتكلفة الإقامة في لندن لسفر العمل؟"),
    ("HYBRID","Chinese","有多少名员工在伦敦工作，出差时伦敦住宿的最高标准是多少？"),
    ("TEXT","English","How many annual leave days are full-time employees entitled to?"),
    ("TEXT","French","À combien de jours de congés annuels les employés à temps plein ont-ils droit ?"),
    ("TEXT","Spanish","¿A cuántos días de vacaciones anuales tienen derecho los empleados a tiempo completo?"),
    ("TEXT","German","Wie viele Jahresurlaubstage stehen Vollzeitbeschäftigten zu?"),
    ("TEXT","Arabic","كم عدد أيام الإجازة السنوية المستحقة للموظفين بدوام كامل؟"),
    ("TEXT","Chinese","全职员工每年有多少天年假？"),
    ("TEXT","English","How many days of remote work per week are employees allowed?"),
    ("TEXT","French","Combien de jours de télétravail par semaine les employés sont-ils autorisés à prendre ?"),
    ("TEXT","Spanish","¿Cuántos días de teletrabajo por semana se permiten a los empleados?"),
    ("TEXT","German","Wie viele Homeoffice-Tage pro Woche sind den Mitarbeitern erlaubt?"),
    ("TEXT","Arabic","كم عدد أيام العمل عن بُعد المسموح بها للموظفين في الأسبوع؟"),
    ("TEXT","Chinese","员工每周可以远程办公几天？"),
    ("TEXT","English","How long is paid maternity leave at Nexora Solutions?"),
    ("TEXT","French","Quelle est la durée du congé maternité payé chez Nexora Solutions ?"),
    ("TEXT","Spanish","¿Cuánto dura la licencia de maternidad remunerada en Nexora Solutions?"),
    ("TEXT","German","Wie lange dauert der bezahlte Mutterschaftsurlaub bei Nexora Solutions?"),
    ("TEXT","Arabic","ما هي مدة إجازة الأمومة المدفوعة في Nexora Solutions؟"),
    ("TEXT","Chinese","Nexora Solutions 的带薪产假有多长？"),
    ("TEXT","English","How many sick days per year are full-time permanent employees entitled to?"),
    ("TEXT","French","À combien de jours de congé maladie par an les employés permanents à temps plein ont-ils droit ?"),
    ("TEXT","Spanish","¿A cuántos días de baja por enfermedad al año tienen derecho los empleados fijos a tiempo completo?"),
    ("TEXT","German","Wie viele Krankheitstage pro Jahr stehen unbefristet Vollzeitbeschäftigten zu?"),
    ("TEXT","Arabic","كم عدد أيام الإجازة المرضية سنويًا المستحقة للموظفين الدائمين بدوام كامل؟"),
    ("TEXT","Chinese","全职正式员工每年有多少天病假？"),
    ("TEXT","English","Within how many days must expense claims be submitted?"),
    ("TEXT","French","Dans quel délai (en jours) les notes de frais doivent-elles être soumises ?"),
    ("TEXT","Spanish","¿Dentro de cuántos días deben presentarse las reclamaciones de gastos?"),
    ("TEXT","German","Innerhalb wie vieler Tage müssen Spesenabrechnungen eingereicht werden?"),
    ("TEXT","Arabic","خلال كم يومًا يجب تقديم مطالبات المصاريف؟"),
    ("TEXT","Chinese","报销申请必须在多少天内提交？"),
]

s=requests.post(BASE+"/api/login", json={"username":USER,"password":PWD}, verify=False, timeout=30)
s.raise_for_status(); sid=s.json()["session_id"]
print("Logged in. Running", len(Q), "questions...")

res=[]; ok=0
for i,(cat,lang,query) in enumerate(Q,1):
    try:
        d=requests.post(BASE+"/api/ask", json={"question":query,"session_id":sid}, verify=False, timeout=180).json()
        det=d.get("category",""); good=(det==cat); ok+=good
        res.append({"category":cat,"language":lang,"query":query,"answer":d.get("answer",""),"detected":det,"routing_ok":good,"blocked":d.get("blocked",False)})
        print("[%3d/%d] %-8s exp=%-9s det=%-9s %s"%(i,len(Q),lang,cat,det,"OK" if good else "MISMATCH"))
    except Exception as e:
        res.append({"category":cat,"language":lang,"query":query,"error":str(e)}); print("[%3d] ERROR %s"%(i,e))
    time.sleep(1)

json.dump(res, open(OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nDONE. Saved", OUT, "| routing accuracy %d/%d = %.1f%%"%(ok,len(Q),100*ok/len(Q)))
print("All questions were also logged into audit_logs.")
