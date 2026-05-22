import time
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.logger import get_logger
logger = get_logger("benchmark_pipeline")

from src.routing.query_router import build_context
from src.routing.query_preprocessor import preprocess_query
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ["qwen2.5:7b", "qwen2.5-q4:7b", "deepseek-r1:7b"]

TEST_CASES = [
    # ── TEXT English ──────────────────────────────────
    {"category": "TEXT", "question": "What is the remote work policy?"},
    {"category": "TEXT", "question": "How many annual leave days do full-time employees get?"},
    {"category": "TEXT", "question": "What are the sick leave entitlements for contractors?"},
    {"category": "TEXT", "question": "What is the maternity leave policy?"},
    {"category": "TEXT", "question": "What is the notice period for resignation?"},
    {"category": "TEXT", "question": "Can I work from home every day?"},
    {"category": "TEXT", "question": "How does the grievance procedure work?"},
    {"category": "TEXT", "question": "What are the travel reimbursement rules?"},
    {"category": "TEXT", "question": "What is the disciplinary procedure?"},
    {"category": "TEXT", "question": "What is the dress code policy?"},
    # TEXT French
    {"category": "TEXT", "question": "Quelle est la politique de teletravail ?"},
    {"category": "TEXT", "question": "Combien de jours de conge annuel pour un employe a temps plein ?"},
    {"category": "TEXT", "question": "Quelle est la politique de maladie ?"},
    {"category": "TEXT", "question": "Comment fonctionne la procedure de reclamation ?"},
    {"category": "TEXT", "question": "Quelle est la politique de voyage d affaires ?"},
    # TEXT Arabic
    {"category": "TEXT", "question": "\u0645\u0627 \u0647\u064a \u0633\u064a\u0627\u0633\u0629 \u0627\u0644\u0625\u062c\u0627\u0632\u0627\u062a\u061f"},
    {"category": "TEXT", "question": "\u0645\u0627 \u0647\u064a \u0633\u064a\u0627\u0633\u0629 \u0627\u0644\u0639\u0645\u0644 \u0639\u0646 \u0628\u0639\u062f\u061f"},
    {"category": "TEXT", "question": "\u0645\u0627 \u0647\u064a \u0633\u064a\u0627\u0633\u0629 \u0627\u0644\u0625\u062c\u0627\u0632\u0629 \u0627\u0644\u0645\u0631\u0636\u064a\u0629\u061f"},
    # TEXT Spanish
    {"category": "TEXT", "question": "Cual es la politica de vacaciones?"},
    {"category": "TEXT", "question": "Cuantos dias de baja por enfermedad tienen los empleados?"},
    {"category": "TEXT", "question": "Como funciona el procedimiento disciplinario?"},
    # TEXT German
    {"category": "TEXT", "question": "Was ist die Fernarbeitsrichtlinie?"},
    {"category": "TEXT", "question": "Wie viele Urlaubstage haben Vollzeitbeschaftigte?"},
    # TEXT Chinese
    {"category": "TEXT", "question": "\u4f11\u5047\u653f\u7b56\u662f\u4ec0\u4e48\uff1f"},
    {"category": "TEXT", "question": "\u8fdc\u7a0b\u529e\u516c\u653f\u7b56\u662f\u4ec0\u4e48\uff1f"},

    # ── STRUCTURED English ────────────────────────────
    {"category": "STRUCTURED", "question": "How many contractors work in Finance?"},
    {"category": "STRUCTURED", "question": "Who is the CEO of Nexora?"},
    {"category": "STRUCTURED", "question": "What department does Julia Jackson work in?"},
    {"category": "STRUCTURED", "question": "How many full-time employees are based in London?"},
    {"category": "STRUCTURED", "question": "How many employees work in HR?"},
    {"category": "STRUCTURED", "question": "Who is Paul Davis?"},
    {"category": "STRUCTURED", "question": "How many interns does Nexora have?"},
    {"category": "STRUCTURED", "question": "How many part-time employees are there?"},
    {"category": "STRUCTURED", "question": "List all employees in Engineering department"},
    {"category": "STRUCTURED", "question": "How many employees work in Berlin?"},
    # STRUCTURED French
    {"category": "STRUCTURED", "question": "Combien d employes travaillent a Londres ?"},
    {"category": "STRUCTURED", "question": "Qui est le PDG de Nexora ?"},
    {"category": "STRUCTURED", "question": "Combien de contractuels dans le departement Finance ?"},
    {"category": "STRUCTURED", "question": "Dans quel departement travaille Julia Jackson ?"},
    # STRUCTURED Arabic
    {"category": "STRUCTURED", "question": "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0645\u062a\u0639\u0627\u0642\u062f\u064a\u0646 \u0641\u064a \u0642\u0633\u0645 \u0627\u0644\u0645\u0627\u0644\u064a\u0629\u061f"},
    {"category": "STRUCTURED", "question": "\u0645\u0646 \u0647\u0648 \u0627\u0644\u0631\u0626\u064a\u0633 \u0627\u0644\u062a\u0646\u0641\u064a\u0630\u064a \u0644\u0634\u0631\u0643\u0629 \u0646\u064a\u0643\u0633\u0648\u0631\u0627\u061f"},
    {"category": "STRUCTURED", "question": "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0645\u0648\u0638\u0641\u064a\u0646 \u0641\u064a \u0642\u0633\u0645 \u0627\u0644\u0647\u0646\u062f\u0633\u0629\u061f"},
    # STRUCTURED Spanish
    {"category": "STRUCTURED", "question": "Cuantos contratistas trabajan en Finanzas?"},
    {"category": "STRUCTURED", "question": "Quien es el CEO de Nexora?"},
    {"category": "STRUCTURED", "question": "Cuantos empleados hay en Londres?"},
    # STRUCTURED German
    {"category": "STRUCTURED", "question": "Wer ist Paul Davis?"},
    {"category": "STRUCTURED", "question": "Wie viele Mitarbeiter arbeiten in London?"},
    # STRUCTURED Chinese
    {"category": "STRUCTURED", "question": "\u8c01\u662fNexora\u7684CEO\uff1f"},
    {"category": "STRUCTURED", "question": "\u4f26\u6566\u529e\u516c\u5ba4\u6709\u591a\u5c11\u540d\u5168\u804c\u5458\u5de5\uff1f"},

    # ── GRAPH English ─────────────────────────────────
    {"category": "GRAPH", "question": "Who manages the Engineering department?"},
    {"category": "GRAPH", "question": "Who reports directly to Paul Davis?"},
    {"category": "GRAPH", "question": "Who manages the HR department?"},
    {"category": "GRAPH", "question": "Who manages the Finance department?"},
    {"category": "GRAPH", "question": "Who are the members of the Engineering team?"},
    {"category": "GRAPH", "question": "Who is the direct supervisor of Paul Davis?"},
    # GRAPH French
    {"category": "GRAPH", "question": "Qui gere le departement Finance ?"},
    {"category": "GRAPH", "question": "Qui gere le departement Engineering ?"},
    {"category": "GRAPH", "question": "Qui rapporte a Paul Davis ?"},
    {"category": "GRAPH", "question": "Qui sont les membres de l equipe HR ?"},
    # GRAPH Arabic
    {"category": "GRAPH", "question": "\u0645\u0646 \u064a\u062f\u064a\u0631 \u0642\u0633\u0645 \u0627\u0644\u0647\u0646\u062f\u0633\u0629\u061f"},
    {"category": "GRAPH", "question": "\u0645\u0646 \u064a\u0631\u0641\u0639 \u062a\u0642\u0627\u0631\u064a\u0631\u0647 \u0625\u0644\u0649 \u0628\u0648\u0644 \u062f\u064a\u0641\u064a\u0633\u061f"},
    {"category": "GRAPH", "question": "\u0645\u0646 \u064a\u062f\u064a\u0631 \u0642\u0633\u0645 \u0627\u0644\u0645\u0627\u0644\u064a\u0629\u061f"},
    # GRAPH Spanish
    {"category": "GRAPH", "question": "Quien gestiona el departamento de Ingenieria?"},
    {"category": "GRAPH", "question": "Quien reporta directamente a Paul Davis?"},
    # GRAPH German
    {"category": "GRAPH", "question": "Wer leitet die Ingenieurswesen-Abteilung?"},
    {"category": "GRAPH", "question": "Wer ist der direkte Vorgesetzte von Paul Davis?"},
    # GRAPH Chinese
    {"category": "GRAPH", "question": "\u8c01\u7ba1\u7406\u5de5\u7a0b\u90e8\u95e8\uff1f"},
    {"category": "GRAPH", "question": "\u8c01\u5411Paul Davis\u6c47\u62a5\uff1f"},

    # ── HYBRID English ────────────────────────────────
    {"category": "HYBRID", "question": "What is the maternity leave policy and who in HR approves it?"},
    {"category": "HYBRID", "question": "Tell me about the Engineering department and their remote work policy"},
    {"category": "HYBRID", "question": "What is Paul Davis role and his team?"},
    {"category": "HYBRID", "question": "What are the leave policies for contractors and how many contractors are there?"},
    {"category": "HYBRID", "question": "Tell me about the HR department structure and their policies"},
    {"category": "HYBRID", "question": "How many employees are in Finance and what is their travel policy?"},
    {"category": "HYBRID", "question": "Who manages Engineering and what is their remote work policy?"},
    # HYBRID French
    {"category": "HYBRID", "question": "Quelle est la politique de conge maternite et qui l approuve dans les RH ?"},
    {"category": "HYBRID", "question": "Parlez moi du departement Engineering et leur politique de teletravail"},
    {"category": "HYBRID", "question": "Quel est le role de Paul Davis et son equipe ?"},
    {"category": "HYBRID", "question": "Quelles sont les politiques de conge pour les contractuels et combien sont ils ?"},
    {"category": "HYBRID", "question": "Qui dirige le departement HR et quelles sont leurs politiques ?"},
    # HYBRID Arabic
    {"category": "HYBRID", "question": "\u0645\u0627 \u0647\u064a \u0633\u064a\u0627\u0633\u0629 \u0625\u062c\u0627\u0632\u0629 \u0627\u0644\u0623\u0645\u0648\u0645\u0629 \u0648\u0645\u0646 \u064a\u0648\u0627\u0641\u0642 \u0639\u0644\u064a\u0647\u0627 \u0641\u064a \u0627\u0644\u0645\u0648\u0627\u0631\u062f \u0627\u0644\u0628\u0634\u0631\u064a\u0629\u061f"},
    {"category": "HYBRID", "question": "\u0623\u062e\u0628\u0631\u0646\u064a \u0639\u0646 \u0642\u0633\u0645 \u0627\u0644\u0647\u0646\u062f\u0633\u0629 \u0648\u0633\u064a\u0627\u0633\u0629 \u0627\u0644\u0639\u0645\u0644 \u0639\u0646 \u0628\u0639\u062f"},
    {"category": "HYBRID", "question": "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0645\u062a\u0639\u0627\u0642\u062f\u064a\u0646 \u0648\u0645\u0627 \u0647\u064a \u062d\u0642\u0648\u0642\u0647\u0645 \u0641\u064a \u0627\u0644\u0625\u062c\u0627\u0632\u0627\u062a\u061f"},
    # HYBRID Spanish
    {"category": "HYBRID", "question": "Cual es la politica de maternidad y cuantos empleados hay en HR?"},
    {"category": "HYBRID", "question": "Cuentame sobre el departamento de Ingenieria y su politica de trabajo remoto"},
    {"category": "HYBRID", "question": "Quien gestiona Finanzas y cuantos empleados tiene?"},
    # HYBRID German
    {"category": "HYBRID", "question": "Was ist die Mutterschaftsurlaubsregelung und wer genehmigt sie in der HR?"},
    {"category": "HYBRID", "question": "Erzahlen Sie mir uber die Ingenieurwesen-Abteilung und ihre Richtlinien"},
    # HYBRID Chinese
    {"category": "HYBRID", "question": "\u4ea7\u5047\u653f\u7b56\u662f\u4ec0\u4e48\uff0c\u8c01\u6279\u51c6\uff1f"},
    {"category": "HYBRID", "question": "\u5de5\u7a0b\u90e8\u95e8\u7684\u7ed3\u6784\u548c\u8fdc\u7a0b\u529e\u516c\u653f\u7b56\u662f\u4ec0\u4e48\uff1f"},
    {"category": "HYBRID", "question": "\u8c01\u7ba1\u7406\u8d22\u52a1\u90e8\u95e8\uff0c\u6709\u591a\u5c11\u540d\u5458\u5de5\uff1f"},
]

def run_with_model(model: str, question: str, category: str) -> dict:
    t0 = time.time()
    try:
        preprocessed = preprocess_query(question)
        query        = preprocessed["reformulated"]
        language     = preprocessed["language"]
        context, sources = build_context(category, query)

        prompt = f"""You are an HR assistant for Nexora Solutions.
Answer the question based ONLY on the context provided below.
IMPORTANT: Always respond in {language}.
CRITICAL: Never mix languages. Respond ONLY in {language}.
Do NOT use markdown formatting or asterisks.
Do NOT include sources in your answer.
If the information is not in the context, say you don't have this information.
Do NOT invent information.

Context:
{context}

Original question: {question}

Answer in {language}:"""

        response = requests.post(OLLAMA_URL, json={
            "model":   model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.1, "num_predict": 512}
        }, timeout=300)

        answer = response.json()["response"].strip()
        if "<think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        elapsed = time.time() - t0
        return {
            "answer":   answer,
            "latency":  round(elapsed, 2),
            "sources":  sources,
            "language": language,
            "error":    None
        }
    except Exception as e:
        return {
            "answer":  "",
            "latency": round(time.time() - t0, 2),
            "sources": [],
            "language": "",
            "error":   str(e)
        }

def run_benchmark():
    print("=" * 70)
    print("PIPELINE BENCHMARK — Qwen2.5:7b vs DeepSeek-R1:7b")
    print("Full RAG context — Multilingual")
    print(f"Total: {len(TEST_CASES)} questions x {len(MODELS)} models = {len(TEST_CASES)*len(MODELS)} calls")
    print("=" * 70)

    results = []
    total = len(TEST_CASES) * len(MODELS)
    done  = 0

    for tc in TEST_CASES:
        category = tc["category"]
        question = tc["question"]
        print(f"\n[{category}] {question[:55]}")
        print("-" * 60)

        row = {"category": category, "question": question, "models": {}}

        for model in MODELS:
            done += 1
            print(f"  [{done}/{total}] {model} ...", end=" ", flush=True)
            result = run_with_model(model, question, category)
            row["models"][model] = result
            status = "ERROR" if result["error"] else "OK"
            print(f"{result['latency']}s | {status}")
            if not result["error"]:
                print(f"  -> {result['answer'][:100]}...")

        results.append(row)

    # ── SUMMARY ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)

    bad_patterns = [
        "i don't know", "i cannot", "no information",
        "relation does not exist", "sql error",
        "je ne sais pas", "aucune information",
        "independently developed", "created by deepseek",
        "i'm an ai", "je suis deepseek"
    ]

    categories = ["TEXT", "STRUCTURED", "GRAPH", "HYBRID"]
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        print(f"\n{cat} ({len(cat_results)} questions):")
        for model in MODELS:
            latencies = [r["models"][model]["latency"] for r in cat_results
                        if not r["models"][model]["error"]]
            bad = sum(1 for r in cat_results
                     if any(p in r["models"][model]["answer"].lower()
                            for p in bad_patterns))
            if latencies:
                avg = round(sum(latencies)/len(latencies), 2)
                mn  = round(min(latencies), 2)
                mx  = round(max(latencies), 2)
                print(f"  {model:<20} avg: {avg}s | min: {mn}s | max: {mx}s | bad: {bad}/{len(cat_results)}")

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/q3_q4_deepseek_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to benchmarks/pipeline_comparison.json")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
