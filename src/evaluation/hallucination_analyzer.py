import psycopg2
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import DB_CONFIG
from src.utils.logger import get_logger

logger = get_logger("hallucination_analyzer")

def analyze_hallucinations():
    """Analyze hallucination patterns from feedback data."""
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # ── GLOBAL STATS ──────────────────────────────
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN score = 1  THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN score = -1 THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN hallucination_flag = true THEN 1 ELSE 0 END) as hallucinations
            FROM feedback
        """)
        row = cursor.fetchone()
        total, positive, negative, hallucinations = row
        hallucination_rate = round(hallucinations / total * 100, 1) if total > 0 else 0

        # ── BY CATEGORY ───────────────────────────────
        cursor.execute("""
            SELECT category,
                COUNT(*) as total,
                SUM(CASE WHEN hallucination_flag = true THEN 1 ELSE 0 END) as hallucinations
            FROM feedback
            GROUP BY category
            ORDER BY hallucinations DESC
        """)
        by_category = []
        for r in cursor.fetchall():
            rate = round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0
            by_category.append({
                "category": r[0],
                "total": r[1],
                "hallucinations": r[2],
                "rate": rate
            })

        # ── BY MODEL ──────────────────────────────────
        cursor.execute("""
            SELECT model_used,
                COUNT(*) as total,
                SUM(CASE WHEN hallucination_flag = true THEN 1 ELSE 0 END) as hallucinations
            FROM feedback
            GROUP BY model_used
            ORDER BY hallucinations DESC
        """)
        by_model = []
        for r in cursor.fetchall():
            rate = round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0
            by_model.append({
                "model": r[0],
                "total": r[1],
                "hallucinations": r[2],
                "rate": rate
            })

        # ── FLAGGED QUESTIONS ─────────────────────────
        cursor.execute("""
            SELECT question, answer, category, model_used, hallucination_reason, created_at
            FROM feedback
            WHERE hallucination_flag = true
            ORDER BY created_at DESC
        """)
        flagged = []
        for r in cursor.fetchall():
            flagged.append({
                "question": r[0],
                "answer": r[1][:200] if r[1] else "",
                "category": r[2],
                "model": r[3],
                "reason": r[4],
                "created_at": str(r[5])
            })

        cursor.close()
        conn.close()

        # ── BUILD REPORT ──────────────────────────────
        report = {
            "global": {
                "total_feedbacks": total,
                "positive": positive,
                "negative": negative,
                "hallucinations": hallucinations,
                "hallucination_rate": hallucination_rate
            },
            "by_category": by_category,
            "by_model": by_model,
            "flagged_questions": flagged
        }

        # ── SAVE REPORT ───────────────────────────────
        os.makedirs("benchmarks", exist_ok=True)
        with open("benchmarks/hallucination_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # ── PRINT SUMMARY ─────────────────────────────
        print("\n" + "="*60)
        print("  HALLUCINATION ANALYSIS REPORT")
        print("="*60)
        print(f"\n📊 Global Stats:")
        print(f"   Total feedbacks    : {total}")
        print(f"   👍 Positive        : {positive}")
        print(f"   👎 Negative        : {negative}")
        print(f"   🚨 Hallucinations  : {hallucinations} ({hallucination_rate}%)")

        print(f"\n📂 By Category:")
        for c in by_category:
            bar = "█" * int(c['rate'] / 10) + "░" * (10 - int(c['rate'] / 10))
            print(f"   {c['category']:<12} {bar} {c['rate']}% ({c['hallucinations']}/{c['total']})")

        print(f"\n🤖 By Model:")
        for m in by_model:
            print(f"   {m['model']:<25} {m['rate']}% ({m['hallucinations']}/{m['total']})")

        print(f"\n🚨 Flagged Questions:")
        for i, q in enumerate(flagged[:5], 1):
            print(f"   {i}. [{q['category']}] {q['question'][:60]}")
            print(f"      Reason: {q['reason']}")

        print(f"\n✅ Report saved to benchmarks/hallucination_report.json")
        print("="*60)

        return report

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {}

if __name__ == "__main__":
    analyze_hallucinations()
