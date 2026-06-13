import json
import requests
import sys
import os
sys.path.insert(0, '/root/project')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

OLLAMA_URL     = "http://localhost:11434/api/generate"
BASE_MODEL     = "/root/models/qwen2.5-7b-hf/Qwen/Qwen2___5-7B-Instruct"
LORA_MODEL     = "/root/project/models/qwen2.5-q4-finetuned"
DATASET_PATH   = "/root/project/data/finetune/alpaca_dataset.json"

# Test questions — celles qui avaient échoué
TEST_QUESTIONS = [
    {"question": "Who is Paul Davis?",                          "expected": "CTO"},
    {"question": "Give me a summary of Engineering department", "expected": "39 employees"},
    {"question": "How many employees are in HR?",               "expected": "38"},
    {"question": "Who is the CEO?",                             "expected": "Alexandra Chen"},
    {"question": "What is the highest salary band in Engineering?", "expected": "Band5"},
    {"question": "List all part-time employees",                "expected": "15"},
    {"question": "Qui est le chef de Nexora?",                  "expected": "Alexandra Chen"},
    {"question": "Who are all Band5 employees?",                "expected": "Alexandra Chen"},
]

print("="*60)
print("  BEFORE/AFTER FINE-TUNING COMPARISON")
print("="*60)

# ── BEFORE — Qwen Q4 via Ollama ───────────────────
print("\n..... Testing BEFORE (qwen2.5-q4:7b via Ollama).....")
before_results = []
for test in TEST_QUESTIONS:
    response = requests.post(OLLAMA_URL, json={
        "model": "qwen2.5-q4:7b",
        "prompt": f"You are an HR assistant for Nexora Solutions. Answer briefly.\n\nQuestion: {test['question']}\nAnswer:",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 150}
    }, timeout=60)
    answer = response.json().get("response", "").strip()
    correct = test["expected"].lower() in answer.lower()
    before_results.append({
        "question": test["question"],
        "answer": answer[:150],
        "correct": correct,
        "expected": test["expected"]
    })
    print(f"  {'✅' if correct else '❌'} {test['question'][:50]}")

# ── AFTER — Fine-tuned model ──────────────────────
print("\n.....Loading fine-tuned model.....")
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True
)
model = PeftModel.from_pretrained(base_model, LORA_MODEL)
model.eval()
print(" Fine-tuned model loaded")

print("\n...... Testing AFTER (qwen2.5 + LoRA fine-tuned)......")
after_results = []
for test in TEST_QUESTIONS:
    prompt = f"""### Instruction:
You are an HR assistant for Nexora Solutions. Answer the question accurately based on the company data.

### Input:
{test['question']}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False
        )
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    correct = test["expected"].lower() in answer.lower()
    after_results.append({
        "question": test["question"],
        "answer": answer[:150],
        "correct": correct,
        "expected": test["expected"]
    })
    print(f"  {'✅' if correct else '❌'} {test['question'][:50]}")

# ── SUMMARY ───────────────────────────────────────
before_score = sum(1 for r in before_results if r["correct"])
after_score  = sum(1 for r in after_results  if r["correct"])
improvement  = after_score - before_score

print(f"\n{'='*60}")
print(f"  RESULTS SUMMARY")
print(f"{'='*60}")
print(f"\n  Before fine-tuning : {before_score}/{len(TEST_QUESTIONS)} correct")
print(f"  After fine-tuning  : {after_score}/{len(TEST_QUESTIONS)} correct")
print(f"  Improvement        : +{improvement} questions")
print(f"\n  {'Question':<45} {'Before':^8} {'After':^8}")
print(f"  {'-'*65}")
for b, a in zip(before_results, after_results):
    print(f"  {b['question'][:45]:<45} {'✅' if b['correct'] else '❌':^8} {'✅' if a['correct'] else '❌':^8}")

# ── SAVE ──────────────────────────────────────────
report = {
    "before_score": before_score,
    "after_score": after_score,
    "improvement": improvement,
    "total": len(TEST_QUESTIONS),
    "before": before_results,
    "after": after_results
}
os.makedirs("benchmarks", exist_ok=True)
with open("benchmarks/finetune_comparison.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n Report saved to benchmarks/finetune_comparison.json")
print("="*60)
