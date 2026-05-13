import sys
sys.path.insert(0, '.')
from src.routing.pipeline import run_pipeline, display_result

test_questions = [
    "What is the remote work policy?",
    "How many contractors work in the Finance department?",
    "Who manages the Engineering department?",
    "What is the maternity leave policy for contractors and who in HR approves it?",
]

for question in test_questions:
    result = run_pipeline(question)
    display_result(result)
    print()
