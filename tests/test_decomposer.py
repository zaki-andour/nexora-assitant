import sys
sys.path.insert(0, '.')
from src.routing.query_decomposer import decompose_query

tests = [
    "What is the remote work policy?",
    "Who manages the Engineering department?",
    "How many contractors are in Finance?",
    "What is the maternity leave policy for contractors and who in HR approves it?",
    "Tell me about the Engineering department structure and their remote work policy.",
    "What is the sick leave policy and how many sick days does Julia Jackson have left?",
]

print("="*65)
print("QUERY DECOMPOSER TEST")
print("="*65)

for question in tests:
    result = decompose_query(question)
    print(f"\nQ: {question}")
    print(f"   Complex : {result['is_complex']} (via {result['method']})")
    for i, sq in enumerate(result['sub_questions'], 1):
        print(f"   Sub-Q {i}: {sq}")
