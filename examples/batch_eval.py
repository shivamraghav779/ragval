"""Batch evaluation over several QA pairs.

    export GROQ_API_KEY=...
    python examples/batch_eval.py
"""

import asyncio

from ragval import RAGEvaluator

CTX = [
    "ADA guidelines recommend metformin as first-line therapy for Type 2 "
    "diabetes, started at 500mg twice daily.",
    "Metformin is contraindicated when eGFR < 30 mL/min/1.73m2.",
    "SGLT2 inhibitors are preferred add-ons when there is established ASCVD or CKD.",
]

QA_PAIRS = [
    {
        "question": "First-line drug for T2DM?",
        "answer": "Metformin, 500mg twice daily.",
        "contexts": CTX,
        "ground_truth": "Metformin is first-line, started at 500mg twice daily.",
    },
    {
        "question": "First-line drug for T2DM?",
        "answer": "Start insulin immediately for all patients.",
        "contexts": CTX,
        "ground_truth": "Metformin is first-line, started at 500mg twice daily.",
    },
    {
        "question": "When is metformin contraindicated?",
        "answer": "When eGFR is below 30.",
        "contexts": CTX,
        "ground_truth": "Metformin is contraindicated when eGFR < 30.",
    },
    {
        "question": "Preferred add-on with established CKD?",
        "answer": "An SGLT2 inhibitor.",
        "contexts": CTX,
        "ground_truth": "SGLT2 inhibitors are preferred with CKD or ASCVD.",
    },
    {
        "question": "What is the capital of France?",
        "answer": "Metformin.",
        "contexts": CTX,
        "ground_truth": "Paris.",
    },
]


async def main() -> None:
    evaluator = RAGEvaluator(
        model="groq/llama-3.3-70b-versatile",
        domain="clinical",
        metrics="all",
    )
    batch = await evaluator.batch_evaluate(QA_PAIRS, concurrency=3)

    print(batch.report())
    print("\n--- worst 2 cases ---")
    for r in batch.worst_cases(2):
        print(f"[{r.verdict}] {r.question!r} -> {r.answer!r}")
        if r.diagnosis:
            print(f"    failed layer: {r.diagnosis.failed_layer}")


if __name__ == "__main__":
    asyncio.run(main())
