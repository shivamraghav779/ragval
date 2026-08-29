"""A case where retrieval is the problem: low context precision.

    export GROQ_API_KEY=...
    python examples/diagnosis_report.py

Expected: diagnosis.failed_layer == "retrieval".
"""

import asyncio

from ragval import RAGEvaluator


async def main() -> None:
    evaluator = RAGEvaluator(
        model="groq/llama-3.3-70b-versatile",
        metrics=["faithfulness", "context_precision", "answer_relevance", "hallucination"],
    )

    question = "What is the first-line treatment for Type 2 diabetes?"
    answer = "Metformin is the first-line treatment for Type 2 diabetes."

    # Only one of five chunks is actually relevant - a noisy retriever.
    contexts = [
        "The mitochondria is the powerhouse of the cell.",
        "Paris is the capital of France.",
        "ADA guidelines recommend metformin as first-line therapy for T2DM.",
        "Photosynthesis converts light energy into chemical energy.",
        "The Great Wall of China is over 13,000 miles long.",
    ]

    result = await evaluator.evaluate(question, answer, contexts)

    print(result)
    d = result.diagnosis
    print("\n=== DIAGNOSIS ===")
    print("failed layer :", d.failed_layer)
    print("confidence   :", d.confidence)
    print("root cause   :", d.root_cause)
    print("suggested fix:", d.suggested_fix)
    for issue in d.secondary_issues:
        print("secondary    :", issue)


if __name__ == "__main__":
    asyncio.run(main())
