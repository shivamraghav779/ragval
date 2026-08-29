"""Minimal three-line usage. Set GROQ_API_KEY before running:

    export GROQ_API_KEY=...
    python examples/basic_eval.py
"""

import asyncio

from ragval import evaluate


async def main() -> None:
    question = "What is the first-line treatment for Type 2 diabetes?"
    answer = (
        "Metformin is the recommended first-line medication for Type 2 diabetes, "
        "typically started at 500mg twice daily."
    )
    contexts = [
        "ADA guidelines recommend metformin as the first-line pharmacologic agent "
        "for Type 2 diabetes. Typical starting dose is 500mg twice daily.",
        "Lifestyle modification remains the foundation of Type 2 diabetes care.",
    ]

    result = await evaluate(question, answer, contexts)

    print(result)  # formatted box display
    print("\nverdict       :", result.verdict)
    print("overall score :", round(result.overall_score, 3))
    print("failed layer  :", result.diagnosis.failed_layer)
    print("root cause    :", result.diagnosis.root_cause)
    print("suggested fix :", result.diagnosis.suggested_fix)


if __name__ == "__main__":
    asyncio.run(main())
