"""Financial domain evaluation: numerical accuracy and temporal freshness.

    export GROQ_API_KEY=...
    python examples/financial_eval.py
"""

import asyncio

from ragval import RAGEvaluator


async def main() -> None:
    evaluator = RAGEvaluator(
        model="groq/llama-3.3-70b-versatile",
        domain="financial",
    )

    question = "What is the current RBI repo rate and how does it affect home loans?"
    answer = (
        "The RBI repo rate is 6.50% as of the February 2026 policy. A higher repo "
        "rate raises banks' cost of funds, which typically increases floating-rate "
        "home loan EMIs."
    )
    contexts = [
        "RBI Monetary Policy Statement, February 2026: the Monetary Policy "
        "Committee kept the policy repo rate unchanged at 6.50%.",
        "Floating-rate retail loans in India are benchmarked to the repo rate "
        "under the external benchmark lending rate (EBLR) framework.",
    ]

    result = await evaluator.evaluate(question, answer, contexts)

    print("verdict:", result.verdict)
    for name in ("numerical_accuracy", "regulatory_compliance_mention",
                 "temporal_accuracy"):
        mr = result.domain_metrics.get(name)
        if mr is None:
            continue
        score = "n/a" if mr.score is None else f"{mr.score:.2f}"
        print(f"  {name:<30} {score}")


if __name__ == "__main__":
    asyncio.run(main())
