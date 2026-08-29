"""Legal domain evaluation: jurisdiction specificity and citation accuracy.

    export GROQ_API_KEY=...
    python examples/legal_eval.py
"""

import asyncio

from ragval import RAGEvaluator


async def main() -> None:
    evaluator = RAGEvaluator(
        model="groq/llama-3.3-70b-versatile",
        domain="legal",
    )

    question = "Is a contract without consideration enforceable in California?"
    answer = (
        "Under California law, a written contract is presumptive consideration "
        "(Cal. Civ. Code section 1614), but generally a contract requires "
        "consideration to be enforceable unless it falls within an exception "
        "such as promissory estoppel."
    )
    contexts = [
        "California Civil Code section 1550 lists the essential elements of a "
        "contract, including 'a sufficient cause or consideration'.",
        "Cal. Civ. Code section 1614: 'A written instrument is presumptive "
        "evidence of a consideration.'",
    ]

    result = await evaluator.evaluate(question, answer, contexts)

    print("verdict:", result.verdict)
    for name in ("jurisdiction_specificity", "citation_accuracy", "statute_currency"):
        mr = result.domain_metrics.get(name)
        if mr is None:
            continue
        score = "n/a" if mr.score is None else f"{mr.score:.2f}"
        print(f"  {name:<26} {score} - {mr.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
