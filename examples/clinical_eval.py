"""Clinical domain evaluation: drug-name precision and dosing accuracy.

    export GROQ_API_KEY=...
    python examples/clinical_eval.py
"""

import asyncio

from ragval import RAGEvaluator


async def main() -> None:
    evaluator = RAGEvaluator(
        model="groq/llama-3.3-70b-versatile",
        domain="clinical",
    )

    question = "What is the starting dose of metformin for a newly diagnosed T2DM patient?"
    answer = (
        "Start metformin at 500mg once or twice daily with meals, titrating up to "
        "a maximum of 2000mg/day as tolerated. Avoid in eGFR < 30."
    )
    contexts = [
        "Metformin immediate-release: initiate 500mg PO once or twice daily with "
        "meals. Titrate weekly. Max 2000-2550mg/day. Contraindicated if eGFR < 30 "
        "mL/min/1.73m2. Source: ADA Standards of Care.",
        "Metformin is associated with a low risk of lactic acidosis, increased in "
        "renal impairment.",
    ]

    result = await evaluator.evaluate(question, answer, contexts)

    print("verdict:", result.verdict)
    for name in ("drug_name_precision", "dosing_accuracy", "contraindication_coverage",
                 "authority_score"):
        mr = result.domain_metrics.get(name)
        if mr is None:
            continue
        score = "n/a" if mr.score is None else f"{mr.score:.2f}"
        print(f"  {name:<26} {score}")
        for v in mr.violations:
            print(f"      ! {v}")


if __name__ == "__main__":
    asyncio.run(main())
