"""Custom metrics: G-Eval, RubricEval, and AspectCritic.

    export GROQ_API_KEY=...
    python examples/custom_metric.py
"""

import asyncio

from ragval.metrics.custom import AspectCritic, GEval, RubricEval
from ragval.providers import get_provider

MODEL = "groq/llama-3.3-70b-versatile"


async def main() -> None:
    provider = get_provider(MODEL)

    question = "What is the starting dose of metformin?"
    answer = "Metformin is usually started at 500mg twice daily with meals."
    contexts = ["Initiate metformin at 500mg PO BID with meals; titrate weekly."]

    g_eval = GEval(
        name="Dosing Specificity",
        criteria=(
            "The answer must include exact dose amounts with units for every "
            "drug mentioned, plus frequency."
        ),
        model=MODEL,
    )
    g = await g_eval.compute(question, answer, contexts, provider)
    print(f"G-Eval [{g_eval.name}]: {g.score:.2f} - {g.reasoning[:120]}")

    rubric = RubricEval(
        name="Clinical Answer Quality",
        rubric={
            1: "Wrong or unsafe.",
            2: "Vague, missing dose or frequency.",
            3: "Correct drug and dose but missing caveats.",
            4: "Correct drug, dose, frequency, and one relevant caveat.",
            5: "Complete: drug, dose, frequency, titration, and contraindications.",
        },
        model=MODEL,
        weight_top_heavy=True,
    )
    r = await rubric.compute(question, answer, contexts, provider)
    print(f"Rubric: level {r.metadata['selected_level']}/5 -> score {r.score:.2f}")

    critic = AspectCritic(
        name="Cites Frequency",
        aspect="dosing frequency",
        description="The answer explicitly states how often the drug is taken.",
        model=MODEL,
    )
    c = await critic.compute(question, answer, contexts, provider)
    print(f"AspectCritic: {'PASS' if c.metadata['passed'] else 'FAIL'} - {c.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
