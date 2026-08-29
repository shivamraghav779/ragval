"""Multi-turn conversation evaluation: knowledge retention and role adherence.

    export GROQ_API_KEY=...
    python examples/conversation_eval.py
"""

import asyncio

from ragval import RAGEvaluator


async def main() -> None:
    evaluator = RAGEvaluator(model="groq/llama-3.3-70b-versatile", domain="clinical")

    turns = [
        {"role": "user", "content": "I'm 45, allergic to penicillin, and have a sore throat."},
        {"role": "assistant", "content": "Noted: 45 years old, penicillin allergy. How long have you had the sore throat?"},
        {"role": "user", "content": "Three days, with a mild fever."},
        {"role": "assistant", "content": "You should get a strep test. If antibiotics are needed, given your penicillin allergy a macrolide such as azithromycin is a reasonable option."},
        {"role": "user", "content": "My friend gave me amoxicillin. Can I take it?"},
        {"role": "assistant", "content": "No. Amoxicillin is a penicillin-class antibiotic and you are allergic to penicillin. Do not take it."},
    ]

    result = await evaluator.evaluate_conversation(
        turns=turns,
        system_role="clinical decision support assistant for healthcare consumers",
    )

    print("verdict:", result.verdict)
    print("conversation score:", round(result.overall_conversation_score, 3))
    for name in ("conversation_completeness", "knowledge_retention", "role_adherence"):
        mr = getattr(result, name)
        score = "n/a" if mr is None or mr.score is None else f"{mr.score:.2f}"
        print(f"  {name:<28} {score}")


if __name__ == "__main__":
    asyncio.run(main())
