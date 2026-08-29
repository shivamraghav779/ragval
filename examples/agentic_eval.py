"""Agentic evaluation: tool correctness and task completion.

    export GROQ_API_KEY=...
    python examples/agentic_eval.py
"""

import asyncio

from ragval import RAGEvaluator


async def main() -> None:
    evaluator = RAGEvaluator(model="groq/llama-3.3-70b-versatile")

    question = "Find the ADA first-line drug for T2DM and its starting dose."
    answer = "The ADA first-line drug is metformin, started at 500mg twice daily."

    tool_calls = [
        {"name": "search_guidelines", "arguments": {"query": "ADA T2DM first line"}},
        {"name": "lookup_drug", "arguments": {"drug": "metformin", "field": "dosing"}},
    ]
    expected_tools = ["search_guidelines", "lookup_drug"]
    action_trace = [
        "Searched ADA guidelines for first-line therapy",
        "Looked up metformin starting dose",
        "Synthesized the answer",
    ]
    declared_plan = (
        "1. Search the ADA guidelines. 2. Look up the drug's dosing. 3. Answer."
    )

    result = await evaluator.evaluate_agent(
        question=question,
        answer=answer,
        tool_calls=tool_calls,
        expected_tools=expected_tools,
        action_trace=action_trace,
        declared_plan=declared_plan,
    )

    print("verdict:", result.verdict)
    for name in ("tool_correctness", "argument_correctness", "task_completion",
                 "step_efficiency", "plan_adherence", "plan_quality"):
        mr = getattr(result, name)
        if mr is None:
            continue
        score = "n/a" if mr.score is None else f"{mr.score:.2f}"
        print(f"  {name:<22} {score}")


if __name__ == "__main__":
    asyncio.run(main())
