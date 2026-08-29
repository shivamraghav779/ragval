"""Use ragval inside a FastAPI route.

    pip install fastapi uvicorn
    export GROQ_API_KEY=...
    uvicorn examples.fastapi_integration:app --reload

Then POST to /rag with {"question": "...", "answer": "...", "contexts": ["..."]}.
"""

from typing import List, Optional

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    raise SystemExit("This example needs: pip install fastapi uvicorn")

from ragval import RAGEvaluator

app = FastAPI(title="ragval + FastAPI")

# One evaluator, reused across requests. It is async-native and safe to share.
evaluator = RAGEvaluator(model="groq/llama-3.3-70b-versatile", domain="general")


class RAGRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@app.post("/rag")
async def evaluate_rag(req: RAGRequest) -> dict:
    """Evaluate a generated answer, then return the verdict and diagnosis."""
    result = await evaluator.evaluate(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth,
    )
    return {
        "verdict": result.verdict,
        "overall_score": result.overall_score,
        "hallucination_detected": result.hallucination_detected,
        "diagnosis": result.diagnosis.to_dict() if result.diagnosis else None,
        "metrics": {
            name: mr.score for name, mr in result.all_metrics().items()
        },
    }


@app.post("/generate-and-evaluate")
async def generate_and_evaluate(question: str, contexts: List[str]) -> dict:
    """Sketch: generate with your own pipeline, then gate on the ragval verdict."""
    answer = f"(your RAG pipeline would answer {question!r} here)"
    result = await evaluator.evaluate(question, answer, contexts)
    return {"answer": answer, "trusted": result.verdict != "FAIL", "verdict": result.verdict}
