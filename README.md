# ragval

**The complete RAG evaluation library. Scores, diagnoses, and domain-aware.**

[![PyPI](https://img.shields.io/pypi/v/ragval?color=1f6feb&label=pypi)](https://pypi.org/project/ragval/)
[![Python](https://img.shields.io/pypi/pyversions/ragval?color=1f6feb)](https://pypi.org/project/ragval/)
[![License: MIT](https://img.shields.io/badge/license-MIT-1f6feb)](https://opensource.org/licenses/MIT)

`ragval` covers every metric category in the RAG evaluation literature —
50 metrics across retrieval, generation, safety, agentic, conversation, and
custom — behind one async-native API. It then does two things most evaluation
tools don't:

1. **Pipeline-layer diagnosis.** When a score is low, `ragval` tells you *which
   layer* failed — knowledge base, retrieval, fusion/ranking, generation, or
   prompt — the likely root cause, and a concrete fix.
2. **Domain-specific metrics.** First-class profiles for **clinical**, **legal**,
   and **financial** RAG, with metrics like `drug_name_precision`,
   `dosing_accuracy`, `jurisdiction_specificity`, `citation_accuracy`, and
   `numerical_accuracy`.

One provider integration: **LiteLLM**. Point a model string at OpenAI, Anthropic,
Groq, Azure, Ollama, or any of 100+ providers.

---

## Install

```bash
pip install ragval
pip install "ragval[pandas]"   # for BatchEvaluationResult.to_dataframe()
```

## Quick start

```python
from ragval import evaluate

result = await evaluate(
    question="What is the first-line treatment for Type 2 diabetes?",
    answer="Metformin, typically started at 500mg twice daily.",
    contexts=["ADA guidelines recommend metformin as first-line therapy..."],
)

print(result.verdict)                    # PASS | WARN | FAIL
print(result.diagnosis.failed_layer)     # e.g. "retrieval"
print(result.diagnosis.suggested_fix)
```

Every metric has async `compute()` and sync `compute_sync()`. The evaluator has
`evaluate()` / `evaluate_sync()`. FastAPI-compatible out of the box.

## Reusable evaluator

```python
from ragval import RAGEvaluator

evaluator = RAGEvaluator(
    model="groq/llama-3.3-70b-versatile",
    domain="clinical",
)

result = await evaluator.evaluate(
    question=question,
    answer=answer,
    contexts=contexts,
    ground_truth=ground_truth,   # optional
)

print(result.domain_metrics["drug_name_precision"].score)
print(result.domain_metrics["dosing_accuracy"].score)
```

Run only the metrics you want:

```python
RAGEvaluator(
    model="groq/llama-3.3-70b-versatile",
    metrics=["faithfulness", "context_precision", "hallucination"],
)
```

## Metrics coverage

**40 registered metrics + 10 domain metrics = 50**, plus 3 custom-metric builders.

| Category | Metric | Ground truth | LLM |
|----------|--------|:------------:|:---:|
| retrieval | `context_precision` | – | ✅ |
| retrieval | `context_recall` | ✅ | ✅ |
| retrieval | `context_relevance` | – | ✅ |
| retrieval | `context_entity_recall` | ✅ | ✅ |
| retrieval | `context_utilization` | – | ✅ |
| retrieval | `context_sufficiency` | – | ✅ |
| retrieval | `retrieval_diversity` | – | – |
| retrieval | `noise_sensitivity` | – | ✅ |
| retrieval | `mrr` | – | ✅ |
| retrieval | `ndcg` | – | ✅ |
| retrieval | `hit_rate` | – | ✅ |
| generation | `faithfulness` | – | ✅ |
| generation | `answer_relevance` | – | ✅ |
| generation | `answer_correctness` | ✅ | ✅ |
| generation | `hallucination` | – | ✅ |
| generation | `factual_correctness` | ✅ | ✅ |
| generation | `answer_semantic_similarity` | ✅ | – |
| generation | `answer_completeness` | – | ✅ |
| generation | `summarization` | – | ✅ |
| generation | `coherence` | – | ✅ |
| generation | `fluency` | – | ✅ |
| generation | `conciseness` | – | ✅ |
| generation | `refusal_appropriateness` | – | ✅ |
| generation | `citation_support` | – | ✅ |
| safety | `bias` | – | ✅ |
| safety | `toxicity` | – | ✅ |
| safety | `topic_adherence` | – | ✅ |
| safety | `pii_leakage` | – | ✅ |
| safety | `tone_professionalism` | – | ✅ |
| agentic | `tool_correctness` | – | – |
| agentic | `argument_correctness` | – | ✅ |
| agentic | `task_completion` | – | ✅ |
| agentic | `step_efficiency` | – | ✅ |
| agentic | `plan_adherence` | – | ✅ |
| agentic | `plan_quality` | – | ✅ |
| agentic | `agent_goal_accuracy` | ✅ | ✅ |
| conversation | `conversation_completeness` | – | ✅ |
| conversation | `knowledge_retention` | – | ✅ |
| conversation | `role_adherence` | – | ✅ |
| conversation | `conversation_relevancy` | – | ✅ |
| custom | `GEval`, `RubricEval`, `AspectCritic` | – | ✅ |

`ragval list-metrics` prints this table from the live registry.

## Domain profiles

| Domain | Extra metrics |
|--------|---------------|
| `clinical` | `drug_name_precision`, `dosing_accuracy`, `contraindication_coverage`, `authority_score` |
| `legal` | `jurisdiction_specificity`, `citation_accuracy`, `statute_currency` |
| `financial` | `numerical_accuracy`, `regulatory_compliance_mention`, `temporal_accuracy` |

## Supported providers

| Model string | Provider | API key env var |
|--------------|----------|-----------------|
| `groq/llama-3.3-70b-versatile` | Groq | `GROQ_API_KEY` |
| `openai/gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `anthropic/claude-3-haiku` | Anthropic | `ANTHROPIC_API_KEY` |
| `azure/gpt-4o` | Azure OpenAI | `AZURE_API_KEY` |
| `ollama/llama3` | Ollama (local) | – (`api_base`) |
| `together_ai/...` | Together | `TOGETHERAI_API_KEY` |

## Full pipeline evaluation

```python
result = await evaluator.evaluate_pipeline(
    question=question,
    answer=answer,
    retrieved_chunks=chunks,
    ground_truth=ground_truth,
    expected_chunks=expected,
    k=5,
)
```

## Batch evaluation

```python
batch = await evaluator.batch_evaluate(qa_pairs=rows, concurrency=3)
print(batch.report())          # markdown table + diagnosis summary
df = batch.to_dataframe()       # needs ragval[pandas]
worst = batch.worst_cases(5)
```

## Diagnosis

```python
result.diagnosis.failed_layer      # "retrieval"
result.diagnosis.root_cause        # "More than half of retrieved chunks are irrelevant..."
result.diagnosis.suggested_fix     # "Improve chunking granularity. Add a reranking step..."
result.diagnosis.confidence        # 0.82
result.diagnosis.secondary_issues  # ["[noise_sensitive] possible fusion_ranking issue: ..."]
```

## Custom metrics

```python
from ragval.metrics.custom import GEval

metric = GEval(
    name="Dosing Specificity",
    criteria="The answer must include exact dose amounts with units for every drug mentioned.",
    model="groq/llama-3.3-70b-versatile",
)
score = await metric.compute(question, answer, contexts, provider)
```

## Agentic and conversation

```python
result = await evaluator.evaluate_agent(
    question=question, answer=answer,
    tool_calls=actual, expected_tools=expected,
    action_trace=steps, declared_plan=plan,
)

conv = await evaluator.evaluate_conversation(
    turns=turns, system_role="clinical decision support assistant",
)
```

## CLI

```bash
ragval evaluate -q "..." -a "..." -c "chunk 1" -c "chunk 2" --domain clinical
ragval batch --file rows.json --concurrency 3 --output markdown
ragval list-metrics --category retrieval
ragval version
```

## Why ragval

**Most evaluation tools score your pipeline. They don't tell you what to fix.**
`ragval`'s diagnosis engine walks an ordered decision tree over the metric
results and isolates the failing layer, so a 0.42 faithfulness score becomes
"retrieval is fine, generation is hallucinating — tighten the grounding
instruction."

**Generic metrics miss domain-critical failures.** A clinically wrong dose can
be perfectly "faithful" to a bad chunk. `ragval`'s clinical profile checks drug
names, doses, and contraindications against the retrieved context specifically,
and flags patient-safety concerns even when the generic scores look fine. The
same applies to jurisdiction and citation currency for legal RAG, and figure
traceability for financial RAG.

**One integration, async-native, no model downloads.** LiteLLM gives you every
provider through a model string. Text metrics use pure-Python TF-IDF — no
`sentence-transformers`, no torch, fast `pip install`.

## Contributing

```bash
git clone https://github.com/shivamraghav779/ragval
cd ragval
pip install -e ".[dev]"
pytest tests/ -v
```

Issues and PRs welcome at <https://github.com/shivamraghav779/ragval>.

## License

MIT © Shivam Raghav
