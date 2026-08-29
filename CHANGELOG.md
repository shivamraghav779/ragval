# Changelog

All notable changes to `ragval` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

`pip install ragval` → `import ragval`.

## [0.0.1] - 2026-08-29

Initial release. **40 registered metrics + 10 domain metrics = 50**, plus 3
custom-metric builders.

### Added

- **Async-native core.** Every metric exposes `async compute()` and sync
  `compute_sync()`. `RAGEvaluator` exposes `evaluate()` / `evaluate_sync()`,
  `evaluate_pipeline()`, `evaluate_agent()`, `evaluate_conversation()`,
  `batch_evaluate()` / `batch_evaluate_sync()`. FastAPI-compatible.
- **One provider integration: LiteLLM.** `LiteLLMProvider` routes to OpenAI,
  Anthropic, Groq, Azure, Ollama, and 100+ providers via model string, with
  auth-error short-circuiting and JSON-mode helpers. Rate limits get a patient
  retry budget (up to 6 attempts) that honors a provider-supplied `Retry-After`;
  other transient errors retry 3× with exponential backoff. `RAGEvaluator`
  accepts `timeout` and `max_tokens` for slower or reasoning models.
- **Retrieval metrics (11):** `context_precision` (rank-aware MAP),
  `context_recall`, `context_relevance` (per-chunk query relevance, not
  rank-aware), `context_entity_recall`, `context_utilization` (reference-free,
  answer-based), `context_sufficiency` (reference-free), `retrieval_diversity`
  (pure-text redundancy), `noise_sensitivity`, `mrr`, `ndcg`, `hit_rate`.
- **Generation metrics (13):** `faithfulness`, `answer_relevance`,
  `answer_correctness`, `hallucination`, `factual_correctness`,
  `answer_semantic_similarity`, `summarization`, `answer_completeness`,
  `coherence`, `fluency`, `conciseness`, `refusal_appropriateness`
  (penalizes both over- and under-refusal), `citation_support`.
- **Safety metrics (5):** `bias`, `toxicity` (professional-context aware),
  `topic_adherence`, `pii_leakage`, `tone_professionalism`.
- **Agentic metrics:** `tool_correctness`, `argument_correctness`,
  `task_completion`, `step_efficiency`, `plan_adherence`, `plan_quality`,
  `agent_goal_accuracy`.
- **Conversation metrics (4):** `conversation_completeness`,
  `knowledge_retention`, `role_adherence`, `conversation_relevancy`.
- **Custom metrics:** `GEval`, `RubricEval`, `AspectCritic`.
- **Pipeline-layer diagnosis engine.** An ordered decision tree over the metric
  results isolates the failing layer (`knowledge_base`, `retrieval`,
  `fusion_ranking`, `generation`, `prompt`), the root cause, a concrete fix, a
  confidence score, and secondary issues.
- **Domain profiles.**
  - `clinical`: `drug_name_precision`, `dosing_accuracy`,
    `contraindication_coverage`, `authority_score`.
  - `legal`: `jurisdiction_specificity`, `citation_accuracy`, `statute_currency`.
  - `financial`: `numerical_accuracy`, `regulatory_compliance_mention`,
    `temporal_accuracy`.
- **Result objects** with `verdict` (PASS/WARN/FAIL), blended `overall_score`,
  `to_dict()` / `to_json()`, and a formatted box display.
  `BatchEvaluationResult.report()` renders a markdown table plus a diagnosis
  summary; `to_dataframe()` needs `ragval[pandas]`.
- **CLI** (`ragval`): `evaluate`, `batch`, `list-metrics`, `version`.
- **Pure-Python text utilities** (TF-IDF cosine, entity/sentence extraction,
  n-grams). No numpy, sklearn, or sentence-transformers.

## Development setup

```bash
git clone https://github.com/shivamraghav779/ragval
cd ragval
pip install -e ".[dev]"

pytest tests/ -v

python -m build
twine check dist/*
twine upload dist/*
```
