"""``ragval`` command-line interface (Typer)."""

from __future__ import annotations

import json
import pathlib
from typing import List, Optional

import typer

from ragval import __version__
from ragval.evaluator import FULL_METRICS, RAGEvaluator
from ragval.metrics.registry import METRIC_REGISTRY, list_metrics

app = typer.Typer(
    add_completion=False,
    help="The complete RAG evaluation library. Scores, diagnoses, and domain-aware.",
)

_COLORS = {"PASS": typer.colors.GREEN, "WARN": typer.colors.YELLOW, "FAIL": typer.colors.RED}


def _parse_metrics(metrics: Optional[str]) -> Optional[List[str]]:
    if not metrics:
        return None
    if metrics.strip().lower() == "all":
        return list(FULL_METRICS)
    return [m.strip() for m in metrics.split(",") if m.strip()]


def _print_result(result, output: str) -> None:
    if output == "json":
        typer.echo(result.to_json())
        return

    color = _COLORS.get(result.verdict, typer.colors.WHITE)
    typer.secho(f"\nVERDICT: {result.verdict}", fg=color, bold=True)
    typer.echo(f"Overall score: {result.overall_score:.3f}")
    typer.echo(f"Model: {result.model}   Domain: {result.domain}\n")

    typer.secho("Metrics", bold=True)
    for name, mr in result.all_metrics().items():
        score = "n/a" if mr.score is None else f"{mr.score:.3f}"
        typer.echo(f"  {name:<32} {score:>8}")
        if mr.violations:
            for v in mr.violations[:3]:
                typer.secho(f"      - {v}", fg=typer.colors.RED)

    if result.domain_metrics:
        typer.secho("\nDomain metrics", bold=True)
        for name, mr in result.domain_metrics.items():
            if mr is None:
                continue
            score = "n/a" if mr.score is None else f"{mr.score:.3f}"
            typer.echo(f"  {name:<32} {score:>8}")

    if result.diagnosis:
        d = result.diagnosis
        typer.secho("\nDiagnosis", bold=True)
        typer.echo(f"  failed_layer : {d.failed_layer}")
        typer.echo(f"  root_cause   : {d.root_cause}")
        typer.echo(f"  suggested_fix: {d.suggested_fix}")
        typer.echo(f"  confidence   : {d.confidence}")
        for issue in d.secondary_issues:
            typer.echo(f"  secondary    : {issue}")


@app.command()
def evaluate(
    question: str = typer.Option(..., "--question", "-q"),
    answer: str = typer.Option(..., "--answer", "-a"),
    context: List[str] = typer.Option([], "--context", "-c", help="Repeatable."),
    model: str = typer.Option("groq/llama-3.3-70b-versatile", "--model", "-m"),
    domain: str = typer.Option("general", "--domain", "-d"),
    ground_truth: Optional[str] = typer.Option(None, "--ground-truth"),
    metrics: Optional[str] = typer.Option(None, "--metrics", help="comma list or 'all'"),
    output: str = typer.Option("text", "--output", "-o", help="text|json"),
) -> None:
    """Evaluate a single question / answer / context set."""
    evaluator = RAGEvaluator(
        model=model, domain=domain, metrics=_parse_metrics(metrics)
    )
    result = evaluator.evaluate_sync(
        question=question,
        answer=answer,
        contexts=list(context),
        ground_truth=ground_truth,
    )
    _print_result(result, output)


@app.command()
def batch(
    file: pathlib.Path = typer.Option(..., "--file", "-f", exists=True),
    model: str = typer.Option("groq/llama-3.3-70b-versatile", "--model", "-m"),
    domain: str = typer.Option("general", "--domain", "-d"),
    concurrency: int = typer.Option(3, "--concurrency"),
    metrics: Optional[str] = typer.Option(None, "--metrics"),
    output: str = typer.Option("text", "--output", "-o", help="text|json|markdown"),
) -> None:
    """Evaluate a JSON file of {question, answer, contexts, ground_truth} objects."""
    data = json.loads(file.read_text())
    if not isinstance(data, list):
        typer.secho("Input file must contain a JSON list.", fg=typer.colors.RED)
        raise typer.Exit(1)

    evaluator = RAGEvaluator(
        model=model, domain=domain, metrics=_parse_metrics(metrics)
    )
    result = evaluator.batch_evaluate_sync(data, concurrency=concurrency)

    if output == "json":
        typer.echo(result.to_json())
    elif output == "markdown":
        typer.echo(result.report())
    else:
        typer.secho(
            f"\n{result.total} evaluated  |  "
            f"PASS {result.pass_count}  WARN {result.warn_count}  "
            f"FAIL {result.fail_count}",
            bold=True,
        )
        typer.echo(f"avg overall score : {result.avg_overall_score:.3f}")
        typer.echo(f"avg faithfulness  : {result.avg_faithfulness:.3f}")
        typer.echo(f"hallucination rate: {result.hallucination_rate:.1%}")
        typer.echo(f"\n{result.report()}")


@app.command(name="list-metrics")
def list_metrics_command(
    category: Optional[str] = typer.Option(None, "--category", "-c"),
) -> None:
    """List every available metric, its category, and whether it needs ground truth."""
    names = list_metrics(category)
    typer.secho(f"{'metric':<30} {'category':<14} {'ground truth'}", bold=True)
    typer.echo("-" * 60)
    for name in names:
        cls = METRIC_REGISTRY[name]
        gt = "required" if getattr(cls, "requires_ground_truth", False) else "-"
        typer.echo(f"{name:<30} {getattr(cls, 'category', '?'):<14} {gt}")
    typer.echo(f"\n{len(names)} metrics.")


@app.command()
def version() -> None:
    """Print the installed ragval version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
