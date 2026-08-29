"""CLI behaviour (Typer CliRunner, MockProvider injected)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import ragval.evaluator as ev_mod
from ragval import __version__
from ragval.cli import app
from tests.conftest import MockProvider

runner = CliRunner()


@pytest.fixture(autouse=True)
def _inject_mock_provider(monkeypatch):
    real_init = ev_mod.RAGEvaluator.__init__

    def patched(self, *a, **kw):
        real_init(self, *a, **kw)
        self.provider = MockProvider()

    monkeypatch.setattr(ev_mod.RAGEvaluator, "__init__", patched)


def test_evaluate_text_output():
    result = runner.invoke(
        app,
        [
            "evaluate",
            "-q", "What is first-line for T2DM?",
            "-a", "Metformin 500mg twice daily.",
            "-c", "ADA guidelines recommend metformin 500mg twice daily.",
            "-c", "Metformin preferred unless contraindicated.",
            "--metrics", "faithfulness,context_precision",
        ],
    )
    assert result.exit_code == 0
    assert "VERDICT" in result.stdout


def test_evaluate_json_output():
    result = runner.invoke(
        app,
        [
            "evaluate",
            "-q", "q", "-a", "Metformin 500mg twice daily.",
            "-c", "ADA guidelines recommend metformin.",
            "--metrics", "faithfulness",
            "--output", "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "verdict" in payload
    assert "diagnosis" in payload


def test_list_metrics():
    result = runner.invoke(app, ["list-metrics"])
    assert result.exit_code == 0
    assert "faithfulness" in result.stdout
    assert "context_precision" in result.stdout
    assert "metrics." in result.stdout


def test_list_metrics_category_filter():
    result = runner.invoke(app, ["list-metrics", "--category", "retrieval"])
    assert result.exit_code == 0
    assert "faithfulness" not in result.stdout
    assert "context_precision" in result.stdout


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_batch_command(tmp_path):
    rows = [
        {
            "question": "q",
            "answer": "Metformin 500mg twice daily.",
            "contexts": ["ADA guidelines recommend metformin 500mg twice daily."],
        }
    ]
    f = tmp_path / "rows.json"
    f.write_text(json.dumps(rows))
    result = runner.invoke(
        app, ["batch", "--file", str(f), "--metrics", "faithfulness", "--output", "markdown"]
    )
    assert result.exit_code == 0
    assert "ragval batch report" in result.stdout
