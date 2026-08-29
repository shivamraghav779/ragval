"""Domain profile behaviour: clinical, legal, financial."""

from __future__ import annotations

import pytest

from ragval.domains import DOMAIN_REGISTRY, get_domain
from ragval.domains.clinical import ClinicalDomain
from ragval.domains.financial import FinancialDomain
from ragval.domains.legal import LegalDomain
from ragval.exceptions import DomainNotFoundError


def test_registry_has_all_domains():
    assert set(DOMAIN_REGISTRY) == {"general", "clinical", "legal", "financial"}


def test_get_unknown_domain_raises():
    with pytest.raises(DomainNotFoundError):
        get_domain("veterinary")


def test_general_domain_has_no_metrics():
    assert get_domain("general").additional_metric_names == []


@pytest.mark.asyncio
async def test_general_domain_metrics_empty(mock_provider):
    out = await get_domain("general").get_domain_metrics("q", "a", ["c"], mock_provider)
    assert out == {}


@pytest.mark.asyncio
async def test_clinical_domain_metrics(mock_provider, sample_question, sample_answer, sample_contexts):
    out = await ClinicalDomain().get_domain_metrics(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert set(out) == {
        "drug_name_precision",
        "dosing_accuracy",
        "contraindication_coverage",
        "authority_score",
    }
    assert out["drug_name_precision"].score == pytest.approx(1.0)
    assert out["dosing_accuracy"].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_clinical_drug_precision_safety_flag(mock_provider, sample_question, sample_answer, sample_contexts):
    mock_provider.set_response(
        "For each drug name from an answer",
        {"verdicts": [{"drug": "Metformin", "verified": False}]},
    )
    out = await ClinicalDomain().get_domain_metrics(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert out["drug_name_precision"].score < 0.9
    assert any("clinical_safety_concern" in v for v in out["drug_name_precision"].violations)


@pytest.mark.asyncio
async def test_clinical_authority_score_who(mock_provider):
    out = await ClinicalDomain().get_domain_metrics(
        "q", "a", ["Per WHO guidance, treatment is X."], mock_provider
    )
    assert out["authority_score"].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_legal_jurisdiction_specificity(mock_provider):
    out = await LegalDomain().get_domain_metrics(
        "Is this contract enforceable in California?",
        "Under California law, yes.",
        ["California Civil Code section 1550 sets out contract requirements."],
        mock_provider,
    )
    assert out["jurisdiction_specificity"].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_legal_jurisdiction_not_applicable(mock_provider):
    mock_provider.set_response(
        "implies or requires a specific jurisdiction",
        {"question_implies_jurisdiction": False,
         "answer_specifies_jurisdiction": "missing", "reasoning": "generic"},
    )
    out = await LegalDomain().get_domain_metrics(
        "What is consideration in contract law?", "A bargained-for exchange.",
        ["Consideration is a bargained-for exchange."], mock_provider,
    )
    assert out["jurisdiction_specificity"].score is None


@pytest.mark.asyncio
async def test_financial_numerical_accuracy(mock_provider):
    mock_provider.set_response(
        "Extract every specific",
        {"numbers": [{"text": "6.5%", "type": "rate"}]},
    )
    out = await FinancialDomain().get_domain_metrics(
        "What is the current repo rate?",
        "The repo rate is 6.5%.",
        ["The RBI repo rate is 6.5% as of 2026."],
        mock_provider,
    )
    assert out["numerical_accuracy"].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_financial_untraceable_figure_flag(mock_provider):
    mock_provider.set_response(
        "Extract every specific",
        {"numbers": [{"text": "9.9%", "type": "rate"}]},
    )
    out = await FinancialDomain().get_domain_metrics(
        "What is the repo rate?", "It is 9.9%.",
        ["The repo rate is 6.5%."], mock_provider,
    )
    assert out["numerical_accuracy"].score == 0.0
    assert any("financial_accuracy_concern" in v for v in out["numerical_accuracy"].violations)


@pytest.mark.asyncio
async def test_domain_metric_error_is_captured(sample_contexts):
    class Boom:
        async def complete_json(self, prompt):
            raise RuntimeError("provider offline")

    out = await ClinicalDomain().get_domain_metrics("q", "a", sample_contexts, Boom())
    # authority_score does not call the provider, so it still succeeds.
    assert out["authority_score"].score is not None
    assert out["drug_name_precision"].score is None
