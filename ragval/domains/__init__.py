"""Domain registry: general, clinical, legal, financial."""

from __future__ import annotations

from typing import Dict, Type

from ragval.domains.base import BaseDomain
from ragval.domains.clinical import ClinicalDomain
from ragval.domains.financial import FinancialDomain
from ragval.domains.general import GeneralDomain
from ragval.domains.legal import LegalDomain
from ragval.exceptions import DomainNotFoundError

DOMAIN_REGISTRY: Dict[str, Type[BaseDomain]] = {
    "general": GeneralDomain,
    "clinical": ClinicalDomain,
    "legal": LegalDomain,
    "financial": FinancialDomain,
}

__all__ = [
    "BaseDomain",
    "GeneralDomain",
    "ClinicalDomain",
    "LegalDomain",
    "FinancialDomain",
    "DOMAIN_REGISTRY",
    "get_domain",
]


def get_domain(name: str) -> BaseDomain:
    """Return an instance of the domain profile registered under ``name``."""
    try:
        return DOMAIN_REGISTRY[name]()
    except KeyError:
        raise DomainNotFoundError(name, list(DOMAIN_REGISTRY)) from None
