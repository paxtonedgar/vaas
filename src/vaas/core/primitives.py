"""
Semantic Core v3: Re-export Facade.

This module re-exports all primitives from their proper layers:
- core_types.py: Ontology primitives (stable)
- execution_types.py: Execution artifacts (evolving)

Import from this module for backward compatibility.
For new code, prefer importing from the specific layer modules.

Design Principles:
1. Authority-agnostic, not form-first
2. EconomicEvent precedes Item
3. Qualification is decomposed (Eligibility + TaxTreatmentChange + ReportingEffect)
4. Precedence is non-linear
5. Uncertainty has lifecycle
6. Decisions are traceable
"""

# =============================================================================
# CORE TYPES (Ontology Layer - Stable)
# =============================================================================

from vaas.core.core_types import (
    # Enums
    AuthorityType,
    EconomicEventType,
    UncertaintyState,
    LifecycleEventType,
    # Primitives
    Authority,
    EconomicEvent,
    Item,
    Classification,
    Actor,
    TemporalConstraint,
    KnowledgeBoundary,
    CrossBorderEffect,
    LifecycleEvent,
    # Canonical instantiations
    AUTHORITY_IRS,
    AUTHORITY_OECD_CRS,
    AUTHORITY_REALTIME_WHT,
)

# =============================================================================
# EXECUTION TYPES (Framework Layer - Evolving)
# =============================================================================

from vaas.core.execution_types import (
    # Enums
    SurfaceType,
    # Qualification decomposition
    Eligibility,
    TaxTreatmentChange,
    ReportingEffect,
    ComposedQualification,
    # Routing and rules
    Routing,
    Exclusion,
    Derivation,
    PrecedenceCondition,
    ConditionalPrecedence,
    RoleActivation,
    ReportingSurface,
    # Audit
    RuleEvaluation,
    DecisionTrace,
)


# =============================================================================
# ADDITIONAL TYPES (for backward compatibility)
# =============================================================================

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass
class AuthorityScopedRule:
    """Base class for authority-scoped rules."""
    rule_id: str
    authority_id: str
    effective_date: Optional[date] = None
    expiration_date: Optional[date] = None


# =============================================================================
# __all__ for explicit exports
# =============================================================================

__all__ = [
    # Core Enums
    "AuthorityType",
    "EconomicEventType",
    "UncertaintyState",
    "LifecycleEventType",
    "SurfaceType",
    # Core Types
    "Authority",
    "AuthorityScopedRule",
    "EconomicEvent",
    "Item",
    "Classification",
    "Actor",
    "TemporalConstraint",
    "KnowledgeBoundary",
    "CrossBorderEffect",
    "LifecycleEvent",
    # Execution Types
    "Eligibility",
    "TaxTreatmentChange",
    "ReportingEffect",
    "ComposedQualification",
    "Routing",
    "Exclusion",
    "Derivation",
    "PrecedenceCondition",
    "ConditionalPrecedence",
    "RoleActivation",
    "ReportingSurface",
    "RuleEvaluation",
    "DecisionTrace",
    # Canonical instances
    "AUTHORITY_IRS",
    "AUTHORITY_OECD_CRS",
    "AUTHORITY_REALTIME_WHT",
]
