"""
Core Ontology Types: The Semantic Foundation.

These are true primitives - irreducible concepts that describe
the domain, not execution artifacts.

Layer: Ontology
- Authority: Who defines rules
- EconomicEvent: What happened
- Item: Reportable unit from event
- Classification: Tax treatment category
- Actor: Party in chain
- TemporalConstraint: Time-bounded condition
- KnowledgeBoundary: Epistemic uncertainty
- LifecycleEvent: Entity/stream termination

These types are stable. Changes here indicate ontology evolution,
not feature additions.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import List, Optional


# =============================================================================
# ENUMS
# =============================================================================

class AuthorityType(Enum):
    """Types of authorities that define reporting rules."""
    SOVEREIGN = "sovereign"           # National government (IRS, HMRC)
    TREATY = "treaty"                 # Bilateral/multilateral (OECD MLI)
    REGULATORY = "regulatory"         # Non-tax regulators (SEC, FCA)
    CONTRACTUAL = "contractual"       # Private agreements
    LEDGER = "ledger"                 # Blockchain/DLT native
    REALTIME = "realtime"             # API-based withholding systems


class EconomicEventType(Enum):
    """Types of economic events that produce Items."""
    DISTRIBUTION = "distribution"     # Dividend, interest, etc.
    DISPOSITION = "disposition"       # Sale, exchange, redemption
    ACCRUAL = "accrual"               # OID, imputed interest
    DEEMED = "deemed"                 # Constructive/deemed events
    TERMINATION = "termination"       # Liquidation, dissolution
    TRANSFER = "transfer"             # Gift, inheritance


class UncertaintyState(Enum):
    """States in uncertainty lifecycle."""
    ACTIVE = "active"                 # Uncertainty is live
    RESOLVED = "resolved"             # Fact became known
    EXPIRED = "expired"               # Deadline passed without resolution
    DISPUTED = "disputed"             # Under dispute
    FINAL = "final"                   # Resolution is binding


class LifecycleEventType(Enum):
    """Types of lifecycle events affecting entities/items."""
    LIQUIDATION = "liquidation"       # Corporate liquidation
    DISSOLUTION = "dissolution"       # Partnership dissolution
    TERMINATION = "termination"       # Trust termination
    MERGER = "merger"                 # Entity combination
    SPINOFF = "spinoff"               # Entity separation
    CONVERSION = "conversion"         # Entity type change
    WIND_UP = "wind_up"               # Cross-border wind-up


# =============================================================================
# AUTHORITY
# =============================================================================

@dataclass
class Authority:
    """
    A jurisdiction or regime that defines reporting rules.

    This is the root of all rule scoping. Every rule, surface, and precedence
    is scoped to an Authority.
    """
    authority_id: str
    authority_type: AuthorityType
    name: str
    jurisdiction: Optional[str] = None

    # What this authority controls
    defines_surfaces: List[str] = field(default_factory=list)
    defines_precedence: List[str] = field(default_factory=list)
    resolves_uncertainty: str = "default"

    # Authority relationships
    supersedes: List[str] = field(default_factory=list)
    superseded_by: List[str] = field(default_factory=list)


# =============================================================================
# ECONOMIC EVENT
# =============================================================================

@dataclass
class EconomicEvent:
    """
    The underlying economic occurrence before reporting classification.

    Separates:
    - Cash flow (actual payment)
    - Legal entitlement (right to receive)
    - Economic benefit (constructive receipt)

    One EconomicEvent may produce multiple Items under different regimes.
    """
    event_id: str
    event_type: EconomicEventType
    occurrence_date: date
    recognition_date: Optional[date] = None

    # Participants
    source_actor: str = ""
    destination_actor: str = ""

    # Economic substance
    amount: Optional[Decimal] = None
    currency: str = "USD"
    underlying_asset: Optional[str] = None

    # Legal characterization dates
    legal_entitlement_date: Optional[date] = None
    cash_flow_date: Optional[date] = None
    economic_benefit_date: Optional[date] = None


# =============================================================================
# ITEM
# =============================================================================

@dataclass
class Item:
    """
    A reportable unit derived from an EconomicEvent.

    Items are authority-specific views of economic events.
    """
    item_id: str
    authority_id: str
    source_event_id: str
    classification_id: str

    amount: Optional[Decimal] = None
    currency: str = "USD"
    recognition_date: Optional[date] = None
    reporting_period: Optional[str] = None

    superseded_by: Optional[str] = None
    is_active: bool = True


# =============================================================================
# CLASSIFICATION
# =============================================================================

@dataclass
class Classification:
    """
    A tax treatment category for Items.

    Classifications are authority-defined categories that determine
    tax treatment (rates, rules, reporting locations).
    """
    classification_id: str
    authority_id: str
    name: str
    description: str = ""

    default_rate: Optional[Decimal] = None
    governed_by: List[str] = field(default_factory=list)

    parent_classification_id: Optional[str] = None
    is_abstract: bool = False


# =============================================================================
# ACTOR
# =============================================================================

@dataclass
class Actor:
    """
    A party in the reporting chain.

    Actors are entities that have obligations, rights, or roles
    in the tax reporting ecosystem.
    """
    actor_id: str
    actor_type: str
    name: Optional[str] = None

    tin: Optional[str] = None
    jurisdiction: Optional[str] = None

    can_withhold: bool = False
    can_file: bool = False
    is_pass_through: bool = False


# =============================================================================
# TEMPORAL CONSTRAINT
# =============================================================================

@dataclass
class TemporalConstraint:
    """
    A time-bounded condition.

    Captures holding periods, windows, and other time-based rules.
    """
    constraint_id: str
    authority_id: str
    description: str

    window_length_days: Optional[int] = None
    window_offset_days: int = 0
    reference_event: str = ""

    minimum_days: Optional[int] = None
    daycount_rule: Optional[str] = None

    governed_by: List[str] = field(default_factory=list)


# =============================================================================
# KNOWLEDGE BOUNDARY
# =============================================================================

@dataclass
class KnowledgeBoundary:
    """
    Epistemic uncertainty with lifecycle.

    Models what is unknown, who is responsible, and how uncertainty resolves.
    """
    boundary_id: str
    authority_id: str

    unknown_fact: str
    responsible_actor: str
    knowledge_holder: Optional[str] = None

    conservative_action: str = ""
    disclosure_required: bool = False
    disclosure_content: Optional[List[str]] = None

    # Lifecycle
    expires_when: Optional[str] = None
    expiration_date: Optional[date] = None
    transitions_to: Optional[str] = None

    resolution_mechanism: str = ""
    dispute_authority: Optional[str] = None
    retroactive_liability: bool = False
    audit_retention_years: Optional[int] = None

    governed_by: List[str] = field(default_factory=list)


# =============================================================================
# LIFECYCLE EVENT
# =============================================================================

@dataclass
class CrossBorderEffect:
    """Effect of lifecycle event in another jurisdiction."""
    target_authority: str
    recognition_date: date
    treatment: str
    withholding_required: bool = False


@dataclass
class LifecycleEvent:
    """
    Events that change entity/item stream lifecycle.

    Not classification changes - fundamentally changes what exists.
    """
    event_id: str
    authority_id: str
    event_type: LifecycleEventType

    affected_entity: str
    affected_item_streams: List[str] = field(default_factory=list)

    effective_date: Optional[date] = None
    announcement_date: Optional[date] = None

    terminates_items: bool = True
    aggregation_override: Optional[str] = None
    final_distribution_treatment: Optional[str] = None
    successor_entity: Optional[str] = None

    cross_border_effects: List[CrossBorderEffect] = field(default_factory=list)


# =============================================================================
# CANONICAL INSTANTIATIONS
# =============================================================================

AUTHORITY_IRS = Authority(
    authority_id="auth:irs",
    authority_type=AuthorityType.SOVEREIGN,
    name="Internal Revenue Service",
    jurisdiction="US",
    defines_surfaces=["1099-DIV", "1099-INT", "Schedule K-1"],
    defines_precedence=["irc_hierarchy"],
    resolves_uncertainty="amended_return",
)

AUTHORITY_OECD_CRS = Authority(
    authority_id="auth:oecd_crs",
    authority_type=AuthorityType.TREATY,
    name="OECD Common Reporting Standard",
    jurisdiction="OECD",
    defines_surfaces=["crs_report"],
    defines_precedence=["crs_hierarchy"],
    resolves_uncertainty="competent_authority_exchange",
)

AUTHORITY_REALTIME_WHT = Authority(
    authority_id="auth:realtime_wht",
    authority_type=AuthorityType.REALTIME,
    name="Real-Time Withholding API",
    jurisdiction="hypothetical",
    defines_surfaces=["api_payload"],
    defines_precedence=["api_precedence"],
    resolves_uncertainty="immediate_correction",
)
