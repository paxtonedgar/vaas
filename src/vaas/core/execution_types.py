"""
Execution Types: Framework Services and Cross-Cutting Structures.

These are NOT ontology primitives. They are execution artifacts that
implement rule evaluation, routing, and tracing.

Layer: Execution
- Eligibility: Condition evaluation
- TaxTreatmentChange: Treatment consequence
- ReportingEffect: Reporting consequence
- Routing: Item → Surface assignment
- Exclusion: Mutual impossibility
- Derivation: Slot arithmetic
- ConditionalPrecedence: Conflict resolution
- RoleActivation: Dynamic role assumption
- ReportingSurface: Output target
- DecisionTrace: Audit trail

These types evolve with the framework. Changes here indicate
feature additions, not ontology shifts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# ENUMS
# =============================================================================

class SurfaceType(Enum):
    """Types of reporting surfaces."""
    FORM_FIELD = "form_field"
    API_ENDPOINT = "api_endpoint"
    LEDGER_ENTRY = "ledger_entry"
    STRUCTURED_MESSAGE = "structured_message"
    FREE_TEXT = "free_text"


# =============================================================================
# ELIGIBILITY / TREATMENT / EFFECT (Qualification Decomposition)
# =============================================================================

@dataclass
class Eligibility:
    """
    Determines whether a condition is met.

    Pure boolean evaluation - no consequences attached.
    """
    eligibility_id: str
    authority_id: str
    description: str

    temporal_constraints: List[str] = field(default_factory=list)
    actor_constraints: List[str] = field(default_factory=list)
    disqualifiers: List[str] = field(default_factory=list)


@dataclass
class TaxTreatmentChange:
    """
    Applies tax treatment consequence when eligibility is met.

    Separates determination from consequence.
    """
    change_id: str
    authority_id: str
    eligibility_id: str

    from_treatment: str
    to_treatment: str
    rate_change: Optional[Decimal] = None
    reclassification_allowed: bool = True


@dataclass
class ReportingEffect:
    """
    Determines reporting consequence of eligibility/treatment.

    Separates reporting from tax treatment.
    """
    effect_id: str
    authority_id: str

    eligibility_id: Optional[str] = None
    treatment_change_id: Optional[str] = None

    target_surface: str = ""
    include_in_aggregate: bool = True
    requires_statement: bool = False
    statement_fields: List[str] = field(default_factory=list)


@dataclass
class ComposedQualification:
    """
    For regimes where eligibility/treatment/reporting are coupled.

    Composes the three primitives into legacy Qualification behavior.
    """
    qualification_id: str
    authority_id: str

    eligibility: Eligibility
    treatment_on_pass: TaxTreatmentChange
    treatment_on_fail: TaxTreatmentChange
    reporting_on_pass: ReportingEffect
    reporting_on_fail: ReportingEffect


# =============================================================================
# ROUTING
# =============================================================================

@dataclass
class Routing:
    """
    Item to Surface assignment.

    Determines which ReportingSurface receives which Items
    under which conditions.
    """
    routing_id: str
    authority_id: str
    source_classification: str
    target_surface: str

    source_eligibility: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    priority: int = 0
    governed_by: List[str] = field(default_factory=list)


# =============================================================================
# EXCLUSION
# =============================================================================

@dataclass
class Exclusion:
    """
    Mutual impossibility constraint.

    Asserts that classifications/treatments cannot coexist within scope.
    """
    exclusion_id: str
    authority_id: str

    classifications: Set[str] = field(default_factory=set)
    scope: str = "per_item"

    governed_by: List[str] = field(default_factory=list)


# =============================================================================
# DERIVATION
# =============================================================================

@dataclass
class Derivation:
    """
    Slot arithmetic - how computed values derive from source values.
    """
    derivation_id: str
    authority_id: str
    target_surface: str

    operator: str  # "SUM", "SUBSET", "DIFFERENCE", "PRODUCT", "MINIMUM", "MAXIMUM"
    source_surfaces: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    governed_by: List[str] = field(default_factory=list)


# =============================================================================
# CONDITIONAL PRECEDENCE
# =============================================================================

@dataclass
class PrecedenceCondition:
    """Condition for precedence to apply."""
    condition_id: str
    condition_type: str
    value: str


@dataclass
class ConditionalPrecedence:
    """
    Non-linear precedence with cycles and context.

    Supports context-dependent and actor-dependent ordering.
    """
    precedence_id: str
    authority_id: str

    higher_rule: str
    lower_rule: str

    conditions: List[PrecedenceCondition] = field(default_factory=list)
    actor_context: Optional[str] = None
    temporal_context: Optional[str] = None

    revives_if: Optional[str] = None
    mutual_exclusion: bool = False


# =============================================================================
# ROLE ACTIVATION
# =============================================================================

@dataclass
class RoleActivation:
    """
    Dynamic role assumption by an actor.

    When context_condition is met, actor assumes role and inherits obligations.
    """
    activation_id: str
    authority_id: str
    actor_type: str
    assumed_role: str

    context_condition: str
    obligations_assumed: List[str] = field(default_factory=list)
    displaces_actor: Optional[str] = None
    governed_by: List[str] = field(default_factory=list)


# =============================================================================
# REPORTING SURFACE
# =============================================================================

@dataclass
class ReportingSurface:
    """
    Abstract target for reported values.

    Authority-defined, not form-centric.
    """
    surface_id: str
    authority_id: str
    surface_type: SurfaceType

    data_type: str = "amount"
    schema: Optional[Dict[str, Any]] = None
    physical_bindings: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# DECISION TRACE (Audit Artifact)
# =============================================================================

@dataclass
class RuleEvaluation:
    """Single rule evaluation within a trace."""
    rule_id: str
    rule_type: str
    evaluated_at: datetime

    conditions_checked: List[str] = field(default_factory=list)
    conditions_results: Dict[str, bool] = field(default_factory=dict)
    triggered: bool = False

    action_taken: Optional[str] = None
    exclusions_checked: List[str] = field(default_factory=list)
    uncertainty_invoked: Optional[str] = None


@dataclass
class DecisionTrace:
    """
    Records how a decision was reached.

    This is an execution artifact for debugging and audits,
    not an ontology primitive.
    """
    trace_id: str
    decision_type: str
    timestamp: datetime

    input_item: str
    input_context: Dict[str, Any] = field(default_factory=dict)

    rules_evaluated: List[RuleEvaluation] = field(default_factory=list)
    precedence_applied: List[str] = field(default_factory=list)

    final_decision: str = ""
    confidence: float = 1.0

    authority_id: str = ""
    human_readable_explanation: str = ""
    machine_checkable_proof: Optional[Dict[str, Any]] = None
