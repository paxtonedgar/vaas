# Semantic Core v3: Jurisdiction-Agnostic Legal Reasoning

**Status:** Architectural Extension
**Version:** 3.0
**Date:** 2026-01-12
**Supersedes:** SEMANTIC_CORE.md (v2.0)

---

## Design Evolution

v2.0 achieved: **Portable abstraction of IRS logic**
v3.0 targets: **Jurisdiction-agnostic legal reasoning infrastructure**

The gap is **architectural**, not semantic. This document closes it.

---

## Architectural Changes from v2.0

| v2.0 Limitation | v3.0 Resolution |
|-----------------|-----------------|
| Implicit IRS Authority | **Authority** primitive - defines reporting surfaces, precedence |
| Atomic Item assumption | **EconomicEvent** pre-Item layer - cash flow ≠ entitlement ≠ benefit |
| Overloaded Qualification | Split into **Eligibility**, **TaxTreatmentChange**, **ReportingEffect** |
| Incomplete uncertainty lifecycle | **KnowledgeBoundary** gains expiration/transition semantics |
| Linear precedence | **ConditionalPrecedence** - contextual, actor-dependent, cyclic-safe |
| Atom inflation risk | **AtomConstraint** - one irreducible fact per atom |
| No decision trace | **DecisionTrace** primitive - explainable, auditable |
| No lifecycle events | **LifecycleEvent** primitive - terminations, dissolutions |

---

## Extended Primitive Set (16 Primitives)

| # | Primitive | New? | Purpose |
|---|-----------|------|---------|
| 1 | **Authority** | NEW | Jurisdiction/regime that defines rules |
| 2 | **EconomicEvent** | NEW | Pre-Item: what happened economically |
| 3 | **Item** | Modified | Reportable unit derived from EconomicEvent |
| 4 | **Classification** | Unchanged | Tax treatment category |
| 5 | **Eligibility** | NEW (from Qualification) | Determines if condition is met |
| 6 | **TaxTreatmentChange** | NEW (from Qualification) | Applies rate/treatment consequence |
| 7 | **ReportingEffect** | NEW (from Qualification) | Determines reporting consequence |
| 8 | **TemporalConstraint** | Unchanged | Time-bounded conditions |
| 9 | **KnowledgeBoundary** | Extended | Uncertainty with lifecycle |
| 10 | **ReportingSurface** | Modified | Abstract target (authority-defined) |
| 11 | **Actor** | Unchanged | Party in chain |
| 12 | **RoleActivation** | Unchanged | Dynamic role assumption |
| 13 | **Routing** | Unchanged | Item → Surface assignment |
| 14 | **Exclusion** | Unchanged | Mutual impossibility |
| 15 | **ConditionalPrecedence** | Modified | Non-linear conflict resolution |
| 16 | **LifecycleEvent** | NEW | Terminations, dissolutions, wind-ups |
| 17 | **DecisionTrace** | NEW | Explainability and audit trail |
| 18 | **Derivation** | Unchanged | Slot arithmetic |

---

## Primitive 1: Authority (NEW)

**The regime/jurisdiction that defines reporting rules.**

This removes implicit IRS assumption. Authority is now explicit.

```python
@dataclass
class Authority:
    """
    A jurisdiction or regime that defines reporting rules.

    Removes implicit IRS bias. Every rule, surface, and precedence
    is now scoped to an Authority.
    """
    authority_id: str
    authority_type: AuthorityType
    name: str
    jurisdiction: Optional[str] = None    # "US", "UK", "OECD", etc.

    # What this authority controls
    defines_surfaces: List[str]           # ReportingSurface IDs
    defines_precedence: List[str]         # Precedence rule IDs
    resolves_uncertainty: str             # How KnowledgeBoundary resolves

    # Authority relationships
    supersedes: List[str] = field(default_factory=list)    # Lower authorities
    superseded_by: List[str] = field(default_factory=list) # Higher authorities

class AuthorityType(Enum):
    """Types of authorities."""
    SOVEREIGN = "sovereign"           # National government (IRS, HMRC)
    TREATY = "treaty"                 # Bilateral/multilateral (OECD MLI)
    REGULATORY = "regulatory"         # Non-tax regulators (SEC, FCA)
    CONTRACTUAL = "contractual"       # Private agreements
    LEDGER = "ledger"                 # Blockchain/DLT native
    REALTIME = "realtime"             # API-based withholding systems

# Instantiations
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
```

### Authority-Scoped References

All rules now require authority scope:

```python
@dataclass
class AuthorityScopedRule:
    """Base class for authority-scoped rules."""
    rule_id: str
    authority_id: str                     # Which authority defines this
    effective_date: Optional[date] = None
    expiration_date: Optional[date] = None
```

---

## Primitive 2: EconomicEvent (NEW)

**Pre-Item layer: what happened economically.**

One EconomicEvent can produce multiple Items (cash flow, legal entitlement, economic benefit).

```python
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
    recognition_date: Optional[date] = None  # May differ (constructive receipt)

    # Participants
    source_actor: str
    destination_actor: str

    # Economic substance
    amount: Optional[Decimal] = None
    currency: str = "USD"
    underlying_asset: Optional[str] = None

    # Legal characterization
    legal_entitlement_date: Optional[date] = None
    cash_flow_date: Optional[date] = None
    economic_benefit_date: Optional[date] = None

    def produces_items(self, authority: Authority) -> List['Item']:
        """
        Generate Items from this event under a specific authority.

        Different authorities may produce different Items from same event.
        """
        pass

class EconomicEventType(Enum):
    """Types of economic events."""
    DISTRIBUTION = "distribution"         # Dividend, interest, etc.
    DISPOSITION = "disposition"           # Sale, exchange, redemption
    ACCRUAL = "accrual"                   # OID, imputed interest
    DEEMED = "deemed"                     # Constructive/deemed events
    TERMINATION = "termination"           # Liquidation, dissolution
    TRANSFER = "transfer"                 # Gift, inheritance
```

### EconomicEvent → Item Derivation

```python
# Single event, multiple items under different recognition rules
event = EconomicEvent(
    event_id="evt:q4_dividend_2024",
    event_type=EconomicEventType.DISTRIBUTION,
    occurrence_date=date(2024, 12, 15),     # Declared
    cash_flow_date=date(2025, 1, 15),       # Paid
    economic_benefit_date=date(2024, 12, 31),  # Constructive receipt
    source_actor="corporation_a",
    destination_actor="shareholder_b",
    amount=Decimal("1000.00"),
)

# Under IRS authority: constructive receipt applies
# → Item dated 2024-12-31, reportable on 2024 Form 1099-DIV

# Under cash-basis regime: actual payment date
# → Item dated 2025-01-15, reportable on 2025 return

# Under accrual regime: declaration date
# → Item dated 2024-12-15
```

---

## Primitives 5-7: Eligibility, TaxTreatmentChange, ReportingEffect (Split from Qualification)

**Qualification did three jobs. Now they're separate.**

### Primitive 5: Eligibility

```python
@dataclass
class Eligibility:
    """
    Determines whether a condition is met.

    Pure boolean evaluation - no consequences attached.
    """
    eligibility_id: str
    authority_id: str
    description: str

    # Evaluation criteria
    temporal_constraints: List['TemporalConstraint']
    actor_constraints: List['ActorConstraint']
    disqualifiers: List['Disqualifier']

    def evaluate(self, item: 'Item', context: 'EvaluationContext') -> bool:
        """Returns True if eligible, False otherwise."""
        pass
```

### Primitive 6: TaxTreatmentChange

```python
@dataclass
class TaxTreatmentChange:
    """
    Applies tax treatment consequence when eligibility is met.

    Separates determination from consequence.
    This allows jurisdictions where:
    - Eligibility affects rate but not reporting
    - Reclassification is prohibited
    """
    change_id: str
    authority_id: str

    # Trigger
    eligibility_id: str                   # Which Eligibility gates this

    # Effect
    from_treatment: str                   # Current classification
    to_treatment: str                     # New classification
    rate_change: Optional[Decimal] = None # Rate adjustment (if any)

    # Constraints
    reclassification_allowed: bool = True # Some jurisdictions prohibit
```

### Primitive 7: ReportingEffect

```python
@dataclass
class ReportingEffect:
    """
    Determines reporting consequence of eligibility/treatment.

    Separates reporting from tax treatment.
    This allows:
    - Same treatment, different reporting
    - Same reporting, different treatment
    """
    effect_id: str
    authority_id: str

    # Trigger
    eligibility_id: Optional[str] = None
    treatment_change_id: Optional[str] = None

    # Consequence
    target_surface: str                   # Where to report
    include_in_aggregate: bool = True
    requires_statement: bool = False
    statement_fields: List[str] = field(default_factory=list)
```

### Composed Qualification (for IRS compatibility)

```python
@dataclass
class ComposedQualification:
    """
    For IRS and similar regimes where eligibility/treatment/reporting are coupled.

    Composes the three primitives into legacy Qualification behavior.
    """
    qualification_id: str
    authority_id: str

    eligibility: Eligibility
    treatment_on_pass: TaxTreatmentChange
    treatment_on_fail: TaxTreatmentChange
    reporting_on_pass: ReportingEffect
    reporting_on_fail: ReportingEffect

# IRS Qualified Dividend as composed qualification
QUALIFIED_DIVIDEND_IRS = ComposedQualification(
    qualification_id="qual:qd_common",
    authority_id="auth:irs",
    eligibility=Eligibility(
        eligibility_id="elig:qd_61day",
        authority_id="auth:irs",
        description="61-day holding within 121-day window",
        temporal_constraints=[...],
        actor_constraints=[...],
        disqualifiers=[...],
    ),
    treatment_on_pass=TaxTreatmentChange(
        change_id="ttc:qd_pass",
        authority_id="auth:irs",
        eligibility_id="elig:qd_61day",
        from_treatment="ordinary_dividend",
        to_treatment="qualified_dividend",
        rate_change=None,  # Rate is implicit in classification
    ),
    treatment_on_fail=TaxTreatmentChange(
        change_id="ttc:qd_fail",
        authority_id="auth:irs",
        eligibility_id="elig:qd_61day",
        from_treatment="ordinary_dividend",
        to_treatment="ordinary_dividend",  # No change
    ),
    reporting_on_pass=ReportingEffect(
        effect_id="re:qd_pass",
        authority_id="auth:irs",
        target_surface="1099-DIV:Box1b",
        include_in_aggregate=True,  # Also in Box 1a
    ),
    reporting_on_fail=ReportingEffect(
        effect_id="re:qd_fail",
        authority_id="auth:irs",
        target_surface="1099-DIV:Box1a_other",
    ),
)
```

---

## Primitive 9: KnowledgeBoundary (Extended)

**Now includes uncertainty lifecycle.**

```python
@dataclass
class KnowledgeBoundary:
    """
    Epistemic uncertainty with lifecycle.

    Extended to model:
    - When uncertainty expires
    - What state it transitions to
    - How disputes are resolved
    """
    boundary_id: str
    authority_id: str

    # What is unknown
    unknown_fact: str
    responsible_actor: str
    knowledge_holder: Optional[str] = None

    # Required response under uncertainty
    conservative_action: str
    disclosure_required: bool = False
    disclosure_content: Optional[List[str]] = None

    # LIFECYCLE (NEW)
    expires_when: Optional[str] = None    # Condition that resolves uncertainty
    expiration_date: Optional[date] = None  # Hard deadline if any
    transitions_to: Optional[str] = None  # State after resolution

    # Resolution mechanisms
    resolution_mechanism: str             # How uncertainty resolves
    dispute_authority: Optional[str] = None  # Who resolves disputes
    retroactive_liability: bool = False   # Can resolution apply retroactively?

    # Audit trail
    audit_retention_years: Optional[int] = None

class UncertaintyState(Enum):
    """States in uncertainty lifecycle."""
    ACTIVE = "active"                     # Uncertainty is live
    RESOLVED = "resolved"                 # Fact became known
    EXPIRED = "expired"                   # Deadline passed without resolution
    DISPUTED = "disputed"                 # Under dispute
    FINAL = "final"                       # Resolution is binding

# Extended IRS example
QSBS_UNCERTAINTY_LIFECYCLE = KnowledgeBoundary(
    boundary_id="kb:section_1202",
    authority_id="auth:irs",
    unknown_fact="recipient_holding_period_for_qsbs_exclusion",
    responsible_actor="filer",
    knowledge_holder="recipient",
    conservative_action="furnish_section_1202_statement",
    disclosure_required=True,

    # LIFECYCLE
    expires_when="recipient_files_return_claiming_exclusion",
    transitions_to="resolved_by_recipient_claim",
    resolution_mechanism="recipient_claims_on_return",
    dispute_authority="auth:irs",
    retroactive_liability=False,  # Filer not liable if recipient misrepresents
    audit_retention_years=7,
)
```

---

## Primitive 15: ConditionalPrecedence (Modified)

**Non-linear precedence with cycles and context.**

```python
@dataclass
class ConditionalPrecedence:
    """
    Precedence that supports:
    - Context-dependent ordering
    - Actor-dependent ordering
    - Conditional cycles (A > B unless C, then B > A)

    Replaces simple linear precedence ladder.
    """
    precedence_id: str
    authority_id: str

    # The relationship
    higher_rule: str
    lower_rule: str

    # Conditions (if any condition is true, precedence applies)
    conditions: List['PrecedenceCondition'] = field(default_factory=list)

    # Context restrictions
    actor_context: Optional[str] = None   # Only applies to this actor type
    temporal_context: Optional[str] = None  # Only applies in time window

    # Cycle handling
    revives_if: Optional[str] = None      # Condition that revives lower_rule
    mutual_exclusion: bool = False        # If true, rules cannot both apply

@dataclass
class PrecedenceCondition:
    """Condition for precedence to apply."""
    condition_id: str
    condition_type: str                   # "fact_present", "fact_absent", "actor_is"
    value: str

    def evaluate(self, context: 'EvaluationContext') -> bool:
        pass

# Example: Treaty override with anti-abuse exception
TREATY_OVERRIDE = ConditionalPrecedence(
    precedence_id="prec:treaty_override",
    authority_id="auth:irs",
    higher_rule="treaty_rate",
    lower_rule="statutory_rate",
    conditions=[],  # Default: treaty wins
)

ANTI_ABUSE_REVIVAL = ConditionalPrecedence(
    precedence_id="prec:anti_abuse",
    authority_id="auth:irs",
    higher_rule="statutory_rate",
    lower_rule="treaty_rate",
    conditions=[
        PrecedenceCondition("cond:ppt", "fact_present", "principal_purpose_is_treaty_benefit"),
    ],
    revives_if="disclosure_safe_harbor_met",  # Treaty can revive
)
```

### Precedence Resolution Algorithm

```python
class PrecedenceResolver:
    """Resolves precedence considering conditions and cycles."""

    def __init__(self, precedences: List[ConditionalPrecedence]):
        self.precedences = precedences

    def resolve(
        self,
        rules: List[str],
        context: 'EvaluationContext'
    ) -> List[str]:
        """
        Order rules by precedence given context.

        Handles:
        - Conditional precedence
        - Revival conditions
        - Cycle detection and resolution
        """
        # Build conditional graph
        graph = self._build_graph(context)

        # Detect cycles
        cycles = self._find_cycles(graph)

        if cycles:
            # Resolve cycles via revival conditions
            graph = self._resolve_cycles(graph, cycles, context)

        # Topological sort with stability
        return self._stable_sort(graph, rules)

    def _build_graph(self, context: 'EvaluationContext') -> Dict:
        """Build precedence graph with only active edges."""
        graph = {}
        for prec in self.precedences:
            if self._conditions_met(prec, context):
                if prec.higher_rule not in graph:
                    graph[prec.higher_rule] = []
                graph[prec.higher_rule].append(prec.lower_rule)
        return graph

    def _find_cycles(self, graph: Dict) -> List[List[str]]:
        """Detect cycles in precedence graph."""
        # Tarjan's algorithm or similar
        pass

    def _resolve_cycles(
        self,
        graph: Dict,
        cycles: List[List[str]],
        context: 'EvaluationContext'
    ) -> Dict:
        """Resolve cycles via revival conditions."""
        for cycle in cycles:
            # Find revival conditions that break cycle
            for i, rule in enumerate(cycle):
                next_rule = cycle[(i + 1) % len(cycle)]
                prec = self._find_precedence(rule, next_rule)
                if prec and prec.revives_if:
                    if self._evaluate_condition(prec.revives_if, context):
                        # Revival breaks cycle
                        graph[rule].remove(next_rule)
        return graph
```

---

## Primitive 16: LifecycleEvent (NEW)

**Terminations, dissolutions, wind-ups.**

```python
@dataclass
class LifecycleEvent:
    """
    Events that change entity/item stream lifecycle.

    Not just classification - fundamentally changes what exists.
    """
    event_id: str
    authority_id: str
    event_type: LifecycleEventType

    # What is affected
    affected_entity: str
    affected_item_streams: List[str]

    # Event details
    effective_date: date
    announcement_date: Optional[date] = None

    # Consequences
    terminates_items: bool = True
    aggregation_override: Optional[str] = None  # Special aggregation rules
    final_distribution_treatment: Optional[str] = None
    successor_entity: Optional[str] = None

    # Cross-border implications
    cross_border_effects: List['CrossBorderEffect'] = field(default_factory=list)

class LifecycleEventType(Enum):
    """Types of lifecycle events."""
    LIQUIDATION = "liquidation"           # Corporate liquidation
    DISSOLUTION = "dissolution"           # Partnership dissolution
    TERMINATION = "termination"           # Trust termination
    MERGER = "merger"                     # Entity combination
    SPINOFF = "spinoff"                   # Entity separation
    CONVERSION = "conversion"             # Entity type change
    WIND_UP = "wind_up"                   # Cross-border wind-up

@dataclass
class CrossBorderEffect:
    """Effect of lifecycle event in another jurisdiction."""
    target_authority: str
    recognition_date: date                # May differ from home jurisdiction
    treatment: str
    withholding_required: bool = False
```

---

## Primitive 17: DecisionTrace (NEW)

**Explainability and audit trail.**

```python
@dataclass
class DecisionTrace:
    """
    Records how a decision was reached.

    Critical for:
    - Debugging
    - Audits
    - ML-assisted review
    - Regulatory explanation requirements
    """
    trace_id: str
    decision_type: str                    # "routing", "classification", "treatment"
    timestamp: datetime

    # Input
    input_item: str
    input_context: Dict[str, Any]

    # Rules evaluated
    rules_evaluated: List['RuleEvaluation']
    precedence_applied: List[str]

    # Outcomes
    final_decision: str
    confidence: float

    # Audit support
    authority_id: str
    human_readable_explanation: str
    machine_checkable_proof: Optional[Dict] = None

@dataclass
class RuleEvaluation:
    """Single rule evaluation within a trace."""
    rule_id: str
    rule_type: str
    evaluated_at: datetime

    # Evaluation
    conditions_checked: List[str]
    conditions_results: Dict[str, bool]
    triggered: bool

    # If triggered
    action_taken: Optional[str] = None
    exclusions_checked: List[str] = field(default_factory=list)
    uncertainty_invoked: Optional[str] = None

# Example trace
DIVIDEND_ROUTING_TRACE = DecisionTrace(
    trace_id="trace:div_001",
    decision_type="routing",
    timestamp=datetime.now(),
    input_item="item:dividend_2024_q4",
    input_context={"holding_days": 65, "issuer_type": "domestic_corp"},
    rules_evaluated=[
        RuleEvaluation(
            rule_id="elig:qd_61day",
            rule_type="eligibility",
            evaluated_at=datetime.now(),
            conditions_checked=["holding_period >= 61"],
            conditions_results={"holding_period >= 61": True},
            triggered=True,
            action_taken="elevate_to_qualified",
        ),
    ],
    precedence_applied=["prec:qualification_before_routing"],
    final_decision="route_to_box_1b",
    confidence=1.0,
    authority_id="auth:irs",
    human_readable_explanation=(
        "Dividend held for 65 days (>61 required) within 121-day window. "
        "No disqualifiers present. Classified as qualified dividend. "
        "Routed to Box 1b per IRS instructions."
    ),
)
```

---

## Atom Constraint: One Irreducible Fact

**Prevents atom inflation.**

```python
@dataclass
class AtomConstraint:
    """
    Enforces: one atom = one irreducible fact.

    An atom is irreducible if:
    1. It cannot be decomposed into smaller semantic units
    2. It expresses exactly one fact about the world
    3. It has no conditional logic (that's for rules)
    """
    pass

class AtomValidator:
    """Validates atoms against irreducibility constraint."""

    VIOLATION_PATTERNS = [
        ("condition", r"\bif\b|\bwhen\b|\bunless\b"),
        ("composite", r"\band\b|\bor\b"),
        ("action", r"\broute\b|\breclassify\b|\breport\b"),
    ]

    def validate(self, atom: 'SemanticAtom') -> List[str]:
        """Returns list of violations, empty if valid."""
        violations = []

        # Check for conditional logic
        atom_repr = str(atom.__dict__)
        for violation_type, pattern in self.VIOLATION_PATTERNS:
            if re.search(pattern, atom_repr, re.IGNORECASE):
                violations.append(
                    f"Atom {atom.atom_id} contains {violation_type} logic"
                )

        # Check for multiple facts
        fact_count = self._count_facts(atom)
        if fact_count > 1:
            violations.append(
                f"Atom {atom.atom_id} expresses {fact_count} facts (max 1)"
            )

        return violations

    def _count_facts(self, atom: 'SemanticAtom') -> int:
        """Count distinct facts in atom."""
        # Heuristic: count non-None, non-default fields
        facts = 0
        for field_name, field_value in atom.__dict__.items():
            if field_name.startswith('_'):
                continue
            if field_value is not None and field_value != []:
                facts += 1
        return max(1, facts - 2)  # Subtract id and type fields

# VALID atoms (one fact each)
VALID_ATOMS = [
    MinimumDurationAtom(atom_id="dur:61", days=61),  # One fact: 61 days
    TemporalWindowAtom(atom_id="win:121", length_days=121),  # One fact: 121 days
]

# INVALID atoms (multiple facts or conditions)
INVALID_ATOMS = [
    # BAD: contains condition
    {"atom_id": "bad:1", "days": 61, "if_held": "domestic_stock"},

    # BAD: contains action
    {"atom_id": "bad:2", "days": 61, "then_route": "box_1b"},

    # BAD: multiple facts
    {"atom_id": "bad:3", "days": 61, "window": 121, "offset": -60},
]
```

---

## ReportingSurface (Replaces ReportingSlot)

**Authority-defined, not form-centric.**

```python
@dataclass
class ReportingSurface:
    """
    Abstract target for reported values.

    Authority-defined, not form-centric.
    Works for forms, APIs, ledgers, real-time systems.
    """
    surface_id: str
    authority_id: str                     # Which authority defines this
    surface_type: SurfaceType

    # Data specification
    data_type: str                        # "amount", "text", "boolean", etc.
    schema: Optional[Dict] = None         # JSON schema if structured

    # Physical manifestation (optional - some surfaces are pure API)
    physical_bindings: Dict[str, str] = field(default_factory=dict)

class SurfaceType(Enum):
    """Types of reporting surfaces."""
    FORM_FIELD = "form_field"             # Traditional box/line
    API_ENDPOINT = "api_endpoint"         # REST/GraphQL endpoint
    LEDGER_ENTRY = "ledger_entry"         # Blockchain/DLT
    STRUCTURED_MESSAGE = "structured_message"  # XML/JSON message
    FREE_TEXT = "free_text"               # Unstructured disclosure

# IRS surface (form-based)
SURFACE_1099DIV_BOX1A = ReportingSurface(
    surface_id="surf:1099div_1a",
    authority_id="auth:irs",
    surface_type=SurfaceType.FORM_FIELD,
    data_type="amount",
    physical_bindings={
        "form": "1099-DIV",
        "field": "Box 1a",
        "label": "Total ordinary dividends",
    },
)

# OECD CRS surface (XML message)
SURFACE_CRS_PAYMENT = ReportingSurface(
    surface_id="surf:crs_payment",
    authority_id="auth:oecd_crs",
    surface_type=SurfaceType.STRUCTURED_MESSAGE,
    data_type="complex",
    schema={
        "type": "object",
        "properties": {
            "PaymentType": {"type": "string"},
            "PaymentAmount": {"type": "number"},
            "Currency": {"type": "string"},
        },
    },
)

# Real-time API surface
SURFACE_REALTIME_WHT = ReportingSurface(
    surface_id="surf:realtime_wht",
    authority_id="auth:realtime_wht",
    surface_type=SurfaceType.API_ENDPOINT,
    data_type="json",
    schema={
        "endpoint": "/api/v1/withhold",
        "method": "POST",
    },
)
```

---

## Summary: 18 Primitives for Jurisdiction-Agnostic Legal Reasoning

| # | Primitive | Layer | Purpose |
|---|-----------|-------|---------|
| 1 | **Authority** | Foundation | Jurisdiction/regime that defines rules |
| 2 | **EconomicEvent** | Pre-Item | What happened economically |
| 3 | **Item** | Core | Reportable unit derived from event |
| 4 | **Classification** | Core | Tax treatment category |
| 5 | **Eligibility** | Determination | Boolean condition evaluation |
| 6 | **TaxTreatmentChange** | Consequence | Treatment effect when eligible |
| 7 | **ReportingEffect** | Consequence | Reporting effect |
| 8 | **TemporalConstraint** | Constraint | Time-bounded conditions |
| 9 | **KnowledgeBoundary** | Uncertainty | Epistemic uncertainty with lifecycle |
| 10 | **ReportingSurface** | Target | Authority-defined reporting target |
| 11 | **Actor** | Participant | Party in chain |
| 12 | **RoleActivation** | Participant | Dynamic role assumption |
| 13 | **Routing** | Flow | Item → Surface assignment |
| 14 | **Exclusion** | Constraint | Mutual impossibility |
| 15 | **ConditionalPrecedence** | Resolution | Non-linear conflict resolution |
| 16 | **LifecycleEvent** | Lifecycle | Terminations, dissolutions |
| 17 | **DecisionTrace** | Audit | Explainability trail |
| 18 | **Derivation** | Computation | Slot arithmetic |

---

## Validation: Can This Model Express...

| Scenario | v2.0 | v3.0 | How |
|----------|------|------|-----|
| IRS 1099-DIV | Yes | Yes | ComposedQualification wraps Eligibility+Treatment+Reporting |
| OECD CRS | No | Yes | Authority defines ReportingSurface as XML message |
| Real-time WHT API | No | Yes | Authority.type=REALTIME, SurfaceType=API_ENDPOINT |
| Constructive receipt | Partial | Yes | EconomicEvent.economic_benefit_date ≠ cash_flow_date |
| Treaty with anti-abuse | No | Yes | ConditionalPrecedence with revives_if |
| Partnership dissolution | No | Yes | LifecycleEvent with cross_border_effects |
| Audit trail | No | Yes | DecisionTrace with machine_checkable_proof |

---

## Migration from v2.0

### Backward Compatibility

v2.0 constructs map to v3.0:

```python
# v2.0 Qualification → v3.0 ComposedQualification
# v2.0 ReportingSlot → v3.0 ReportingSurface with surface_type=FORM_FIELD
# v2.0 Precedence → v3.0 ConditionalPrecedence with empty conditions
# v2.0 Item → v3.0 Item (derived from implicit EconomicEvent)
```

### New Capabilities

v3.0 enables:

1. **Multi-authority systems**: Same event, different Items per authority
2. **Non-form reporting**: APIs, ledgers, real-time systems
3. **Uncertainty lifecycle**: Track from uncertainty through resolution
4. **Non-linear precedence**: Handle legal complexity
5. **Decision explainability**: Full audit trail
6. **Lifecycle modeling**: Corporate events properly handled

---

## Conclusion

v3.0 transforms the semantic core from:

> **US tax execution ontology**

to:

> **Jurisdiction-agnostic legal reasoning infrastructure**

The 18 primitives can express any tax/regulatory reporting regime that follows the pattern:

```
Authority defines rules
EconomicEvents produce Items
Items are evaluated for Eligibility
Eligibility triggers TaxTreatmentChange and ReportingEffect
ConditionalPrecedence resolves conflicts
KnowledgeBoundary handles uncertainty (with lifecycle)
ReportingSurface receives output
DecisionTrace records reasoning
LifecycleEvent handles terminations
```

This is no longer tax infrastructure. This is **regulatory reasoning infrastructure**.
