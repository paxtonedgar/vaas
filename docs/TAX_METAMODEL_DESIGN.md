# Tax Reporting Meta-Model Design

**Status:** Design Proposal (Revised)
**Version:** 2.0
**Date:** 2026-01-12
**Foundation:** [SEMANTIC_CORE.md](./SEMANTIC_CORE.md) - 12 Primitives

---

## Executive Summary

This document describes the **architectural integration** of the semantic core into the VaaS pipeline. The semantic core (12 primitives) provides the foundational types; this document covers:

1. **How primitives map to graph schema** (nodes, edges)
2. **How templates compose primitives** (avoiding semantic diffusion)
3. **How form bindings decouple semantics from IRS identifiers**
4. **How validation enforces constraints** (exclusions, derivations, boundaries)

**Key design principle:** IRS concepts (Qualified Dividend, Section 199A) are **instances** of generic primitives, not types in the schema.

---

## 1. Architecture Overview

### Layer Separation

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Semantic Core (12 Primitives)                     │
│  - Item, Classification, Qualification, TemporalConstraint  │
│  - KnowledgeBoundary, ReportingSlot, Actor, RoleActivation  │
│  - Routing, Exclusion, Derivation, Precedence               │
│  [Defined in SEMANTIC_CORE.md]                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Semantic Atoms (Composable Units)                 │
│  - temporal_window, minimum_duration, daycount_rule         │
│  - uncertainty_resolution, reroute_target, exclusion_scope  │
│  - actor_requirement, disclosure_obligation                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Form Bindings (IRS-Specific Mappings)             │
│  - SemanticBinding: semantic_id → form:physical_id          │
│  - "ordinary_dividends_total" → "1099-DIV:Box 1a"           │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Form Instantiations (Configuration)               │
│  - F1099DIV_QUALIFICATIONS, F1099DIV_ROUTINGS               │
│  - F1099INT_QUALIFICATIONS, F1099INT_ROUTINGS               │
└─────────────────────────────────────────────────────────────┘
```

### What Changed from v1.0

| v1.0 Issue | v2.0 Resolution |
|------------|-----------------|
| IRS concepts as types (`Classification.QUALIFIED_DIVIDEND`) | Qualification instances compose from primitives |
| Semantic logic split across operators/templates/precedence | Semantic Atoms layer consolidates logic |
| Epistemic uncertainty as "just another rule" | KnowledgeBoundary is first-class primitive |
| Box-centric model | ReportingSlot with `carrier_type` (box, line, attachment, statement) |
| Precedence without exclusion | MutualExclusion as explicit constraint |
| Actor types without role switching | RoleActivation for dynamic roles |
| IRS language in identifiers | SemanticBinding adapters decouple names |

---

## 2. Graph Schema Mapping

### 2.1 Node Types (Extended)

The graph stores **instances** of primitives, not the primitives themselves.

```python
# src/vaas/graph/nodes.py

# Existing node types (unchanged)
NODE_TYPE_DOC_ROOT = "doc_root"
NODE_TYPE_BOX_SECTION = "box_section"
NODE_TYPE_SECTION = "section"
NODE_TYPE_CONCEPT = "concept"
NODE_TYPE_PREAMBLE = "preamble"
NODE_TYPE_REGIME = "regime"
NODE_TYPE_PARAGRAPH = "paragraph"

# NEW: Primitive-derived node types
NODE_TYPE_QUALIFICATION = "qualification"    # Instance of Qualification primitive
NODE_TYPE_TEMPORAL_WINDOW = "temporal_window"  # Instance of TemporalConstraint
NODE_TYPE_KNOWLEDGE_BOUNDARY = "knowledge_boundary"  # Instance of KnowledgeBoundary
NODE_TYPE_EXCLUSION = "exclusion"            # Instance of Exclusion primitive
NODE_TYPE_ACTOR = "actor"                    # Instance of Actor primitive
NODE_TYPE_ROUTING = "routing"                # Instance of Routing primitive
NODE_TYPE_DERIVATION = "derivation"          # Instance of Derivation primitive

# Schema: Qualification node
QUALIFICATION_NODE_SCHEMA = {
    "node_id": str,                          # e.g., "1099div:qual_qualified_dividend_common"
    "node_type": "qualification",
    "qualification_id": str,                 # Semantic ID
    "base_classification": str,              # Starting classification (semantic)
    "target_classification": str,            # If qualified (semantic)
    "failure_classification": str,           # If not qualified (semantic)
    "governed_by": List[str],                # IRC sections
    # Composed from atoms (references, not embedded)
    "temporal_constraint_ids": List[str],
    "disqualifier_ids": List[str],
    "actor_constraint_ids": List[str],
}

# Schema: Knowledge Boundary node
KNOWLEDGE_BOUNDARY_NODE_SCHEMA = {
    "node_id": str,
    "node_type": "knowledge_boundary",
    "boundary_id": str,
    "unknown_fact": str,
    "responsible_actor": str,
    "knowledge_holder": Optional[str],
    "conservative_action": str,
    "disclosure_required": bool,
    "disclosure_content": Optional[List[str]],
    "governed_by": List[str],
}

# Schema: Exclusion node
EXCLUSION_NODE_SCHEMA = {
    "node_id": str,
    "node_type": "exclusion",
    "exclusion_id": str,
    "classifications": List[str],           # Mutually exclusive set
    "scope": str,                            # "per_item", "per_slot", "per_form"
    "governed_by": List[str],
}
```

### 2.2 Edge Types (Extended)

```python
# src/vaas/graph/edges.py

# Existing edge types (unchanged)
EDGE_TYPE_PARENT_OF = "parent_of"
EDGE_TYPE_FOLLOWS = "follows"
EDGE_TYPE_IN_SECTION = "in_section"
EDGE_TYPE_REFERENCES_BOX = "references_box"
EDGE_TYPE_SAME_GROUP = "same_group"
# ... existing semantic edges

# NEW: Primitive-derived edge types
EDGE_TYPE_QUALIFIES = "qualifies"            # qualification → classification
EDGE_TYPE_CONSTRAINED_BY = "constrained_by"  # qualification → temporal_constraint
EDGE_TYPE_DISQUALIFIED_BY = "disqualified_by"  # qualification → disqualifier
EDGE_TYPE_BOUNDED_BY = "bounded_by"          # action → knowledge_boundary
EDGE_TYPE_EXCLUDES_MUTUAL = "excludes_mutual"  # exclusion → classifications
EDGE_TYPE_ACTIVATES_ROLE = "activates_role"  # context → role_activation
EDGE_TYPE_ROUTES_TO = "routes_to"            # routing → slot
EDGE_TYPE_DERIVES_FROM = "derives_from"      # derivation → source slots
EDGE_TYPE_OVERRIDES = "overrides"            # rule → rule (precedence)
EDGE_TYPE_GOVERNED_BY = "governed_by"        # any → irc_section
```

### 2.3 Semantic Binding Nodes

Form-specific identifiers are stored as **bindings**, not embedded in semantic nodes.

```python
# Schema: Semantic Binding node
SEMANTIC_BINDING_NODE_SCHEMA = {
    "node_id": str,                          # e.g., "binding:ordinary_dividends_total"
    "node_type": "semantic_binding",
    "semantic_id": str,                      # Canonical semantic name
    "form_bindings": Dict[str, str],         # {"1099-DIV": "Box 1a", "Schedule B": "Line 5"}
}

# Edge: binding → slot
EDGE_TYPE_BINDS_TO = "binds_to"              # semantic_id → form:physical_id
```

---

## 3. Semantic Atoms: Preventing Diffusion

**Problem:** In v1.0, holding period logic appeared in WINDOW, DAYCOUNT, template conditions, classification overrides, and fallback rules.

**Solution:** Define **Semantic Atoms** as the single source of truth for each meaning-bearing unit.

### 3.1 Atom Registry

```python
# src/vaas/core/atoms.py

class AtomType(Enum):
    """The 8 canonical semantic atoms."""
    TEMPORAL_WINDOW = "temporal_window"
    MINIMUM_DURATION = "minimum_duration"
    DAYCOUNT_RULE = "daycount_rule"
    UNCERTAINTY_RESOLUTION = "uncertainty_resolution"
    REROUTE_TARGET = "reroute_target"
    EXCLUSION_SCOPE = "exclusion_scope"
    ACTOR_REQUIREMENT = "actor_requirement"
    DISCLOSURE_OBLIGATION = "disclosure_obligation"

@dataclass
class SemanticAtom:
    """Base for all atoms. Atoms are immutable once defined."""
    atom_id: str
    atom_type: AtomType
    form_agnostic: bool = True        # True if reusable across forms

# Central registry - ONE definition per semantic concept
ATOM_REGISTRY: Dict[str, SemanticAtom] = {}

def register_atom(atom: SemanticAtom) -> None:
    """Register atom. Fails if atom_id already exists."""
    if atom.atom_id in ATOM_REGISTRY:
        raise ValueError(f"Atom {atom.atom_id} already registered")
    ATOM_REGISTRY[atom.atom_id] = atom

def get_atom(atom_id: str) -> SemanticAtom:
    """Get atom by ID. Fails if not found."""
    if atom_id not in ATOM_REGISTRY:
        raise ValueError(f"Atom {atom_id} not registered")
    return ATOM_REGISTRY[atom_id]
```

### 3.2 Canonical Atom Definitions

```python
# src/vaas/core/atoms_canonical.py

# ONE definition of "61-day holding requirement" - used everywhere
ATOM_61_DAY_HOLDING = MinimumDurationAtom(
    atom_id="holding:61_day",
    days=61,
    daycount_rule="daycount:diminished_risk_excluded",
)

# ONE definition of "121-day window around ex-dividend"
ATOM_121_DAY_WINDOW = TemporalWindowAtom(
    atom_id="window:121_day_ex_div",
    length_days=121,
    offset_days=-60,
    reference="ex_dividend_date",
)

# ONE definition of "diminished risk day counting"
ATOM_DAYCOUNT_DIMINISHED_RISK = DaycountRuleAtom(
    atom_id="daycount:diminished_risk_excluded",
    excludes=["diminished_risk_days", "short_sale_days", "put_obligation_days"],
    governed_by="IRC_246(c)(4)",
)

# Register all canonical atoms
for atom in [ATOM_61_DAY_HOLDING, ATOM_121_DAY_WINDOW, ATOM_DAYCOUNT_DIMINISHED_RISK]:
    register_atom(atom)
```

### 3.3 Templates Compose Atoms (Never Embed Logic)

```python
# src/vaas/core/templates.py

@dataclass
class QualificationTemplate:
    """
    Templates reference atoms by ID - they never embed logic.

    This ensures:
    - One definition of "61-day holding" (in atom registry)
    - Templates are pure composition
    - No semantic drift between forms
    """
    template_id: str
    base_classification: str
    target_classification: str
    failure_classification: str

    # References to atoms (by ID, not embedded)
    temporal_window_atom: str         # atom_id
    minimum_duration_atom: str        # atom_id
    daycount_atom: str                # atom_id
    disqualifier_atoms: List[str]     # atom_ids
    actor_constraint_atoms: List[str] # atom_ids
    failure_routing_atom: str         # atom_id

    governed_by: List[str]

    def resolve_atoms(self) -> Dict[str, SemanticAtom]:
        """Resolve all atom references to actual atoms."""
        return {
            "window": get_atom(self.temporal_window_atom),
            "duration": get_atom(self.minimum_duration_atom),
            "daycount": get_atom(self.daycount_atom),
            "disqualifiers": [get_atom(a) for a in self.disqualifier_atoms],
            "actor_constraints": [get_atom(a) for a in self.actor_constraint_atoms],
            "failure_routing": get_atom(self.failure_routing_atom),
        }
```

---

## 4. Form Binding Adapters

**Problem:** v1.0 used IRS identifiers ("Box 1a", "Section 199A Dividend") as semantic names.

**Solution:** Canonical semantic names bind to form-specific identifiers via adapters.

### 4.1 Binding Registry

```python
# src/vaas/core/bindings.py

@dataclass
class SemanticBinding:
    """Maps canonical semantic concept to form-specific locations."""
    semantic_id: str              # Canonical name (form-agnostic)
    description: str              # Human-readable description
    form_bindings: Dict[str, str] # form_type → physical_id

# Central binding registry
BINDING_REGISTRY: Dict[str, SemanticBinding] = {
    # Income items
    "ordinary_dividends_total": SemanticBinding(
        semantic_id="ordinary_dividends_total",
        description="Total ordinary dividends from all sources",
        form_bindings={
            "1099-DIV": "Box 1a",
            "Schedule B": "Line 5",
            "K-1 (1065)": "Box 6a",
            "K-1 (1120S)": "Box 5a",
        },
    ),
    "qualified_dividends": SemanticBinding(
        semantic_id="qualified_dividends",
        description="Dividends qualifying for preferential rate",
        form_bindings={
            "1099-DIV": "Box 1b",
            "Schedule B": "Line 6",
            "K-1 (1065)": "Box 6b",
        },
    ),
    "interest_income_total": SemanticBinding(
        semantic_id="interest_income_total",
        description="Total taxable interest income",
        form_bindings={
            "1099-INT": "Box 1",
            "1099-OID": "Box 1",
            "Schedule B": "Line 1",
        },
    ),
    "capital_gain_distributions": SemanticBinding(
        semantic_id="capital_gain_distributions",
        description="Total capital gain distributions",
        form_bindings={
            "1099-DIV": "Box 2a",
            "Schedule D": "Line 13",
            "K-1 (1065)": "Box 9a",
        },
    ),
    # ... more bindings
}

def resolve_binding(semantic_id: str, form_type: str) -> Optional[str]:
    """Get form-specific physical ID for semantic concept."""
    binding = BINDING_REGISTRY.get(semantic_id)
    if binding:
        return binding.form_bindings.get(form_type)
    return None

def semantic_id_for_form_slot(form_type: str, physical_id: str) -> Optional[str]:
    """Reverse lookup: form slot → semantic ID."""
    for semantic_id, binding in BINDING_REGISTRY.items():
        if binding.form_bindings.get(form_type) == physical_id:
            return semantic_id
    return None
```

### 4.2 Usage in Graph Construction

```python
# When building graph nodes, use semantic IDs
def build_slot_node(semantic_id: str, form_type: str) -> Dict:
    """Build slot node with semantic ID, form binding as metadata."""
    binding = BINDING_REGISTRY.get(semantic_id)
    physical_id = binding.form_bindings.get(form_type) if binding else None

    return {
        "node_id": f"{form_type}:{semantic_id}",
        "node_type": "reporting_slot",
        "semantic_id": semantic_id,              # Canonical
        "form_type": form_type,
        "physical_id": physical_id,              # Form-specific (metadata)
        "carrier_type": infer_carrier_type(physical_id),  # box, line, etc.
    }
```

---

## 5. ReportingSlot with Carrier Types

**Problem:** v1.0 was box-centric. Other forms use lines, attachments, statements.

**Solution:** Abstract `carrier_type` distinguishes physical manifestations.

### 5.1 Carrier Types

```python
# src/vaas/core/reporting.py

class CarrierType(Enum):
    """Physical manifestation of a reporting slot."""
    BOX = "box"                       # 1099 series boxes
    LINE = "line"                     # Schedule/form lines (1040, Schedule D)
    ATTACHMENT = "attachment"         # Supplemental attachments
    STATEMENT = "statement"           # Freeform disclosure statements
    FIELD = "field"                   # Structured data field (TIN, dates)
    CHECKBOX = "checkbox"             # Yes/no indicators

@dataclass
class ReportingSlot:
    """
    Abstract target for reported values.
    Semantic ID is canonical; carrier_type and physical_id are form-specific.
    """
    semantic_id: str                  # Canonical semantic name
    carrier_type: CarrierType
    data_type: str                    # "amount", "text", "boolean", "tin", "date"

    # Aggregation (if this slot derives from others)
    aggregation: Optional['Derivation'] = None

    # Constraints
    exclusions: List[str] = field(default_factory=list)  # Mutually exclusive semantic_ids
```

### 5.2 Statement as Carrier

```python
# Section 1202 statement is a ReportingSlot with carrier_type=STATEMENT
SECTION_1202_STATEMENT = ReportingSlot(
    semantic_id="section_1202_qsbs_statement",
    carrier_type=CarrierType.STATEMENT,
    data_type="text",
)

# It binds to multiple forms
BINDING_REGISTRY["section_1202_qsbs_statement"] = SemanticBinding(
    semantic_id="section_1202_qsbs_statement",
    description="Statement for potential QSBS exclusion",
    form_bindings={
        "1099-DIV": "Supplemental Statement",
        "1099-B": "Supplemental Statement",
    },
)
```

---

## 6. Mutual Exclusion Constraints

**Problem:** v1.0 relied on precedence to avoid collisions but didn't assert impossibility.

**Solution:** Explicit MutualExclusion constraints that validation enforces.

### 6.1 Exclusion Definitions

```python
# src/vaas/core/exclusions.py

@dataclass
class MutualExclusion:
    """
    Asserts that classifications cannot coexist within scope.

    Validation fails if both classifications appear on same item/slot/form.
    """
    exclusion_id: str
    classifications: FrozenSet[str]  # Semantic IDs (immutable set)
    scope: str                        # "per_item", "per_slot", "per_form"
    governed_by: List[str]

    def validate(self, item_classifications: Set[str]) -> bool:
        """Returns True if constraint satisfied (at most one match)."""
        matches = self.classifications.intersection(item_classifications)
        return len(matches) <= 1

# Canonical exclusions
EXCLUSIONS = [
    MutualExclusion(
        exclusion_id="dividend_interest_mutual",
        classifications=frozenset({"dividend", "interest"}),
        scope="per_item",
        governed_by=["IRC_61"],
    ),
    MutualExclusion(
        exclusion_id="capgain_ordinary_mutual",
        classifications=frozenset({"capital_gain_distribution", "ordinary_dividend"}),
        scope="per_item",
        governed_by=["IRC_852(b)(3)"],
    ),
    MutualExclusion(
        exclusion_id="foreign_tax_credit_deduction",
        classifications=frozenset({"foreign_tax_credit", "foreign_tax_deduction"}),
        scope="per_form",  # Taxpayer elects one per return
        governed_by=["IRC_901", "IRC_164"],
    ),
    MutualExclusion(
        exclusion_id="short_long_term_gain",
        classifications=frozenset({"short_term_capital_gain", "long_term_capital_gain"}),
        scope="per_item",
        governed_by=["IRC_1222"],
    ),
]
```

### 6.2 Validation Integration

```python
# src/vaas/evaluation/validation.py

class ExclusionValidator:
    """Validates mutual exclusion constraints against graph."""

    def __init__(self, exclusions: List[MutualExclusion]):
        self.exclusions = exclusions

    def validate_graph(self, nodes_df: pd.DataFrame) -> List[str]:
        """Check all exclusions against graph nodes."""
        violations = []

        # Group items by scope
        for exclusion in self.exclusions:
            if exclusion.scope == "per_item":
                violations.extend(self._validate_per_item(exclusion, nodes_df))
            elif exclusion.scope == "per_slot":
                violations.extend(self._validate_per_slot(exclusion, nodes_df))
            elif exclusion.scope == "per_form":
                violations.extend(self._validate_per_form(exclusion, nodes_df))

        return violations

    def _validate_per_item(
        self, exclusion: MutualExclusion, nodes_df: pd.DataFrame
    ) -> List[str]:
        """Check exclusion at item level."""
        violations = []
        # Implementation: group by item_id, check classifications
        return violations
```

---

## 7. Role Activation Semantics

**Problem:** v1.0 modeled actor types but not intentional role switching.

**Solution:** RoleActivation captures dynamic role assumption.

### 7.1 Role Activation Schema

```python
# src/vaas/core/actors.py

class Role(Enum):
    """Roles actors can dynamically assume."""
    PAYER = "payer"
    FILER = "filer"
    RECIPIENT = "recipient"
    NOMINEE = "nominee"
    MIDDLEMAN = "middleman"
    TRUSTEE = "trustee"
    BENEFICIAL_OWNER = "beneficial_owner"

@dataclass
class RoleActivation:
    """
    Dynamic role assumption by an actor.

    When context_condition is met, actor assumes role and inherits obligations.
    """
    activation_id: str
    actor_type: str                   # Which actor type can activate
    assumed_role: Role
    context_condition: str            # When activation occurs

    obligations_assumed: List[str]    # Obligations that transfer with role
    displaces_actor: Optional[str]    # Does this supersede another actor?

    governed_by: List[str]

# Canonical role activations
ROLE_ACTIVATIONS = [
    RoleActivation(
        activation_id="broker_as_nominee",
        actor_type="broker",
        assumed_role=Role.NOMINEE,
        context_condition="broker_holds_securities_in_street_name",
        obligations_assumed=[
            "furnish_1099_to_beneficial_owner",
            "aggregate_payments_by_cusip",
        ],
        displaces_actor=None,  # Payer still files to broker
        governed_by=["Treas_Reg_1.6045-1"],
    ),
    RoleActivation(
        activation_id="middleman_as_filer",
        actor_type="middleman",
        assumed_role=Role.FILER,
        context_condition="middleman_receives_payment_on_behalf_of_payee",
        obligations_assumed=[
            "file_1099_to_payee",
            "backup_withhold_if_required",
        ],
        displaces_actor="original_payer",
        governed_by=["IRC_6041", "IRC_3406"],
    ),
    RoleActivation(
        activation_id="trustee_whfit_filer",
        actor_type="trustee",
        assumed_role=Role.FILER,
        context_condition="trust_is_widely_held_fixed_investment_trust",
        obligations_assumed=[
            "provide_tax_information_statement_to_tih",
            "report_directly_to_tih",
        ],
        displaces_actor="issuer",
        governed_by=["Treas_Reg_1.671-5"],
    ),
]
```

---

## 8. Precedence Model (Extended)

### 8.1 Complete Precedence Hierarchy

```python
# src/vaas/core/precedence.py

class PrecedenceClass(Enum):
    """
    Complete semantic precedence tiers.
    Lower value = higher priority.
    """
    # Tier 1: Access control (who can act)
    GATING = (1, "Who can complete this slot/form")

    # Tier 2: Negative constraints (what cannot happen)
    PROHIBITION = (2, "Must NOT include/report")
    EXCLUSION = (3, "Cannot coexist with")  # NEW: Mutual exclusivity

    # Tier 3: Exception handling (overrides)
    EXPLICIT_OVERRIDE = (4, "Named override relationship")
    FALLBACK = (5, "When primary rule cannot apply")

    # Tier 4: Classification changes
    RECLASSIFICATION = (6, "Changes item classification")
    QUALIFICATION = (7, "Conditional classification elevation")

    # Tier 5: Routing decisions
    FORM_ROUTING = (8, "Which form receives item")
    SLOT_ROUTING = (9, "Which slot on form")

    # Tier 6: Computation
    AGGREGATION = (10, "Slot arithmetic/inclusion")
    DERIVATION = (11, "Computed values")

    # Tier 7: Role dynamics  # NEW
    ROLE_ACTIVATION = (12, "Dynamic role assumption")
    OBLIGATION_TRANSFER = (13, "Obligations moving with role")

# Precedence computation unchanged from v1.0
PRECEDENCE_SCALE = 1_000_000_000

def compute_precedence(
    precedence_class: PrecedenceClass,
    reading_order: int,
    sentence_idx: int
) -> int:
    """Compute precedence value. Lower = higher priority."""
    base = precedence_class.value[0]
    return base * PRECEDENCE_SCALE + reading_order * 1000 + sentence_idx
```

---

## 9. Form Instantiation Pattern

### 9.1 1099-DIV as Configuration

```python
# src/vaas/forms/f1099div.py

"""
1099-DIV Form Definition

This file contains ONLY:
- Template instantiations (referencing canonical atoms)
- Derivation definitions
- Form-specific exclusions

NO embedded logic - all semantics come from atoms.
"""

from vaas.core.atoms import get_atom
from vaas.core.templates import QualificationTemplate
from vaas.core.derivations import Derivation, DerivationOperator
from vaas.core.exclusions import MutualExclusion

# Qualification instances (reference atoms, don't embed logic)
F1099DIV_QUALIFICATIONS = [
    QualificationTemplate(
        template_id="1099div:qualified_dividend_common",
        base_classification="ordinary_dividend",
        target_classification="qualified_dividend",
        failure_classification="ordinary_dividend",
        temporal_window_atom="window:121_day_ex_div",        # Reference
        minimum_duration_atom="holding:61_day",              # Reference
        daycount_atom="daycount:diminished_risk_excluded",   # Reference
        disqualifier_atoms=["disq:hedge_position", "disq:short_sale"],
        actor_constraint_atoms=["actor:issuer_not_pfic"],
        failure_routing_atom="routing:to_ordinary_dividends",
        governed_by=["IRC_1(h)(11)", "IRC_246(c)"],
    ),
    QualificationTemplate(
        template_id="1099div:qualified_dividend_preferred",
        base_classification="ordinary_dividend",
        target_classification="qualified_dividend",
        failure_classification="ordinary_dividend",
        temporal_window_atom="window:181_day_ex_div",        # Different window
        minimum_duration_atom="holding:91_day",              # Different duration
        daycount_atom="daycount:diminished_risk_excluded",   # Same daycount
        disqualifier_atoms=["disq:hedge_position", "disq:short_sale"],
        actor_constraint_atoms=["actor:issuer_not_pfic"],
        failure_routing_atom="routing:to_ordinary_dividends",
        governed_by=["IRC_1(h)(11)", "IRC_246(c)"],
    ),
]

# Derivations (slot arithmetic)
F1099DIV_DERIVATIONS = [
    Derivation(
        derivation_id="1099div:box_1a_sum",
        target_slot="ordinary_dividends_total",
        operator=DerivationOperator.SUM,
        source_slots=[
            "qualified_dividends",
            "exempt_interest_dividends",
            "section_199a_dividends",
            "ordinary_dividends_other",
        ],
    ),
    Derivation(
        derivation_id="1099div:box_1b_subset",
        target_slot="ordinary_dividends_total",
        operator=DerivationOperator.SUBSET,
        source_slots=["qualified_dividends"],
    ),
    Derivation(
        derivation_id="1099div:box_2f_subset",
        target_slot="capital_gain_distributions",
        operator=DerivationOperator.SUBSET,
        source_slots=["section_897_capital_gain"],
    ),
]

# Knowledge boundaries for 1099-DIV
F1099DIV_KNOWLEDGE_BOUNDARIES = [
    # Section 1202 uncertainty
    {
        "boundary_id": "1099div:section_1202_recipient_holding",
        "unknown_fact": "recipient_holding_period_for_qsbs_exclusion",
        "responsible_actor": "filer",
        "conservative_action": "furnish_section_1202_statement",
        "governed_by": ["IRC_1202"],
    },
    # Classification unknown at deadline
    {
        "boundary_id": "1099div:classification_deadline",
        "unknown_fact": "dividend_qualified_status",
        "responsible_actor": "filer",
        "conservative_action": "report_as_ordinary_dividend",
        "governed_by": ["IRC_6042"],
    },
]
```

### 9.2 1099-INT Reuses Same Patterns

```python
# src/vaas/forms/f1099int.py

"""
1099-INT Form Definition

Demonstrates reuse: same templates, different parameters.
"""

F1099INT_QUALIFICATIONS = [
    # Interest doesn't have holding period qualification
    # but has other conditional treatments
]

F1099INT_DERIVATIONS = [
    Derivation(
        derivation_id="1099int:box_1_sum",
        target_slot="interest_income_total",
        operator=DerivationOperator.SUM,
        source_slots=[
            "interest_on_us_savings_bonds",
            "interest_on_treasury_obligations",
            "other_interest_income",
        ],
    ),
]

F1099INT_KNOWLEDGE_BOUNDARIES = [
    # OID uncertainty
    {
        "boundary_id": "1099int:oid_calculation",
        "unknown_fact": "correct_oid_amount_for_complex_instrument",
        "responsible_actor": "filer",
        "conservative_action": "use_constant_yield_method",
        "governed_by": ["IRC_1272"],
    },
]
```

---

## 10. Updated File Structure

```
src/vaas/
├── core/                             # NEW: Semantic core implementation
│   ├── __init__.py
│   ├── primitives.py                 # 12 primitive definitions
│   ├── atoms.py                      # Semantic atom base + registry
│   ├── atoms_canonical.py            # Canonical atom definitions
│   ├── templates.py                  # Template composition
│   ├── bindings.py                   # Form binding adapters
│   ├── exclusions.py                 # Mutual exclusion constraints
│   ├── precedence.py                 # Precedence model
│   └── validation.py                 # Core validation logic
├── forms/                            # Form instantiations (configuration)
│   ├── __init__.py
│   ├── f1099div.py
│   ├── f1099int.py
│   ├── f1099b.py
│   └── schedule_k1.py
├── graph/
│   ├── nodes.py                      # Extended with primitive-derived types
│   ├── edges.py                      # Extended with new edge types
│   └── builders.py                   # Graph construction from forms
├── extraction/                       # Unchanged
│   └── ...
└── evaluation/
    ├── quality_checks.py
    ├── ontology_coverage.py
    └── semantic_validation.py        # NEW: Validates against core
```

---

## 11. Success Metrics (Revised)

| Metric | v1.0 Target | v2.0 Target | Measurement |
|--------|-------------|-------------|-------------|
| Ontology coverage | >50% | >70% | Predicates implemented / ontology predicates |
| Code reuse | >70% | >85% | Shared code between 1099-DIV and 1099-INT |
| Template count | 5 | 8-10 | Distinct templates in registry |
| Atom reuse | N/A | >60% | Atoms used by 2+ forms / total atoms |
| Exclusion coverage | N/A | 100% | Documented exclusions / actual exclusions |
| Binding coverage | N/A | 100% | Semantic IDs with bindings / total IDs |

---

## 12. Migration Path from v1.0

### Phase 1: Core Foundation (Week 1)

1. Implement `core/primitives.py` with 12 primitive dataclasses
2. Implement `core/atoms.py` registry
3. Define canonical atoms for 1099-DIV
4. No changes to existing pipeline

### Phase 2: Binding Layer (Week 2)

1. Implement `core/bindings.py` registry
2. Map existing box_registry → semantic bindings
3. Update graph builders to use semantic IDs
4. Physical IDs become metadata

### Phase 3: Exclusion + Validation (Week 3)

1. Implement `core/exclusions.py`
2. Add ExclusionValidator to quality checks
3. Define canonical exclusions
4. Validation fails on exclusion violations

### Phase 4: Template Refactor (Week 4)

1. Refactor existing templates to reference atoms
2. Move embedded logic to atom registry
3. Verify no semantic drift between templates

### Phase 5: Portability Test (Week 5)

1. Define 1099-INT using same infrastructure
2. Measure actual code reuse
3. Identify any 1099-DIV-specific leakage
4. Iterate on atom/binding design

---

## Appendix: Primitive Quick Reference

| # | Primitive | Purpose | Graph Representation |
|---|-----------|---------|---------------------|
| 1 | Item | Base reportable unit | Not a node (ephemeral) |
| 2 | Classification | Tax treatment category | Edge attribute |
| 3 | Qualification | Conditional elevation | `qualification` node |
| 4 | TemporalConstraint | Time-bounded condition | `temporal_window` node |
| 5 | KnowledgeBoundary | Epistemic uncertainty | `knowledge_boundary` node |
| 6 | ReportingSlot | Where items land | `reporting_slot` node |
| 7 | Actor | Party in chain | `actor` node |
| 8 | RoleActivation | Dynamic role assumption | `role_activation` node |
| 9 | Routing | Item → Slot assignment | `routes_to` edge |
| 10 | Exclusion | Mutual impossibility | `exclusion` node |
| 11 | Derivation | Slot arithmetic | `derivation` node |
| 12 | Precedence | Conflict resolution | Edge attribute (`precedence`) |

See [SEMANTIC_CORE.md](./SEMANTIC_CORE.md) for complete primitive definitions.
