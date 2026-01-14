"""
Form Binding Adapters: Decoupling Semantics from IRS Identifiers.

Problem: Early designs used IRS identifiers ("Box 1a", "Section 199A Dividend")
as semantic names, creating tight coupling to form-specific terminology.

Solution: Canonical semantic names bind to form-specific identifiers via adapters.
This allows:
- Same semantic concept mapped to different physical locations across forms
- Reverse lookup: form slot → semantic ID
- Form-agnostic rule definitions
- Conditional bindings based on context
- Version-aware bindings for form revisions

From TAX_METAMODEL_DESIGN.md and SEMANTIC_CORE_V3.md.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Any


# =============================================================================
# CARRIER TYPES
# =============================================================================

class CarrierType(Enum):
    """Physical manifestation of a reporting slot."""
    BOX = "box"                       # 1099 series boxes
    LINE = "line"                     # Schedule/form lines (1040, Schedule D)
    ATTACHMENT = "attachment"         # Supplemental attachments
    STATEMENT = "statement"           # Freeform disclosure statements
    FIELD = "field"                   # Structured data field (TIN, dates)
    CHECKBOX = "checkbox"             # Yes/no indicators


# =============================================================================
# BINDING RESULT (Conditional/Versioned Resolution)
# =============================================================================

@dataclass
class BindingCondition:
    """
    Condition that must be met for a binding to apply.

    Examples:
    - filer_is_ric=True → different box for RIC filers
    - amount > 10 → threshold-based routing
    - tax_year >= 2024 → version-specific binding
    """
    condition_id: str
    condition_type: str              # "flag", "threshold", "version", "actor_type"
    field: str                       # Field to check in context
    operator: str                    # "eq", "gt", "gte", "lt", "lte", "in", "not_in"
    value: Any                       # Expected value

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        actual = context.get(self.field)
        if actual is None:
            return False

        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "gt":
            return actual > self.value
        elif self.operator == "gte":
            return actual >= self.value
        elif self.operator == "lt":
            return actual < self.value
        elif self.operator == "lte":
            return actual <= self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "not_in":
            return actual not in self.value
        return False


@dataclass
class BindingResult:
    """
    Result of a conditional binding resolution.

    Contains everything needed to emit to a surface, including:
    - Physical location
    - Carrier type
    - Conditions that matched
    - Version information
    """
    semantic_id: str
    surface_id: str                  # Full surface identifier
    carrier_type: CarrierType
    slot_id: str                     # Physical slot (e.g., "Box 1a", "Line 5")

    # Resolution context
    authority_id: str
    form_type: str
    form_version: Optional[str] = None
    effective_date: Optional[date] = None
    expiration_date: Optional[date] = None

    # Conditions that matched
    conditions_matched: List[str] = field(default_factory=list)

    # Priority for conflict resolution
    priority: int = 0
    is_primary: bool = True          # False if secondary/overflow binding

    # Additional constraints
    requires_statement: bool = False
    requires_attachment: bool = False
    max_occurrences: Optional[int] = None


@dataclass
class ConditionalBinding:
    """
    A binding with conditions and version constraints.

    This replaces the simple form_bindings dict with a richer model
    that supports conditional resolution.
    """
    binding_id: str
    semantic_id: str
    authority_id: str
    surface_id: str
    carrier_type: CarrierType
    slot_id: str

    # Version constraints
    effective_date: Optional[date] = None
    expiration_date: Optional[date] = None
    form_version: Optional[str] = None

    # Conditions (all must match)
    conditions: List[BindingCondition] = field(default_factory=list)

    # Priority and flags
    priority: int = 0
    is_primary: bool = True
    requires_statement: bool = False

    def matches(self, context: Dict[str, Any], asof_date: Optional[date] = None) -> bool:
        """Check if binding matches context and date."""
        # Check date constraints
        if asof_date:
            if self.effective_date and asof_date < self.effective_date:
                return False
            if self.expiration_date and asof_date > self.expiration_date:
                return False

        # Check all conditions
        return all(c.evaluate(context) for c in self.conditions)

    def to_result(self, context: Dict[str, Any]) -> BindingResult:
        """Convert to BindingResult."""
        return BindingResult(
            semantic_id=self.semantic_id,
            surface_id=self.surface_id,
            carrier_type=self.carrier_type,
            slot_id=self.slot_id,
            authority_id=self.authority_id,
            form_type=self.surface_id.split(":")[0] if ":" in self.surface_id else self.surface_id,
            form_version=self.form_version,
            effective_date=self.effective_date,
            expiration_date=self.expiration_date,
            conditions_matched=[c.condition_id for c in self.conditions],
            priority=self.priority,
            is_primary=self.is_primary,
            requires_statement=self.requires_statement,
        )


# =============================================================================
# SEMANTIC BINDING (Simplified for common cases)
# =============================================================================

@dataclass
class SemanticBinding:
    """
    Maps canonical semantic concept to form-specific locations.

    Attributes:
        semantic_id: Canonical name (form-agnostic)
        description: Human-readable description
        form_bindings: form_type → physical_id mapping
        carrier_type: Physical manifestation type
        data_type: Type of data ("amount", "text", "boolean", etc.)
    """
    semantic_id: str
    description: str
    form_bindings: Dict[str, str] = field(default_factory=dict)
    carrier_type: CarrierType = CarrierType.BOX
    data_type: str = "amount"

    def get_physical_id(self, form_type: str) -> Optional[str]:
        """Get form-specific physical ID."""
        return self.form_bindings.get(form_type)

    def supported_forms(self) -> Set[str]:
        """Get set of form types with bindings."""
        return set(self.form_bindings.keys())


# =============================================================================
# BINDING REGISTRY
# =============================================================================

class BindingRegistry:
    """
    Central registry for semantic bindings.

    Provides:
    - Forward lookup: semantic_id → form binding
    - Reverse lookup: (form_type, physical_id) → semantic_id
    - Conditional resolution with context and date
    """

    def __init__(self):
        self._bindings: Dict[str, SemanticBinding] = {}
        self._conditional_bindings: Dict[str, List[ConditionalBinding]] = {}
        self._reverse_index: Dict[str, str] = {}  # "form:physical" → semantic_id

    def register(self, binding: SemanticBinding) -> None:
        """Register a simple binding."""
        self._bindings[binding.semantic_id] = binding

        # Build reverse index
        for form_type, physical_id in binding.form_bindings.items():
            key = f"{form_type}:{physical_id}"
            self._reverse_index[key] = binding.semantic_id

    def register_conditional(self, binding: ConditionalBinding) -> None:
        """Register a conditional binding."""
        if binding.semantic_id not in self._conditional_bindings:
            self._conditional_bindings[binding.semantic_id] = []
        self._conditional_bindings[binding.semantic_id].append(binding)

        # Build reverse index
        key = f"{binding.surface_id}:{binding.slot_id}"
        self._reverse_index[key] = binding.semantic_id

    def get(self, semantic_id: str) -> Optional[SemanticBinding]:
        """Get simple binding by semantic ID."""
        return self._bindings.get(semantic_id)

    def resolve(self, semantic_id: str, form_type: str) -> Optional[str]:
        """
        Get form-specific physical ID for semantic concept.

        Simple resolution - use resolve_conditional for context-aware resolution.
        """
        binding = self._bindings.get(semantic_id)
        if binding:
            return binding.get_physical_id(form_type)
        return None

    def resolve_conditional(
        self,
        semantic_id: str,
        surface_id: str,
        authority_id: str,
        asof_date: Optional[date] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[BindingResult]:
        """
        Resolve binding with full context awareness.

        Args:
            semantic_id: Semantic concept to resolve
            surface_id: Target surface (e.g., "1099-DIV")
            authority_id: Authority context (e.g., "auth:irs")
            asof_date: Date for version filtering
            context: Dict with evaluation context (flags, values, etc.)

        Returns:
            List of matching BindingResults, sorted by priority (highest first)
        """
        context = context or {}
        results: List[BindingResult] = []

        # Check conditional bindings first
        if semantic_id in self._conditional_bindings:
            for cb in self._conditional_bindings[semantic_id]:
                if cb.authority_id != authority_id:
                    continue
                if not cb.surface_id.startswith(surface_id):
                    continue
                if cb.matches(context, asof_date):
                    results.append(cb.to_result(context))

        # Fall back to simple binding if no conditional matches
        if not results and semantic_id in self._bindings:
            binding = self._bindings[semantic_id]
            # Extract form_type from surface_id
            form_type = surface_id.split(":")[0] if ":" in surface_id else surface_id
            slot_id = binding.get_physical_id(form_type)
            if slot_id:
                results.append(BindingResult(
                    semantic_id=semantic_id,
                    surface_id=surface_id,
                    carrier_type=binding.carrier_type,
                    slot_id=slot_id,
                    authority_id=authority_id,
                    form_type=form_type,
                    priority=0,
                    is_primary=True,
                ))

        # Sort by priority (highest first)
        results.sort(key=lambda r: -r.priority)
        return results

    def reverse_lookup(self, form_type: str, physical_id: str) -> Optional[str]:
        """Get semantic ID from form-specific identifier."""
        key = f"{form_type}:{physical_id}"
        return self._reverse_index.get(key)

    def all_ids(self) -> Set[str]:
        """Get all semantic IDs."""
        simple_ids = set(self._bindings.keys())
        conditional_ids = set(self._conditional_bindings.keys())
        return simple_ids | conditional_ids

    def for_form(self, form_type: str) -> List[SemanticBinding]:
        """Get all simple bindings that have mappings for a specific form."""
        return [
            b for b in self._bindings.values()
            if form_type in b.form_bindings
        ]

    def conditional_for_surface(
        self,
        surface_id: str,
        authority_id: str,
    ) -> List[ConditionalBinding]:
        """Get all conditional bindings for a surface."""
        results = []
        for bindings in self._conditional_bindings.values():
            for cb in bindings:
                if cb.authority_id == authority_id and cb.surface_id.startswith(surface_id):
                    results.append(cb)
        return results

    def clear(self) -> None:
        """Clear all bindings. Use only in tests."""
        self._bindings.clear()
        self._conditional_bindings.clear()
        self._reverse_index.clear()


# Global registry instance
BINDING_REGISTRY = BindingRegistry()


def resolve_binding(semantic_id: str, form_type: str) -> Optional[str]:
    """Get form-specific physical ID for semantic concept."""
    return BINDING_REGISTRY.resolve(semantic_id, form_type)


def semantic_id_for_form_slot(form_type: str, physical_id: str) -> Optional[str]:
    """Reverse lookup: form slot → semantic ID."""
    return BINDING_REGISTRY.reverse_lookup(form_type, physical_id)


# =============================================================================
# CANONICAL BINDINGS
# =============================================================================

# Income Items
BINDING_ORDINARY_DIVIDENDS_TOTAL = SemanticBinding(
    semantic_id="ordinary_dividends_total",
    description="Total ordinary dividends from all sources",
    form_bindings={
        "1099-DIV": "Box 1a",
        "Schedule B": "Line 5",
        "K-1 (1065)": "Box 6a",
        "K-1 (1120S)": "Box 5a",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_QUALIFIED_DIVIDENDS = SemanticBinding(
    semantic_id="qualified_dividends",
    description="Dividends qualifying for preferential rate",
    form_bindings={
        "1099-DIV": "Box 1b",
        "Schedule B": "Line 6",
        "K-1 (1065)": "Box 6b",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_CAPITAL_GAIN_DISTRIBUTIONS = SemanticBinding(
    semantic_id="capital_gain_distributions",
    description="Total capital gain distributions",
    form_bindings={
        "1099-DIV": "Box 2a",
        "Schedule D": "Line 13",
        "K-1 (1065)": "Box 9a",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_UNRECAPTURED_1250_GAIN = SemanticBinding(
    semantic_id="unrecaptured_section_1250_gain",
    description="Unrecaptured Section 1250 gain",
    form_bindings={
        "1099-DIV": "Box 2b",
        "Schedule D": "Line 19",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_SECTION_1202_GAIN = SemanticBinding(
    semantic_id="section_1202_gain",
    description="Section 1202 gain",
    form_bindings={
        "1099-DIV": "Box 2c",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_COLLECTIBLES_GAIN = SemanticBinding(
    semantic_id="collectibles_28_percent_gain",
    description="Collectibles (28%) gain",
    form_bindings={
        "1099-DIV": "Box 2d",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_SECTION_897_ORDINARY = SemanticBinding(
    semantic_id="section_897_ordinary_dividends",
    description="Section 897 ordinary dividends (FIRPTA)",
    form_bindings={
        "1099-DIV": "Box 2e",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_SECTION_897_CAPITAL_GAIN = SemanticBinding(
    semantic_id="section_897_capital_gain",
    description="Section 897 capital gain (FIRPTA)",
    form_bindings={
        "1099-DIV": "Box 2f",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_NONDIVIDEND_DISTRIBUTIONS = SemanticBinding(
    semantic_id="nondividend_distributions",
    description="Nondividend distributions (return of capital)",
    form_bindings={
        "1099-DIV": "Box 3",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_FEDERAL_TAX_WITHHELD = SemanticBinding(
    semantic_id="federal_income_tax_withheld",
    description="Federal income tax withheld",
    form_bindings={
        "1099-DIV": "Box 4",
        "1099-INT": "Box 4",
        "1099-B": "Box 4",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_SECTION_199A_DIVIDENDS = SemanticBinding(
    semantic_id="section_199a_dividends",
    description="Section 199A dividends (qualified REIT dividends)",
    form_bindings={
        "1099-DIV": "Box 5",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_INVESTMENT_EXPENSES = SemanticBinding(
    semantic_id="investment_expenses",
    description="Investment expenses",
    form_bindings={
        "1099-DIV": "Box 6",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_FOREIGN_TAX_PAID = SemanticBinding(
    semantic_id="foreign_tax_paid",
    description="Foreign tax paid",
    form_bindings={
        "1099-DIV": "Box 7",
        "1099-INT": "Box 6",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_FOREIGN_COUNTRY = SemanticBinding(
    semantic_id="foreign_country_or_possession",
    description="Foreign country or U.S. possession",
    form_bindings={
        "1099-DIV": "Box 8",
        "1099-INT": "Box 7",
    },
    carrier_type=CarrierType.BOX,
    data_type="text",
)

BINDING_CASH_LIQUIDATION = SemanticBinding(
    semantic_id="cash_liquidation_distributions",
    description="Cash liquidation distributions",
    form_bindings={
        "1099-DIV": "Box 9",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_NONCASH_LIQUIDATION = SemanticBinding(
    semantic_id="noncash_liquidation_distributions",
    description="Noncash liquidation distributions",
    form_bindings={
        "1099-DIV": "Box 10",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_EXEMPT_INTEREST_DIVIDENDS = SemanticBinding(
    semantic_id="exempt_interest_dividends",
    description="Exempt-interest dividends",
    form_bindings={
        "1099-DIV": "Box 12",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_SPECIFIED_PRIVATE_BOND = SemanticBinding(
    semantic_id="specified_private_activity_bond_interest",
    description="Specified private activity bond interest dividends (AMT)",
    form_bindings={
        "1099-DIV": "Box 13",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

# Interest income bindings
BINDING_INTEREST_INCOME_TOTAL = SemanticBinding(
    semantic_id="interest_income_total",
    description="Total taxable interest income",
    form_bindings={
        "1099-INT": "Box 1",
        "1099-OID": "Box 1",
        "Schedule B": "Line 1",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_EARLY_WITHDRAWAL_PENALTY = SemanticBinding(
    semantic_id="early_withdrawal_penalty",
    description="Early withdrawal penalty",
    form_bindings={
        "1099-INT": "Box 2",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_US_SAVINGS_BOND_INTEREST = SemanticBinding(
    semantic_id="us_savings_bond_interest",
    description="Interest on U.S. Savings Bonds and Treasury obligations",
    form_bindings={
        "1099-INT": "Box 3",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_TAX_EXEMPT_INTEREST = SemanticBinding(
    semantic_id="tax_exempt_interest",
    description="Tax-exempt interest",
    form_bindings={
        "1099-INT": "Box 8",
    },
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

# Statement bindings
BINDING_SECTION_1202_STATEMENT = SemanticBinding(
    semantic_id="section_1202_qsbs_statement",
    description="Statement for potential QSBS exclusion",
    form_bindings={
        "1099-DIV": "Supplemental Statement",
        "1099-B": "Supplemental Statement",
    },
    carrier_type=CarrierType.STATEMENT,
    data_type="text",
)


def register_canonical_bindings() -> None:
    """Register all canonical bindings."""
    canonical_bindings = [
        # 1099-DIV
        BINDING_ORDINARY_DIVIDENDS_TOTAL,
        BINDING_QUALIFIED_DIVIDENDS,
        BINDING_CAPITAL_GAIN_DISTRIBUTIONS,
        BINDING_UNRECAPTURED_1250_GAIN,
        BINDING_SECTION_1202_GAIN,
        BINDING_COLLECTIBLES_GAIN,
        BINDING_SECTION_897_ORDINARY,
        BINDING_SECTION_897_CAPITAL_GAIN,
        BINDING_NONDIVIDEND_DISTRIBUTIONS,
        BINDING_FEDERAL_TAX_WITHHELD,
        BINDING_SECTION_199A_DIVIDENDS,
        BINDING_INVESTMENT_EXPENSES,
        BINDING_FOREIGN_TAX_PAID,
        BINDING_FOREIGN_COUNTRY,
        BINDING_CASH_LIQUIDATION,
        BINDING_NONCASH_LIQUIDATION,
        BINDING_EXEMPT_INTEREST_DIVIDENDS,
        BINDING_SPECIFIED_PRIVATE_BOND,
        # 1099-INT
        BINDING_INTEREST_INCOME_TOTAL,
        BINDING_EARLY_WITHDRAWAL_PENALTY,
        BINDING_US_SAVINGS_BOND_INTEREST,
        BINDING_TAX_EXEMPT_INTEREST,
        # Statements
        BINDING_SECTION_1202_STATEMENT,
    ]

    for binding in canonical_bindings:
        if binding.semantic_id not in BINDING_REGISTRY.all_ids():
            BINDING_REGISTRY.register(binding)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def infer_carrier_type(physical_id: Optional[str]) -> CarrierType:
    """
    Infer carrier type from physical identifier.

    Args:
        physical_id: Physical identifier like "Box 1a", "Line 5", etc.

    Returns:
        Inferred CarrierType.
    """
    if not physical_id:
        return CarrierType.FIELD

    pid_lower = physical_id.lower()

    if pid_lower.startswith("box"):
        return CarrierType.BOX
    elif pid_lower.startswith("line"):
        return CarrierType.LINE
    elif "statement" in pid_lower:
        return CarrierType.STATEMENT
    elif "attachment" in pid_lower:
        return CarrierType.ATTACHMENT
    elif "checkbox" in pid_lower or "check" in pid_lower:
        return CarrierType.CHECKBOX
    else:
        return CarrierType.FIELD


def build_slot_node(semantic_id: str, form_type: str) -> Dict:
    """
    Build slot node with semantic ID, form binding as metadata.

    Args:
        semantic_id: Canonical semantic name.
        form_type: Form type for binding resolution.

    Returns:
        Node dictionary with semantic_id and physical_id.
    """
    binding = BINDING_REGISTRY.get(semantic_id)
    physical_id = binding.get_physical_id(form_type) if binding else None

    return {
        "node_id": f"{form_type}:{semantic_id}",
        "node_type": "reporting_slot",
        "semantic_id": semantic_id,
        "form_type": form_type,
        "physical_id": physical_id,
        "carrier_type": infer_carrier_type(physical_id).value,
    }
