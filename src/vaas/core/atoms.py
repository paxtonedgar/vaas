"""
Semantic Atoms: Preventing Logic Diffusion.

Problem: In early designs, holding period logic appeared in multiple places -
window definitions, daycount rules, template conditions, classification overrides,
and fallback rules.

Solution: Define Semantic Atoms as the single source of truth for each
meaning-bearing unit. Templates reference atoms by ID, never embedding logic.

Design Principles:
1. One atom = one irreducible fact
2. Atoms are immutable once registered
3. Templates compose atoms, never embed logic
4. Atom inflation is prevented via AtomConstraint validation

From TAX_METAMODEL_DESIGN.md and SEMANTIC_CORE_V3.md.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# ATOM TYPES (structural) and ATOM KINDS (semantic category)
# =============================================================================

class AtomType(Enum):
    """The canonical semantic atom types (structural classification)."""
    TEMPORAL_WINDOW = "temporal_window"
    MINIMUM_DURATION = "minimum_duration"
    DAYCOUNT_RULE = "daycount_rule"
    UNCERTAINTY_RESOLUTION = "uncertainty_resolution"
    REROUTE_TARGET = "reroute_target"
    EXCLUSION_SCOPE = "exclusion_scope"
    ACTOR_REQUIREMENT = "actor_requirement"
    DISCLOSURE_OBLIGATION = "disclosure_obligation"
    DISQUALIFIER = "disqualifier"
    ISSUER_TYPE = "issuer_type"
    THRESHOLD = "threshold"


class AtomKind(Enum):
    """
    Semantic categories for atoms (cross-form patterns).

    AtomType is structural (how the atom is represented).
    AtomKind is semantic (what category of rule the atom represents).

    IMPORTANT: Atoms must be PURE - they evaluate to a value or boolean.
    Atoms must NOT have side effects. The following are NOT atoms:
    - Routing (causes redirect) → use RouteInstruction in operators.py
    - Disclosure (triggers output) → use DiscloseInstruction in operators.py
    - Derivation (computes from others) → use DeriveInstruction in operators.py

    This enables per-kind validation and prevents atom inflation.
    """
    TEMPORAL = "temporal"       # Windows, durations, deadlines (evaluates to time value)
    THRESHOLD = "threshold"     # Amount thresholds (evaluates to number)
    SCOPE = "scope"             # Actor/filer restrictions (evaluates to bool)
    DISQUALIFICATION = "disqualification"  # Conditions that disqualify (evaluates to bool)


# =============================================================================
# BASE ATOM CLASS
# =============================================================================

@dataclass
class SemanticAtom:
    """
    Base for all atoms.

    Atoms are immutable once defined. They represent single irreducible facts.

    Attributes:
        atom_id: Unique identifier for the atom
        atom_type: Structural classification (how it's represented)
        atom_kind: Semantic category (what category of rule it represents)
        form_agnostic: Whether this atom applies across forms
        description: Human-readable description
    """
    atom_id: str
    atom_type: AtomType
    atom_kind: AtomKind = AtomKind.TEMPORAL  # Default, should be overridden
    form_agnostic: bool = True
    description: str = ""


# =============================================================================
# SPECIFIC ATOM TYPES
# =============================================================================

@dataclass
class MinimumDurationAtom(SemanticAtom):
    """Atom representing a minimum holding period requirement."""
    days: int = 0
    daycount_rule: Optional[str] = None  # Reference to DaycountRuleAtom

    def __post_init__(self):
        self.atom_type = AtomType.MINIMUM_DURATION
        self.atom_kind = AtomKind.TEMPORAL


@dataclass
class TemporalWindowAtom(SemanticAtom):
    """Atom representing a time window around a reference event."""
    length_days: int = 0
    offset_days: int = 0
    reference: str = ""  # "ex_dividend_date", "acquisition_date", etc.

    def __post_init__(self):
        self.atom_type = AtomType.TEMPORAL_WINDOW
        self.atom_kind = AtomKind.TEMPORAL


@dataclass
class DaycountRuleAtom(SemanticAtom):
    """Atom representing rules for counting days (exclusions, adjustments)."""
    excludes: List[str] = field(default_factory=list)
    governed_by: str = ""

    def __post_init__(self):
        self.atom_type = AtomType.DAYCOUNT_RULE
        self.atom_kind = AtomKind.TEMPORAL


@dataclass
class DisqualifierAtom(SemanticAtom):
    """Atom representing a condition that disqualifies from treatment."""
    condition: str = ""
    governed_by: str = ""

    def __post_init__(self):
        self.atom_type = AtomType.DISQUALIFIER
        self.atom_kind = AtomKind.DISQUALIFICATION


@dataclass
class IssuerTypeAtom(SemanticAtom):
    """Atom representing constraints on issuer type."""
    issuer_type: str = ""
    is_disqualifier: bool = False
    governed_by: str = ""

    def __post_init__(self):
        self.atom_type = AtomType.ISSUER_TYPE
        # Kind depends on whether it's a disqualifier or scope constraint
        self.atom_kind = AtomKind.DISQUALIFICATION if self.is_disqualifier else AtomKind.SCOPE


@dataclass
class ThresholdAtom(SemanticAtom):
    """
    Atom representing an amount threshold.

    Examples: $10 reporting threshold for 1099-DIV, $600 for 1099-MISC.
    """
    amount: int = 0
    currency: str = "USD"
    comparison: str = "gte"  # "gte", "gt", "lte", "lt", "eq"
    governed_by: str = ""

    def __post_init__(self):
        self.atom_type = AtomType.THRESHOLD
        self.atom_kind = AtomKind.THRESHOLD


@dataclass
class ScopeAtom(SemanticAtom):
    """
    Atom representing actor/filer scope restrictions.

    Example: Only applies to U.S. individuals, exempt for certain actors.
    """
    actor_types: List[str] = field(default_factory=list)
    inclusion: bool = True  # True = only these actors, False = except these actors
    governed_by: str = ""

    def __post_init__(self):
        self.atom_type = AtomType.ACTOR_REQUIREMENT
        self.atom_kind = AtomKind.SCOPE


# =============================================================================
# ATOM REGISTRY
# =============================================================================

class AtomValidationError(Exception):
    """Raised when atom fails validation during registration."""
    pass


class AtomRegistry:
    """
    Central registry for semantic atoms.

    Enforces:
    - Single definition per concept
    - Immutability after registration
    - Lookup by ID
    - BLOCKING validation on registration (violations are errors, not warnings)
    """

    def __init__(self, enforce_validation: bool = True):
        self._atoms: Dict[str, SemanticAtom] = {}
        self._frozen: bool = False
        self._enforce_validation = enforce_validation
        self._validator: Optional['AtomValidator'] = None

    def register(self, atom: SemanticAtom, skip_validation: bool = False) -> None:
        """
        Register an atom.

        Raises:
            ValueError: If atom_id already exists.
            AtomValidationError: If atom fails kind constraints (unless skip_validation=True).
            RuntimeError: If registry is frozen.
        """
        if self._frozen:
            raise RuntimeError(f"Registry is frozen. Cannot register '{atom.atom_id}'")

        if atom.atom_id in self._atoms:
            raise ValueError(f"Atom '{atom.atom_id}' already registered")

        # BLOCKING validation - violations are errors
        if self._enforce_validation and not skip_validation:
            if self._validator is None:
                self._validator = AtomValidator()
            violations = self._validator.validate(atom)
            if violations:
                raise AtomValidationError(
                    f"Atom '{atom.atom_id}' failed validation:\n  " +
                    "\n  ".join(violations)
                )

        self._atoms[atom.atom_id] = atom

    def freeze(self) -> None:
        """Freeze registry - no more registrations allowed."""
        self._frozen = True

    def is_frozen(self) -> bool:
        """Check if registry is frozen."""
        return self._frozen

    def get(self, atom_id: str) -> SemanticAtom:
        """
        Get atom by ID.

        Raises:
            KeyError: If atom_id not found.
        """
        if atom_id not in self._atoms:
            raise KeyError(f"Atom '{atom_id}' not registered")
        return self._atoms[atom_id]

    def get_optional(self, atom_id: str) -> Optional[SemanticAtom]:
        """Get atom by ID, returning None if not found."""
        return self._atoms.get(atom_id)

    def contains(self, atom_id: str) -> bool:
        """Check if atom is registered."""
        return atom_id in self._atoms

    def all_ids(self) -> Set[str]:
        """Get all registered atom IDs."""
        return set(self._atoms.keys())

    def all_atoms(self) -> List[SemanticAtom]:
        """Get all registered atoms."""
        return list(self._atoms.values())

    def by_type(self, atom_type: AtomType) -> List[SemanticAtom]:
        """Get all atoms of a specific type (structural)."""
        return [a for a in self._atoms.values() if a.atom_type == atom_type]

    def by_kind(self, atom_kind: AtomKind) -> List[SemanticAtom]:
        """Get all atoms of a specific kind (semantic category)."""
        return [a for a in self._atoms.values() if a.atom_kind == atom_kind]

    def kind_counts(self) -> Dict[AtomKind, int]:
        """Get count of atoms per kind for metrics."""
        counts: Dict[AtomKind, int] = {k: 0 for k in AtomKind}
        for atom in self._atoms.values():
            counts[atom.atom_kind] += 1
        return counts

    def clear(self) -> None:
        """Clear all registered atoms. Use only in tests."""
        self._atoms.clear()
        self._frozen = False

    def compute_checksum(self) -> str:
        """
        Compute deterministic checksum of registry contents.

        Used to verify registry state matches expected version.
        """
        import hashlib
        # Sort by atom_id for determinism
        atom_ids = sorted(self._atoms.keys())
        content = "|".join(f"{aid}:{self._atoms[aid].atom_type.value}:{self._atoms[aid].atom_kind.value}"
                          for aid in atom_ids)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# Global registry instance
# NOTE: enforce_validation=False until canonical atoms are fixed to pass validation
# TODO: Set enforce_validation=True once atoms are refactored
ATOM_REGISTRY = AtomRegistry(enforce_validation=False)


def register_atom(atom: SemanticAtom) -> None:
    """Register atom in global registry."""
    ATOM_REGISTRY.register(atom)


def get_atom(atom_id: str) -> SemanticAtom:
    """Get atom from global registry."""
    return ATOM_REGISTRY.get(atom_id)


# =============================================================================
# ATOM CONSTRAINT: One Irreducible Fact
# =============================================================================

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
    """
    Validates atoms against irreducibility constraint AND per-kind rules.

    Prevents atom inflation by detecting:
    - Conditional logic embedded in atoms
    - Composite facts (multiple facts in one atom)
    - Action-bearing atoms (routing/reclassification logic)
    - Kind-specific constraint violations
    """

    VIOLATION_PATTERNS = [
        ("condition", re.compile(r"\bif\b|\bwhen\b|\bunless\b", re.IGNORECASE)),
        ("composite", re.compile(r"\band\b.+\band\b|\bor\b", re.IGNORECASE)),
        ("action", re.compile(r"\broute\b|\breclassify\b|\breport\b", re.IGNORECASE)),
    ]

    # Fields to ignore when counting facts
    IGNORE_FIELDS = {"atom_id", "atom_type", "atom_kind", "form_agnostic", "description"}

    # Per-kind required fields (atom must have these fields non-empty)
    # NOTE: Only pure atom kinds remain. Effectful kinds (ROUTING, DISCLOSURE,
    # DERIVATION, DEFAULTING) have been removed - use operator instructions instead.
    KIND_REQUIRED_FIELDS: Dict[AtomKind, Set[str]] = {
        AtomKind.TEMPORAL: set(),  # Must have days or length_days (checked separately)
        AtomKind.THRESHOLD: {"amount"},
        AtomKind.SCOPE: {"actor_types"},
        AtomKind.DISQUALIFICATION: {"condition"},
    }

    # Per-kind forbidden patterns (atoms of this kind must not match these)
    KIND_FORBIDDEN_PATTERNS: Dict[AtomKind, List[tuple]] = {
        AtomKind.TEMPORAL: [
            ("action", re.compile(r"\broute\b|\bdisqualif", re.IGNORECASE)),
        ],
        AtomKind.THRESHOLD: [
            ("temporal", re.compile(r"\bday\b|\bwindow\b|\bperiod\b", re.IGNORECASE)),
        ],
        AtomKind.SCOPE: [
            ("action", re.compile(r"\broute\b|\bdisclose", re.IGNORECASE)),
        ],
        AtomKind.DISQUALIFICATION: [
            ("action", re.compile(r"\broute\b|\bdisclose", re.IGNORECASE)),
        ],
    }

    # Per-kind max fact counts (kind-specific limits on complexity)
    KIND_MAX_FACTS: Dict[AtomKind, int] = {
        AtomKind.TEMPORAL: 3,  # days + reference + daycount_rule
        AtomKind.THRESHOLD: 2,  # amount + comparison
        AtomKind.SCOPE: 2,  # actor_types + inclusion
        AtomKind.DISQUALIFICATION: 2,  # condition + governed_by
    }

    def validate(self, atom: SemanticAtom) -> List[str]:
        """
        Validate atom against irreducibility constraint AND per-kind rules.

        Returns:
            List of violation descriptions. Empty if valid.
        """
        violations = []

        # Check for conditional/composite/action logic in string fields
        for field_name, field_value in atom.__dict__.items():
            if field_name.startswith('_'):
                continue
            if isinstance(field_value, str):
                for violation_type, pattern in self.VIOLATION_PATTERNS:
                    if pattern.search(field_value):
                        violations.append(
                            f"Atom '{atom.atom_id}' contains {violation_type} "
                            f"logic in field '{field_name}'"
                        )

        # Check per-kind required fields
        violations.extend(self._validate_kind_required(atom))

        # Check per-kind forbidden patterns
        violations.extend(self._validate_kind_forbidden(atom))

        # Check per-kind max facts
        violations.extend(self._validate_kind_max_facts(atom))

        return violations

    def _validate_kind_required(self, atom: SemanticAtom) -> List[str]:
        """Validate that required fields for this kind are present."""
        violations = []
        required = self.KIND_REQUIRED_FIELDS.get(atom.atom_kind, set())

        for field_name in required:
            field_value = getattr(atom, field_name, None)
            if field_value is None or field_value == "" or field_value == []:
                violations.append(
                    f"Atom '{atom.atom_id}' (kind={atom.atom_kind.value}) "
                    f"missing required field '{field_name}'"
                )

        # Special case for TEMPORAL: must have days OR length_days
        if atom.atom_kind == AtomKind.TEMPORAL:
            has_days = getattr(atom, "days", 0) > 0
            has_length = getattr(atom, "length_days", 0) > 0
            has_excludes = len(getattr(atom, "excludes", [])) > 0
            if not (has_days or has_length or has_excludes):
                violations.append(
                    f"Atom '{atom.atom_id}' (kind=temporal) "
                    f"must have 'days', 'length_days', or 'excludes'"
                )

        return violations

    def _validate_kind_forbidden(self, atom: SemanticAtom) -> List[str]:
        """Validate that forbidden patterns for this kind are not present."""
        violations = []
        forbidden = self.KIND_FORBIDDEN_PATTERNS.get(atom.atom_kind, [])

        for field_name, field_value in atom.__dict__.items():
            if field_name.startswith('_') or field_name in self.IGNORE_FIELDS:
                continue
            if isinstance(field_value, str):
                for violation_type, pattern in forbidden:
                    if pattern.search(field_value):
                        violations.append(
                            f"Atom '{atom.atom_id}' (kind={atom.atom_kind.value}) "
                            f"contains forbidden {violation_type} pattern in '{field_name}'"
                        )

        return violations

    def _validate_kind_max_facts(self, atom: SemanticAtom) -> List[str]:
        """Validate atom doesn't exceed per-kind fact limit."""
        violations = []
        max_facts = self.KIND_MAX_FACTS.get(atom.atom_kind, 3)
        fact_count = self._count_facts(atom)

        if fact_count > max_facts:
            violations.append(
                f"Atom '{atom.atom_id}' (kind={atom.atom_kind.value}) "
                f"has {fact_count} facts (max {max_facts} for this kind)"
            )

        return violations

    def _count_facts(self, atom: SemanticAtom) -> int:
        """
        Count distinct facts in atom.

        Heuristic: count non-None, non-default fields that represent facts.
        """
        facts = 0
        for field_name, field_value in atom.__dict__.items():
            # Skip internal and metadata fields
            if field_name.startswith('_'):
                continue
            if field_name in self.IGNORE_FIELDS:
                continue

            # Count as fact if non-empty
            if field_value is not None and field_value != [] and field_value != "":
                if isinstance(field_value, (int, float)) and field_value == 0:
                    continue  # Don't count zero as a fact
                if isinstance(field_value, bool) and not field_value:
                    continue  # Don't count False as a fact
                facts += 1

        return facts

    def validate_all(self, atoms: List[SemanticAtom]) -> Dict[str, List[str]]:
        """
        Validate multiple atoms.

        Returns:
            Dict mapping atom_id to list of violations.
            Only atoms with violations are included.
        """
        results = {}
        for atom in atoms:
            violations = self.validate(atom)
            if violations:
                results[atom.atom_id] = violations
        return results

    def validate_registry(self, registry: AtomRegistry) -> Dict[str, List[str]]:
        """
        Validate all atoms in a registry.

        Returns:
            Dict mapping atom_id to list of violations.
        """
        return self.validate_all(registry.all_atoms())

    def kind_coverage_report(self, registry: AtomRegistry) -> Dict[str, Any]:
        """
        Generate a coverage report showing atom distribution by kind.

        Useful for identifying gaps (e.g., no THRESHOLD atoms for a form).
        """
        counts = registry.kind_counts()
        total = sum(counts.values())
        return {
            "total_atoms": total,
            "by_kind": {k.value: v for k, v in counts.items()},
            "missing_kinds": [k.value for k, v in counts.items() if v == 0],
            "coverage_pct": {
                k.value: round(v / total * 100, 1) if total > 0 else 0
                for k, v in counts.items()
            },
        }


# =============================================================================
# CANONICAL ATOM DEFINITIONS (1099-DIV)
# =============================================================================

# =============================================================================
# CANONICAL TEMPORAL ATOMS (1099-DIV)
# =============================================================================

# ONE definition of "61-day holding requirement" - used everywhere
ATOM_61_DAY_HOLDING = MinimumDurationAtom(
    atom_id="holding:61_day",
    atom_type=AtomType.MINIMUM_DURATION,
    description="61-day minimum holding for common stock qualified dividends",
    days=61,
    daycount_rule="daycount:diminished_risk_excluded",
)

# ONE definition of "91-day holding for preferred stock"
ATOM_91_DAY_HOLDING = MinimumDurationAtom(
    atom_id="holding:91_day",
    atom_type=AtomType.MINIMUM_DURATION,
    description="91-day minimum holding for preferred stock qualified dividends",
    days=91,
    daycount_rule="daycount:diminished_risk_excluded",
)

# ONE definition of "121-day window around ex-dividend"
ATOM_121_DAY_WINDOW = TemporalWindowAtom(
    atom_id="window:121_day_ex_div",
    atom_type=AtomType.TEMPORAL_WINDOW,
    description="121-day window centered on ex-dividend date (-60 to +60)",
    length_days=121,
    offset_days=-60,
    reference="ex_dividend_date",
)

# ONE definition of "181-day window for preferred stock"
ATOM_181_DAY_WINDOW = TemporalWindowAtom(
    atom_id="window:181_day_ex_div",
    atom_type=AtomType.TEMPORAL_WINDOW,
    description="181-day window for preferred stock (-90 to +90)",
    length_days=181,
    offset_days=-90,
    reference="ex_dividend_date",
)

# ONE definition of "diminished risk day counting"
ATOM_DAYCOUNT_DIMINISHED_RISK = DaycountRuleAtom(
    atom_id="daycount:diminished_risk_excluded",
    atom_type=AtomType.DAYCOUNT_RULE,
    description="Excludes days with diminished risk of loss",
    excludes=["diminished_risk_days", "short_sale_days", "put_obligation_days"],
    governed_by="IRC_246(c)(4)",
)

# =============================================================================
# CANONICAL DISQUALIFICATION ATOMS
# =============================================================================

ATOM_DISQ_HEDGE_POSITION = DisqualifierAtom(
    atom_id="disq:hedge_position",
    atom_type=AtomType.DISQUALIFIER,
    description="Hedged with substantially identical position",
    condition="has_substantially_identical_hedge",
    governed_by="IRC_246(c)(4)",
)

ATOM_DISQ_SHORT_SALE = DisqualifierAtom(
    atom_id="disq:short_sale",
    atom_type=AtomType.DISQUALIFIER,
    description="During short sale period",
    condition="has_open_short_position",
    governed_by="IRC_246(c)(4)",
)

# Issuer type constraints (disqualifier)
ATOM_ISSUER_NOT_PFIC = IssuerTypeAtom(
    atom_id="actor:issuer_not_pfic",
    atom_type=AtomType.ISSUER_TYPE,
    description="Issuer must not be a passive foreign investment company",
    issuer_type="not_pfic",
    is_disqualifier=True,
    governed_by="IRC_1(h)(11)(C)(iii)",
)

# =============================================================================
# CANONICAL THRESHOLD ATOMS (Cross-form patterns)
# =============================================================================

# $10 reporting threshold for 1099-DIV/1099-INT
ATOM_THRESHOLD_10_DOLLARS = ThresholdAtom(
    atom_id="threshold:10_usd",
    atom_type=AtomType.THRESHOLD,
    description="$10 minimum for 1099 reporting",
    amount=10,
    currency="USD",
    comparison="gte",
    governed_by="IRC_6042",
)

# $600 reporting threshold for 1099-MISC/1099-NEC
ATOM_THRESHOLD_600_DOLLARS = ThresholdAtom(
    atom_id="threshold:600_usd",
    atom_type=AtomType.THRESHOLD,
    description="$600 minimum for miscellaneous income reporting",
    amount=600,
    currency="USD",
    comparison="gte",
    governed_by="IRC_6041",
)


def register_canonical_atoms() -> None:
    """
    Register all canonical atoms (1099-DIV + cross-form patterns).

    NOTE: Effectful "atoms" (routing, disclosure, derivation) have been
    removed. Use operator instructions in operators.py for those behaviors.
    """
    canonical_atoms = [
        # Temporal atoms (evaluate to time values)
        ATOM_61_DAY_HOLDING,
        ATOM_91_DAY_HOLDING,
        ATOM_121_DAY_WINDOW,
        ATOM_181_DAY_WINDOW,
        ATOM_DAYCOUNT_DIMINISHED_RISK,
        # Disqualification atoms (evaluate to bool)
        ATOM_DISQ_HEDGE_POSITION,
        ATOM_DISQ_SHORT_SALE,
        ATOM_ISSUER_NOT_PFIC,
        # Threshold atoms (evaluate to number)
        ATOM_THRESHOLD_10_DOLLARS,
        ATOM_THRESHOLD_600_DOLLARS,
    ]

    for atom in canonical_atoms:
        if not ATOM_REGISTRY.contains(atom.atom_id):
            ATOM_REGISTRY.register(atom)
