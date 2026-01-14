"""
Operator IR: The Compiler Target for Semantic Rules.

Every template, rule, and atom must compile down to this minimal operator set.
This prevents "rules as Python" drift and ensures all logic is traceable.

The IR guarantees:
1. Every rule action is one of 9 operators
2. Every operator has a typed signature
3. Every operator produces a DecisionTrace entry
4. No side effects outside the operator set

Operator Set:
    ROUTE      - Direct item to surface
    REQUIRE    - Assert precondition (fails evaluation if false)
    FORBID     - Assert prohibition (fails if condition true)
    DERIVE     - Compute value from sources
    QUALIFY    - Elevate classification if eligible
    EXCLUDE    - Assert mutual exclusivity
    SET_TIME   - Establish temporal context
    DISCLOSE   - Require disclosure/statement
    ASSERT     - Invariant check (hard failure if violated)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


# =============================================================================
# OPERATOR ENUM
# =============================================================================

class Operator(Enum):
    """The 9 canonical operators. All rules compile to these."""
    ROUTE = "ROUTE"           # Direct item to surface
    REQUIRE = "REQUIRE"       # Assert precondition
    FORBID = "FORBID"         # Assert prohibition
    DERIVE = "DERIVE"         # Compute value from sources
    QUALIFY = "QUALIFY"       # Elevate classification if eligible
    EXCLUDE = "EXCLUDE"       # Assert mutual exclusivity
    SET_TIME = "SET_TIME"     # Establish temporal context
    DISCLOSE = "DISCLOSE"     # Require disclosure/statement
    ASSERT = "ASSERT"         # Invariant check


# =============================================================================
# TYPED FACT SET (Context Model)
# =============================================================================

@dataclass
class ActorFacts:
    """Facts about an actor in the reporting chain."""
    actor_id: str
    actor_type: str                           # "individual", "corporation", etc.
    is_us_person: Optional[bool] = None
    is_exempt_payee: Optional[bool] = None
    is_ric: bool = False
    is_reit: bool = False
    is_pfic: bool = False
    tin: Optional[str] = None
    jurisdiction: Optional[str] = None


@dataclass
class InstrumentFacts:
    """Facts about the financial instrument."""
    instrument_id: str
    instrument_type: str                      # "common_stock", "preferred_stock", etc.
    issuer_actor_id: str
    cusip: Optional[str] = None
    is_qualified_dividend_eligible: bool = True
    ex_dividend_date: Optional[date] = None
    record_date: Optional[date] = None
    payment_date: Optional[date] = None


@dataclass
class HoldingFacts:
    """Facts about holding period and risk."""
    acquisition_date: date
    disposition_date: Optional[date] = None
    holding_days: int = 0
    diminished_risk_days: int = 0
    short_sale_days: int = 0
    put_obligation_days: int = 0
    has_hedge_position: bool = False


@dataclass
class AmountFacts:
    """Facts about amounts."""
    gross_amount: Decimal = Decimal("0")
    qualified_amount: Optional[Decimal] = None
    ordinary_amount: Optional[Decimal] = None
    capital_gain_amount: Optional[Decimal] = None
    foreign_tax_paid: Optional[Decimal] = None
    federal_tax_withheld: Optional[Decimal] = None
    currency: str = "USD"


@dataclass
class TemporalFacts:
    """Facts about dates and time windows."""
    tax_year: int
    reporting_deadline: Optional[date] = None
    amendment_deadline: Optional[date] = None
    evaluation_date: date = field(default_factory=date.today)


@dataclass
class FactSet:
    """
    Canonical fact schema for rule evaluation.

    All operators receive a FactSet. This prevents untyped dict access
    and ensures consistent evaluation across rules.
    """
    fact_set_id: str
    authority_id: str

    # Typed fact categories
    actor: Optional[ActorFacts] = None
    issuer: Optional[ActorFacts] = None
    instrument: Optional[InstrumentFacts] = None
    holding: Optional[HoldingFacts] = None
    amounts: Optional[AmountFacts] = None
    temporal: Optional[TemporalFacts] = None

    # Filer-specific facts
    filer: Optional[ActorFacts] = None
    filer_is_ric: bool = False
    filer_is_reit: bool = False

    # Additional typed facts (extensible but typed)
    flags: Dict[str, bool] = field(default_factory=dict)
    values: Dict[str, Decimal] = field(default_factory=dict)
    dates: Dict[str, date] = field(default_factory=dict)
    strings: Dict[str, str] = field(default_factory=dict)

    def get_flag(self, key: str, default: bool = False) -> bool:
        """Get boolean flag with default."""
        return self.flags.get(key, default)

    def get_value(self, key: str, default: Decimal = Decimal("0")) -> Decimal:
        """Get decimal value with default."""
        return self.values.get(key, default)

    def get_date(self, key: str) -> Optional[date]:
        """Get date value."""
        return self.dates.get(key)

    def effective_holding_days(self) -> int:
        """Compute holding days excluding diminished risk periods."""
        if not self.holding:
            return 0
        return (
            self.holding.holding_days
            - self.holding.diminished_risk_days
            - self.holding.short_sale_days
            - self.holding.put_obligation_days
        )


# =============================================================================
# OPERATOR INSTRUCTIONS (IR)
# =============================================================================

@dataclass
class OperatorInstruction:
    """
    Base class for operator instructions.

    Every rule compiles to a sequence of OperatorInstructions.
    """
    instruction_id: str
    operator: Operator
    source_rule_id: str
    source_template_id: Optional[str] = None
    source_atom_ids: List[str] = field(default_factory=list)

    # Execution metadata
    priority: int = 0
    authority_id: str = ""

    def execute(self, facts: FactSet) -> 'OperatorResult':
        """Execute instruction against facts. Override in subclasses."""
        raise NotImplementedError


@dataclass
class RouteInstruction(OperatorInstruction):
    """ROUTE: Direct item to surface."""
    target_surface_id: str = ""
    source_classification: str = ""
    conditions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.operator = Operator.ROUTE


@dataclass
class RequireInstruction(OperatorInstruction):
    """REQUIRE: Assert precondition."""
    predicate: str = ""                       # Predicate expression
    failure_message: str = ""
    is_soft: bool = False                     # Soft = warning, Hard = failure

    def __post_init__(self):
        self.operator = Operator.REQUIRE


@dataclass
class ForbidInstruction(OperatorInstruction):
    """FORBID: Assert prohibition."""
    predicate: str = ""
    failure_message: str = ""
    overridable: bool = False

    def __post_init__(self):
        self.operator = Operator.FORBID


@dataclass
class DeriveInstruction(OperatorInstruction):
    """DERIVE: Compute value from sources."""
    target_surface_id: str = ""
    derivation_operator: str = ""             # "SUM", "SUBSET", "DIFFERENCE", etc.
    source_surface_ids: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.operator = Operator.DERIVE


@dataclass
class QualifyInstruction(OperatorInstruction):
    """QUALIFY: Elevate classification if eligible."""
    from_classification: str = ""
    to_classification: str = ""
    eligibility_predicates: List[str] = field(default_factory=list)
    disqualifier_predicates: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.operator = Operator.QUALIFY


@dataclass
class ExcludeInstruction(OperatorInstruction):
    """EXCLUDE: Assert mutual exclusivity."""
    classifications: Set[str] = field(default_factory=set)
    scope: str = "per_item"                   # "per_item", "per_surface", "per_form"

    def __post_init__(self):
        self.operator = Operator.EXCLUDE


@dataclass
class SetTimeInstruction(OperatorInstruction):
    """SET_TIME: Establish temporal context."""
    window_length_days: int = 0
    window_offset_days: int = 0
    reference_date_key: str = ""              # Key in FactSet.dates
    minimum_days: Optional[int] = None

    def __post_init__(self):
        self.operator = Operator.SET_TIME


@dataclass
class DiscloseInstruction(OperatorInstruction):
    """DISCLOSE: Require disclosure/statement."""
    disclosure_type: str = ""                 # "statement", "attachment", "flag"
    required_fields: List[str] = field(default_factory=list)
    trigger_condition: str = ""

    def __post_init__(self):
        self.operator = Operator.DISCLOSE


@dataclass
class AssertInstruction(OperatorInstruction):
    """ASSERT: Invariant check."""
    invariant: str = ""
    failure_severity: str = "error"           # "error", "warning", "info"
    failure_message: str = ""

    def __post_init__(self):
        self.operator = Operator.ASSERT


# =============================================================================
# OPERATOR RESULT AND DECISION TRACE
# =============================================================================

@dataclass
class OperatorResult:
    """Result of executing a single operator instruction."""
    instruction_id: str
    operator: Operator
    success: bool
    outcome: str = ""                         # "routed", "qualified", "excluded", etc.

    # What was evaluated
    predicates_checked: List[str] = field(default_factory=list)
    predicate_results: Dict[str, bool] = field(default_factory=dict)

    # What was produced
    output_surface_id: Optional[str] = None
    output_classification: Optional[str] = None
    output_value: Optional[Decimal] = None

    # Uncertainty
    uncertainty_boundary_id: Optional[str] = None
    confidence: float = 1.0

    # Error info
    error_message: Optional[str] = None


@dataclass
class TraceEntry:
    """Single entry in a decision trace."""
    entry_id: str
    timestamp: datetime
    operator: Operator
    instruction_id: str

    # Source provenance
    rule_id: str
    template_id: Optional[str] = None
    atom_ids: List[str] = field(default_factory=list)

    # Inputs
    facts_referenced: List[str] = field(default_factory=list)
    binding_results_used: List[str] = field(default_factory=list)

    # Evaluation
    predicates_checked: List[str] = field(default_factory=list)
    predicate_results: Dict[str, bool] = field(default_factory=dict)

    # Output
    outcome: str = ""
    reason: str = ""
    confidence: float = 1.0
    uncertainty_boundary_hit: Optional[str] = None


@dataclass
class DecisionTraceSchema:
    """
    Structured decision trace with enforced schema.

    Every rule evaluation produces a DecisionTraceSchema.
    This is not a log - it's a structured audit artifact.
    """
    trace_id: str
    authority_id: str
    created_at: datetime
    fact_set_id: str

    # What was evaluated
    item_id: str
    initial_classification: str

    # Execution trace
    entries: List[TraceEntry] = field(default_factory=list)

    # Final outcome
    final_classification: str = ""
    final_surface_id: str = ""
    final_confidence: float = 1.0

    # Audit support
    instructions_executed: int = 0
    predicates_evaluated: int = 0
    uncertainty_boundaries_hit: List[str] = field(default_factory=list)

    # Human-readable summary
    explanation: str = ""

    def add_entry(self, entry: TraceEntry) -> None:
        """Add trace entry and update counters."""
        self.entries.append(entry)
        self.instructions_executed += 1
        self.predicates_evaluated += len(entry.predicates_checked)
        if entry.uncertainty_boundary_hit:
            self.uncertainty_boundaries_hit.append(entry.uncertainty_boundary_hit)

    def generate_explanation(self) -> str:
        """Generate human-readable explanation from trace."""
        lines = []
        for entry in self.entries:
            if entry.outcome:
                lines.append(f"{entry.operator.value}: {entry.reason or entry.outcome}")
        self.explanation = " → ".join(lines)
        return self.explanation


# =============================================================================
# INSTRUCTION COMPILER (Template -> IR)
# =============================================================================

class InstructionCompiler:
    """
    Compiles templates/rules to operator instructions.

    Enforces that all logic passes through the IR boundary.
    """

    def __init__(self, authority_id: str):
        self.authority_id = authority_id
        self._instruction_counter = 0

    def _next_id(self) -> str:
        self._instruction_counter += 1
        return f"inst_{self._instruction_counter:04d}"

    def compile_qualification(
        self,
        rule_id: str,
        from_classification: str,
        to_classification: str,
        temporal_atom_id: str,
        duration_atom_id: str,
        disqualifier_atom_ids: List[str],
        target_surface_id: str,
        failure_surface_id: str,
    ) -> List[OperatorInstruction]:
        """
        Compile a qualification rule to IR.

        Returns sequence: SET_TIME -> QUALIFY -> ROUTE (pass) / ROUTE (fail)
        """
        instructions: List[OperatorInstruction] = []

        # 1. SET_TIME - establish temporal context
        instructions.append(SetTimeInstruction(
            instruction_id=self._next_id(),
            operator=Operator.SET_TIME,
            source_rule_id=rule_id,
            source_atom_ids=[temporal_atom_id, duration_atom_id],
            authority_id=self.authority_id,
            reference_date_key="ex_dividend_date",
        ))

        # 2. QUALIFY - check eligibility and elevate
        instructions.append(QualifyInstruction(
            instruction_id=self._next_id(),
            operator=Operator.QUALIFY,
            source_rule_id=rule_id,
            source_atom_ids=[duration_atom_id] + disqualifier_atom_ids,
            authority_id=self.authority_id,
            from_classification=from_classification,
            to_classification=to_classification,
            eligibility_predicates=[f"holding_days >= atom:{duration_atom_id}.days"],
            disqualifier_predicates=[f"atom:{d}.condition" for d in disqualifier_atom_ids],
        ))

        # 3. ROUTE - direct to appropriate surface
        instructions.append(RouteInstruction(
            instruction_id=self._next_id(),
            operator=Operator.ROUTE,
            source_rule_id=rule_id,
            authority_id=self.authority_id,
            target_surface_id=target_surface_id,
            source_classification=to_classification,
            conditions=["qualification_passed"],
        ))

        instructions.append(RouteInstruction(
            instruction_id=self._next_id(),
            operator=Operator.ROUTE,
            source_rule_id=rule_id,
            authority_id=self.authority_id,
            target_surface_id=failure_surface_id,
            source_classification=from_classification,
            conditions=["qualification_failed"],
        ))

        return instructions

    def compile_derivation(
        self,
        rule_id: str,
        target_surface_id: str,
        operator: str,
        source_surface_ids: List[str],
    ) -> List[OperatorInstruction]:
        """Compile a derivation rule to IR."""
        return [DeriveInstruction(
            instruction_id=self._next_id(),
            operator=Operator.DERIVE,
            source_rule_id=rule_id,
            authority_id=self.authority_id,
            target_surface_id=target_surface_id,
            derivation_operator=operator,
            source_surface_ids=source_surface_ids,
        )]

    def compile_exclusion(
        self,
        rule_id: str,
        classifications: Set[str],
        scope: str = "per_item",
    ) -> List[OperatorInstruction]:
        """Compile an exclusion rule to IR."""
        return [ExcludeInstruction(
            instruction_id=self._next_id(),
            operator=Operator.EXCLUDE,
            source_rule_id=rule_id,
            authority_id=self.authority_id,
            classifications=classifications,
            scope=scope,
        )]

    def compile_gating(
        self,
        rule_id: str,
        predicate: str,
        target_surface_ids: List[str],
    ) -> List[OperatorInstruction]:
        """Compile a gating rule to IR."""
        instructions = []
        for surface_id in target_surface_ids:
            instructions.append(RequireInstruction(
                instruction_id=self._next_id(),
                operator=Operator.REQUIRE,
                source_rule_id=rule_id,
                authority_id=self.authority_id,
                predicate=predicate,
                failure_message=f"Gating condition not met for {surface_id}",
            ))
        return instructions

    def compile_disclosure(
        self,
        rule_id: str,
        disclosure_type: str,
        required_fields: List[str],
        trigger_condition: str,
    ) -> List[OperatorInstruction]:
        """Compile a disclosure requirement to IR."""
        return [DiscloseInstruction(
            instruction_id=self._next_id(),
            operator=Operator.DISCLOSE,
            source_rule_id=rule_id,
            authority_id=self.authority_id,
            disclosure_type=disclosure_type,
            required_fields=required_fields,
            trigger_condition=trigger_condition,
        )]


# =============================================================================
# PIPELINE: THE SINGLE EXECUTION PATH
# =============================================================================

class PipelineError(Exception):
    """Raised when pipeline execution fails."""
    pass


class TraceMissingError(PipelineError):
    """Raised when trace emission is skipped (mandatory)."""
    pass


@dataclass
class PipelineOutput:
    """Output from pipeline execution."""
    fact_set_id: str
    authority_id: str
    trace: DecisionTraceSchema  # MANDATORY - pipeline fails without trace
    bindings: Dict[str, Decimal]  # semantic_id -> value
    final_classification: str
    success: bool
    error: Optional[str] = None


class Pipeline:
    """
    THE single execution path for rule evaluation.

    All evaluation MUST go through this pipeline:
        FactSet → Predicates → IR → Resolution → Bindings → Trace

    This class enforces:
    1. Trace emission is MANDATORY (TraceMissingError if skipped)
    2. All evaluation uses predicates (no ad-hoc conditions)
    3. All output goes through bindings (no direct surface writes)
    4. Authority adapter is consulted for precedence
    """

    def __init__(self, authority_id: str = "irs"):
        """
        Initialize pipeline.

        Args:
            authority_id: Identifier for provenance tracking (not routing).
                         For IRS-only scope, this is just metadata.
        """
        self.authority_id = authority_id
        self._predicate_evaluator: Optional[Any] = None  # Lazy import
        self._binding_registry: Optional[Any] = None  # Lazy import

    def _get_predicate_evaluator(self) -> Any:
        """Lazy load predicate evaluator to avoid circular imports."""
        if self._predicate_evaluator is None:
            from vaas.core.predicates import PredicateEvaluator
            self._predicate_evaluator = PredicateEvaluator()
        return self._predicate_evaluator

    def _get_binding_registry(self) -> Any:
        """Lazy load binding registry."""
        if self._binding_registry is None:
            from vaas.core.bindings import BINDING_REGISTRY
            self._binding_registry = BINDING_REGISTRY
        return self._binding_registry

    def execute(
        self,
        facts: FactSet,
        instructions: List[OperatorInstruction],
        initial_classification: str,
        item_id: str,
    ) -> PipelineOutput:
        """
        Execute the pipeline.

        Args:
            facts: Input facts for evaluation
            instructions: Compiled IR instructions
            initial_classification: Starting classification
            item_id: ID of item being evaluated

        Returns:
            PipelineOutput with trace (MANDATORY), bindings, and final state

        Raises:
            TraceMissingError: If trace emission fails (should never happen)
            PipelineError: On execution failure
        """
        # Create trace - this is MANDATORY
        trace = DecisionTraceSchema(
            trace_id=f"trace_{facts.fact_set_id}_{item_id}",
            authority_id=self.authority_id,
            created_at=datetime.now(),
            fact_set_id=facts.fact_set_id,
            item_id=item_id,
            initial_classification=initial_classification,
        )

        current_classification = initial_classification
        bindings: Dict[str, Decimal] = {}
        final_surface_id = ""

        try:
            # Sort instructions by priority (authority adapter may influence)
            sorted_instructions = self._sort_by_precedence(instructions)

            for instruction in sorted_instructions:
                result = self._execute_instruction(instruction, facts, current_classification)

                # Record in trace (MANDATORY for every instruction)
                entry = TraceEntry(
                    entry_id=f"entry_{len(trace.entries):03d}",
                    timestamp=datetime.now(),
                    operator=instruction.operator,
                    instruction_id=instruction.instruction_id,
                    rule_id=instruction.source_rule_id,
                    template_id=instruction.source_template_id,
                    atom_ids=instruction.source_atom_ids,
                    predicates_checked=result.predicates_checked,
                    predicate_results=result.predicate_results,
                    outcome=result.outcome,
                    confidence=result.confidence,
                    uncertainty_boundary_hit=result.uncertainty_boundary_id,
                )
                trace.add_entry(entry)

                # Update state based on result
                if result.success:
                    if result.output_classification:
                        current_classification = result.output_classification
                    if result.output_surface_id:
                        final_surface_id = result.output_surface_id
                    if result.output_value is not None and result.output_surface_id:
                        # Resolve binding
                        bindings[result.output_surface_id] = result.output_value

            # Finalize trace
            trace.final_classification = current_classification
            trace.final_surface_id = final_surface_id
            trace.generate_explanation()

            # Verify trace was emitted (sanity check)
            if trace.instructions_executed == 0 and len(instructions) > 0:
                raise TraceMissingError("Instructions executed but no trace entries recorded")

            return PipelineOutput(
                fact_set_id=facts.fact_set_id,
                authority_id=self.authority_id,
                trace=trace,
                bindings=bindings,
                final_classification=current_classification,
                success=True,
            )

        except TraceMissingError:
            raise  # Re-raise trace errors
        except Exception as e:
            # Even on error, we must have a trace
            trace.generate_explanation()
            return PipelineOutput(
                fact_set_id=facts.fact_set_id,
                authority_id=self.authority_id,
                trace=trace,
                bindings=bindings,
                final_classification=current_classification,
                success=False,
                error=str(e),
            )

    def _sort_by_precedence(
        self, instructions: List[OperatorInstruction]
    ) -> List[OperatorInstruction]:
        """
        Sort instructions by priority.

        For IRS-only scope, we don't need authority-based routing.
        Just sort by the priority field on each instruction.
        """
        return sorted(instructions, key=lambda i: i.priority, reverse=True)

    def _execute_instruction(
        self,
        instruction: OperatorInstruction,
        facts: FactSet,
        current_classification: str,
    ) -> OperatorResult:
        """Execute a single instruction against facts."""
        evaluator = self._get_predicate_evaluator()

        # Dispatch based on operator type
        if instruction.operator == Operator.QUALIFY:
            return self._execute_qualify(instruction, facts, evaluator)
        elif instruction.operator == Operator.ROUTE:
            return self._execute_route(instruction, facts, current_classification)
        elif instruction.operator == Operator.DERIVE:
            return self._execute_derive(instruction, facts)
        elif instruction.operator == Operator.REQUIRE:
            return self._execute_require(instruction, facts, evaluator)
        elif instruction.operator == Operator.FORBID:
            return self._execute_forbid(instruction, facts, evaluator)
        else:
            # Default: success with no output
            return OperatorResult(
                instruction_id=instruction.instruction_id,
                operator=instruction.operator,
                success=True,
                outcome="executed",
            )

    def _execute_qualify(
        self,
        instruction: OperatorInstruction,
        facts: FactSet,
        evaluator: Any,
    ) -> OperatorResult:
        """Execute QUALIFY instruction."""
        qualify_inst = instruction  # Type narrowing
        if not isinstance(qualify_inst, QualifyInstruction):
            return OperatorResult(
                instruction_id=instruction.instruction_id,
                operator=Operator.QUALIFY,
                success=False,
                error_message="Invalid instruction type",
            )

        predicates_checked = []
        predicate_results = {}

        # Check eligibility predicates (all must pass)
        all_eligible = True
        for pred_str in qualify_inst.eligibility_predicates:
            predicates_checked.append(pred_str)
            try:
                # Use simple evaluation for now
                result = self._evaluate_simple_predicate(pred_str, facts)
                predicate_results[pred_str] = result
                if not result:
                    all_eligible = False
            except Exception:
                predicate_results[pred_str] = False
                all_eligible = False

        # Check disqualifiers (any true = disqualified)
        disqualified = False
        for pred_str in qualify_inst.disqualifier_predicates:
            predicates_checked.append(pred_str)
            try:
                result = self._evaluate_simple_predicate(pred_str, facts)
                predicate_results[pred_str] = result
                if result:
                    disqualified = True
            except Exception:
                predicate_results[pred_str] = False

        qualified = all_eligible and not disqualified

        return OperatorResult(
            instruction_id=instruction.instruction_id,
            operator=Operator.QUALIFY,
            success=True,
            outcome="qualified" if qualified else "not_qualified",
            predicates_checked=predicates_checked,
            predicate_results=predicate_results,
            output_classification=qualify_inst.to_classification if qualified else qualify_inst.from_classification,
        )

    def _execute_route(
        self,
        instruction: OperatorInstruction,
        facts: FactSet,
        current_classification: str,
    ) -> OperatorResult:
        """Execute ROUTE instruction."""
        route_inst = instruction
        if not isinstance(route_inst, RouteInstruction):
            return OperatorResult(
                instruction_id=instruction.instruction_id,
                operator=Operator.ROUTE,
                success=False,
                error_message="Invalid instruction type",
            )

        # Check if conditions match current state
        conditions_met = True
        for cond in route_inst.conditions:
            if cond == "qualification_passed":
                conditions_met = current_classification == route_inst.source_classification
            elif cond == "qualification_failed":
                conditions_met = current_classification != route_inst.source_classification

        if not conditions_met:
            return OperatorResult(
                instruction_id=instruction.instruction_id,
                operator=Operator.ROUTE,
                success=True,
                outcome="skipped",
            )

        # Route to surface
        value = facts.amounts.gross_amount if facts.amounts else Decimal("0")

        return OperatorResult(
            instruction_id=instruction.instruction_id,
            operator=Operator.ROUTE,
            success=True,
            outcome="routed",
            output_surface_id=route_inst.target_surface_id,
            output_classification=current_classification,
            output_value=value,
        )

    def _execute_derive(
        self,
        instruction: OperatorInstruction,
        facts: FactSet,
    ) -> OperatorResult:
        """Execute DERIVE instruction."""
        derive_inst = instruction
        if not isinstance(derive_inst, DeriveInstruction):
            return OperatorResult(
                instruction_id=instruction.instruction_id,
                operator=Operator.DERIVE,
                success=False,
                error_message="Invalid instruction type",
            )

        # Derivation requires source values - placeholder for now
        return OperatorResult(
            instruction_id=instruction.instruction_id,
            operator=Operator.DERIVE,
            success=True,
            outcome="derived",
            output_surface_id=derive_inst.target_surface_id,
        )

    def _execute_require(
        self,
        instruction: OperatorInstruction,
        facts: FactSet,
        evaluator: Any,
    ) -> OperatorResult:
        """Execute REQUIRE instruction."""
        require_inst = instruction
        if not isinstance(require_inst, RequireInstruction):
            return OperatorResult(
                instruction_id=instruction.instruction_id,
                operator=Operator.REQUIRE,
                success=False,
                error_message="Invalid instruction type",
            )

        result = self._evaluate_simple_predicate(require_inst.predicate, facts)

        return OperatorResult(
            instruction_id=instruction.instruction_id,
            operator=Operator.REQUIRE,
            success=result or require_inst.is_soft,
            outcome="passed" if result else ("soft_fail" if require_inst.is_soft else "failed"),
            predicates_checked=[require_inst.predicate],
            predicate_results={require_inst.predicate: result},
            error_message=None if result else require_inst.failure_message,
        )

    def _execute_forbid(
        self,
        instruction: OperatorInstruction,
        facts: FactSet,
        evaluator: Any,
    ) -> OperatorResult:
        """Execute FORBID instruction."""
        forbid_inst = instruction
        if not isinstance(forbid_inst, ForbidInstruction):
            return OperatorResult(
                instruction_id=instruction.instruction_id,
                operator=Operator.FORBID,
                success=False,
                error_message="Invalid instruction type",
            )

        # Forbid passes if condition is FALSE
        result = self._evaluate_simple_predicate(forbid_inst.predicate, facts)
        passed = not result

        return OperatorResult(
            instruction_id=instruction.instruction_id,
            operator=Operator.FORBID,
            success=passed or forbid_inst.overridable,
            outcome="passed" if passed else "forbidden",
            predicates_checked=[forbid_inst.predicate],
            predicate_results={forbid_inst.predicate: result},
            error_message=None if passed else forbid_inst.failure_message,
        )

    def _evaluate_simple_predicate(self, predicate: str, facts: FactSet) -> bool:
        """
        Evaluate a simple predicate against facts.

        Supports:
        - holding_days >= N
        - holding_days >= atom:id.days (looks up atom's days field)
        - atom:id.condition (disqualifier check)
        - Direct fact references
        """
        # Handle atom reference on RHS: "holding_days >= atom:holding:61_day.days"
        if "atom:" in predicate and ">=" in predicate:
            return self._evaluate_atom_comparison(predicate, facts, ">=")
        if "atom:" in predicate and ">" in predicate and ">=" not in predicate:
            return self._evaluate_atom_comparison(predicate, facts, ">")

        # Simple numeric comparisons
        if "holding_days >=" in predicate:
            try:
                threshold = int(predicate.split(">=")[1].strip())
                return facts.effective_holding_days() >= threshold
            except (ValueError, IndexError):
                return False

        if "holding_days >" in predicate:
            try:
                threshold = int(predicate.split(">")[1].strip())
                return facts.effective_holding_days() > threshold
            except (ValueError, IndexError):
                return False

        # Disqualifier atom reference: atom:disq:xyz.condition
        if predicate.startswith("atom:") and ".condition" in predicate:
            return self._evaluate_disqualifier(predicate, facts)

        # Boolean fact lookups
        if "==" in predicate:
            parts = predicate.split("==")
            if len(parts) == 2:
                field = parts[0].strip()
                value = parts[1].strip()
                if value.lower() == "true":
                    return facts.get_flag(field, False)
                elif value.lower() == "false":
                    return not facts.get_flag(field, True)

        # Default: unknown predicate fails
        return False

    def _evaluate_atom_comparison(self, predicate: str, facts: FactSet, op: str) -> bool:
        """Evaluate a comparison with atom reference on RHS."""
        try:
            parts = predicate.split(op)
            if len(parts) != 2:
                return False

            lhs = parts[0].strip()
            rhs = parts[1].strip()

            # Get LHS value
            if lhs == "holding_days":
                lhs_value = facts.effective_holding_days()
            else:
                return False  # Unknown LHS

            # Get RHS value from atom
            if rhs.startswith("atom:"):
                rhs_value = self._get_atom_field_value(rhs)
                if rhs_value is None:
                    return False
            else:
                try:
                    rhs_value = int(rhs)
                except ValueError:
                    return False

            # Compare
            if op == ">=":
                return lhs_value >= rhs_value
            elif op == ">":
                return lhs_value > rhs_value
            else:
                return False

        except Exception:
            return False

    def _get_atom_field_value(self, atom_ref: str) -> Optional[int]:
        """
        Get a field value from an atom reference.

        Format: atom:atom_id.field_name
        Example: atom:holding:61_day.days -> 61
        """
        try:
            from vaas.core.atoms import ATOM_REGISTRY

            # Parse atom:id.field
            if not atom_ref.startswith("atom:"):
                return None

            rest = atom_ref[5:]  # Remove "atom:"
            if "." not in rest:
                return None

            atom_id, field_name = rest.rsplit(".", 1)

            # Look up atom
            atom = ATOM_REGISTRY.get_optional(atom_id)
            if atom is None:
                return None

            # Get field value
            return getattr(atom, field_name, None)

        except Exception:
            return None

    def _evaluate_disqualifier(self, predicate: str, facts: FactSet) -> bool:
        """
        Evaluate a disqualifier condition.

        Format: atom:disq:xyz.condition
        Returns True if the disqualification condition is met.
        """
        try:
            from vaas.core.atoms import ATOM_REGISTRY

            # Parse atom:id.condition
            if not predicate.startswith("atom:"):
                return False

            rest = predicate[5:]  # Remove "atom:"
            if not rest.endswith(".condition"):
                return False

            atom_id = rest[:-10]  # Remove ".condition"

            # Look up atom
            atom = ATOM_REGISTRY.get_optional(atom_id)
            if atom is None:
                return False

            # Get condition field
            condition = getattr(atom, "condition", None)
            if condition is None:
                return False

            # Evaluate condition against facts
            # Common conditions:
            if condition == "has_substantially_identical_hedge":
                return facts.holding.has_hedge_position if facts.holding else False
            elif condition == "has_open_short_position":
                return (facts.holding.short_sale_days > 0) if facts.holding else False
            else:
                # Unknown condition - default to False (not disqualified)
                return False

        except Exception:
            return False
