"""
Golden E2E Fixture: Proves the Pipeline Works End-to-End.

This test is the BEHAVIORAL PROOF that the semantic core works.
It tests the full path: FactSet -> IR -> Bindings -> Trace -> Output

If this test passes, the pipeline is BEHAVIOR_PROVEN.
If this test fails, we have STRUCTURE_ONLY.

Test Case: Qualified Dividend Evaluation
- Input: $1000 dividend, 70 holding days, no disqualifiers
- Expected: Qualified (70 >= 61 days)
- Expected Output: Routes to qualified_dividends surface
"""

from datetime import date
from decimal import Decimal

import pytest

from vaas.core import (
    # FactSet
    FactSet,
    ActorFacts,
    InstrumentFacts,
    HoldingFacts,
    AmountFacts,
    TemporalFacts,
    # Instructions
    QualifyInstruction,
    RouteInstruction,
    Operator,
    # Pipeline
    Pipeline,
    PipelineOutput,
    TraceMissingError,
    # Compiler
    InstructionCompiler,
    # Authority
    initialize_core,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def initialized_core():
    """Initialize registries once per module."""
    initialize_core()
    return True


@pytest.fixture
def qualified_dividend_facts():
    """
    Facts representing a qualified dividend scenario.

    - 70 holding days (exceeds 61-day minimum)
    - No diminished risk days
    - U.S. person receiving from domestic corporation
    """
    return FactSet(
        fact_set_id="facts_qdiv_001",
        authority_id="auth:irs",
        actor=ActorFacts(
            actor_id="actor_001",
            actor_type="individual",
            is_us_person=True,
            tin="123-45-6789",
        ),
        instrument=InstrumentFacts(
            instrument_id="inst_001",
            instrument_type="common_stock",
            issuer_actor_id="issuer_001",
            cusip="ABC123456",
            is_qualified_dividend_eligible=True,
            ex_dividend_date=date(2025, 3, 15),
        ),
        holding=HoldingFacts(
            acquisition_date=date(2025, 1, 1),
            holding_days=70,
            diminished_risk_days=0,
            short_sale_days=0,
            put_obligation_days=0,
            has_hedge_position=False,
        ),
        amounts=AmountFacts(
            gross_amount=Decimal("1000.00"),
            currency="USD",
        ),
        temporal=TemporalFacts(
            tax_year=2025,
            evaluation_date=date(2025, 6, 15),
        ),
    )


@pytest.fixture
def non_qualified_dividend_facts():
    """
    Facts representing a NON-qualified dividend scenario.

    - Only 30 holding days (below 61-day minimum)
    """
    return FactSet(
        fact_set_id="facts_non_qdiv_001",
        authority_id="auth:irs",
        actor=ActorFacts(
            actor_id="actor_002",
            actor_type="individual",
            is_us_person=True,
        ),
        instrument=InstrumentFacts(
            instrument_id="inst_002",
            instrument_type="common_stock",
            issuer_actor_id="issuer_002",
            is_qualified_dividend_eligible=True,
        ),
        holding=HoldingFacts(
            acquisition_date=date(2025, 5, 1),
            holding_days=30,  # Less than 61 days
            diminished_risk_days=0,
        ),
        amounts=AmountFacts(
            gross_amount=Decimal("500.00"),
        ),
        temporal=TemporalFacts(
            tax_year=2025,
        ),
    )


@pytest.fixture
def qualification_instructions():
    """
    Compiled IR for qualified dividend evaluation.

    Sequence: QUALIFY -> ROUTE (pass) -> ROUTE (fail)
    """
    compiler = InstructionCompiler(authority_id="auth:irs")
    return compiler.compile_qualification(
        rule_id="rule:qdiv:1",
        from_classification="ordinary_dividend",
        to_classification="qualified_dividend",
        temporal_atom_id="window:121_day_ex_div",
        duration_atom_id="holding:61_day",
        disqualifier_atom_ids=["disq:hedge_position", "disq:short_sale"],
        target_surface_id="qualified_dividends",
        failure_surface_id="ordinary_dividends",
    )


# =============================================================================
# GOLDEN E2E TESTS
# =============================================================================

class TestGoldenE2E:
    """
    Golden E2E tests that prove BEHAVIOR, not just STRUCTURE.

    Each test follows the pattern:
    1. Create FactSet with specific inputs
    2. Execute through Pipeline
    3. Verify expected bindings
    4. Verify trace was emitted
    """

    def test_qualified_dividend_passes(
        self,
        initialized_core,
        qualified_dividend_facts,
        qualification_instructions,
    ):
        """
        GOLDEN TEST: Qualified dividend with sufficient holding period.

        Input: 70 holding days, $1000 dividend
        Expected: Classification = qualified_dividend
        Expected: Routes to qualified_dividends surface
        Expected: Trace has entries for all instructions
        """
        # Execute pipeline
        pipeline = Pipeline(authority_id="auth:irs")
        output = pipeline.execute(
            facts=qualified_dividend_facts,
            instructions=qualification_instructions,
            initial_classification="ordinary_dividend",
            item_id="item_qdiv_001",
        )

        # BEHAVIORAL ASSERTIONS

        # 1. Pipeline succeeded
        assert output.success, f"Pipeline failed: {output.error}"

        # 2. Classification changed to qualified
        assert output.final_classification == "qualified_dividend", (
            f"Expected qualified_dividend, got {output.final_classification}"
        )

        # 3. Trace was emitted (MANDATORY)
        assert output.trace is not None, "Trace must be emitted"
        assert output.trace.instructions_executed > 0, "Trace must have entries"

        # 4. Trace has correct provenance
        assert output.trace.authority_id == "auth:irs"
        assert output.trace.fact_set_id == "facts_qdiv_001"

        # 5. Bindings were produced (routed to correct surface)
        assert "qualified_dividends" in output.bindings, (
            f"Expected qualified_dividends in bindings, got {output.bindings.keys()}"
        )
        assert output.bindings["qualified_dividends"] == Decimal("1000.00")

    def test_non_qualified_dividend_fails_holding_period(
        self,
        initialized_core,
        non_qualified_dividend_facts,
        qualification_instructions,
    ):
        """
        GOLDEN TEST: Non-qualified dividend due to insufficient holding period.

        Input: 30 holding days (< 61), $500 dividend
        Expected: Classification = ordinary_dividend (unchanged)
        Expected: Routes to ordinary_dividends surface
        """
        pipeline = Pipeline(authority_id="auth:irs")
        output = pipeline.execute(
            facts=non_qualified_dividend_facts,
            instructions=qualification_instructions,
            initial_classification="ordinary_dividend",
            item_id="item_non_qdiv_001",
        )

        # BEHAVIORAL ASSERTIONS

        # 1. Pipeline succeeded (even though qualification failed)
        assert output.success, f"Pipeline failed: {output.error}"

        # 2. Classification stayed ordinary (did not qualify)
        assert output.final_classification == "ordinary_dividend", (
            f"Expected ordinary_dividend, got {output.final_classification}"
        )

        # 3. Trace was emitted
        assert output.trace is not None
        assert output.trace.instructions_executed > 0

        # 4. Bindings route to ordinary
        assert "ordinary_dividends" in output.bindings, (
            f"Expected ordinary_dividends in bindings, got {output.bindings.keys()}"
        )

    def test_trace_is_mandatory(self, initialized_core):
        """
        Verify that trace emission cannot be bypassed.

        Even with empty instructions, trace must exist.
        """
        facts = FactSet(
            fact_set_id="facts_empty",
            authority_id="auth:irs",
        )

        pipeline = Pipeline(authority_id="auth:irs")
        output = pipeline.execute(
            facts=facts,
            instructions=[],
            initial_classification="unknown",
            item_id="item_empty",
        )

        # Even with no instructions, trace must exist
        assert output.trace is not None
        assert output.trace.trace_id is not None
        assert output.trace.authority_id == "auth:irs"

    def test_trace_records_predicate_results(
        self,
        initialized_core,
        qualified_dividend_facts,
        qualification_instructions,
    ):
        """
        Verify trace records predicate evaluation results.

        This is critical for auditability.
        """
        pipeline = Pipeline(authority_id="auth:irs")
        output = pipeline.execute(
            facts=qualified_dividend_facts,
            instructions=qualification_instructions,
            initial_classification="ordinary_dividend",
            item_id="item_audit_001",
        )

        # Find QUALIFY entry
        qualify_entries = [
            e for e in output.trace.entries
            if e.operator == Operator.QUALIFY
        ]
        assert len(qualify_entries) > 0, "Expected QUALIFY entry in trace"

        qualify_entry = qualify_entries[0]

        # Verify predicates were checked
        assert len(qualify_entry.predicates_checked) > 0, (
            "QUALIFY must check predicates"
        )

        # Verify predicate results were recorded
        assert len(qualify_entry.predicate_results) > 0, (
            "QUALIFY must record predicate results"
        )


# =============================================================================
# EXPECTED OUTPUT STRUCTURE (for documentation)
# =============================================================================

"""
Expected output for qualified_dividend_facts:

PipelineOutput(
    fact_set_id="facts_qdiv_001",
    authority_id="auth:irs",
    trace=DecisionTraceSchema(
        trace_id="trace_facts_qdiv_001_item_qdiv_001",
        authority_id="auth:irs",
        fact_set_id="facts_qdiv_001",
        item_id="item_qdiv_001",
        initial_classification="ordinary_dividend",
        final_classification="qualified_dividend",
        final_surface_id="qualified_dividends",
        instructions_executed=4,  # SET_TIME, QUALIFY, ROUTE, ROUTE
        predicates_evaluated=3,   # holding_days >= 61, disq checks
        entries=[
            TraceEntry(operator=SET_TIME, outcome="executed"),
            TraceEntry(operator=QUALIFY, outcome="qualified"),
            TraceEntry(operator=ROUTE, outcome="routed"),
            TraceEntry(operator=ROUTE, outcome="skipped"),
        ],
        explanation="SET_TIME: executed -> QUALIFY: qualified -> ROUTE: routed",
    ),
    bindings={
        "qualified_dividends": Decimal("1000.00"),
    },
    final_classification="qualified_dividend",
    success=True,
)
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
