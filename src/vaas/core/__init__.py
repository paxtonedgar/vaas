"""
VaaS Semantic Core: 18 Primitives for Jurisdiction-Agnostic Legal Reasoning.

This module implements the foundational semantic infrastructure from:
- SEMANTIC_CORE_V3.md (18 primitives)
- TAX_METAMODEL_DESIGN.md (integration architecture)
- INDEX.md (documentation hub)

Layer Architecture:
    Authority Layer:   Authority defines all rules below
                           ↓
    Economic Layer:    EconomicEvent → produces → Item(s)
                           ↓
    Evaluation Layer:  Eligibility → TaxTreatmentChange → ReportingEffect
                           ↓
    Resolution Layer:  ConditionalPrecedence (non-linear, cyclic-safe)
                           ↓
    Output Layer:      ReportingSurface (form, API, ledger)
                           ↓
    Audit Layer:       DecisionTrace (explainability)

Usage:
    from vaas.core import (
        Authority,
        EconomicEvent,
        Item,
        Eligibility,
        TaxTreatmentChange,
        ReportingEffect,
        ATOM_REGISTRY,
        BINDING_REGISTRY,
    )
"""

# Primitives - 18 dataclasses
from vaas.core.primitives import (
    # Enums
    AuthorityType,
    EconomicEventType,
    SurfaceType,
    UncertaintyState,
    LifecycleEventType,
    # Primitives
    Authority,
    AuthorityScopedRule,
    EconomicEvent,
    Item,
    Classification,
    Eligibility,
    TaxTreatmentChange,
    ReportingEffect,
    ComposedQualification,
    TemporalConstraint,
    KnowledgeBoundary,
    ReportingSurface,
    Actor,
    RoleActivation,
    Routing,
    Exclusion,
    PrecedenceCondition,
    ConditionalPrecedence,
    CrossBorderEffect,
    LifecycleEvent,
    RuleEvaluation,
    DecisionTrace,
    Derivation,
    # Canonical instantiations
    AUTHORITY_IRS,
    AUTHORITY_OECD_CRS,
    AUTHORITY_REALTIME_WHT,
)

# Atoms - registry and validation
# NOTE: Effectful atom types (RerouteTargetAtom, DisclosureAtom, DerivationAtom,
# DefaultingAtom) have been removed. Use operator instructions for those behaviors.
from vaas.core.atoms import (
    # Types
    AtomType,
    AtomKind,
    SemanticAtom,
    MinimumDurationAtom,
    TemporalWindowAtom,
    DaycountRuleAtom,
    DisqualifierAtom,
    IssuerTypeAtom,
    ThresholdAtom,
    ScopeAtom,
    # Registry
    AtomRegistry,
    ATOM_REGISTRY,
    register_atom,
    get_atom,
    # Validation
    AtomConstraint,
    AtomValidator,
    AtomValidationError,
    # Canonical atoms - Temporal
    ATOM_61_DAY_HOLDING,
    ATOM_91_DAY_HOLDING,
    ATOM_121_DAY_WINDOW,
    ATOM_181_DAY_WINDOW,
    ATOM_DAYCOUNT_DIMINISHED_RISK,
    # Canonical atoms - Disqualification
    ATOM_DISQ_HEDGE_POSITION,
    ATOM_DISQ_SHORT_SALE,
    ATOM_ISSUER_NOT_PFIC,
    # Canonical atoms - Threshold
    ATOM_THRESHOLD_10_DOLLARS,
    ATOM_THRESHOLD_600_DOLLARS,
    register_canonical_atoms,
)

# Bindings - semantic to form mapping
from vaas.core.bindings import (
    # Types
    CarrierType,
    BindingCondition,
    BindingResult,
    ConditionalBinding,
    SemanticBinding,
    # Registry
    BindingRegistry,
    BINDING_REGISTRY,
    resolve_binding,
    semantic_id_for_form_slot,
    # Canonical bindings
    BINDING_ORDINARY_DIVIDENDS_TOTAL,
    BINDING_QUALIFIED_DIVIDENDS,
    BINDING_CAPITAL_GAIN_DISTRIBUTIONS,
    BINDING_SECTION_199A_DIVIDENDS,
    BINDING_FEDERAL_TAX_WITHHELD,
    BINDING_FOREIGN_TAX_PAID,
    BINDING_INTEREST_INCOME_TOTAL,
    BINDING_SECTION_1202_STATEMENT,
    register_canonical_bindings,
    # Helpers
    infer_carrier_type,
    build_slot_node,
)

# Operators - IR boundary and typed context
from vaas.core.operators import (
    # Operator enum
    Operator,
    # Typed FactSet
    ActorFacts,
    InstrumentFacts,
    HoldingFacts,
    AmountFacts,
    TemporalFacts,
    FactSet,
    # Operator instructions
    OperatorInstruction,
    RouteInstruction,
    RequireInstruction,
    ForbidInstruction,
    DeriveInstruction,
    QualifyInstruction,
    ExcludeInstruction,
    SetTimeInstruction,
    DiscloseInstruction,
    AssertInstruction,
    # Results and traces
    OperatorResult,
    TraceEntry,
    DecisionTraceSchema,
    # Compiler
    InstructionCompiler,
    # Pipeline - THE single execution path
    Pipeline,
    PipelineOutput,
    PipelineError,
    TraceMissingError,
)

# Predicates - rule evaluation expressions
from vaas.core.predicates import (
    # Types
    Comparator,
    BoolOperator,
    PredicateAST,
    ComparisonPredicate,
    BooleanPredicate,
    AtomReference,
    FactReference,
    # Errors
    PredicateParseError,
    PredicateEvalError,
    PredicateComplexityError,
    # Limits
    MAX_FACT_REFERENCES,
    MAX_NESTING_DEPTH,
    # Parser and evaluator
    PredicateParser,
    PredicateEvaluator,
    CompiledPredicate,
    # Functions
    compile_predicate,
    evaluate_predicate,
)

# Corpus Profile - static config for document extraction (replaces authority adapters)
from vaas.core.corpus_profile import (
    # Enums
    CitationType,
    SurfaceType,
    # Config classes
    CitationScheme,
    SurfaceModel,
    CorpusProfile,
    AuthorityProvenance,
    # Canonical configs
    IRS_CITATION_SCHEME,
    IRS_FORM_SURFACE,
    IRS_1099DIV_2024,
    IRS_1099INT_2024,
)


def initialize_core() -> None:
    """
    Initialize the semantic core with canonical atoms and bindings.

    Call this once at application startup to populate registries.
    """
    register_canonical_atoms()
    register_canonical_bindings()


__all__ = [
    # Enums
    "AuthorityType",
    "EconomicEventType",
    "SurfaceType",
    "UncertaintyState",
    "LifecycleEventType",
    "AtomType",
    "AtomKind",
    "CarrierType",
    "Operator",
    # Primitives
    "Authority",
    "AuthorityScopedRule",
    "EconomicEvent",
    "Item",
    "Classification",
    "Eligibility",
    "TaxTreatmentChange",
    "ReportingEffect",
    "ComposedQualification",
    "TemporalConstraint",
    "KnowledgeBoundary",
    "ReportingSurface",
    "Actor",
    "RoleActivation",
    "Routing",
    "Exclusion",
    "PrecedenceCondition",
    "ConditionalPrecedence",
    "CrossBorderEffect",
    "LifecycleEvent",
    "RuleEvaluation",
    "DecisionTrace",
    "Derivation",
    # Atoms - Pure atom types (evaluate to values or booleans, no side effects)
    "SemanticAtom",
    "MinimumDurationAtom",
    "TemporalWindowAtom",
    "DaycountRuleAtom",
    "DisqualifierAtom",
    "IssuerTypeAtom",
    "ThresholdAtom",
    "ScopeAtom",
    # Atoms - Registry and validation
    "AtomRegistry",
    "ATOM_REGISTRY",
    "register_atom",
    "get_atom",
    "AtomConstraint",
    "AtomValidator",
    "AtomValidationError",
    "register_canonical_atoms",
    # Atoms - Canonical instances
    "ATOM_61_DAY_HOLDING",
    "ATOM_91_DAY_HOLDING",
    "ATOM_121_DAY_WINDOW",
    "ATOM_181_DAY_WINDOW",
    "ATOM_DAYCOUNT_DIMINISHED_RISK",
    "ATOM_DISQ_HEDGE_POSITION",
    "ATOM_DISQ_SHORT_SALE",
    "ATOM_ISSUER_NOT_PFIC",
    "ATOM_THRESHOLD_10_DOLLARS",
    "ATOM_THRESHOLD_600_DOLLARS",
    # Bindings
    "BindingCondition",
    "BindingResult",
    "ConditionalBinding",
    "SemanticBinding",
    "BindingRegistry",
    "BINDING_REGISTRY",
    "resolve_binding",
    "semantic_id_for_form_slot",
    "register_canonical_bindings",
    "infer_carrier_type",
    "build_slot_node",
    # Operators - Typed FactSet
    "ActorFacts",
    "InstrumentFacts",
    "HoldingFacts",
    "AmountFacts",
    "TemporalFacts",
    "FactSet",
    # Operators - Instructions
    "OperatorInstruction",
    "RouteInstruction",
    "RequireInstruction",
    "ForbidInstruction",
    "DeriveInstruction",
    "QualifyInstruction",
    "ExcludeInstruction",
    "SetTimeInstruction",
    "DiscloseInstruction",
    "AssertInstruction",
    # Operators - Results
    "OperatorResult",
    "TraceEntry",
    "DecisionTraceSchema",
    "InstructionCompiler",
    # Pipeline - THE single execution path
    "Pipeline",
    "PipelineOutput",
    "PipelineError",
    "TraceMissingError",
    # Predicates - Types
    "Comparator",
    "BoolOperator",
    "PredicateAST",
    "ComparisonPredicate",
    "BooleanPredicate",
    "AtomReference",
    "FactReference",
    # Predicates - Errors
    "PredicateParseError",
    "PredicateEvalError",
    "PredicateComplexityError",
    # Predicates - Limits
    "MAX_FACT_REFERENCES",
    "MAX_NESTING_DEPTH",
    # Predicates - Classes
    "PredicateParser",
    "PredicateEvaluator",
    "CompiledPredicate",
    # Predicates - Functions
    "compile_predicate",
    "evaluate_predicate",
    # Corpus Profile - static config (replaces authority adapters)
    "CitationType",
    "SurfaceType",
    "CitationScheme",
    "SurfaceModel",
    "CorpusProfile",
    "AuthorityProvenance",
    "IRS_CITATION_SCHEME",
    "IRS_FORM_SURFACE",
    "IRS_1099DIV_2024",
    "IRS_1099INT_2024",
    # Initialization
    "initialize_core",
]
