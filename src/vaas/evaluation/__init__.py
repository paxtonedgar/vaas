"""
Evaluation module for VaaS pipeline.

Provides:
- Internal sanity checks (validate_internal)
- Corpus-grounded checks (validate_corpus)
- Ontology coverage analysis
"""

from .validate_internal import (
    validate_internal,
    CheckResult as InternalCheckResult,
    Finding as InternalFinding,
)

from .validate_corpus import (
    validate_corpus_grounded,
    CheckResult as CorpusCheckResult,
    Finding as CorpusFinding,
)

from .ontology_parser import (
    ParsedOntology,
    OntologyElement,
    OntologyElementType,
    Relationship,
    parse_ontology,
    extract_predicate_vocabulary,
)

from .ontology_coverage import (
    CoverageReport,
    PredicateMapping,
    NodeTypeGap,
    run_coverage_analysis,
    generate_coverage_report_md,
)

__all__ = [
    # Internal sanity
    "validate_internal",
    "InternalCheckResult",
    "InternalFinding",
    # Corpus-grounded
    "validate_corpus_grounded",
    "CorpusCheckResult",
    "CorpusFinding",
    # Ontology
    "ParsedOntology",
    "OntologyElement",
    "OntologyElementType",
    "Relationship",
    "parse_ontology",
    "extract_predicate_vocabulary",
    "CoverageReport",
    "PredicateMapping",
    "NodeTypeGap",
    "run_coverage_analysis",
    "generate_coverage_report_md",
]
