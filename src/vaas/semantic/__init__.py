"""
Semantic analysis module for VaaS.

Provides concept role classification, typed edge extraction, and pair generation.
"""

from .concept_roles import classify_concept_role, CONCEPT_ROLES
from .typed_edges import extract_typed_edges_from_section, TypedEdgeCandidate
from .pair_generation import (
    generate_pairs_from_edges,
    generate_stratified_pairs,
    compute_pair_statistics,
    pairs_to_dataframe,
    TrainingPair,
    PairGenerationConfig,
    EDGE_TYPE_CATEGORIES,
)

__all__ = [
    "classify_concept_role",
    "CONCEPT_ROLES",
    "extract_typed_edges_from_section",
    "TypedEdgeCandidate",
    "generate_pairs_from_edges",
    "generate_stratified_pairs",
    "compute_pair_statistics",
    "pairs_to_dataframe",
    "TrainingPair",
    "PairGenerationConfig",
    "EDGE_TYPE_CATEGORIES",
]
