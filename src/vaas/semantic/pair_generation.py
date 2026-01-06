"""
Training pair generation with edge-type stratified sampling.

Generates contrastive learning pairs from the knowledge graph by traversing
edges and sampling to ensure representation across edge types.

Pair Types:
- hierarchical: parent_of, includes edges (structural)
- cross_reference: references_box edges (document cross-refs)
- semantic: defines, qualifies, applies_if, requires edges (meaning)
- negative: excludes edges (what NOT to associate)

Stratification ensures the model learns all relationship types,
not just the most frequent ones.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import pandas as pd


@dataclass
class TrainingPair:
    """A training pair for contrastive learning."""
    anchor_id: str
    anchor_text: str
    positive_id: str
    positive_text: str
    edge_type: str
    pair_category: str  # hierarchical, cross_reference, semantic, negative
    confidence: float
    evidence_text: Optional[str] = None


@dataclass
class PairGenerationConfig:
    """Configuration for pair generation."""
    # Target counts per category (stratification)
    target_per_category: Dict[str, int] = field(default_factory=lambda: {
        "hierarchical": 50,
        "cross_reference": 100,
        "semantic": 50,
        "negative": 30,
    })
    # Minimum confidence threshold
    min_confidence: float = 0.7
    # Random seed for reproducibility
    seed: int = 42
    # Whether to include reverse pairs (B->A for each A->B)
    include_reverse: bool = False


# Edge type to pair category mapping
EDGE_TYPE_CATEGORIES = {
    # Structural (hierarchical)
    "parent_of": "hierarchical",
    "includes": "hierarchical",
    "same_group": "hierarchical",

    # Cross-reference
    "references_box": "cross_reference",
    "same_field": "cross_reference",

    # Semantic (typed edges from Phase 2a)
    "defines": "semantic",
    "qualifies": "semantic",
    "applies_if": "semantic",
    "requires": "semantic",

    # Negative knowledge
    "excludes": "negative",
}


def _get_pair_category(edge_type: str) -> str:
    """Get the pair category for an edge type."""
    return EDGE_TYPE_CATEGORIES.get(edge_type, "cross_reference")


def _get_node_text(node_id: str, nodes_df: pd.DataFrame) -> Optional[str]:
    """Get text content for a node."""
    # Determine which column to use for matching
    if "node_id" in nodes_df.columns:
        id_col = "node_id"
    elif "anchor_id" in nodes_df.columns:
        id_col = "anchor_id"
        # Strip doc prefix if present (e.g., "1099div_filer:box_1a" -> "box_1a")
        if ":" in node_id:
            node_id = node_id.split(":", 1)[1]
    else:
        return None

    matches = nodes_df[nodes_df[id_col] == node_id]
    if len(matches) > 0:
        row = matches.iloc[0]
        # Prefer text, then body_text, then header_text
        text = row.get("text") or row.get("body_text") or row.get("header_text") or ""
        return text if text else None
    return None


def generate_pairs_from_edges(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    config: Optional[PairGenerationConfig] = None,
) -> List[TrainingPair]:
    """
    Generate training pairs from graph edges.

    Args:
        edges_df: DataFrame with columns [source_node_id, target_node_id, edge_type, confidence, ...]
        nodes_df: DataFrame with columns [node_id, body_text, header_text, ...]
        config: Configuration for pair generation

    Returns:
        List of TrainingPair objects
    """
    if config is None:
        config = PairGenerationConfig()

    random.seed(config.seed)

    # Group edges by category
    edges_by_category: Dict[str, List[dict]] = defaultdict(list)

    for _, edge in edges_df.iterrows():
        edge_type = edge["edge_type"]
        confidence = edge.get("confidence", 1.0)

        if confidence < config.min_confidence:
            continue

        category = _get_pair_category(edge_type)
        edges_by_category[category].append({
            "source_id": edge["source_node_id"],
            "target_id": edge["target_node_id"],
            "edge_type": edge_type,
            "confidence": confidence,
            "evidence": edge.get("source_evidence") or edge.get("evidence_text"),
        })

    # Generate pairs with stratified sampling
    all_pairs = []

    for category, target_count in config.target_per_category.items():
        category_edges = edges_by_category.get(category, [])

        if not category_edges:
            continue

        # Sample up to target_count edges (or all if fewer)
        sampled = category_edges
        if len(category_edges) > target_count:
            sampled = random.sample(category_edges, target_count)

        # Convert to pairs
        for edge_data in sampled:
            source_text = _get_node_text(edge_data["source_id"], nodes_df)
            target_text = _get_node_text(edge_data["target_id"], nodes_df)

            if not source_text or not target_text:
                continue

            pair = TrainingPair(
                anchor_id=edge_data["source_id"],
                anchor_text=source_text,
                positive_id=edge_data["target_id"],
                positive_text=target_text,
                edge_type=edge_data["edge_type"],
                pair_category=category,
                confidence=edge_data["confidence"],
                evidence_text=edge_data["evidence"],
            )
            all_pairs.append(pair)

            # Optionally add reverse pair
            if config.include_reverse and category != "negative":
                reverse_pair = TrainingPair(
                    anchor_id=edge_data["target_id"],
                    anchor_text=target_text,
                    positive_id=edge_data["source_id"],
                    positive_text=source_text,
                    edge_type=edge_data["edge_type"],
                    pair_category=category,
                    confidence=edge_data["confidence"],
                    evidence_text=edge_data["evidence"],
                )
                all_pairs.append(reverse_pair)

    return all_pairs


def compute_pair_statistics(pairs: List[TrainingPair]) -> Dict:
    """
    Compute statistics about generated pairs.

    Returns dict with counts by category and edge type.
    """
    by_category = defaultdict(int)
    by_edge_type = defaultdict(int)
    confidence_sum = defaultdict(float)

    for pair in pairs:
        by_category[pair.pair_category] += 1
        by_edge_type[pair.edge_type] += 1
        confidence_sum[pair.pair_category] += pair.confidence

    avg_confidence = {
        cat: confidence_sum[cat] / count if count > 0 else 0
        for cat, count in by_category.items()
    }

    return {
        "total_pairs": len(pairs),
        "by_category": dict(by_category),
        "by_edge_type": dict(by_edge_type),
        "avg_confidence_by_category": avg_confidence,
    }


def pairs_to_dataframe(pairs: List[TrainingPair]) -> pd.DataFrame:
    """Convert pairs list to DataFrame for storage."""
    records = []
    for i, pair in enumerate(pairs):
        records.append({
            "pair_id": f"pair_{i:04d}",
            "anchor_id": pair.anchor_id,
            "anchor_text": pair.anchor_text,
            "positive_id": pair.positive_id,
            "positive_text": pair.positive_text,
            "edge_type": pair.edge_type,
            "pair_category": pair.pair_category,
            "confidence": pair.confidence,
            "evidence_text": pair.evidence_text,
        })
    return pd.DataFrame(records)


def generate_stratified_pairs(
    edges_parquet_path: str,
    nodes_parquet_path: str,
    output_path: Optional[str] = None,
    config: Optional[PairGenerationConfig] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    High-level function to generate stratified pairs from parquet files.

    Args:
        edges_parquet_path: Path to graph_edges.parquet
        nodes_parquet_path: Path to graph_nodes.parquet (or sections.parquet)
        output_path: Optional path to save pairs parquet
        config: Pair generation configuration

    Returns:
        Tuple of (pairs DataFrame, statistics dict)
    """
    edges_df = pd.read_parquet(edges_parquet_path)
    nodes_df = pd.read_parquet(nodes_parquet_path)

    pairs = generate_pairs_from_edges(edges_df, nodes_df, config)
    stats = compute_pair_statistics(pairs)

    pairs_df = pairs_to_dataframe(pairs)

    if output_path:
        pairs_df.to_parquet(output_path, index=False)

    return pairs_df, stats
