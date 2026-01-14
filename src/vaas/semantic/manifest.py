"""
Semantic KG manifest utilities.

The manifest records which semantic artifacts were produced during a pipeline
run. Downstream validation can rely on it to ensure all required tables exist
for the declared schema_version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from vaas.schemas.semantic_contract import SchemaVersion

SEMKG_REQUIRED_TABLES_V1 = [
    "graph_nodes.parquet",
    "graph_edges.parquet",
    "struct_nodes.parquet",
    "sections.parquet",
    "anchors.parquet",
    "element_sentences.parquet",
    "references.parquet",
    "authorities.parquet",
    "paragraph_authority_edges.parquet",
    "authority_mentions.parquet",
    "typed_edges.parquet",
    "typed_edge_claims.parquet",
    "claims.parquet",
    "constraints.parquet",
    "claim_edges.parquet",
    "claim_rejections.parquet",
    "claim_authority_mentions.parquet",
    "claim_precedence.parquet",
    "resolved_claims.parquet",
    "resolution_groups.parquet",
    "compiled_directives.parquet",
    "constraints_resolved.parquet",
    "claim_constraints.parquet",
    "constraint_attributes.parquet",
]

MANIFEST_FILENAME = "semkg_manifest.json"


def build_manifest(doc_id: str, tables: List[str]) -> Dict[str, object]:
    """Create a manifest dictionary for the given doc."""
    return {
        "schema_version": SchemaVersion,
        "doc_id": doc_id,
        "tables": sorted(tables),
    }


def write_manifest(output_dir: Path, manifest: Dict[str, object]) -> Path:
    """Persist manifest to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return path


def load_manifest(output_dir: Path) -> Dict[str, object]:
    """Load manifest from disk."""
    path = output_dir / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No semantic manifest found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def required_tables_for_version(version: str) -> List[str]:
    """Return the required table set for the requested schema version."""
    if version == SchemaVersion:
        return SEMKG_REQUIRED_TABLES_V1
    raise ValueError(f"Unknown semantic KG schema version: {version}")


__all__ = [
    "MANIFEST_FILENAME",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "required_tables_for_version",
]
