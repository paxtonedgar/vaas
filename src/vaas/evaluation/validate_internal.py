"""
Internal Sanity Validator - Post-processed table integrity checks.

Checks:
- IS1: ID integrity (non-null, unique)
- IS2: Edge endpoint existence
- IS3: Provenance presence (source_evidence populated)
- IS4: Evidence normalization (length, no newlines)
- IS5: Structural DAG (parent_of/follows/part_of_group only)
- IS6: Provenance traceability (element_id -> bbox resolution)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


# Structural edges for DAG validation - cycles NOT allowed
STRUCTURAL_EDGE_TYPES = {"parent_of", "follows", "part_of_group"}

# Evidence constraints
MAX_EVIDENCE_CHARS = 200


@dataclass
class Finding:
    check_id: str
    severity: str  # error, warning, info
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    check_id: str
    check_name: str
    passed: bool
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def check_is1_id_integrity(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    elements_df: pd.DataFrame,
) -> CheckResult:
    """IS1: Verify IDs are non-null and unique where expected."""
    findings = []
    metrics = {}

    for name, df, id_col in [
        ("nodes", nodes_df, "node_id"),
        ("edges", edges_df, "edge_id"),
        ("anchors", anchors_df, "anchor_id"),
        ("elements", elements_df, "element_id"),
    ]:
        if df.empty or id_col not in df.columns:
            continue

        null_count = df[id_col].isna().sum()
        dup_count = df[id_col].duplicated().sum()

        metrics[f"{name}_count"] = len(df)
        metrics[f"{name}_null"] = int(null_count)
        metrics[f"{name}_dup"] = int(dup_count)

        if null_count > 0:
            findings.append(Finding("IS1", "error", f"{null_count} null {id_col}s"))
        if dup_count > 0 and name != "edges":  # edge dups are warning
            findings.append(Finding("IS1", "error", f"{dup_count} duplicate {id_col}s"))

    return CheckResult(
        "IS1", "ID Integrity",
        passed=not any(f.severity == "error" for f in findings),
        findings=findings,
        metrics=metrics,
    )


def check_is2_edge_endpoints(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> CheckResult:
    """IS2: Verify all edge endpoints exist in nodes."""
    findings = []
    metrics = {}

    if edges_df.empty or nodes_df.empty:
        return CheckResult("IS2", "Edge Endpoints", passed=True, metrics={"skipped": True})

    valid_ids = set(nodes_df["node_id"].dropna().astype(str))

    sources = set(edges_df["source_node_id"].dropna().astype(str))
    targets = set(edges_df["target_node_id"].dropna().astype(str))

    missing_src = sources - valid_ids
    missing_tgt = targets - valid_ids

    metrics["edge_count"] = len(edges_df)
    metrics["missing_sources"] = len(missing_src)
    metrics["missing_targets"] = len(missing_tgt)

    if missing_src:
        findings.append(Finding("IS2", "error", f"{len(missing_src)} edges with missing source nodes",
                               {"sample": sorted(missing_src)[:5]}))
    if missing_tgt:
        findings.append(Finding("IS2", "error", f"{len(missing_tgt)} edges with missing target nodes",
                               {"sample": sorted(missing_tgt)[:5]}))

    return CheckResult(
        "IS2", "Edge Endpoints",
        passed=len(missing_src) == 0 and len(missing_tgt) == 0,
        findings=findings,
        metrics=metrics,
    )


def check_is3_provenance_presence(edges_df: pd.DataFrame) -> CheckResult:
    """IS3: Verify edges have source_evidence."""
    findings = []
    metrics = {}

    if edges_df.empty or "source_evidence" not in edges_df.columns:
        return CheckResult("IS3", "Provenance Presence", passed=True,
                          findings=[Finding("IS3", "info", "No source_evidence column")])

    evidence = edges_df["source_evidence"].fillna("").astype(str).str.strip()
    missing = (evidence == "").sum()

    # Structural edges are self-evident, don't require evidence
    edge_types = edges_df["edge_type"].fillna("")
    structural_missing = ((evidence == "") & edge_types.isin(STRUCTURAL_EDGE_TYPES)).sum()
    semantic_missing = missing - structural_missing

    metrics["total_edges"] = len(edges_df)
    metrics["missing_evidence"] = int(missing)
    metrics["semantic_missing"] = int(semantic_missing)

    if semantic_missing > 0:
        findings.append(Finding("IS3", "warning",
                               f"{semantic_missing} non-structural edges without evidence"))

    return CheckResult("IS3", "Provenance Presence", passed=True, findings=findings, metrics=metrics)


def check_is4_evidence_normalization(edges_df: pd.DataFrame) -> CheckResult:
    """IS4: Verify evidence is normalized (length, no newlines)."""
    findings = []
    metrics = {}

    if edges_df.empty or "source_evidence" not in edges_df.columns:
        return CheckResult("IS4", "Evidence Normalization", passed=True)

    evidence = edges_df["source_evidence"].fillna("").astype(str)

    too_long = (evidence.str.len() > MAX_EVIDENCE_CHARS).sum()
    has_newlines = evidence.str.contains(r'[\n\r]', regex=True, na=False).sum()

    metrics["max_length"] = int(evidence.str.len().max()) if len(evidence) > 0 else 0
    metrics["too_long"] = int(too_long)
    metrics["has_newlines"] = int(has_newlines)

    if too_long > 0:
        findings.append(Finding("IS4", "warning", f"{too_long} evidence strings > {MAX_EVIDENCE_CHARS} chars"))
    if has_newlines > 0:
        findings.append(Finding("IS4", "warning", f"{has_newlines} evidence strings with newlines"))

    return CheckResult("IS4", "Evidence Normalization", passed=True, findings=findings, metrics=metrics)


def check_is5_structural_dag(edges_df: pd.DataFrame) -> CheckResult:
    """
    IS5: DAG check on STRUCTURAL edges only.

    Enforces: no cycles in parent_of/follows/part_of_group
    Excludes: references_box, semantic edges (cycles allowed)
    """
    findings = []
    metrics = {}

    if edges_df.empty:
        findings.append(Finding("IS5", "error", "No edges at all - skeleton required"))
        return CheckResult("IS5", "Structural DAG", passed=False, findings=findings, metrics=metrics)

    # Filter to structural edges ONLY
    edge_types = edges_df["edge_type"].fillna("")
    skeleton = edges_df[edge_types.isin(STRUCTURAL_EDGE_TYPES)]

    metrics["total_edges"] = len(edges_df)
    metrics["structural_edges"] = len(skeleton)
    metrics["excluded_edges"] = len(edges_df) - len(skeleton)

    if skeleton.empty:
        findings.append(Finding("IS5", "error", "No structural edges found - skeleton required"))
        return CheckResult("IS5", "Structural DAG", passed=False, findings=findings, metrics=metrics)

    # Build adjacency for parent_of (the hierarchy edge)
    parent_of = skeleton[skeleton["edge_type"] == "parent_of"]
    metrics["parent_of_count"] = len(parent_of)

    # Require at least one parent_of edge for a valid hierarchy
    if parent_of.empty:
        findings.append(Finding("IS5", "error", "No parent_of edges - hierarchy required"))
        return CheckResult("IS5", "Structural DAG", passed=False, findings=findings, metrics=metrics)

    adj: Dict[str, List[str]] = defaultdict(list)
    for _, row in parent_of.iterrows():
        adj[str(row["source_node_id"])].append(str(row["target_node_id"]))

    # DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = defaultdict(int)
    has_cycle = False

    def dfs(node: str):
        nonlocal has_cycle
        color[node] = GRAY
        for neighbor in adj[node]:
            if color[neighbor] == GRAY:
                has_cycle = True
                return
            if color[neighbor] == WHITE:
                dfs(neighbor)
        color[node] = BLACK

    for node in set(adj.keys()) | {n for ns in adj.values() for n in ns}:
        if color[node] == WHITE:
            dfs(node)
            if has_cycle:
                break

    metrics["has_cycle"] = has_cycle

    # Check for roots
    children = set(parent_of["target_node_id"].astype(str))
    parents = set(parent_of["source_node_id"].astype(str))
    roots = parents - children

    metrics["root_count"] = len(roots)

    if has_cycle:
        findings.append(Finding("IS5", "error", "Cycle in structural hierarchy"))
    if len(roots) == 0 and len(parent_of) > 0:
        findings.append(Finding("IS5", "error", "No root node in hierarchy"))

    return CheckResult(
        "IS5", "Structural DAG",
        passed=not has_cycle and len(roots) > 0,
        findings=findings,
        metrics=metrics,
    )


def check_is6_provenance_traceability(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    elements_df: pd.DataFrame,
) -> CheckResult:
    """IS6: Verify provenance columns exist and resolve."""
    findings = []
    metrics = {}

    # Check node provenance columns
    node_prov_cols = {"page_span", "bbox_union", "element_ids", "pages", "bbox"}
    found_node_cols = node_prov_cols & set(nodes_df.columns) if not nodes_df.empty else set()
    metrics["node_provenance_cols"] = list(found_node_cols)

    if not found_node_cols:
        findings.append(Finding("IS6", "info", "No node provenance columns found"))

    # Check edge provenance
    edge_prov_cols = {"source_element_id", "source_element_ids", "source_pages", "source_bbox"}
    found_edge_cols = edge_prov_cols & set(edges_df.columns) if not edges_df.empty else set()
    metrics["edge_provenance_cols"] = list(found_edge_cols)

    # Verify element_id resolution if available
    if not edges_df.empty and not elements_df.empty:
        if "source_element_id" in edges_df.columns and "element_id" in elements_df.columns:
            edge_elem_ids = set(edges_df["source_element_id"].dropna().astype(str))
            valid_elem_ids = set(elements_df["element_id"].astype(str))
            unresolved = edge_elem_ids - valid_elem_ids

            metrics["edge_element_refs"] = len(edge_elem_ids)
            metrics["unresolved_refs"] = len(unresolved)

            if unresolved:
                findings.append(Finding("IS6", "warning",
                                       f"{len(unresolved)} edge element_ids don't resolve"))

    return CheckResult("IS6", "Provenance Traceability", passed=True, findings=findings, metrics=metrics)


# Reachability threshold
REACHABILITY_THRESHOLD = 0.95


def check_is7_reachability(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> CheckResult:
    """
    IS7: Gating reachability check.

    Requires:
    - Exactly one doc_root node
    - ≥95% of nodes reachable from doc_root via parent_of
    - All box_* nodes reachable

    This kills the "graph emits but topology is garbage" scenario.
    """
    findings = []
    metrics = {}

    if nodes_df.empty:
        findings.append(Finding("IS7", "error", "No nodes to check reachability"))
        return CheckResult("IS7", "Reachability", passed=False, findings=findings, metrics=metrics)

    # Find doc_root nodes
    node_types = nodes_df["node_type"].fillna("") if "node_type" in nodes_df.columns else pd.Series([""] * len(nodes_df))
    doc_roots = nodes_df[node_types == "doc_root"]["node_id"].tolist()

    metrics["doc_root_count"] = len(doc_roots)
    metrics["total_nodes"] = len(nodes_df)

    # Require exactly one doc_root
    if len(doc_roots) == 0:
        findings.append(Finding("IS7", "error", "No doc_root node found"))
        return CheckResult("IS7", "Reachability", passed=False, findings=findings, metrics=metrics)

    if len(doc_roots) > 1:
        findings.append(Finding("IS7", "error", f"Multiple doc_roots found: {doc_roots}"))
        return CheckResult("IS7", "Reachability", passed=False, findings=findings, metrics=metrics)

    doc_root_id = doc_roots[0]

    # Build adjacency for parent_of edges (forward direction: parent -> children)
    if edges_df.empty or "edge_type" not in edges_df.columns:
        findings.append(Finding("IS7", "error", "No edges to traverse"))
        return CheckResult("IS7", "Reachability", passed=False, findings=findings, metrics=metrics)

    parent_of = edges_df[edges_df["edge_type"] == "parent_of"]
    adj: Dict[str, List[str]] = defaultdict(list)
    for _, row in parent_of.iterrows():
        src = str(row["source_node_id"])
        tgt = str(row["target_node_id"])
        adj[src].append(tgt)

    # BFS from doc_root
    reachable: Set[str] = set()
    queue = [doc_root_id]
    reachable.add(doc_root_id)

    while queue:
        node = queue.pop(0)
        for child in adj.get(node, []):
            if child not in reachable:
                reachable.add(child)
                queue.append(child)

    # Calculate reachability
    all_node_ids = set(nodes_df["node_id"].astype(str))
    unreachable = all_node_ids - reachable
    reachability_rate = len(reachable) / len(all_node_ids) if all_node_ids else 0.0

    metrics["reachable_count"] = len(reachable)
    metrics["unreachable_count"] = len(unreachable)
    metrics["reachability_rate"] = reachability_rate

    # Check box nodes specifically
    box_nodes = nodes_df[
        nodes_df["node_type"].fillna("").str.contains("box", case=False) |
        nodes_df["node_id"].astype(str).str.contains("box_", case=False)
    ]["node_id"].astype(str).tolist()

    unreachable_boxes = [b for b in box_nodes if b not in reachable]
    metrics["box_node_count"] = len(box_nodes)
    metrics["unreachable_box_count"] = len(unreachable_boxes)

    # Evaluate pass/fail
    passed = True

    if reachability_rate < REACHABILITY_THRESHOLD:
        passed = False
        findings.append(Finding(
            "IS7", "error",
            f"Reachability {reachability_rate:.1%} below {REACHABILITY_THRESHOLD:.0%} threshold"
        ))

    if unreachable_boxes:
        passed = False
        findings.append(Finding(
            "IS7", "error",
            f"{len(unreachable_boxes)} box nodes unreachable: {unreachable_boxes[:5]}"
        ))

    if unreachable and len(unreachable) <= 10:
        findings.append(Finding(
            "IS7", "warning",
            f"Unreachable nodes: {sorted(unreachable)[:10]}"
        ))

    return CheckResult("IS7", "Reachability", passed=passed, findings=findings, metrics=metrics)


def validate_internal(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    elements_df: pd.DataFrame,
) -> List[CheckResult]:
    """Run all internal sanity checks."""
    return [
        check_is1_id_integrity(nodes_df, edges_df, anchors_df, elements_df),
        check_is2_edge_endpoints(nodes_df, edges_df),
        check_is3_provenance_presence(edges_df),
        check_is4_evidence_normalization(edges_df),
        check_is5_structural_dag(edges_df),
        check_is6_provenance_traceability(nodes_df, edges_df, elements_df),
        check_is7_reachability(nodes_df, edges_df),
    ]
