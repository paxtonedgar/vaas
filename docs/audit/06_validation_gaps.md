# Validation Upgrade Review

**Generated:** 2026-01-08
**File Reviewed:** `validate_graph.py`
**LOC:** 1,023

---

## 1. Current Checks Summary

| Check | Name | What It Validates |
|-------|------|-------------------|
| A1 | Anchor Coverage + Uniqueness | Expected boxes present, content per anchor, no duplicate element assignments |
| A2 | Artifact Contamination | No page markers, header/footer text in nodes; evidence quality |
| A3 | Structure Completeness | No monolith nodes (>4000 chars) |
| A4 | Edge Integrity | Valid edge types, confidence bands, source/target exist, evidence present |
| A5 | Skeleton Edge Coverage | Skeleton edge count >0, connectivity, no orphans, no self-edges |
| A6 | Provenance Completeness | Required fields (doc_id, pages, anchor_id) present |
| B1 | Anchor Assignment (LLM) | Placeholder - boundary case judgment |
| B2 | Edge Correctness (LLM) | Placeholder - semantic edge validation |
| B3 | Pair Suitability (LLM) | Placeholder - training pair quality |

---

## 2. Identified Gaps

### 2.1 Graph Skeleton: `parent_of` Coverage

**Current (A5):** Counts `parent_of` edges but doesn't verify structural invariants.

**Missing checks:**
1. Every non-root node has **exactly one** incoming `parent_of` edge
2. No cycles in `parent_of` subgraph (must be a tree)
3. Hierarchy depth is bounded (max ~5 for tax docs)

**Impact:** If a node has 0 or 2+ parents, n-pair mining produces invalid traversals.

### 2.2 Typed Edge Distribution

**Current (A4):** Checks edge types are in allowed set, but doesn't check **distribution**.

**Missing checks:**
1. Semantic edges (`includes`, `excludes`, `defines`, etc.) exist and are diverse
2. Not all semantic edges are `references_box` (would indicate pattern failures)
3. Concept→box edges exist (critical for retrieval)

**Impact:** If 100% of semantic edges are `references_box`, typed edge extraction failed silently.

### 2.3 Negative Knowledge Coverage

**Current:** No checks for `excludes` edges.

**Missing checks:**
1. At least N `excludes` edges exist (known mutual exclusions in 1099-DIV)
2. `excludes` edges connect box nodes (not random paragraphs)
3. Polarity is correct (A excludes B should NOT also have A includes B)

**Impact:** Without `excludes` edges, retrieval can't filter out incorrect context.

### 2.4 Follows Edge DAG

**Current (A5):** Checks for self-edges, but not cycles in `follows`.

**Missing check:** `follows` edges form a DAG (no A→B→A cycles).

---

## 3. Proposed New Checks

### A7: Hierarchy Integrity

**Purpose:** Verify `parent_of` edges form a valid tree.

**Threshold:** 100% of non-root nodes have exactly 1 parent

**Algorithm:**
```python
def audit_a7_hierarchy_integrity(nodes_df, edges_df) -> CheckResult:
    """
    Checks:
    1. Every non-root node has exactly one parent_of incoming edge
    2. No cycles in parent_of subgraph
    3. Depth ≤ MAX_DEPTH (5)
    """
    parent_of_edges = edges_df[edges_df["edge_type"] == "parent_of"]

    # Build child → parent map
    child_to_parent = {}
    for _, row in parent_of_edges.iterrows():
        child = row["target_node_id"]  # parent_of: source → target means source is parent
        parent = row["source_node_id"]
        if child in child_to_parent:
            # Multiple parents - ERROR
            findings.append(...)
        child_to_parent[child] = parent

    # Check all non-root nodes have a parent
    root_nodes = nodes_df[nodes_df["node_type"] == "doc_root"]["node_id"]
    non_root = set(nodes_df["node_id"]) - set(root_nodes)

    for node_id in non_root:
        if node_id not in child_to_parent:
            # Orphan - ERROR
            findings.append(...)

    # Check for cycles using DFS
    # ... (cycle detection)

    # Check max depth
    # ... (depth calculation)
```

**Sample Output:**
```
A7: Hierarchy Integrity
Status: PASS (120/120 nodes have exactly 1 parent)
- Max depth: 4 (threshold: 5)
- Cycles: 0
```

---

### A8: Typed Edge Distribution

**Purpose:** Verify semantic edges are diverse, not dominated by one type.

**Thresholds:**
- `references_box` ≤ 80% of semantic edges
- At least 2 distinct semantic edge types present
- Concept→box edges > 0 (if concepts exist)

**Algorithm:**
```python
def audit_a8_typed_edge_distribution(edges_df, nodes_df) -> CheckResult:
    """
    Checks:
    1. Semantic edge type diversity (not all references_box)
    2. At least 2 distinct semantic types
    3. Concept→box edges exist if concept nodes exist
    """
    semantic_types = {"includes", "excludes", "defines", "qualifies",
                      "applies_if", "requires", "portion_of"}
    ref_types = {"references_box", "references_section", "same_field", "same_group"}

    semantic_edges = edges_df[edges_df["edge_type"].isin(semantic_types)]
    ref_edges = edges_df[edges_df["edge_type"].isin(ref_types)]

    # Distribution check
    type_counts = edges_df["edge_type"].value_counts()

    # Dominance check
    if len(ref_edges) > 0:
        ref_box_count = len(edges_df[edges_df["edge_type"] == "references_box"])
        ref_pct = ref_box_count / (len(semantic_edges) + len(ref_edges))
        if ref_pct > 0.80:
            findings.append(Finding(
                severity="warning",
                message=f"references_box dominates ({ref_pct:.0%})",
                recommendation="Check typed_edges.py pattern tables"
            ))

    # Diversity check
    semantic_type_count = semantic_edges["edge_type"].nunique()
    if semantic_type_count < 2 and len(semantic_edges) > 0:
        findings.append(...)
```

**Sample Output:**
```
A8: Typed Edge Distribution
Status: PASS
- Edge type breakdown:
  - parent_of: 120 (39%)
  - follows: 97 (31%)
  - references_box: 61 (20%)
  - in_section: 22 (7%)
  - includes: 2 (1%)
  - excludes: 2 (1%)
  - portion_of: 1 (<1%)
- references_box: 61/66 non-structural = 92% (WARNING: >80%)
- Semantic types present: 3 (includes, excludes, portion_of)
```

---

### A9: Negative Knowledge Coverage

**Purpose:** Verify `excludes` edges exist for known mutual exclusions.

**Thresholds:**
- `excludes` edge count ≥ 1 (for 1099-DIV, known exclusions exist)
- No conflicting polarity (A excludes B AND A includes B)
- `excludes` sources are box/concept nodes (not paragraphs)

**Algorithm:**
```python
def audit_a9_negative_knowledge(edges_df, nodes_df) -> CheckResult:
    """
    Checks:
    1. At least 1 excludes edge exists
    2. No polarity conflicts (excludes + includes between same nodes)
    3. excludes edges connect meaningful nodes (not paragraphs)
    """
    excludes = edges_df[edges_df["edge_type"] == "excludes"]
    includes = edges_df[edges_df["edge_type"] == "includes"]

    # Check existence
    if len(excludes) == 0:
        findings.append(Finding(
            severity="warning",
            message="No excludes edges found",
            recommendation="1099-DIV has known exclusions (e.g., 'does not include' phrases)"
        ))

    # Check polarity conflicts
    excludes_pairs = set(zip(excludes["source_node_id"], excludes["target_node_id"]))
    includes_pairs = set(zip(includes["source_node_id"], includes["target_node_id"]))

    conflicts = excludes_pairs & includes_pairs
    for src, tgt in conflicts:
        findings.append(Finding(
            severity="error",
            message=f"Polarity conflict: {src} both includes and excludes {tgt}"
        ))

    # Check node types
    paragraph_nodes = set(nodes_df[nodes_df["node_type"] == "paragraph"]["node_id"])
    for _, row in excludes.iterrows():
        if row["source_node_id"] in paragraph_nodes:
            findings.append(Finding(
                severity="info",
                message=f"excludes edge sourced from paragraph (expected: box/concept)"
            ))
```

**Sample Output:**
```
A9: Negative Knowledge Coverage
Status: PASS
- excludes edges: 2
- Polarity conflicts: 0
- Known exclusion coverage: 2/3 (67%)
  - Present: "2e excludes 2a", "1b excludes foreign source"
  - Missing: "qualified dividends excludes Section 404(k)"
```

---

### A10: Follows Edge DAG

**Purpose:** Verify `follows` edges don't form cycles.

**Threshold:** 0 cycles

**Algorithm:**
```python
def audit_a10_follows_dag(edges_df) -> CheckResult:
    """
    Verify follows edges form a DAG (no cycles).
    """
    follows = edges_df[edges_df["edge_type"] == "follows"]

    # Build adjacency
    adj = defaultdict(list)
    for _, row in follows.iterrows():
        adj[row["source_node_id"]].append(row["target_node_id"])

    # Detect cycles using DFS with coloring
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    cycles = []

    def dfs(node, path):
        color[node] = GRAY
        for neighbor in adj[node]:
            if color[neighbor] == GRAY:
                # Cycle found
                cycles.append(path + [neighbor])
            elif color[neighbor] == WHITE:
                dfs(neighbor, path + [neighbor])
        color[node] = BLACK

    for node in adj:
        if color[node] == WHITE:
            dfs(node, [node])

    if cycles:
        for cycle in cycles[:3]:
            findings.append(Finding(
                severity="error",
                message=f"Cycle in follows: {' -> '.join(cycle)}"
            ))
```

**Sample Output:**
```
A10: Follows Edge DAG
Status: PASS
- follows edges: 97
- Cycles detected: 0
```

---

## 4. Implementation Priority

| Check | Priority | Effort | Impact |
|-------|----------|--------|--------|
| A7 (Hierarchy Integrity) | **P1** | Medium | High - breaks n-pair if wrong |
| A9 (Negative Knowledge) | **P1** | Low | High - retrieval quality |
| A8 (Typed Edge Distribution) | **P2** | Low | Medium - early warning |
| A10 (Follows DAG) | **P3** | Low | Low - rare failure mode |

---

## 5. Integration Notes

### Add to `run_validation()`

```python
# After A6
print("  A7: Hierarchy Integrity...")
report.add_check(audit_a7_hierarchy_integrity(nodes_df, edges_df))

print("  A8: Typed Edge Distribution...")
report.add_check(audit_a8_typed_edge_distribution(edges_df, nodes_df))

print("  A9: Negative Knowledge...")
report.add_check(audit_a9_negative_knowledge(edges_df, nodes_df))

print("  A10: Follows DAG...")
report.add_check(audit_a10_follows_dag(edges_df))
```

### Update Constants

```python
# Expected exclusions for 1099-DIV (for A9 coverage check)
EXPECTED_EXCLUSIONS_1099DIV = {
    ("box_2e", "box_2a"),  # "Section 897 ordinary dividends" excludes "Total ordinary dividends"
    ("box_1b", "foreign_source"),  # Qualified dividends must be domestic
}

# Thresholds for A8
MAX_SINGLE_TYPE_PCT = 0.80  # No edge type > 80% of semantic edges
MIN_SEMANTIC_TYPES = 2
```

---

## 6. What's Already Good

| Aspect | Status |
|--------|--------|
| Edge type allowlist | ✅ ALLOWED_EDGE_TYPES in A4 |
| Confidence band checking | ✅ per created_by in A4 |
| Connectivity check | ✅ Connected components in A5 |
| Orphan detection | ✅ Nodes with no edges in A5 |
| Self-edge detection | ✅ source == target in A5 |
| Evidence quality | ✅ Length, newlines, findability in A4 |

---

## 7. Not Proposing

| Check | Why Not |
|-------|---------|
| `same_field` coverage | Document-specific, not generalizable |
| Box label accuracy | Requires ground truth, better for B-check |
| Cross-document edges | Out of scope (single-doc pipeline) |
| Embedding quality | Post-training concern |

---

*End of Validation Gaps Review*
