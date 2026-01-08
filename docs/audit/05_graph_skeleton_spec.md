# Graph Skeleton Specification: Hierarchy Edges

**Generated:** 2026-01-08
**Status:** Specification (not implemented yet — documenting current + proposed)

---

## 1. Current State

### 1.1 Edge Counts (1099-DIV Document)

| Edge Type | Count | Status |
|-----------|-------|--------|
| `parent_of` | 120 | ✅ Implemented |
| `follows` | 97 | ✅ Implemented |
| `in_section` | 22 | ✅ Implemented |
| `same_group` | 3 | ✅ Implemented |
| `references_box` | 61 | ✅ Implemented |
| `excludes` | 2 | ✅ Implemented |
| `includes` | 2 | ✅ Implemented |
| `portion_of` | 1 | ✅ Implemented |
| **Total** | **308** | |

### 1.2 Node Hierarchy

```
doc_root (1)
├── preamble (1)
├── section (3)
│   └── concept/subsection (18)
└── box_section (21)
    └── paragraph (78)
```

---

## 2. `parent_of` Edges (Already Implemented)

### 2.1 Algorithm

**Location:** `src/vaas/graph/edges.py:194-278`

```python
def build_section_hierarchy_edges(sections_df, doc_id):
    """
    Uses "nearest preceding" containment:

    1. Sort sections by (page, y0) to establish reading order
    2. Track state: current_section, current_box
    3. For each section:
       - preamble → doc_root
       - section → doc_root (resets current_box)
       - box → current_section OR doc_root
       - subsection → current_box OR current_section OR doc_root
    """
```

**Key Fields Used:**
- `anchor_type`: Determines hierarchy level
- `pages[0]`: Primary sort key
- `bbox[1]` (y0): Secondary sort key

### 2.2 Parent Assignment Rules

| Source Type | Parent | Condition |
|-------------|--------|-----------|
| `preamble` | `doc_root` | Always |
| `section` | `doc_root` | Always (resets box state) |
| `box` | `current_section` | If section exists |
| `box` | `doc_root` | If no section |
| `subsection` | `current_box` | If box exists |
| `subsection` | `current_section` | If no box, section exists |
| `subsection` | `doc_root` | If neither exists |

### 2.3 Failure Modes

| Failure | Cause | Detection | Fix |
|---------|-------|-----------|-----|
| Orphan box | Box before any section | `parent_id == doc_root` for box | Acceptable (document structure) |
| Wrong parent | Incorrect y0 (multi-column) | Manual inspection | Add column-aware sorting |
| Missing edges | Empty `anchor_type` | Count edges vs nodes | Fix element classification |

### 2.4 Validation Query

```sql
-- All non-root nodes should have exactly one parent_of edge
SELECT n.node_id, COUNT(e.edge_id) as parent_count
FROM graph_nodes n
LEFT JOIN graph_edges e ON e.target_node_id = n.node_id AND e.edge_type = 'parent_of'
WHERE n.node_type != 'doc_root'
GROUP BY n.node_id
HAVING parent_count != 1
```

---

## 3. `follows` Edges (Already Implemented)

### 3.1 Algorithm

**Location:** `src/vaas/graph/edges.py:281-329` (section-level)
**Location:** `src/vaas/graph/edges.py:383-423` (paragraph-level)

```python
def build_section_follows_edges(sections_df, doc_id):
    """
    Section-level reading order:
    1. Sort by (page, y0)
    2. Emit edge: prev_section → current_section
    """

def build_paragraph_follows_edges(paragraph_nodes_df, doc_id):
    """
    Paragraph-level reading order (within same anchor):
    1. Group by anchor_id
    2. Sort by reading_order field
    3. Emit edge: prev_para → current_para
    """
```

### 3.2 Edge Direction

```
follows: A → B means "A comes before B in reading order"
```

### 3.3 Scope Rules

| Level | Scope | Example |
|-------|-------|---------|
| Section `follows` | Document-wide | `sec_reminders` → `box_1a` |
| Paragraph `follows` | Within anchor | `el_1:2:seg0` → `el_1:2:seg1` |

### 3.4 Failure Modes

| Failure | Cause | Detection | Fix |
|---------|-------|-----------|-----|
| Missing edge | Only 1 paragraph in anchor | N/A | Acceptable (no predecessor) |
| Cross-page jump | No column awareness | Pages span > 1 | Add column to sort key |
| Cycle | Duplicate reading_order | Graph cycle detection | Dedupe reading_order |

### 3.5 Validation Query

```sql
-- Follows edges should form a DAG (no cycles)
-- Check: no edge where source appears as target of same target
SELECT e1.edge_id, e2.edge_id
FROM graph_edges e1
JOIN graph_edges e2 ON e1.target_node_id = e2.source_node_id
                   AND e1.source_node_id = e2.target_node_id
WHERE e1.edge_type = 'follows' AND e2.edge_type = 'follows'
```

---

## 4. `in_section` Edges (Already Implemented)

### 4.1 Algorithm

**Location:** `src/vaas/graph/edges.py:426-492`

```python
def build_in_section_edges(sections_df, doc_id):
    """
    Denormalized containment for fast queries:
    1. Track current_section as we iterate
    2. Only emit for subsection → section (not box → section)
    """
```

### 4.2 Purpose

Enables fast queries like:
```sql
-- All concepts in "Specific Instructions" section
SELECT * FROM graph_nodes n
JOIN graph_edges e ON e.source_node_id = n.node_id
WHERE e.edge_type = 'in_section'
  AND e.target_node_id = '1099div_filer:sec_specific_instructions'
```

### 4.3 Note

Currently only emits for `subsection` → `section`. Does NOT emit for `box` → `section` because boxes can span sections in this document.

---

## 5. `contains_element` Edges (Optional — Not Implemented)

### 5.1 Proposal

Currently, paragraph nodes have `element_id` field but no explicit edge to elements. If we want fine-grained provenance:

```
anchor → contains_element → element_id
```

### 5.2 Algorithm (Proposed)

```python
def build_contains_element_edges(
    sections_df: pd.DataFrame,
    doc_id: str,
) -> List[Edge]:
    """
    Build anchor → element containment edges.

    Uses sections_df.element_ids field (list of element IDs per section).
    """
    edges = []
    for _, section in sections_df.iterrows():
        anchor_id = section["anchor_id"]
        element_ids = section.get("element_ids", [])

        for eid in element_ids:
            edges.append(Edge(
                edge_id=generate_edge_id("contains_element", ...),
                source_node_id=f"{doc_id}:{anchor_id}",
                target_node_id=f"{doc_id}:el_{eid}",
                edge_type="contains_element",
                confidence=1.0,
                created_by="structural",
            ))

    return edges
```

### 5.3 Decision

**Recommendation:** Do NOT implement `contains_element` edges.

**Reason:**
- Element IDs are already stored in `paragraph_nodes.element_id`
- Adding 200+ edges would bloat the graph
- Queries can join on element_id field instead

---

## 6. Incremental Delivery Plan

### Phase 1: Validate Current Implementation (Complete)

**Status:** ✅ Done

- `parent_of`: 120 edges, passes validation
- `follows`: 97 edges, passes validation
- `in_section`: 22 edges, passes validation

### Phase 2: Add Column-Aware Sorting (Optional Enhancement)

**Status:** 🔶 Not Started

**Problem:** Current sort is (page, y0), but 1099-DIV is two-column.

**Fix:**
```python
def get_sort_key_for_section(row) -> Tuple[int, int, float, float]:
    """
    Column-aware sort key: (page, column, y0, x0)
    """
    page = extract_page(row)
    x0 = extract_x0(row)
    y0 = extract_y0(row)
    col = 0 if x0 < PAGE_MID_X else 1
    return (page, col, y0, x0)
```

**Impact:** Would fix potential cross-column parent assignment issues.

### Phase 3: Add `same_box_group` Edge (Already Implemented)

**Status:** ✅ Done (called `same_group`)

Handles grouped boxes like "Boxes 14-16":
```
box_14 ←same_group→ box_15 ←same_group→ box_16
```

### Phase 4: Enhance Validation (Proposed)

**Status:** 🔶 Not Started

Add to `validate_graph.py`:

```python
def audit_a7_hierarchy_integrity(nodes_df, edges_df):
    """
    Check hierarchy invariants:
    1. Every non-root node has exactly one parent_of incoming edge
    2. No cycles in parent_of edges
    3. All follows edges are DAG (no cycles)
    4. Follows edges connect same-type nodes at paragraph level
    """
```

---

## 7. Schema Reference

### 7.1 Edge Fields

| Field | Type | Description |
|-------|------|-------------|
| `edge_id` | str | Unique ID: `e_{type}_{hash}` |
| `source_node_id` | str | Fully qualified: `doc_id:anchor_id` |
| `target_node_id` | str | Fully qualified |
| `edge_type` | str | `parent_of`, `follows`, `in_section`, etc. |
| `direction` | str | `directed` or `bidirectional` |
| `confidence` | float | 1.0 for structural edges |
| `source_evidence` | str | Human-readable explanation |
| `created_by` | str | `structural`, `regex`, `llm` |

### 7.2 Node Fields (Relevant for Hierarchy)

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | str | `doc_id:anchor_id` or `doc_id:el_{element_id}` |
| `node_type` | str | `doc_root`, `section`, `box_section`, `concept`, `paragraph` |
| `anchor_type` | str | `preamble`, `section`, `box`, `subsection` |
| `anchor_id` | str | Anchor identifier |
| `pages` | list[int] | Page numbers |
| `bbox` | list[float] | [x0, y0, x1, y1] |
| `reading_order` | float | For paragraphs: order within anchor |

---

## 8. Query Examples

### 8.1 Get All Children of a Node

```python
def get_children(edges_df, parent_node_id):
    """Get immediate children via parent_of edges."""
    mask = (edges_df["edge_type"] == "parent_of") & \
           (edges_df["source_node_id"] == parent_node_id)
    return edges_df[mask]["target_node_id"].tolist()
```

### 8.2 Get Reading Order Sequence

```python
def get_reading_sequence(edges_df, start_node_id):
    """Follow follows edges to build reading sequence."""
    sequence = [start_node_id]
    current = start_node_id

    while True:
        mask = (edges_df["edge_type"] == "follows") & \
               (edges_df["source_node_id"] == current)
        next_edges = edges_df[mask]
        if next_edges.empty:
            break
        current = next_edges.iloc[0]["target_node_id"]
        sequence.append(current)

    return sequence
```

### 8.3 Get All Paragraphs Under Box

```python
def get_paragraphs_under_box(nodes_df, edges_df, box_node_id):
    """Get all paragraph nodes under a box."""
    # Get direct children
    mask = (edges_df["edge_type"] == "parent_of") & \
           (edges_df["source_node_id"] == box_node_id)
    children = edges_df[mask]["target_node_id"].tolist()

    # Filter to paragraphs
    para_mask = (nodes_df["node_id"].isin(children)) & \
                (nodes_df["node_type"] == "paragraph")
    return nodes_df[para_mask]
```

---

## 9. Success Criteria

| Criterion | Target | Current |
|-----------|--------|---------|
| Every node has ≤1 parent | 100% | ✅ 100% |
| No orphan section nodes | 100% | ✅ 100% |
| Follows edges form DAG | Yes | ✅ Yes |
| Validation A4 passes | 100% | ✅ 100% |
| Edge count reasonable | 2-3x node count | ✅ 308/122 = 2.5x |

---

## 10. Not Implementing

| Feature | Reason |
|---------|--------|
| `contains_element` edges | Bloat; use element_id field instead |
| Cross-document edges | Single-doc scope for now |
| Bidirectional `follows` | Use directed + reverse query |
| `sibling_of` edges | Derivable from parent_of |

---

*End of Graph Skeleton Specification*
