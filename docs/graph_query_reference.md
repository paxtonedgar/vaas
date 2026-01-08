# Graph Query Reference

This document details the queries used to evaluate the VaaS knowledge graph and their results.

## Graph Structure

The graph is stored as two Parquet files:
- `output/graph_nodes.parquet` - Node definitions
- `output/graph_edges.parquet` - Edge relationships

### Loading the Graph

```python
import pandas as pd

nodes = pd.read_parquet('output/graph_nodes.parquet')
edges = pd.read_parquet('output/graph_edges.parquet')
```

### Node Schema

| Column | Type | Description |
|--------|------|-------------|
| `node_id` | str | Unique identifier (e.g., `1099div_filer:box_1a`) |
| `doc_id` | str | Document identifier |
| `node_type` | str | One of: `doc_root`, `preamble`, `section`, `concept`, `box_section`, `paragraph` |
| `anchor_id` | str | Anchor identifier within document |
| `box_key` | str | Box number if applicable (e.g., `1a`, `2e`) |
| `label` | str | Human-readable label/header |
| `text` | str | Full text content |
| `pages` | list | Page numbers where content appears |
| `element_count` | int | Number of PDF elements |
| `char_count` | int | Character count |
| `concept_role` | str | Role classification (`definition`, `condition`, etc.) |
| `paragraph_kind` | str | `body` or `list` for paragraph nodes |
| `anchor_type` | str | `box`, `section`, `subsection`, `preamble` |

### Edge Schema

| Column | Type | Description |
|--------|------|-------------|
| `edge_id` | str | Unique identifier |
| `source_node_id` | str | Source node reference |
| `target_node_id` | str | Target node reference |
| `edge_type` | str | Relationship type |
| `direction` | str | `directed` or `bidirectional` |
| `confidence` | float | Confidence score (0.0-1.0) |
| `source_evidence` | str | Text snippet supporting the edge |
| `source_element_id` | str | PDF element ID for provenance |
| `created_by` | str | `structural` or `regex` |
| `pattern_matched` | str | Regex pattern name that matched |
| `polarity` | str | `positive` or `negative` |

### Edge Types

**Structural (Skeleton):**
- `parent_of` - Hierarchical containment
- `follows` - Reading order sequence
- `in_section` - Section membership

**Reference:**
- `references_box` - Cross-reference to another box
- `same_group` - Grouped box relationship (e.g., boxes 14-16)

**Semantic:**
- `includes` - Box containment (e.g., "Box 1a includes amounts in box 1b")
- `excludes` - Negation/exception (e.g., "Do not include in box 1a")
- `defines` - Semantic definition
- `applies_if` - Conditional applicability
- `requires` - Computational dependency
- `qualifies` - Scope/constraint

---

## Diagnostic Queries

### Q1: Find Orphan Nodes

Nodes not connected to any edge (neither source nor target).

```python
all_node_ids = set(nodes['node_id'].tolist())
sources = set(edges['source_node_id'].tolist())
targets = set(edges['target_node_id'].tolist())
connected = sources | targets
orphans = all_node_ids - connected

print(f'Orphan nodes: {len(orphans)}')
for o in orphans:
    print(f'  - {o}')
```

**Result:**
```
Orphan nodes: 1
  - 1099div_filer:el_1099div_filer:1:2:seg0
```

### Q2: Count Skeleton Edges

Structural edges that form the document skeleton.

```python
skeleton_types = {'parent_of', 'follows', 'in_section'}
skeleton_edges = edges[edges['edge_type'].isin(skeleton_types)]

print(f'Total skeleton edges: {len(skeleton_edges)}')
for et in skeleton_types:
    count = len(edges[edges['edge_type'] == et])
    print(f'  - {et}: {count}')
```

**Result:**
```
Total skeleton edges: 239
  - parent_of: 120
  - follows: 97
  - in_section: 22
```

### Q3: Count Connected Components

Uses union-find to determine graph connectivity.

```python
parent = {n: n for n in all_node_ids}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

for _, row in edges.iterrows():
    s, t = row['source_node_id'], row['target_node_id']
    if s in parent and t in parent:
        union(s, t)

components = len(set(find(n) for n in all_node_ids))
print(f'Connected components: {components}')
```

**Result:**
```
Connected components: 2
```

### Q4: Find Self-Edges

Edges where source equals target (should be 0).

```python
self_edges = edges[edges['source_node_id'] == edges['target_node_id']]
print(f'Self-edges: {len(self_edges)}')
```

**Result:**
```
Self-edges: 0
```

### Q5: Check Provenance Coverage

Edges with source_element_id for traceability.

```python
has_provenance = edges['source_element_id'].notna().sum()
total = len(edges)
print(f'Provenance: {has_provenance}/{total} ({100*has_provenance/total:.1f}%)')
```

**Result:**
```
Provenance: 184/316 (58.2%)
```

---

## Semantic Edge Queries

### Get Semantic Edges for a Box

Helper function to find semantic relationships involving a specific box.

```python
def get_semantic_edges(box_key, edge_type):
    """
    Get edges of a specific type involving a box.

    Args:
        box_key: Box identifier (e.g., '1a', '2e')
        edge_type: Edge type (e.g., 'includes', 'excludes')

    Returns:
        List of tuples: (direction, related_box, evidence)
    """
    box_node_ids = nodes[nodes['box_key'] == box_key]['node_id'].tolist()
    results = []

    for nid in box_node_ids:
        # Outgoing edges (this box -> other)
        out_edges = edges[
            (edges['source_node_id'] == nid) &
            (edges['edge_type'] == edge_type)
        ]
        for _, e in out_edges.iterrows():
            target_box = nodes[nodes['node_id'] == e['target_node_id']]['box_key'].values
            if len(target_box) > 0 and pd.notna(target_box[0]):
                results.append(('→', target_box[0], e.get('source_evidence', '')))

        # Incoming edges (other -> this box)
        in_edges = edges[
            (edges['target_node_id'] == nid) &
            (edges['edge_type'] == edge_type)
        ]
        for _, e in in_edges.iterrows():
            source_box = nodes[nodes['node_id'] == e['source_node_id']]['box_key'].values
            if len(source_box) > 0 and pd.notna(source_box[0]):
                results.append(('←', source_box[0], e.get('source_evidence', '')))

    return results
```

### Query: Box 1a Includes Which Boxes?

```python
includes_1a = get_semantic_edges('1a', 'includes')
print(f'Box 1a includes edges: {len(includes_1a)}')
for direction, box, evidence in includes_1a:
    print(f'  {direction} Box {box}')
    print(f'    Evidence: {evidence[:80]}...')
```

**Result:**
```
Box 1a includes edges: 2
  → Box 1b
    Evidence: ...nds paid directly from the corporation. Box 1a includes amounts entered in boxe...
  → Box 2e
    Evidence: ...nds paid directly from the corporation. Box 1a includes amounts entered in boxe...
```

### Query: Box 1b Conditions (applies_if)

```python
applies_if_1b = get_semantic_edges('1b', 'applies_if')
print(f'Box 1b applies_if edges: {len(applies_if_1b)}')
for direction, box, evidence in applies_if_1b:
    print(f'  {direction}')
    print(f'    Evidence: {evidence[:80]}...')
```

**Result:**
```
Box 1b applies_if edges: 1
  ←
    Evidence: ...an 60 days before the ex-dividend date. See the instructions for box 1b, later...
```

### Query: All Excludes Relationships

```python
excludes_edges = edges[edges['edge_type'] == 'excludes']
print(f'Total excludes edges: {len(excludes_edges)}')

for _, e in excludes_edges.iterrows():
    source = e['source_node_id'].split(':')[-1][:40]
    target_box = nodes[nodes['node_id'] == e['target_node_id']]['box_key'].values
    target = target_box[0] if len(target_box) > 0 else 'N/A'
    evidence = e.get('source_evidence', '')[:60]
    print(f'  {source} → Box {target}')
    print(f'    Evidence: {evidence}...')
```

**Result:**
```
Total excludes edges: 5
  sub_rics_special_reporting_instruc_5af5 → Box 1a
    Evidence: ...dation. Do not include these amounts in box 1a or 1b. ! C...
  sub_rics_special_reporting_instruc_5af5 → Box 1b
    Evidence: ...dation. Do not include these amounts in box 1a or 1b. ! C...
  box_5 → Box 1a
    Evidence: ...t is included in the amount reported in box 1a. Include R...
  box_6 → Box 1a
    Evidence: ...tion 67(c) and must also be included in box 1a. Do not in...
  box_6 → Box 1b
    Evidence: ...not include any investment expenses in box 1b....
```

---

## N-Pair Mining Test Queries

### Q1: Box 1a Includes Which Other Boxes?

```python
includes_1a = get_semantic_edges('1a', 'includes')
print(f'Actual: Box 1a includes {[x[1] for x in includes_1a]}')
print(f'Expected: [1b, 2a, 2b, 2e]')
```

**Result:**
```
Actual: Box 1a includes ['1b', '2e']
Expected: [1b, 2a, 2b, 2e]
Coverage: PARTIAL (document text explicitly mentions 1b and 2e)
```

### Q2: Box 2a Subset Of Which Boxes?

```python
includes_2a = get_semantic_edges('2a', 'includes')
incoming = [x for x in includes_2a if x[0] == '←']
print(f'Incoming includes to 2a: {incoming}')
print(f'Expected: 1a')
```

**Result:**
```
Incoming includes to 2a: []
Expected: 1a
Note: No explicit "Box 1a includes 2a" text found in document
```

### Q3: Box 1b Holding Period Condition

```python
applies_if_1b = get_semantic_edges('1b', 'applies_if')
print(f'Conditions for Box 1b: {len(applies_if_1b)} edges')
print(f'Expected: 61-day holding period within 121-day window')
```

**Result:**
```
Conditions for Box 1b: 1 edges
  Evidence: "...60 days before the ex-dividend date. See the instructions for box 1b..."
Status: CAPTURED via see_instructions_for_box pattern
```

### Q4: Box 3 Relationships

```python
includes_3 = get_semantic_edges('3', 'includes')
excludes_3 = get_semantic_edges('3', 'excludes')
print(f'Box 3 includes: {len(includes_3)}, excludes: {len(excludes_3)}')
print(f'Expected: None (Box 3 is nontaxable distributions)')
```

**Result:**
```
Box 3 includes: 0, excludes: 0
Expected: None
Status: CORRECT
```

### Q7: All Holding Period Rules

```python
applies_if = edges[edges['edge_type'] == 'applies_if']
print(f'Holding period / condition edges: {len(applies_if)}')

for _, e in applies_if.iterrows():
    target_box = nodes[nodes['node_id'] == e['target_node_id']]['box_key'].values
    target = target_box[0] if len(target_box) > 0 else 'N/A'
    pattern = e.get('pattern_matched', '')
    evidence = e.get('source_evidence', '')[:60]
    print(f'  → Box {target} ({pattern})')
    print(f'    {evidence}...')
```

**Result:**
```
Holding period / condition edges: 3
  → Box 8 (see_instructions_for_box)
    ...t of a loan of a customer's securities, see the instructi...
  → Box 1b (see_instructions_for_box)
    ...an 60 days before the ex-dividend date. See the instructi...
  → Box 13 (see_instructions_for_box)
    ...in box 13 and in the total for box 12. See the instructio...
```

---

## Edge Type Distribution Query

```python
print('Edge Type Distribution:')
for et in sorted(edges['edge_type'].unique()):
    count = len(edges[edges['edge_type'] == et])
    pct = 100 * count / len(edges)
    print(f'  {et}: {count} ({pct:.1f}%)')
```

**Result:**
```
Edge Type Distribution:
  applies_if: 3 (0.9%)
  defines: 3 (0.9%)
  excludes: 5 (1.6%)
  follows: 97 (30.7%)
  in_section: 22 (7.0%)
  includes: 2 (0.6%)
  parent_of: 120 (38.0%)
  references_box: 61 (19.3%)
  same_group: 3 (0.9%)
```

---

## Summary Statistics

```python
print(f'Graph Statistics:')
print(f'  Nodes: {len(nodes)}')
print(f'  Edges: {len(edges)}')
print(f'  Orphans: {len(orphans)}')
print(f'  Components: {components}')
print(f'  Self-edges: {len(self_edges)}')

semantic_types = {'includes', 'excludes', 'applies_if', 'defines', 'requires', 'qualifies'}
semantic_count = len(edges[edges['edge_type'].isin(semantic_types)])
print(f'  Semantic edges: {semantic_count}')
```

**Result:**
```
Graph Statistics:
  Nodes: 122
  Edges: 316
  Orphans: 1
  Components: 2
  Self-edges: 0
  Semantic edges: 13
```
