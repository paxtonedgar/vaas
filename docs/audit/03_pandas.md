# Pandas Correctness & Performance Audit

**Generated:** 2026-01-08
**Scope:** `src/vaas/**`, `validate_graph.py`

---

## 1. Summary

| Issue Type | Count | Severity |
|------------|-------|----------|
| `iterrows()` usage | 36 | Medium |
| Nested `iterrows()` | 1 | **High** |
| Chained assignment | 0 | ✅ OK |
| SettingWithCopy risk | 3 | Low |
| `fillna` dtype warnings | 0 | ✅ Fixed |
| Repeated merge patterns | 3 | Low |
| `.copy()` defensiveness | 27 | ✅ OK |

---

## 2. `iterrows()` Usage Analysis

### 2.1 All `iterrows()` Occurrences

| File | Count | Acceptable? |
|------|-------|-------------|
| `graph/edges.py` | 7 | ⚠️ Check |
| `graph/nodes.py` | 3 | ⚠️ Check |
| `extraction/anchors.py` | 6 | **❌ Critical** |
| `extraction/references.py` | 2 | ✅ OK |
| `extraction/sections.py` | 2 | ✅ OK |
| `extraction/merge.py` | 2 | ✅ OK |
| `extraction/elements.py` | 1 | ✅ OK (debug print) |
| `semantic/pair_generation.py` | 1 | ✅ OK |
| `semantic/concept_roles.py` | 0 | ✅ OK |
| `validate_graph.py` | 12 | ⚠️ Check |

### 2.2 Critical: Nested `iterrows()` in `anchors.py`

**Location:** `src/vaas/extraction/anchors.py:806-822`

```python
# BEFORE: O(n*m) nested iteration
for idx, row in page_elements.iterrows():      # O(n)
    ro = row["reading_order"]
    for _, anchor in page_anchors.iterrows():  # O(m)
        start = anchor["start_reading_order"]
        end = anchor["end_reading_order"]
        if start <= ro <= end:
            elements_df.loc[idx, "anchor_id"] = anchor_id
            break
```

**Problem:**
- For a page with 100 elements and 10 anchors: 1,000 iterations
- Total across 12 pages: ~12,000 iterations
- Each `.loc[]` assignment is another DataFrame access

**AFTER: Vectorized interval assignment using `pd.IntervalIndex`**

```python
def assign_elements_to_anchors_vectorized(
    elements_df: pd.DataFrame,
    anchor_timeline: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorized anchor assignment using IntervalIndex."""
    elements_df = elements_df.copy()
    elements_df["anchor_id"] = "unassigned"

    for page in elements_df["page"].unique():
        page_mask = elements_df["page"] == page
        page_anchors = anchor_timeline[anchor_timeline["page"] == page]

        if page_anchors.empty:
            elements_df.loc[page_mask, "anchor_id"] = "preamble"
            continue

        # Build IntervalIndex from anchor ranges
        intervals = pd.IntervalIndex.from_arrays(
            page_anchors["start_reading_order"].values,
            page_anchors["end_reading_order"].values,
            closed="both",
        )
        anchor_ids = page_anchors["anchor_id"].values

        # Vectorized lookup
        page_ro = elements_df.loc[page_mask, "reading_order"].values

        # Find which interval each reading_order falls into
        # Returns -1 for no match
        idx_matches = intervals.get_indexer(page_ro)

        # Assign anchor_ids (vectorized)
        assigned = np.where(
            idx_matches >= 0,
            anchor_ids[idx_matches],
            "unassigned"
        )
        elements_df.loc[page_mask, "anchor_id"] = assigned

    return elements_df
```

**Expected speedup:** 10-50x for typical documents

### 2.3 Acceptable `iterrows()` Patterns

These are acceptable because they build new data structures (not modifying in-place):

```python
# Building a list of new objects - OK
nodes = []
for _, section in sections_df.iterrows():
    nodes.append(Node(...))

# Debug printing - OK
for _, r in box_headers.head(25).iterrows():
    print(f"  {r['box_key']}: {r['label'][:40]}...")
```

---

## 3. Chained Assignment Analysis

### ✅ No Chained Assignment Found

The codebase correctly uses `.loc[]` for all assignments:

```python
# CORRECT patterns found:
df.loc[mask, "column"] = value        # Direct loc assignment
df.loc[idx, "anchor_id"] = anchor_id  # Single-cell loc
df.at[idx, "anchor_ids"] = value      # Single-cell at
```

### 3.1 Potential SettingWithCopy Risks

These patterns are safe because `.copy()` is called:

| File | Line | Pattern | Safe? |
|------|------|---------|-------|
| `anchors.py:462` | `subsections = anchors_df[...].copy()` | ✅ |
| `anchors.py:790` | `page_anchors = anchor_timeline[...].copy()` | ✅ |
| `sections.py:353` | `anchor_elements = elements_df.loc[...].copy()` | ✅ |

### 3.2 Recommendation: Add `SettingWithCopyWarning` check

```python
# In tests or CI
import pandas as pd
pd.options.mode.chained_assignment = "raise"  # Fail fast on issues
```

---

## 4. `fillna` Dtype Downcast Warnings

### ✅ Already Fixed

**Location:** `src/vaas/extraction/layout_detection.py:62-64`

```python
# CORRECT pattern (already implemented)
df["next_bold"] = df["_next_bold_raw"].astype("boolean").fillna(False).astype(bool)
df["next_char_count"] = df["_next_char_count_raw"].astype("Int64").fillna(0).astype(int)
```

This pattern uses nullable dtypes to avoid the FutureWarning:
1. Cast to nullable dtype (`boolean`, `Int64`)
2. `fillna()` with appropriate default
3. Cast back to native dtype (`bool`, `int`)

---

## 5. Merge/GroupBy Consolidation Opportunities

### 5.1 Repeated Merges in `lines.py`

**Current:** 3 sequential merges

```python
# lines.py:144
df = df.merge(block_geom, on=["doc_id", "page", "block_id"], how="left")

# lines.py:165
df = df.merge(page_geom[...], on=["doc_id", "page"], how="left")

# lines.py:285
df = df.merge(page_y_bounds, on=["doc_id", "page"], how="left")
```

**Opportunity:** Combine page-level merges

```python
# Consolidate page_geom and page_y_bounds computation
page_stats = df.groupby(["doc_id", "page"], as_index=False).agg(
    page_x0=("geom_x0", "min"),
    page_y0=("geom_y0", "min"),
    page_x1=("geom_x1", "max"),
    page_y1=("geom_y1", "max"),
    page_mid_x=("geom_x0", lambda x: (x.min() + x.max()) / 2),
    min_y0=("geom_y0", "min"),
    max_y1=("geom_y1", "max"),
)
df = df.merge(page_stats, on=["doc_id", "page"], how="left")
```

**Impact:** Minor (3 merges → 2, ~10% faster on merge-heavy paths)

### 5.2 Repeated `groupby().agg()` Patterns

**Location:** `lines.py:107-172`

Four similar aggregations:
1. Line aggregation (spans → lines)
2. Block geometry
3. Page geometry
4. Page Y bounds

**Verdict:** Keep separate — each serves a distinct pipeline stage. Consolidating would reduce clarity.

---

## 6. Performance Hotspots

### 6.1 High Impact

| Location | Issue | Fix | Impact |
|----------|-------|-----|--------|
| `anchors.py:806-822` | Nested iterrows | IntervalIndex | **10-50x** |

### 6.2 Medium Impact

| Location | Issue | Fix | Impact |
|----------|-------|-----|--------|
| `edges.py:229-260` | iterrows for hierarchy | Vectorize parent assignment | 2-5x |
| `nodes.py:348-436` | iterrows for paragraphs | `df.apply()` with axis=1 | 2-3x |
| `validate_graph.py` | 12 iterrows | Vectorize checks | 2-3x |

### 6.3 Low Impact (Not Worth Optimizing)

| Location | Issue | Why OK |
|----------|-------|--------|
| `references.py:465` | iterrows for extraction | Builds complex objects |
| `sections.py:284` | iterrows for assignment | Single pass, small N |
| `concept_roles.py` | iterrows for classification | Calls external function |

---

## 7. Best Practices Checklist

| Practice | Status | Notes |
|----------|--------|-------|
| Use `.loc[]` for assignment | ✅ | Consistently applied |
| Call `.copy()` after slice | ✅ | 27 occurrences |
| Avoid chained indexing | ✅ | None found |
| Use nullable dtypes for fillna | ✅ | Already implemented |
| Avoid nested iterrows | ❌ | 1 critical instance |
| Use vectorized operations | ⚠️ | Room for improvement |

---

## 8. Recommended Fixes (Prioritized)

### P1: Fix Nested `iterrows()` in `anchors.py`

```python
# File: src/vaas/extraction/anchors.py
# Replace lines 806-822 with IntervalIndex-based assignment
```

**Effort:** Medium (30-60 min)
**Impact:** High (10-50x speedup for anchor assignment)

### P2: Vectorize Edge Building in `edges.py`

```python
# Current: iterrows to build hierarchy edges
for _, row in ordered.iterrows():
    if anchor_type == "section":
        parent_id = doc_root_id
    elif anchor_type == "box":
        parent_id = ...

# Better: Use np.select for vectorized parent assignment
conditions = [
    ordered["anchor_type"] == "section",
    ordered["anchor_type"] == "box",
    ordered["anchor_type"] == "subsection",
]
choices = [
    doc_root_id,
    ordered["current_section_id"],
    ordered["current_box_id"].fillna(ordered["current_section_id"]),
]
ordered["parent_id"] = np.select(conditions, choices, default=doc_root_id)
```

**Effort:** High (requires tracking "current" state)
**Impact:** Medium (2-5x speedup)

### P3: Add CI Check for SettingWithCopy

```yaml
# In pytest conftest.py or CI setup
import pandas as pd
pd.options.mode.chained_assignment = "raise"
```

**Effort:** Low (5 min)
**Impact:** Prevents future bugs

---

## 9. Code Samples

### 9.1 Safe `iterrows()` Pattern (Current)

```python
# Building list of objects - safe and readable
edges = []
for _, row in sections_df.iterrows():
    edges.append({
        "source_node_id": f"{doc_id}:{row['anchor_id']}",
        "target_node_id": parent_id,
        "edge_type": "parent_of",
    })
graph_edges = pd.DataFrame(edges)
```

### 9.2 Vectorized Alternative (When Applicable)

```python
# When logic is simple and vectorizable
graph_edges = pd.DataFrame({
    "source_node_id": doc_id + ":" + sections_df["anchor_id"],
    "target_node_id": parent_id,
    "edge_type": "parent_of",
})
```

---

## 10. Summary

**Critical Fix:** Nested `iterrows()` in `anchors.py:806-822`

**Already Good:**
- No chained assignment
- Proper `.copy()` usage
- `fillna` dtype warnings fixed

**Optional Improvements:**
- Vectorize edge building (medium effort)
- Consolidate page-level merges (low effort)
- Add SettingWithCopy CI check (trivial)

---

*End of Pandas Audit*
