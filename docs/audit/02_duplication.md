# Duplication Audit

**Generated:** 2026-01-08
**Scope:** `src/vaas/**`, `run_pipeline_v2.py`, `validate_graph.py`

---

## 1. Summary

| Category | Count | Debt Estimate |
|----------|-------|---------------|
| Exact duplicates (≥5 lines) | 2 | Low |
| Near duplicates (same structure) | 5 | Medium |
| Pattern duplicates (try/except, DataFrame ops) | 8 | High |
| **Total** | **15** | **Medium-High** |

---

## 2. Exact Duplicates

### 2.1 Role Constants

**Defined in 2 places:**

| File | Lines |
|------|-------|
| `src/vaas/extraction/elements.py` | 24-34 |
| `src/vaas/graph/nodes.py` | 30-41 |

```python
# elements.py:24-28
ROLE_BOX_HEADER = "BoxHeader"
ROLE_SECTION_HEADER = "SectionHeader"
ROLE_SUBSECTION_HEADER = "SubsectionHeader"
ROLE_LIST_BLOCK = "ListBlock"
ROLE_PAGE_ARTIFACT = "PageArtifact"

# nodes.py:30-34 (identical)
ROLE_BOX_HEADER = "BoxHeader"
ROLE_SECTION_HEADER = "SectionHeader"
...
```

**Fix:** Single source in `vaas/constants.py`

---

## 3. Near Duplicates (Same Structure, Different Literals)

### 3.1 Sort Key Functions

**3 implementations of the same pattern:**

| File | Function | Lines |
|------|----------|-------|
| `src/vaas/graph/edges.py` | `get_sort_key_for_section()` | 153-180 |
| `src/vaas/extraction/geometry.py` | `reading_order_sort_key()` | 177-223 |
| `src/vaas/extraction/merge.py` | inline lambda | 194 |

**Structure:**
```python
def get_sort_key(row):
    pages = row.get("pages")
    if isinstance(pages, (list, tuple, np.ndarray)) and len(pages) > 0:
        page = int(pages[0])
    elif isinstance(pages, (int, float, np.integer)):
        page = int(pages)
    else:
        page = 0
    bbox = row.get("bbox", [0, 0, 0, 0])
    ...
    return (page, col, y0, x0)
```

**Fix:** Use `geometry.reading_order_sort_key()` everywhere

### 3.2 Anchor Type Filtering

**Repeated pattern across 6+ files:**

```python
# Pattern: filter by anchor_type
anchors_df[anchors_df["anchor_type"] == "box"]
anchors_df[anchors_df["anchor_type"] == "section"]
anchors_df[anchors_df["anchor_type"] == "subsection"]
```

| File | Occurrences |
|------|-------------|
| `validate_graph.py` | 2 |
| `src/vaas/extraction/anchors.py` | 3 |
| `src/vaas/extraction/merge.py` | 2 |
| `src/vaas/extraction/references.py` | 1 |
| `src/vaas/graph/edges.py` | 5 |
| `src/vaas/semantic/concept_roles.py` | 1 |

**Fix:** Helper `filter_by_anchor_type(df, "box")` in `vaas/utils/dataframe.py`

### 3.3 Box Key Normalization

**2 similar functions:**

| File | Function |
|------|----------|
| `src/vaas/extraction/anchors.py:91` | `parse_box_keys()` |
| `src/vaas/extraction/references.py:153` | `parse_box_ref_keys()` |

Both parse "1a", "14-16", "1a and 1b" patterns. Different contexts (header vs reference) but 80% overlap.

**Fix:** Extract shared `expand_box_range()` helper

---

## 4. Pattern Duplicates

### 4.1 `isinstance(x, (list, tuple, np.ndarray))` Guard

**15 occurrences** of this defensive pattern:

| File | Lines |
|------|-------|
| `geometry.py` | 49, 142, 169, 208 |
| `edges.py` | 164, 172, 183 |
| `merge.py` | 171, 185 |
| `serialize.py` | 152 |

```python
# Pattern
if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
    return float(bbox[0]), ...
```

**Fix:** Already extracted to `geometry.safe_bbox()` and `serialize.normalize_bbox()` — but not consistently used

### 4.2 DataFrame `sort_values().reset_index()` Chain

**14 occurrences:**

| File | Lines |
|------|-------|
| `lines.py` | 303 |
| `merge.py` | 194 |
| `anchors.py` | 709 |
| `edges.py` | 223, 303, 404, 451 |

```python
df = df.sort_values(["page", "reading_order"]).reset_index(drop=True)
```

**Verdict:** Acceptable — idiomatic pandas. Not worth extracting.

### 4.3 DataFrame `groupby().agg()` Pattern

**4 identical aggregations in `lines.py`:**

| Lines | Purpose |
|-------|---------|
| 107-130 | Line aggregation |
| 140-155 | Block geometry |
| 159-172 | Page geometry |
| 280-290 | Page Y bounds |

```python
# Pattern (repeated 4x with different columns)
df.groupby(["doc_id", "page", "block_id"], as_index=False).agg(
    geom_x0=("x0", "min"),
    geom_y0=("y0", "min"),
    geom_x1=("x1", "max"),
    geom_y1=("y1", "max"),
)
```

**Fix:** Extract `aggregate_bbox(df, group_cols)` helper

### 4.4 Try/Except Float Extraction

**Same pattern in 3 functions in `geometry.py`:**

```python
# safe_bbox() lines 48-53
try:
    if isinstance(bbox, ...) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), ...
    return np.nan, np.nan, np.nan, np.nan
except (TypeError, ValueError, IndexError):
    return np.nan, np.nan, np.nan, np.nan

# bbox_y0() lines 141-147 (identical structure)
# bbox_x0() lines 168-174 (identical structure)
```

**Fix:** Already factored — but could use `safe_bbox()` internally

### 4.5 Page Extraction from `pages` Field

**6 occurrences of defensive page extraction:**

| File | Lines |
|------|-------|
| `geometry.py` | 206-213 |
| `edges.py` | 164-168, 183-187 |
| `merge.py` | 97-105 |
| `serialize.py` | 147-157 |

```python
# Pattern
if isinstance(pages, (list, tuple, np.ndarray)) and len(pages) > 0:
    page = int(pages[0])
elif isinstance(pages, (int, float)):
    page = int(pages)
else:
    page = 0
```

**Fix:** Use `serialize.safe_extract_page()` consistently

---

## 5. Duplication Debt by Priority

| Priority | Issue | Occurrences | Fix | Effort |
|----------|-------|-------------|-----|--------|
| **P1** | Role constants in 2 places | 2 | Create `vaas/constants.py` | Low |
| **P1** | Sort key functions (3 versions) | 3 | Use `geometry.reading_order_sort_key()` | Low |
| **P2** | Box parsing overlap | 2 | Extract `expand_box_range()` | Medium |
| **P2** | Page extraction pattern | 6 | Use `safe_extract_page()` | Low |
| **P2** | Bbox guard pattern | 15 | Consistent use of `safe_bbox()` | Medium |
| **P3** | `groupby().agg()` bbox patterns | 4 | Extract helper (optional) | Low |
| **P3** | Anchor type filtering | 14 | Helper function (optional) | Low |

---

## 6. Proposed Extractions

### 6.1 `vaas/constants.py` (NEW)

```python
"""Shared constants across vaas modules."""

# Role constants
ROLE_BOX_HEADER = "BoxHeader"
ROLE_SECTION_HEADER = "SectionHeader"
ROLE_SUBSECTION_HEADER = "SubsectionHeader"
ROLE_LIST_BLOCK = "ListBlock"
ROLE_PAGE_ARTIFACT = "PageArtifact"
ROLE_BODY_TEXT = "BodyTextBlock"

# Anchor types
ANCHOR_TYPE_BOX = "box"
ANCHOR_TYPE_SECTION = "section"
ANCHOR_TYPE_SUBSECTION = "subsection"
ANCHOR_TYPE_PREAMBLE = "preamble"

# Header roles set (for filtering)
HEADER_ROLES = {ROLE_BOX_HEADER, ROLE_SECTION_HEADER, ROLE_SUBSECTION_HEADER}
SKIP_ROLES = {ROLE_PAGE_ARTIFACT}
```

### 6.2 `vaas/utils/dataframe.py` (NEW)

```python
"""DataFrame utility functions."""

def filter_by_anchor_type(
    df: pd.DataFrame,
    anchor_type: str,
) -> pd.DataFrame:
    """Filter DataFrame by anchor_type column."""
    if df.empty or "anchor_type" not in df.columns:
        return pd.DataFrame()
    return df[df["anchor_type"] == anchor_type].copy()


def aggregate_bbox(
    df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """Aggregate bbox coordinates by group."""
    return df.groupby(group_cols, as_index=False).agg(
        geom_x0=("geom_x0", "min"),
        geom_y0=("geom_y0", "min"),
        geom_x1=("geom_x1", "max"),
        geom_y1=("geom_y1", "max"),
    )
```

### 6.3 Enhance `vaas/extraction/anchors.py`

```python
def expand_box_range(start: str, end: str) -> List[str]:
    """
    Expand a box range like "14"-"16" to ["14", "15", "16"].

    Shared between parse_box_keys() and parse_box_ref_keys().
    """
    try:
        lo = int(re.match(r"(\d+)", start).group(1))
        hi = int(re.match(r"(\d+)", end).group(1))
        return [str(k) for k in range(min(lo, hi), max(lo, hi) + 1)]
    except (ValueError, AttributeError):
        return [start, end]
```

---

## 7. Migration Notes

1. **Role constants migration:**
   - Create `vaas/constants.py`
   - Update imports in `elements.py`, `nodes.py`, `edges.py`
   - Delete duplicates

2. **Sort key consolidation:**
   - `edges.py:get_sort_key_for_section()` → use `geometry.reading_order_sort_key()`
   - `merge.py` inline lambda → use geometry function

3. **Page extraction:**
   - `serialize.safe_extract_page()` already exists
   - Replace 6 inline occurrences with calls to it

---

## 8. Not Worth Extracting

| Pattern | Reason |
|---------|--------|
| `sort_values().reset_index()` | Idiomatic pandas, clear intent |
| `isinstance(x, pd.DataFrame)` checks | Too context-specific |
| `f"{doc_id}:{anchor_id}"` ID construction | Trivial, clear |

---

*End of Duplication Audit*
