# SRP Audit: run_pipeline_v2.py

**File:** `run_pipeline_v2.py`
**LOC:** 429
**Generated:** 2026-01-08

---

## 1. Executive Summary

`run_pipeline_v2.py` is a thin orchestrator that delegates to modular components. Overall architecture is sound, but there are a few SRP concerns worth addressing.

**Status:** ✅ PASS with minor issues

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Max function LOC | 69 | <100 | ✅ |
| Category mixing | 1 function | 0 | ⚠️ |
| Inline logic | Minimal | None | ✅ |

---

## 2. Function Audit

| Function | LOC | Category | Issue |
|----------|-----|----------|-------|
| `PipelineConfig.for_1099div()` | 7 | Config | - |
| `stage_1_extract_spans()` | 11 | IO | - |
| `stage_2_infer_body_font()` | 10 | Transform | - |
| `stage_3_build_lines()` | 41 | Transform | - |
| `stage_4_split_and_classify()` | 18 | Transform | - |
| `stage_5_extract_anchors()` | 36 | Extract + **Validate** | ⚠️ Mixed |
| `stage_6_materialize_sections()` | 33 | Transform | - |
| `stage_7_extract_references()` | 22 | Extract | - |
| `stage_8_build_graph()` | 30 | Graph-build | - |
| `run_pipeline()` | 69 | Orchestration | - |
| `main()` | 23 | CLI | - |

---

## 3. Issues Found

### 3.1 `stage_5_extract_anchors()` mixes extraction and validation

**Lines 187-222**

```python
def stage_5_extract_anchors(elements_df, expected_boxes):
    extraction = extract_anchors(...)           # Extraction

    if expected_boxes and not anchors_df.empty:
        validation = validate_box_coverage(...) # <-- Validation concern
        coverage = len(validation.found) / ...
        if validation.missing:
            print(f"Missing boxes: ...")
```

**Problem:** Validation logic embedded in extraction stage.

**Fix:** Move validation to `run_pipeline()` or separate stage:

```python
def stage_5_extract_anchors(elements_df):
    """Pure extraction - no validation."""
    extraction = extract_anchors(...)
    return extraction.anchors_df, elements_df

# In run_pipeline():
anchors_df, elements_df = stage_5_extract_anchors(elements_df)
if config.expected_boxes:
    validation = validate_box_coverage(anchors_df, config.expected_boxes)
    _report_coverage(validation)
```

### 3.2 `run_pipeline()` returns dict instead of typed result

**Lines 373-388**

```python
results = {
    "spans": len(spans_df),
    "lines": len(line_df),
    ...
}
return results
```

**Problem:** Untyped dict loses DataFrames, only returns counts.

**Fix:** Return a typed dataclass:

```python
@dataclass
class PipelineResult:
    spans_df: pd.DataFrame
    elements_df: pd.DataFrame
    anchors_df: pd.DataFrame
    sections_df: pd.DataFrame
    references_df: pd.DataFrame
    graph_nodes: pd.DataFrame
    graph_edges: pd.DataFrame
```

### 3.3 Hardcoded doc_label in `stage_8_build_graph()`

**Line 300**

```python
graph_nodes, paragraph_nodes_df = build_nodes_legacy(
    ...
    doc_label="1099-DIV Filer Instructions",  # <-- Hardcoded
)
```

**Problem:** Should come from config, not hardcoded.

**Fix:** Add `doc_label` to `PipelineConfig`:

```python
@dataclass
class PipelineConfig:
    ...
    doc_label: str = "Document"
```

### 3.4 Local import inside `stage_5_extract_anchors()` and `stage_6_materialize_sections()`

**Lines 196-199, 235**

```python
def stage_5_extract_anchors(...):
    from vaas.extraction import (
        ROLE_BOX_HEADER, ROLE_SECTION_HEADER, ...
    )
```

**Problem:** Local imports obscure dependencies.

**Fix:** Move to top-level imports or pass as parameters.

---

## 4. Category Distribution

| Category | Stages | Assessment |
|----------|--------|------------|
| Config | PipelineConfig | Clean |
| IO | stage_1, run_pipeline (save) | Clean |
| Transform | stage_2, stage_3, stage_4, stage_6 | Clean |
| Extract | stage_5, stage_7 | ⚠️ stage_5 mixed |
| Graph-build | stage_8 | Clean |
| Orchestration | run_pipeline, main | Clean |

---

## 5. Recommendations

| Priority | Issue | Fix | Effort |
|----------|-------|-----|--------|
| P1 | stage_5 mixes validation | Extract to run_pipeline() | Low |
| P2 | Untyped return | Add PipelineResult dataclass | Low |
| P2 | Hardcoded doc_label | Add to PipelineConfig | Trivial |
| P3 | Local imports | Move to top-level | Trivial |

---

## 6. What's Good

- Clear stage separation (1-8)
- Each stage is a thin wrapper calling module functions
- No business logic in orchestrator
- CLI is minimal
- Config is a dataclass

---

## 7. Proposed Refactor

```python
# Fix 3.1: Pure extraction stage
def stage_5_extract_anchors(elements_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract anchors. No validation."""
    extraction = extract_anchors(elements_df, ...)
    timeline = build_anchor_timeline(extraction.anchors_df, elements_df)
    elements_df = assign_elements_to_anchors(elements_df, timeline)
    return extraction.anchors_df, elements_df


# Fix 3.2: Typed result
@dataclass
class PipelineResult:
    graph_nodes: pd.DataFrame
    graph_edges: pd.DataFrame
    sections_df: pd.DataFrame
    anchors_df: pd.DataFrame
    counts: Dict[str, int]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    ...
    # Validation separate from extraction
    anchors_df, elements_df = stage_5_extract_anchors(elements_df)
    if config.expected_boxes:
        _validate_coverage(anchors_df, config.expected_boxes)
    ...
    return PipelineResult(
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        sections_df=sections_df,
        anchors_df=anchors_df,
        counts={...},
    )
```

---

*End of SRP Audit*
