# Config & Hardcoded Values Audit

**Generated:** 2026-01-08
**Goal:** Make Databricks + local runs consistent

---

## 1. Current State

### 1.1 Duplicate Config Classes

**Problem:** Two separate `PipelineConfig` classes exist:

| Location | LOC | Used By |
|----------|-----|---------|
| `src/vaas/config.py:15-148` | 134 | Not imported anywhere |
| `run_pipeline_v2.py:72-93` | 22 | Active pipeline |

**Impact:** The comprehensive config in `src/vaas/config.py` is dead code. `run_pipeline_v2.py` has a minimal config that doesn't include all settings.

### 1.2 Hardcoded Values Summary

| Category | Count | Severity |
|----------|-------|----------|
| File paths | 4 | **High** |
| Page geometry | 5 | Medium |
| Thresholds (merge) | 4 | Medium |
| Thresholds (validation) | 3 | Medium |
| Confidence values | 12 | Low |
| Column detection | 5 | Low |
| Layout detection | 4 | Low |
| Document metadata | 3 | **High** |

---

## 2. Complete Hardcoded Values Inventory

### 2.1 File Paths (High Priority)

| File | Line | Value | Context |
|------|------|-------|---------|
| `run_pipeline_v2.py` | 84, 401 | `"data/i1099div.pdf"` | Default PDF path |
| `run_pipeline_v2.py` | 85, 405 | `"output"` | Default output dir |
| `config.py` | 90, 123 | `"data/i1099div.pdf"` | Factory default |
| `legacy/run_pipeline.py` | 48 | `"data/i1099div.pdf"` | Legacy hardcode |

**Databricks Problem:** These paths don't exist on Databricks. Need:
```
/Volumes/112557_prefetch_ctg_prd_exp/112557_prefetch_raw/irs_raw/i1099div.pdf
```

### 2.2 Page Geometry

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `run_pipeline_v2.py` | 78, 170 | `300.0` | page_mid_x |
| `config.py` | 50 | `306.0` | page_mid_x_fallback |
| `geometry.py` | 179 | `306.0` | reading_order_sort_key default |
| `merge.py` | 54, 87, 135 | `306.0` | default_split_x |
| `columns.py` | 115 | `612.0` | Standard letter width |

**Issue:** `300.0` vs `306.0` inconsistency. Standard US Letter is 612pt wide, so midpoint should be `306.0`.

### 2.3 Merge-Forward Thresholds

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `merge.py` | 50 | `160` | thin_char_thresh |
| `merge.py` | 51 | `2` | thin_elem_thresh |
| `merge.py` | 52 | `120` | body_char_thresh |
| `merge.py` | 53 | `10` | max_iterations |
| `config.py` | 53-55 | same | Duplicated in config |

### 2.4 Validation Thresholds

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `validate_graph.py` | 31 | `4000` | MAX_NODE_CHARS |
| `validate_graph.py` | 32 | `200` | MAX_EVIDENCE_CHARS |
| `validate_graph.py` | 33 | `20` | MIN_CONTENT_CHARS |

### 2.5 Confidence Values

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `references.py` | 89, 335 | `0.95 / 0.70` | Box ref confidence (target exists/not) |
| `references.py` | 359, 383, 407 | `0.90` | Pub/IRC/Form ref confidence |
| `elements.py` | 247-285 | `0.7-0.99` | Role classification confidence |
| `concept_roles.py` | 161, 173, 181 | `0.6, 0.5, 0.0` | Role fallback confidences |
| `pair_generation.py` | 48 | `0.7` | min_confidence threshold |

### 2.6 Confidence Bands (Validation)

| File | Line | Band | For |
|------|------|------|-----|
| `validate_graph.py` | 44 | `(0.95, 1.0)` | structural edges |
| `validate_graph.py` | 45 | `(0.85, 1.0)` | regex edges |
| `validate_graph.py` | 46 | `(0.5, 1.0)` | llm edges |

### 2.7 Column Detection

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `columns.py` | 65 | `0.25` | min_peak_ratio |
| `columns.py` | 66 | `0.25` | min_distance_pct |
| `columns.py` | 67 | `4.0` | x0_bucket_size |
| `columns.py` | 68 | `3` | min_peak_occurrences |
| `columns.py` | 69 | `20` | min_text_length |

### 2.8 Layout Detection

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `layout_detection.py` | 69 | `1.2` | Next line length ratio |
| `layout_detection.py` | 106 | `2` | structural_confirms threshold |
| `geometry.py` | 264 | `0.15` | is_centered tolerance_pct |
| `geometry.py` | 89, 99, 100 | `2.0, 0.02` | margin_tolerance defaults |

### 2.9 Reference Extraction

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `references.py` | 198, 268, 420 | `50` | context_chars |
| `config.py` | 58 | `50` | evidence_context_chars |

### 2.10 Document Metadata

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `run_pipeline_v2.py` | 90 | `"1099div_filer"` | doc_id |
| `run_pipeline_v2.py` | 300 | `"1099-DIV Filer Instructions"` | doc_label |
| `validate_graph.py` | 24-28 | `EXPECTED_BOXES_1099DIV` | Expected box keys |

---

## 3. Proposed Unified Config Module

### 3.1 Structure

```python
# src/vaas/config.py (REVISED)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set
import os


@dataclass
class PathConfig:
    """File path configuration."""
    pdf_path: Path
    output_dir: Path = field(default_factory=lambda: Path("output"))

    @classmethod
    def for_local(cls, pdf_name: str = "i1099div.pdf") -> "PathConfig":
        """Local development paths."""
        return cls(
            pdf_path=Path("data") / pdf_name,
            output_dir=Path("output"),
        )

    @classmethod
    def for_databricks(cls, pdf_name: str = "i1099div.pdf") -> "PathConfig":
        """Databricks paths."""
        base = Path("/Volumes/112557_prefetch_ctg_prd_exp/112557_prefetch_raw/irs_raw")
        return cls(
            pdf_path=base / pdf_name,
            output_dir=Path("/tmp/vaas_output"),  # Or dbfs path
        )

    @classmethod
    def auto_detect(cls, pdf_name: str = "i1099div.pdf") -> "PathConfig":
        """Auto-detect environment."""
        if os.path.exists("/Volumes"):
            return cls.for_databricks(pdf_name)
        return cls.for_local(pdf_name)


@dataclass
class GeometryConfig:
    """Page geometry configuration."""
    page_width: float = 612.0          # US Letter width in points
    page_mid_x: float = 306.0          # page_width / 2
    margin_min_tol: float = 2.0        # Minimum margin tolerance
    margin_pct: float = 0.02           # Margin as % of block width
    is_centered_tolerance: float = 0.15


@dataclass
class ColumnConfig:
    """Column detection configuration."""
    min_peak_ratio: float = 0.25
    min_distance_pct: float = 0.25
    x0_bucket_size: float = 4.0
    min_peak_occurrences: int = 3
    min_text_length: int = 20


@dataclass
class MergeConfig:
    """Merge-forward thresholds."""
    thin_char_thresh: int = 160
    thin_elem_thresh: int = 2
    body_char_thresh: int = 120
    max_iterations: int = 10


@dataclass
class ValidationConfig:
    """Validation thresholds."""
    max_node_chars: int = 4000
    max_evidence_chars: int = 200
    min_content_chars: int = 20

    # Confidence bands by created_by
    structural_confidence_band: tuple = (0.95, 1.0)
    regex_confidence_band: tuple = (0.85, 1.0)
    llm_confidence_band: tuple = (0.5, 1.0)


@dataclass
class ConfidenceConfig:
    """Confidence score settings."""
    # Reference extraction
    box_ref_exists: float = 0.95
    box_ref_missing: float = 0.70
    pub_ref: float = 0.90
    irc_ref: float = 0.90
    form_ref: float = 0.90

    # Role classification
    role_header_match: float = 0.95
    role_pattern_match: float = 0.85
    role_default: float = 0.70

    # Pair generation
    min_pair_confidence: float = 0.70


@dataclass
class LayoutConfig:
    """Layout detection configuration."""
    next_line_length_ratio: float = 1.2
    structural_confirms_threshold: int = 2
    evidence_context_chars: int = 50


@dataclass
class DocumentConfig:
    """Document-specific configuration."""
    doc_id: str
    doc_label: str
    expected_boxes: Set[str]
    section_id_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_1099div(cls) -> "DocumentConfig":
        return cls(
            doc_id="1099div_filer",
            doc_label="1099-DIV Filer Instructions",
            expected_boxes={
                "1a", "1b", "2a", "2b", "2c", "2d", "2e", "2f",
                "3", "4", "5", "6", "7", "8", "9", "10",
                "11", "12", "13", "14", "15", "16"
            },
            section_id_map={
                "future developments": "sec_future_developments",
                "reminders": "sec_reminders",
                "general instructions": "sec_general_instructions",
                "specific instructions": "sec_specific_instructions",
            },
        )

    @classmethod
    def for_1099int(cls) -> "DocumentConfig":
        return cls(
            doc_id="1099int_filer",
            doc_label="1099-INT Filer Instructions",
            expected_boxes={
                "1", "2", "3", "4", "5", "6", "7", "8",
                "9", "10", "11", "12", "13", "14", "15", "16", "17"
            },
        )


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    paths: PathConfig
    document: DocumentConfig
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)

    @classmethod
    def for_1099div_local(cls) -> "PipelineConfig":
        """1099-DIV config for local development."""
        return cls(
            paths=PathConfig.for_local("i1099div.pdf"),
            document=DocumentConfig.for_1099div(),
        )

    @classmethod
    def for_1099div_databricks(cls) -> "PipelineConfig":
        """1099-DIV config for Databricks."""
        return cls(
            paths=PathConfig.for_databricks("i1099div.pdf"),
            document=DocumentConfig.for_1099div(),
        )

    @classmethod
    def auto(cls, form: str = "1099div") -> "PipelineConfig":
        """Auto-detect environment and form."""
        paths = PathConfig.auto_detect(f"i{form}.pdf")
        if form == "1099div":
            doc = DocumentConfig.for_1099div()
        elif form == "1099int":
            doc = DocumentConfig.for_1099int()
        else:
            raise ValueError(f"Unknown form: {form}")
        return cls(paths=paths, document=doc)
```

### 3.2 Databricks Override Pattern

```python
# In Databricks notebook cell 0:

from vaas.config import PipelineConfig, PathConfig, DocumentConfig

# Option 1: Use factory method
config = PipelineConfig.for_1099div_databricks()

# Option 2: Auto-detect (recommended)
config = PipelineConfig.auto("1099div")

# Option 3: Override specific values
config = PipelineConfig.auto("1099div")
config.merge.thin_char_thresh = 200  # Adjust threshold
config.validation.max_node_chars = 5000  # Larger monolith threshold

# Option 4: Full override for custom paths
config = PipelineConfig(
    paths=PathConfig(
        pdf_path=Path("/dbfs/mnt/custom/i1099div.pdf"),
        output_dir=Path("/dbfs/mnt/custom/output"),
    ),
    document=DocumentConfig.for_1099div(),
)
```

### 3.3 Local Override Pattern

```python
# In local development:

from vaas.config import PipelineConfig

# Standard local run
config = PipelineConfig.for_1099div_local()

# Or with auto-detection
config = PipelineConfig.auto("1099div")

# Run pipeline
from run_pipeline_v2 import run_pipeline
results = run_pipeline(config)
```

---

## 4. Migration Plan

### 4.1 Phase 1: Consolidate Config (Effort: Low)

1. Delete `run_pipeline_v2.py` PipelineConfig (lines 72-93)
2. Update imports to use `from vaas.config import PipelineConfig`
3. Update `PipelineConfig.for_1099div()` to match new structure
4. Update `run_pipeline()` to accept new config structure

### 4.2 Phase 2: Wire Sub-Configs (Effort: Medium)

| Module | Changes |
|--------|---------|
| `merge.py` | Accept `MergeConfig` instead of inline `MergeConfig` |
| `columns.py` | Accept `ColumnConfig` |
| `validate_graph.py` | Accept `ValidationConfig` |
| `references.py` | Use `ConfidenceConfig.box_ref_exists` etc. |
| `geometry.py` | Use `GeometryConfig.page_mid_x` |

### 4.3 Phase 3: Environment Detection (Effort: Low)

```python
# Add to config.py
def detect_environment() -> str:
    """Detect execution environment."""
    if os.path.exists("/databricks"):
        return "databricks"
    if os.path.exists("/Volumes"):
        return "databricks"
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return "databricks"
    return "local"
```

---

## 5. Inconsistencies Found

### 5.1 page_mid_x: 300.0 vs 306.0

| Location | Value |
|----------|-------|
| `run_pipeline_v2.py` | `300.0` |
| `geometry.py`, `merge.py`, `config.py` | `306.0` |

**Recommendation:** Standardize on `306.0` (half of 612pt letter width).

### 5.2 Duplicate EXPECTED_BOXES

| Location | Description |
|----------|-------------|
| `run_pipeline_v2.py:52` | Imports from anchors.py |
| `validate_graph.py:24-28` | Hardcoded inline |
| `config.py:103-107` | In DocumentConfig |
| `anchors.py` (if exists) | Original definition |

**Recommendation:** Single source in `DocumentConfig.for_1099div()`.

### 5.3 context_chars vs evidence_context_chars

| Location | Name | Value |
|----------|------|-------|
| `references.py` | `context_chars` | `50` |
| `config.py` | `evidence_context_chars` | `50` |

**Recommendation:** Use `layout.evidence_context_chars` from config.

---

## 6. Environment Variables (Alternative)

For simpler Databricks override without code changes:

```python
# config.py addition
import os

def get_path_from_env(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))

@dataclass
class PathConfig:
    pdf_path: Path = field(
        default_factory=lambda: get_path_from_env(
            "VAAS_PDF_PATH", "data/i1099div.pdf"
        )
    )
    output_dir: Path = field(
        default_factory=lambda: get_path_from_env(
            "VAAS_OUTPUT_DIR", "output"
        )
    )
```

**Databricks usage:**
```python
# In notebook cell 0:
import os
os.environ["VAAS_PDF_PATH"] = "/Volumes/.../i1099div.pdf"
os.environ["VAAS_OUTPUT_DIR"] = "/tmp/vaas_output"

# Then import config (reads from env)
from vaas.config import PipelineConfig
config = PipelineConfig.auto("1099div")
```

---

## 7. Summary

| Action | Priority | Effort |
|--------|----------|--------|
| Consolidate two PipelineConfig classes | **P1** | Low |
| Fix page_mid_x inconsistency (300 → 306) | **P1** | Trivial |
| Add PathConfig.for_databricks() | **P1** | Low |
| Add environment auto-detection | **P2** | Low |
| Wire sub-configs to modules | **P2** | Medium |
| Add environment variable overrides | **P3** | Low |
| Move EXPECTED_BOXES to single location | **P3** | Trivial |

---

*End of Config Audit*
