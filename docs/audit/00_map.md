# VaaS Codebase Map

**Generated:** 2026-01-08

---

## 1. Top 5 Largest Modules by LOC

| Rank | Module | File | LOC |
|------|--------|------|-----|
| 1 | **edges.py** | `src/vaas/graph/edges.py` | 1,033 |
| 2 | **validate_graph.py** | `validate_graph.py` | 1,023 |
| 3 | **typed_edges.py** | `src/vaas/semantic/typed_edges.py` | 994 |
| 4 | **nodes.py** | `src/vaas/graph/nodes.py` | 550 |
| 5 | **run_pipeline_v2.py** | `run_pipeline_v2.py` | 429 |

---

## 2. Top 10 Longest Functions

| Rank | Function | File | Start Line | Length |
|------|----------|------|------------|--------|
| 1 | `build_paragraph_nodes` | `src/vaas/graph/nodes.py` | 305 | ~135 lines |
| 2 | `extract_typed_edges_from_section` | `src/vaas/semantic/typed_edges.py` | 750 | ~98 lines |
| 3 | `audit_a4_edge_integrity` | `validate_graph.py` | 368 | ~112 lines |
| 4 | `audit_a2_artifact_contamination` | `validate_graph.py` | 197 | ~99 lines |
| 5 | `generate_markdown_report` | `validate_graph.py` | 771 | ~142 lines |
| 6 | `build_all_edges` | `src/vaas/graph/edges.py` | 864 | ~120 lines |
| 7 | `audit_a5_skeleton_coverage` | `validate_graph.py` | 482 | ~115 lines |
| 8 | `extract_concept_to_box_edges` | `src/vaas/semantic/typed_edges.py` | 856 | ~85 lines |
| 9 | `build_section_nodes` | `src/vaas/graph/nodes.py` | 242 | ~61 lines |
| 10 | `split_sentences_with_offsets` | `src/vaas/semantic/typed_edges.py` | 53 | ~92 lines |

---

## 3. Module Responsibilities

### Orchestration Layer

| Module | Responsibility |
|--------|----------------|
| `run_pipeline_v2.py` | Thin orchestrator: calls modular components in sequence, handles CLI, saves outputs |
| `validate_graph.py` | Graph quality validator: 9 checks (A1-A6 deterministic, B1-B3 LLM placeholders) |

### Extraction Modules (`src/vaas/extraction/`)

| Module | Responsibility |
|--------|----------------|
| `pdf.py` | PDF span extraction using PyMuPDF |
| `lines.py` | Line building from spans; block/page geometry; column detection |
| `elements.py` | Element construction with role classification |
| `anchors.py` | Anchor detection: box/section/subsection headers |
| `sections.py` | Section assembly: assigns elements to anchors |
| `merge.py` | Merge-forward: folds thin subsections into subsequent anchors |
| `references.py` | Cross-reference extraction via regex |
| `layout_detection.py` | Layout-driven subsection detection |
| `columns.py` | Column detection via x0 peak analysis |
| `geometry.py` | Bbox utilities |

### Graph Modules (`src/vaas/graph/`)

| Module | Responsibility |
|--------|----------------|
| `nodes.py` | Node construction: doc_root, section nodes, paragraph nodes |
| `edges.py` | Edge construction: structural, reference, typed semantic |

### Semantic Modules (`src/vaas/semantic/`)

| Module | Responsibility |
|--------|----------------|
| `typed_edges.py` | Pattern-based semantic edge extraction with sentence gating |
| `concept_roles.py` | Concept role classification for subsections |
| `pair_generation.py` | Training pair generation (stub) |

### Utilities (`src/vaas/utils/`)

| Module | Responsibility |
|--------|----------------|
| `text.py` | Stable hashing, text normalization |
| `serialize.py` | Parquet serialization helpers |

---

## 4. Hotspot Analysis

### Critical (High Risk)

| Location | Risk |
|----------|------|
| `typed_edges.py` pattern tables (192-299) | Pattern interactions cause FP/FN |
| `edges.py` `build_typed_edges()` (791-858) | Dedupe policy affects downstream |
| `nodes.py` `generate_paragraph_node_id()` (166-181) | ID format drift breaks joins |

### Moderate (Medium Risk)

| Location | Risk |
|----------|------|
| `validate_graph.py` confidence bands (43-47) | Threshold mismatch causes false alarms |
| `extraction/sections.py` content assignment | Wrong element-to-anchor assignment |
| `typed_edges.py` sentence splitting (53-144) | Over/under-splitting affects patterns |

### Stable (Low Risk)

| Location | Notes |
|----------|-------|
| `edges.py` structural builders (194-493) | Well-tested |
| `extraction/pdf.py` | Stable PyMuPDF wrapper |

---

## 5. Architecture Diagram

```
run_pipeline_v2.py
       │
       ├─► vaas.extraction.pdf          ──► spans_df
       ├─► vaas.extraction.lines        ──► line_df
       ├─► vaas.extraction.elements     ──► elements_df
       ├─► vaas.extraction.anchors      ──► anchors_df
       ├─► vaas.extraction.sections     ──► sections_df
       ├─► vaas.extraction.merge        ──► sections_df (merged)
       ├─► vaas.extraction.references   ──► references_df
       ├─► vaas.semantic.concept_roles  ──► sections_df (with roles)
       ├─► vaas.graph.nodes             ──► graph_nodes_df
       └─► vaas.graph.edges             ──► graph_edges_df
                 │
                 └─► vaas.semantic.typed_edges (internal)

       ▼
   validate_graph.py                    ──► quality report
```

---

## 6. Key Statistics

| Metric | Value |
|--------|-------|
| Python modules | ~20 |
| Total LOC | ~6,500 |
| Edge types | 11 |
| Pattern tables | 7 |
| Validation checks | 9 |
| Graph nodes (1099-DIV) | 122 |
| Graph edges (1099-DIV) | 308 |

---

## 7. Implementation Notes

- **Semantic edges** source from paragraph nodes (Phase B architecture)
- **Sentence gating** is active for concept→box edges
- **Dedupe policy**: highest-confidence wins per (type, source, target)
- **Confidence floor**: 0.85 minimum for semantic edges
- **LLM judge** (B1-B3): placeholders only

---

*End of codebase map*
