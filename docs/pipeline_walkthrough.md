# VaaS Pipeline Walkthrough: PDF to Knowledge Graph

This document walks through the complete extraction pipeline that transforms a PDF (IRS 1099-DIV Instructions) into a validated knowledge graph suitable for RAG retrieval and contrastive training.

---

## The Problem

Standard RAG fails on tax documents because:

1. **Cross-referential structure**: "Box 1a includes amounts in boxes 1b and 2e" - embeddings don't capture these dependencies
2. **Hierarchical content**: Box headers contain sections, which contain subsections, which contain paragraphs
3. **Precise terminology**: "Box 2e" vs "Box 2f" are lexically similar but semantically different
4. **Polarity matters**: "Box 1a does NOT include..." is the opposite of "Box 1a includes..."

**Solution**: Build a knowledge graph that captures document structure AND semantic relationships with polarity tracking, then use graph edges for training pair generation and retrieval augmentation.

---

## Pipeline Overview

```
PDF (i1099div.pdf)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 1-2: Span Extraction + Font Analysis                   │
│  • PyMuPDF extracts spans with position, font, size           │
│  • Body font inference (9.0pt) anchors all classification     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 3: Line Building + Structure Detection                 │
│  • Two-column layout detection (pages 2-5)                    │
│  • Emphasis flags (bold, larger-than-body)                    │
│  • Split trigger detection (content boundaries)               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 4: Element Classification                              │
│  • Split blocks at triggers into elements                     │
│  • Role classification with confidence scores                 │
│  • Roles: BoxHeader, SectionHeader, SubsectionHeader,         │
│           ListBlock, BodyTextBlock, PageArtifact              │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 5-6: Anchor Detection + Content Assignment             │
│  • Detect anchors from headers (22 boxes, 3 sections, 24 sub) │
│  • Build anchor timeline respecting column boundaries         │
│  • Assign elements to governing anchors                       │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 7: Section Materialization + Concept Classification    │
│  • Aggregate elements under anchors                           │
│  • Merge thin subsections (<100 chars)                        │
│  • Classify concept roles via regex + position heuristics     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 8: Reference Extraction                                │
│  • Box references: "see Box 1a", "boxes 14-16"                │
│  • Publication refs: "Pub. 550"                               │
│  • IRC section refs: "section 301(c)"                         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 9: Graph Construction                                  │
│  • Node building: doc_root, sections, concepts, paragraphs    │
│  • Structural edges: parent_of, follows, in_section           │
│  • Reference edges: references_box, same_group                │
│  • Typed semantic edges: includes, excludes, defines, etc.    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  VALIDATION: Graph Quality Checks (A1-A8, B1-B3)              │
│  • Deterministic: coverage, integrity, distribution           │
│  • LLM-as-judge: boundary accuracy, edge correctness          │
└───────────────────────────────────────────────────────────────┘
```

---

## Stage Details

### Stages 1-4: Physical Extraction

These stages transform raw PDF into classified elements. Key techniques:

| Stage | Input | Output | Key Logic |
|-------|-------|--------|-----------|
| 1 | PDF | Spans | PyMuPDF extraction with bbox, font metadata |
| 2 | Spans | Body size | Histogram analysis (modal font size = body) |
| 3 | Spans | Lines | Y-clustering, column detection, emphasis flags |
| 4 | Lines | Elements | Split at triggers, role classification |

**Role distribution for 1099-DIV**:
```
BodyTextBlock       51
SubsectionHeader    24
ListBlock           21
BoxHeader           20
PageArtifact         7
SectionHeader        3
```

---

### Stages 5-6: Anchor Detection + Assignment

Anchors are the structural skeleton. Each anchor "owns" content until the next anchor.

**Anchor detection** uses regex patterns:
```python
# Box patterns
BOX_RX_SINGLE = r"^Box\s*(\d+[a-z]?)\.?\s+"
BOX_RX_RANGE = r"^Boxes?\s*(\d+[a-z]?)\s*(through|[-–])\s*(\d+[a-z]?)"

# Section patterns
SECTION_HEADER_RX = r"^(What|Who|When|How|General|Specific|Instructions|...)"
```

**Content assignment** respects column boundaries:
- Elements sorted by `(page, column, y0, x0)` - the column-aware sort key
- Assignment uses IntervalIndex for O(1) lookup of governing anchor
- Grouped boxes (14-16) share content legitimately

**Validation checkpoint**: 100% box coverage required (all 22 expected boxes detected).

---

### Stage 7: Concept Role Classification

Subsections get semantic role labels via **regex-first heuristics with position fallbacks**:

```python
# Priority order (first match wins):

# 1. Exception (highest - explicit negation)
r"(?i)^.{0,80}(?:does\s+not\s+include|except\s+|excluding\s+)"
→ role="exception", confidence=0.90

# 2. Condition (explicit conditional logic)
r"(?i)^.{0,60}(?:if\s+(?:you|the)|when\s+(?:you|the)|only\s+if)"
→ role="condition", confidence=0.90

# 3. Procedure (imperative verbs)
r"(?i)^(?:Enter|Report|File|Include|Use|Complete)"
→ role="procedure", confidence=0.85

# 4. Qualification (scope constraints)
r"(?i)^.{0,40}(?:the\s+following|these\s+dividends|applies\s+to)"
→ role="qualification", confidence=0.75

# 5. Definition (noun phrase headers - fallback)
r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$"
→ role="definition", confidence=0.80
```

**Position-based tiebreakers** when regex fails:
- Before first box → likely `definition`
- After box header with short body → likely `qualification`
- No match → `role=NULL` (explicit uncertainty)

---

### Stage 8-9: Typed Semantic Edge Extraction

This is where semantic intelligence lives. The typed edge system extracts **relationship meaning with polarity**.

#### Edge Types and Polarity

| Edge Type | Direction | Polarity | Example |
|-----------|-----------|----------|---------|
| `includes` | box → box | positive | "Box 1a includes amounts in boxes 1b and 2e" |
| `excludes` | concept → box | **negative** | "Box 1a does NOT include nondividend distributions" |
| `defines` | concept → box | positive | "Box 1b is the portion that qualifies..." |
| `qualifies` | concept → box | positive | "Box 1a includes ONLY ordinary dividends" |
| `applies_if` | concept → box | positive | "Report in box 2e IF holding period met" |
| `requires` | box → box | positive | "Include amounts from box 1b" |
| `portion_of` | concept → box | positive | "If any PART of box 1a is qualified..." |

#### Why Polarity Matters

**Negative knowledge prevents false associations**:
- "Box 1a does NOT include nondividend distributions"
- Without `excludes` edge, embedding might associate "nondividend" with Box 1a
- Training pairs use `excludes` as **hard negatives** (things that should NOT be close)

#### Sentence Gating for Precision

Typed edges are extracted **per-sentence** to prevent cross-sentence bleed:

```python
def split_sentences_with_offsets(text):
    """
    IRS-tuned, precision-first splitter that:
    - Handles abbreviations (U.S., No., Sec., Pub., etc.)
    - Treats bullet/numbered lists as boundaries
    - Under-splits rather than over-splits (safe for extraction)
    """
```

Each sentence is processed independently, with character offsets preserved for evidence pointers.

#### Subset Cue Gating (portion_of precision)

The `portion_of` edge type requires **explicit subset cues** to fire:

```python
SUBSET_CUES = [
    r'(?i)\bany\s+part\s+of\b',      # "if any part of box 1a..."
    r'(?i)\bportion\s+of\b',          # "portion of the amount..."
    r'(?i)\bto\s+the\s+extent\b',     # "to the extent attributable..."
    r'(?i)\battributable\s+to\b',
]

# Narration filter - reject false positives
NARRATION_FILTERS = [
    r'(?i)\bwas\s+determined\b',      # Past tense = description, not rule
    r'(?i)\bhas\s+been\s+allocated\b',
]
```

A sentence must have a subset cue AND pass the narration filter to emit `portion_of`.

#### Edge Precedence (Negation Wins)

When multiple edge types could apply, **excludes always wins**:

```python
def extract_typed_edges_from_section(anchor_id, body_text, valid_box_keys):
    # PASS 1: Extract excludes first (global precedence)
    excludes_edges = extract_excludes_edges(...)
    excluded_boxes = {ed.target_box_key for ed in excludes_edges}

    # PASS 2: Other edges skip excluded targets
    applies_if_edges = extract_applies_if_edges(..., excluded_boxes)
    portion_of_edges = extract_portion_of_edges(..., excluded_boxes)

    # portion_of also suppresses defines (more specific wins)
    excluded_from_defines = excluded_boxes | portion_of_boxes
    defines_edges = extract_defines_edges(..., excluded_from_defines)
```

#### Negation Context Detection

Even for non-excludes edges, we check surrounding context:

```python
def _has_negation_context(text, pos, window=80):
    """Check if position has negation context (excludes should win)."""
    context = text[max(0, pos-80) : min(len(text), pos+80)]

    for _, pattern, _ in EXCLUDES_PATTERNS:
        if pattern.search(context):
            return True  # Skip this edge, let excludes handle it
    return False
```

---

## Validation Framework

### Phase A: Deterministic Checks

| Check | What it validates | Pass criteria |
|-------|-------------------|---------------|
| **A1: Anchor Coverage** | All expected boxes detected | 100% coverage |
| **A2: Artifact Contamination** | No page markers in content | 0 artifacts |
| **A3: Structure Completeness** | No monolith nodes (>4000 chars) | Size limits |
| **A4: Edge Integrity** | All edges reference valid nodes | 0 dangling |
| **A5: Skeleton Coverage** | Graph is connected | <3 components |
| **A6: Provenance** | Source tracking for highlighting | Required fields |
| **A7: Hierarchy Integrity** | DAG, single parent per node | No cycles |
| **A8: Edge Distribution** | Balanced edge types | refs_box < 80% |

### A7: Hierarchy Integrity (Deep Dive)

Three independent checks ensure the graph is traversable:

1. **Parent cardinality**: Non-root nodes must have exactly 1 parent
2. **Acyclicity**: DFS with WHITE/GRAY/BLACK coloring detects back edges
3. **Depth bounds**: No node deeper than MAX_HIERARCHY_DEPTH (5)

```python
# Cycle detection via DFS coloring
WHITE, GRAY, BLACK = 0, 1, 2

def dfs_cycle_detect(node, path):
    color[node] = GRAY  # Currently visiting
    for child in children[node]:
        if color[child] == GRAY:
            # Back edge = cycle!
            cycles_found.append(path + [child])
        elif color[child] == WHITE:
            dfs_cycle_detect(child, path + [child])
    color[node] = BLACK  # Done visiting
```

### A8: Edge-Type Distribution (The Current Failure)

Checks that semantic edges aren't drowned out by references:

```python
# Current state:
# references_box: 61 edges (88%)
# semantic edges: 5 edges (7%)
# Threshold: references_box must be < 80%

REF_DOMINANCE_FAIL = 0.80  # FAIL if > 80%
MIN_SEMANTIC_TYPES = 2     # Need at least 2 distinct types
```

**Fix**: Improve typed edge extraction patterns to capture more `defines`, `qualifies`, `applies_if`, etc.

### Phase B: LLM-as-Judge (Planned)

| Check | What LLM validates |
|-------|-------------------|
| **B1: Anchor Assignment** | Elements near boundaries assigned correctly |
| **B2: Edge Correctness** | Semantic edges supported by evidence |
| **B3: Pair Suitability** | Training pairs are valid for contrastive learning |

Each LLM judgment requires **evidence pointers with quotes**:
```json
{
  "decision": "edge_correct",
  "confidence": 0.85,
  "evidence": [
    {"doc_id": "1099div_filer", "page": 2, "quote": "Box 1a includes..."}
  ],
  "unsupported_claims": []  // Must be empty to pass
}
```

---

## Training Pair Generation

Graph edges become training pairs for contrastive learning:

### Stratified Sampling

```python
EDGE_TYPE_CATEGORIES = {
    # Structural (hierarchical)
    "parent_of": "hierarchical",
    "includes": "hierarchical",

    # Cross-reference
    "references_box": "cross_reference",

    # Semantic
    "defines": "semantic",
    "qualifies": "semantic",
    "applies_if": "semantic",

    # Negative knowledge
    "excludes": "negative",  # Used for hard negatives!
}

# Target counts per category (ensures balance)
target_per_category = {
    "hierarchical": 50,
    "cross_reference": 100,
    "semantic": 50,
    "negative": 30,
}
```

### Hard Negative Mining

The `excludes` edges are gold for hard negatives:
- High lexical similarity (both mention same boxes)
- Graph says they should NOT be associated
- Perfect for teaching the model precision

```python
# Example: Box 1a excludes edge
anchor_text: "Box 1a. Ordinary dividends..."
positive_text: "Box 1b. Qualified dividends..."  # from includes edge
HARD_NEGATIVE: "Nondividend distributions are NOT included in box 1a"  # from excludes edge
```

---

## Confidence Bands

All edges have confidence scores with expected bands:

| Creation Method | Expected Band | Example |
|-----------------|---------------|---------|
| `structural` | 0.95-1.0 | parent_of, follows |
| `regex` | 0.85-1.0 | references_box, typed edges |
| `llm` | 0.5-1.0 | Ambiguous relationships |

Validation (A4) flags edges outside expected bands:
```python
if not (lo <= confidence <= hi):
    findings.append(Finding(
        severity="warning",
        message=f"Confidence {confidence} outside band [{lo}, {hi}]"
    ))
```

---

## Current Status

```
Overall Status: FAIL
Checks: 10/11 passed

✅ A1: Anchor Coverage (22/22 boxes)
✅ A2: Artifact Contamination (clean)
✅ A3: Structure Completeness (no monoliths)
✅ A4: Edge Integrity (308 valid edges)
✅ A5: Skeleton Coverage (connected)
✅ A6: Provenance (tracking works)
✅ A7: Hierarchy Integrity (valid DAG)
❌ A8: Edge Distribution (refs_box at 88% > 80%)
✅ B1-B3: LLM Judge (placeholders)
```

**Next step**: Improve typed edge extraction to generate more semantic edges and bring `references_box` below 80%.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/vaas/extraction/elements.py` | Element splitting + role classification |
| `src/vaas/extraction/anchors.py` | Anchor detection + content assignment |
| `src/vaas/semantic/concept_roles.py` | Subsection role classification |
| `src/vaas/semantic/typed_edges.py` | Semantic edge extraction with polarity |
| `src/vaas/semantic/pair_generation.py` | Stratified training pair generation |
| `src/vaas/graph/nodes.py` | Node construction |
| `src/vaas/graph/edges.py` | Edge construction (all types) |
| `validate_graph.py` | A1-A8, B1-B3 validation framework |
