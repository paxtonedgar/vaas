# Stages 8-9 Dialectical Analysis: Graph Construction & Validation

This document interrogates the design decisions in stages 8-9 of the VaaS extraction pipeline: graph construction (nodes, edges, typed edges) and validation.

---

## Overview: What We're Building

> **What Stages 8-9 are**: Transform extraction artifacts into a validated knowledge graph with typed nodes, structural/reference/semantic edges, and quality checks.
>
> **Role in bigger picture**: The graph is the final output of extraction - used for training pair generation and retrieval augmentation. Validation ensures the graph is correct before downstream use.
>
> **What we'd lose without it**: No graph = no training pairs, no graph expansion during retrieval. No validation = silent failures, bad training data, incorrect retrieval.
>
> **How we do it**: Build nodes at multiple granularities, construct edges by type (structural, reference, semantic), validate with deterministic audits and LLM-as-judge placeholders.

**Stage 8** transforms extraction artifacts (sections, elements, references, anchors) into a knowledge graph:
- **Nodes**: Chunks at various granularities (doc_root, box_section, section, concept, paragraph)
- **Edges**: Typed relationships between nodes (structural, reference, semantic)

**Stage 9** validates the graph through deterministic audits (A1-A8) and LLM-as-judge checks (B1-B3 placeholders).

**Goal**: A graph suitable for:
1. Training pair generation (contrastive learning on graph edges)
2. Retrieval augmentation (graph expansion on top-K results)

---

## Stage 8: Graph Construction

> **What it is**: Build the knowledge graph from sections, elements, references, and anchors.
>
> **Role in bigger picture**: The graph is the semantic representation of the document. It encodes hierarchy (parent_of), sequence (follows), cross-references (references_box), and semantic relationships (excludes, defines, etc.).
>
> **What we'd lose without it**: No structured representation. Couldn't traverse from "Box 1a" to related concepts. Couldn't generate training pairs from edges.
>
> **How we do it**: Build nodes from sections (coarse) and elements (fine), then build edges in layers: structural → reference → semantic. Filter to active nodes.

---

## Part 1: Node Construction (`nodes.py`)

> **What it is**: Create graph nodes at multiple granularities from sections (coarse) and elements (fine).
>
> **Role in bigger picture**: Nodes are the retrieval units. Different granularities serve different query types.
>
> **What we'd lose without it**: No graph nodes = no graph. Couldn't represent document structure.
>
> **How we do it**: Build doc_root (1), section nodes from sections_df, paragraph nodes from elements_df. Combine into single DataFrame.

### Q1: Why have multiple node types (doc_root, box_section, section, concept, paragraph)?

**Design Decision**: Five node types at different granularities:
```
doc_root (1)
  └── section/box_section (22 boxes + 3 sections)
        └── concept (24 subsections)
              └── paragraph (many)
```

**Why?**

1. **Different retrieval use cases demand different granularities**:
   - "What goes in Box 1a?" → box_section node (complete answer)
   - "What are qualified dividends?" → concept node (definition)
   - "If any part of box 1a is qualified..." → paragraph node (specific rule)

2. **Training pair quality depends on granularity matching**:
   - Positive pairs should be at similar granularity
   - Hard negatives work best at paragraph level (high lexical overlap, semantic difference)

3. **Graph expansion needs coarse-to-fine navigation**:
   - Start at box_section → expand to child concepts → expand to paragraphs
   - Without hierarchy, expansion is just adjacency walking

**Alternative rejected**: Single node type with `granularity` field. This complicates edge semantics (does `parent_of` mean hierarchy or containment?) and makes type-aware queries harder.

---

### Q2: Why separate paragraph nodes from section nodes? Why not just have section text?

**Code Reference** (`nodes.py:290-424`):
```python
def build_paragraph_nodes(
    elements_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    doc_id: str,
    skip_roles: Optional[Set[str]] = None,
) -> List[Node]:
    # Creates one node per non-header element
    for _, elem in elements_df.iterrows():
        # Skip headers - already represented as anchor nodes
        if role in HEADER_ROLES:
            continue
        # ... build paragraph node
```

**Why?**

1. **Semantic edges point to rules, not containers**:
   - The sentence "Box 1a does NOT include nondividend distributions" is the rule-holder
   - That sentence lives in a paragraph, not the section as a whole
   - Typed edges (`excludes`, `applies_if`) should point FROM paragraphs TO boxes

2. **Evidence provenance requires element-level tracking**:
   - When an edge is created, we need `source_element_id` for highlighting
   - If we only had section text, we'd lose the element boundary

3. **Training pairs at paragraph level are more precise**:
   - A section might be 2000 chars with 10 different concepts
   - A paragraph is 100-300 chars with one focused idea
   - Contrastive learning works better with focused anchors

**Alternative rejected**: Just store full_text in section nodes. This loses the fine-grained structure we worked hard to extract in stages 1-4.

---

### Q3: Why is node_id format `{doc_id}:{anchor_id}` or `{doc_id}:el_{element_id}`?

**Code Reference** (`nodes.py:137-166`):
```python
def generate_node_id(doc_id: str, anchor_id: str) -> str:
    return f"{doc_id}:{anchor_id}"

def generate_paragraph_node_id(doc_id: str, element_id: str) -> str:
    # Strip doc_id prefix if already present
    if element_id.startswith(f"{doc_id}:"):
        element_id = element_id[len(doc_id) + 1:]
    return f"{doc_id}:el_{element_id}"
```

**Why?**

1. **Cross-document uniqueness**:
   - When we have multiple documents (1099-DIV filer + recipient), node IDs must be globally unique
   - `doc_id:` prefix ensures "box_1a" from filer instructions doesn't collide with "box_1a" from recipient instructions

2. **Distinguishing section vs paragraph nodes**:
   - `1099div_filer:box_1a` is a section node
   - `1099div_filer:el_1:2:seg0` is a paragraph node (the `el_` prefix)
   - This makes it easy to filter/query by node granularity

3. **Stable edge references**:
   - Edges store `source_node_id` and `target_node_id` as strings
   - The colon convention makes parsing straightforward: `split(':')[0]` = doc_id, `split(':')[1:]` = local id

**Alternative considered**: UUID node IDs. Rejected because:
- Harder to debug (no semantic meaning in ID)
- Requires lookup table to resolve references
- Loses document provenance

---

### Q4: Why does `build_paragraph_nodes` skip unassigned elements?

**Code Reference** (`nodes.py:350-352`):
```python
# Skip unassigned/invalid anchor_id
aid = elem.get("anchor_id")
if not isinstance(aid, str) or not aid or aid == "unassigned":
    continue
```

**Why?**

1. **Unassigned elements are structural orphans**:
   - They weren't claimed by any anchor during content assignment (stage 5-6)
   - Creating a node for them would leave it dangling in the graph
   - Better to drop them than create disconnected components

2. **Validation catches missing content**:
   - A1 (Anchor Coverage) checks that all boxes have content
   - If important content is unassigned, it shows up as sparse coverage
   - This is a signal to fix content assignment, not to hack around it here

3. **Garbage in, garbage out principle**:
   - If element assignment failed, creating a node doesn't fix the problem
   - It just propagates bad data into the graph

**What happens to unassigned elements?**: They're logged as warnings in stage 5-6 and should be investigated. Usually they're page artifacts that slipped through classification.

---

### Q5: Why does the Node dataclass have so many optional fields?

**Code Reference** (`nodes.py:42-101`):
```python
@dataclass
class Node:
    node_id: str
    doc_id: str
    node_type: str
    anchor_id: Optional[str] = None
    box_key: Optional[str] = None
    label: str = ""
    text: str = ""
    pages: List[int] = field(default_factory=list)
    bbox: Optional[List[float]] = None
    element_id: Optional[str] = None
    element_count: int = 0
    char_count: int = 0
    reading_order: Optional[int] = None
    paragraph_kind: Optional[str] = None
    anchor_type: Optional[str] = None
    concept_role: Optional[str] = None
```

**Why?**

1. **Different node types have different attributes**:
   - `box_key` only applies to box_section nodes
   - `reading_order` only applies to paragraph nodes
   - `concept_role` only applies to concept nodes
   - Single dataclass handles all types with optional fields

2. **DataFrame export requires consistent columns**:
   - `nodes_df = pd.DataFrame([n.to_dict() for n in all_nodes])`
   - If we had different classes per type, we'd need complex union logic
   - Optional fields become NaN in DataFrame (handled naturally by pandas)

3. **Provenance completeness (A6 validation)**:
   - Fields like `pages`, `bbox`, `element_id` enable source highlighting
   - Even if not always populated, having the field signals the capability

**Alternative considered**: Type-specific node classes with inheritance. Rejected because:
- Adds complexity without benefit (no polymorphic behavior)
- DataFrame export becomes multi-step merge
- Most consumers just want the DataFrame anyway

---

## Part 2: Edge Construction (`edges.py`)

> **What it is**: Build typed edges between nodes - structural (parent_of, follows), reference (references_box), and semantic (excludes, defines, etc.).
>
> **Role in bigger picture**: Edges are the relationships that make the graph useful. Without edges, nodes are isolated. Edges enable traversal, training pairs, and graph expansion.
>
> **What we'd lose without it**: Flat list of nodes with no relationships. No hierarchy, no cross-references, no semantic connections.
>
> **How we do it**: Build edges in layers - structural first (hierarchy + sequence), then references, then semantic. Filter to active nodes. Deduplicate.

### Q6: Why are there so many edge types? Couldn't we simplify?

**Edge Types** (from `validate_graph.py:35-43`):
```python
ALLOWED_EDGE_TYPES = {
    # Structural edges
    "parent_of", "follows", "in_section",
    # Reference edges
    "references_box", "same_group", "same_field",
    # Semantic edges
    "includes", "excludes", "applies_if", "defines", "qualifies", "requires",
    "portion_of",
}
```

**Why distinct types?**

1. **Training pair generation uses edge semantics**:
   - `parent_of` → hierarchical pairs (box → concept)
   - `references_box` → cross-reference pairs (section mentioning another box)
   - `excludes` → **hard negatives** (things that should NOT be associated)
   - Different edge types generate different pair categories

2. **Graph expansion uses edge weights**:
   - Structural edges (confidence 1.0) are always safe to follow
   - Reference edges (confidence 0.8-0.95) need validation before ranking boost
   - Semantic edges (varies) carry meaning that affects expansion strategy

3. **Validation checks edge diversity** (A8):
   - A graph with only `references_box` edges is semantic-poor
   - Need `defines`, `qualifies`, `applies_if` for meaningful pairs
   - Current failure: 88% references_box, only 5 semantic edges

**The edge type taxonomy maps to training pair categories**:
```python
EDGE_TYPE_CATEGORIES = {
    "parent_of": "hierarchical",
    "includes": "hierarchical",
    "references_box": "cross_reference",
    "defines": "semantic",
    "qualifies": "semantic",
    "applies_if": "semantic",
    "excludes": "negative",  # Hard negatives!
}
```

---

### Q7: What is "nearest preceding containment" and why use it for hierarchy?

**Code Reference** (`edges.py:184-268`):
```python
def build_section_hierarchy_edges(sections_df: pd.DataFrame, doc_id: str) -> List[Edge]:
    # Track hierarchy state
    current_section = None
    current_box = None

    for _, row in ordered.iterrows():
        anchor_type = row.get("anchor_type", "")

        if anchor_type == "section":
            parent_id = doc_root_id
            current_section = anchor_id
            current_box = None  # Reset box when entering new section

        elif anchor_type == "box":
            if current_section:
                parent_id = f"{doc_id}:{current_section}"
            else:
                parent_id = doc_root_id
            current_box = anchor_id

        elif anchor_type == "subsection":  # concept
            if current_box:
                parent_id = f"{doc_id}:{current_box}"
            elif current_section:
                parent_id = f"{doc_id}:{current_section}"
            else:
                parent_id = doc_root_id
```

**Why "nearest preceding"?**

1. **IRS documents have implicit scope from layout**:
   - "Box 1a. Ordinary Dividends" followed by subsection headers means those subsections belong to Box 1a
   - The scope continues until the next box header (Box 1b)
   - This is "nearest preceding" - the most recent structural header governs

2. **State machine tracks open containers**:
   - `current_section`: The open section (e.g., "Specific Instructions")
   - `current_box`: The open box within that section (e.g., Box 1a)
   - When a new section opens, box resets to None
   - When a new box opens, it becomes the current container for concepts

3. **Fallback chain ensures no orphans**:
   - Concept → current_box → current_section → doc_root
   - Box → current_section → doc_root
   - Every node gets a parent

**This is NOT explicit containment** - the PDF doesn't say "Subsection X belongs to Box Y". We infer it from reading order and structural headers. This is why validation (A7) checks that the hierarchy is acyclic and single-parent.

---

### Q8: Why do we build `in_section` edges separately from `parent_of`?

**Code Reference** (`edges.py:416-482`):
```python
def build_in_section_edges(sections_df: pd.DataFrame, doc_id: str) -> List[Edge]:
    """
    Denormalized edges for fast "all boxes in section X" queries.
    Only emitted for subsection (concept) → section relationships.
    """
    # Build anchor → section map
    for _, row in ordered.iterrows():
        if anchor_type == "section":
            current_section = anchor_id
        elif anchor_type in ("box", "subsection") and current_section:
            anchor_to_section[anchor_id] = current_section

    # Emit in_section edges only for subsections (concepts)
    for anchor_id, section_id in anchor_to_section.items():
        if anchor_type == "subsection":
            edges.append(Edge(
                edge_type="in_section",
                ...
            ))
```

**Why?**

1. **Query optimization for section-based retrieval**:
   - "Get all concepts in Specific Instructions" should be O(1) edge lookup
   - With only `parent_of`, you'd need to traverse: section → box → concept
   - `in_section` provides a direct shortcut

2. **Denormalization for common access patterns**:
   - Training pair generation often groups by section
   - Graph expansion might want "all related nodes in same section"
   - The extra edges cost storage but save query time

3. **Only for concepts, not boxes**:
   - Box → section is already one hop via `parent_of`
   - Concept → section would be two hops (concept → box → section)
   - The skip-level edge only helps where depth > 1

**Trade-off**: More edges = more storage, but queries like "section membership" become O(1).

---

### Q9: Why do `follows` edges exist? Isn't reading order implicit in sort?

**Code Reference** (`edges.py:271-319`):
```python
def build_section_follows_edges(sections_df: pd.DataFrame, doc_id: str) -> List[Edge]:
    """Build follows edges between sections in reading order."""
    for _, row in ordered.iterrows():
        if prev_anchor_id and prev_anchor_id != anchor_id:
            edges.append(Edge(
                edge_type="follows",
                source_node_id=f"{doc_id}:{prev_anchor_id}",
                target_node_id=node_id,
                ...
            ))
        prev_anchor_id = anchor_id
```

**Why explicit edges?**

1. **Graph queries need edges, not sort keys**:
   - "What comes after Box 1a?" is a graph traversal
   - Without `follows` edges, you'd need to reload the DataFrame and sort
   - The graph should be self-contained

2. **Training pairs from sequential context**:
   - "Box 1a" → "Box 1b" (adjacent boxes, likely related)
   - Sequential pairs are valuable for teaching document structure
   - The `follows` edge makes these pairs easy to generate

3. **Linearity check in validation**:
   - A8 checks for skeleton coverage including `follows`
   - If `follows` edges are missing, the graph has fragmented sequences
   - This catches sorting or ordering bugs

**Alternative considered**: Store `reading_order` as node attribute, compute adjacency on demand. Rejected because:
- Every traversal would need full sort
- Graph expansion algorithms expect edges, not attributes
- Validation (A5) specifically checks `follows` presence

---

### Q10: How do reference edges differ from typed edges?

**Reference Edges** (`edges.py:489-545`):
```python
def build_box_reference_edges(references_df: pd.DataFrame, ...) -> List[Edge]:
    for _, ref in references_df.iterrows():
        if ref.get("ref_type") != "box_reference":
            continue
        edges.append(Edge(
            edge_type="references_box",
            confidence=float(ref.get("confidence", 0.9)),
            source_evidence=ref.get("evidence_text"),
            created_by=ref.get("created_by", "regex"),
        ))
```

**Typed Edges** (via `typed_edges.py`):
```python
# Examples:
#   excludes: "Box 1a does NOT include nondividend distributions"
#   applies_if: "Report in box 2e IF holding period met"
#   defines: "Box 1b is the portion that qualifies..."
```

**Key Differences**:

| Aspect | Reference Edges | Typed Edges |
|--------|-----------------|-------------|
| What they capture | "Box X mentions Box Y" | "Box X has semantic relationship R to Box Y" |
| Direction semantics | Source mentions target | Source defines/excludes/qualifies target |
| Polarity | Always neutral | Can be negative (excludes) |
| Confidence | 0.85-0.95 (regex match quality) | 0.80-0.95 (pattern specificity) |
| Training use | Cross-reference pairs | Semantic pairs + hard negatives |

**Why both?**

1. **References are abundant but shallow**:
   - "See Box 1a" appears dozens of times
   - This tells you boxes are related, not HOW they're related
   - Good for coverage, not for deep semantics

2. **Typed edges are sparse but semantic**:
   - "Box 1a includes amounts in Box 1b" is a containment relationship
   - "Box 1a does NOT include..." is a negation (hard negative gold)
   - Current problem: only 5 semantic edges vs 61 references

3. **A8 validation checks the balance**:
   - `references_box` at 88% > 80% threshold → FAIL
   - Need more typed edges to pass validation
   - This is the current gap

---

## Part 3: Typed Edge Extraction (`typed_edges.py`)

> **What it is**: Extract semantic edges with polarity from text using template-first matching.
>
> **Role in bigger picture**: Typed edges are the semantic intelligence of the graph. They capture HOW things are related, not just THAT they're related. Critical for hard negative mining.
>
> **What we'd lose without it**: Graph would be reference-only (shallow). No hard negatives from `excludes`. Training pairs would lack semantic diversity.
>
> **How we do it**: Regex pattern tables for each edge type, sentence gating to prevent bleed, excludes-first precedence, negation context checks.

### Q11: Why template-first matching instead of LLM extraction?

**Code Reference** (`typed_edges.py:191-268`):
```python
# Pattern tables - evaluated in priority order
EXCLUDES_PATTERNS = [
    ("does_not_include", re.compile(r"(?i)does\s+not\s+include"), 0.95),
    ("do_not_include", re.compile(r"(?i)do\s+not\s+include"), 0.95),
    ...
]

APPLIES_IF_PATTERNS = [
    ("report_if", re.compile(r"(?i)report\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)\s+if\b"), 0.90),
    ...
]
```

**Why templates?**

1. **Determinism**:
   - Regex patterns give the same output for the same input
   - LLMs can hallucinate edges or miss obvious patterns
   - Templates are auditable: you can see exactly why an edge was created

2. **Speed**:
   - Regex runs in microseconds
   - LLM API calls take seconds and cost money
   - For 1000+ paragraphs, templates are practical

3. **Confidence calibration**:
   - Each pattern has a calibrated confidence (0.85-0.95)
   - We know "does not include" is 95% reliable
   - LLM confidence is harder to calibrate

4. **Prioritized precedence**:
   - Patterns are ordered by specificity
   - First match wins within a type
   - This prevents duplicate edges from overlapping patterns

**When would LLM be better?**

- Implicit relationships ("This amount is usually larger than Box 1b")
- Complex conditionals spanning multiple sentences
- Term resolution ("the qualified amount" → Box 1b)

The architecture supports LLM via `created_by="llm"` field, but Phase 1 uses templates only.

---

### Q12: What is "sentence gating" and why is it critical?

**Code Reference** (`typed_edges.py:750-847`):
```python
def extract_typed_edges_from_section(anchor_id, body_text, valid_box_keys, source_box_key=None):
    # Split once; keep offsets stable vs full_text
    sents = split_sentences_with_offsets(body_text)

    # PASS 1: EXCLUDES (global precedence) across all sentences
    for i, (sent, s, e) in enumerate(sents):
        edges_i = extract_excludes_edges(anchor_id, sent, valid_box_keys)
        edges_i = _attach_sentence_provenance(edges_i, i, s, e)
        excludes_edges.extend(edges_i)
```

**Why process per-sentence?**

1. **Prevents cross-sentence bleed**:
   - "Box 1a includes qualified dividends. Box 2a does NOT include capital gains."
   - If we process as one block, "does NOT include" might incorrectly attach to Box 1a
   - Sentence gating ensures patterns match within their sentence

2. **Evidence precision**:
   - `evidence_char_start` and `evidence_char_end` locate the exact sentence
   - For highlighting, we want to show the supporting sentence, not the whole section
   - Offsets are relative to `full_text` for stable provenance

3. **Polarity isolation**:
   - Negation in sentence 2 shouldn't affect edge extraction in sentence 1
   - Each sentence is an independent extraction context
   - The `excludes` pass first, then other types skip `excluded_boxes`

**Sentence splitter is IRS-tuned** (`typed_edges.py:44-144`):
- Handles abbreviations (U.S., No., Sec., Pub.)
- Treats bullet points as sentence boundaries
- Under-splits rather than over-splits (conservative)

---

### Q13: Why does `excludes` have global precedence over other edge types?

**Code Reference** (`typed_edges.py:786-800`):
```python
# PASS 1: EXCLUDES (global precedence) across all sentences
excludes_edges: List[TypedEdgeCandidate] = []
for i, (sent, s, e) in enumerate(sents):
    edges_i = extract_excludes_edges(anchor_id, sent, valid_box_keys)
    excludes_edges.extend(edges_i)

excluded_boxes = {ed.target_box_key for ed in excludes_edges}

# PASS 2: Other edges skip excluded targets
applies_if_edges = extract_applies_if_edges(..., excluded_boxes)
```

**Why?**

1. **Negative knowledge is more precise than positive**:
   - "Box 1a does NOT include nondividend distributions" is definitive
   - "Box 1a includes..." might be overridden by a later exclusion
   - When both apply, the exclusion is the authoritative statement

2. **Prevents contradictory edges**:
   - Without precedence: `defines(concept → box_1a)` AND `excludes(concept → box_1a)`
   - With precedence: only `excludes` is emitted
   - The `excluded_boxes` set blocks other edge types from those targets

3. **Hard negative mining depends on clean exclusions**:
   - `excludes` edges are gold for hard negatives
   - If we also had `defines` edges to the same target, the negative would be ambiguous
   - Clean exclusions = clean hard negatives

**The precedence chain**:
1. `excludes` (wins over everything)
2. `portion_of` (wins over `defines` - more specific)
3. `applies_if`, `defines`, `qualifies` (parallel, after exclusions)
4. `requires`, `includes` (box→box only, section-level)

---

### Q14: What is the `portion_of` edge type and why is it special?

**Code Reference** (`typed_edges.py:271-299, 672-733`):
```python
PORTION_OF_PATTERNS = [
    ("any_part_of_box", re.compile(
        r"(?i)(?:if\s+)?any\s+part\s+of\s+(?:the\s+)?"
        r"(?:amount|total|dividends?|distribution|gain|income)?\s*"
        r"(?:reported\s+in\s+)?box\s+(\d+[a-z]?)"
    ), 0.95),
    ...
]

def extract_portion_of_edges(anchor_id, text, valid_box_keys, excluded_boxes):
    # SENTENCE-GATED FIXES (precision-first)
    if not has_subset_cue(text):
        return []
    if has_narration_filter(text):
        return []
```

**What it captures**:
- "If any part of box 1a is qualified dividends..."
- "The portion of the amount reported in box 2a"
- "Some of box 1a is attributable to..."

**Why special treatment?**

1. **Subset semantics differ from containment**:
   - `includes`: Box 1a contains all of Box 1b
   - `portion_of`: Some of Box 1a qualifies for special treatment
   - The difference matters for reasoning (not all dividends in 1a are qualified)

2. **Precision-first with subset cues**:
   - Must have explicit cue: "any part", "portion", "some of"
   - Without cue, the pattern might fire on false positives
   - The `has_subset_cue()` gate prevents over-matching

3. **Narration filter for disambiguation**:
   - "The portion was determined by..." is description, not rule
   - "The portion must be reported in..." is a rule
   - Past tense narration is filtered to avoid false positives

4. **Wins over `defines`**:
   - If something is a "portion of Box 1a", it's more specific than just "defining" 1a
   - The `portion_of_boxes` set suppresses `defines` for those targets

---

### Q15: Why is negation context checked even for non-excludes edges?

**Code Reference** (`typed_edges.py:328-337`):
```python
def _has_negation_context(text: str, pos: int, window: int = 80) -> bool:
    """Check if position has negation context (excludes should win)."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    context = text[start:end]

    for _, pattern, _ in EXCLUDES_PATTERNS:
        if pattern.search(context):
            return True  # Skip this edge, let excludes handle it
    return False
```

**Why check context for non-excludes patterns?**

1. **Precedence enforcement via context window**:
   - Even if we're extracting `applies_if`, nearby negation changes semantics
   - "Report in box 2e if... but does NOT include..." → excludes should win
   - The 80-char window catches adjacent negation

2. **Prevents contradictory edges within sentence**:
   - Sentence: "Include amounts from box 1b but do not include nondividend distributions"
   - Without context check: both `includes` and `excludes` edges emitted
   - With context check: only `excludes` survives (the `includes` pattern skips)

3. **Defense against pattern overlap**:
   - Some positive patterns have fragments that appear in negation contexts
   - "Box 2e includes" might appear after "does not include...see Box 2e includes"
   - Context check catches these edge cases

---

## Stage 9: Validation Framework

> **What it is**: Comprehensive quality checks on the graph - deterministic audits (A1-A8) and LLM-as-judge placeholders (B1-B3).
>
> **Role in bigger picture**: Validation gates the pipeline. If checks fail, the graph shouldn't be used for training or retrieval. Catches extraction bugs, ensures quality.
>
> **What we'd lose without it**: Silent failures. Bad training data. Incorrect retrieval. No confidence in graph quality.
>
> **How we do it**: Deterministic checks (coverage, integrity, distribution) run always. LLM checks (semantic correctness) planned for Phase B.

---

## Part 4: Validation Framework (`validate_graph.py`)

> **What it is**: Quality checks organized in two phases - fast deterministic audits and expensive LLM semantic validation.
>
> **Role in bigger picture**: Validation is the quality gate. A1-A8 run on every pipeline execution. B1-B3 will validate semantic correctness.
>
> **What we'd lose without it**: No way to know if extraction succeeded. Bad graphs would propagate to training and retrieval.
>
> **How we do it**: Phase A checks are pure Python (coverage, contamination, integrity, distribution). Phase B checks will call Claude API for semantic judgment.

### Q16: Why two phases of validation (A = deterministic, B = LLM)?

**Code Reference** (`validate_graph.py:131-1142`):
```python
# Phase A: Deterministic Audits
audit_a1_anchor_coverage(nodes_df, sections_df)
audit_a2_artifact_contamination(nodes_df, edges_df)
...
audit_a8_edge_type_distribution(edges_df, nodes_df)

# Phase B: LLM-as-Judge (Placeholders)
judge_b1_anchor_assignment(samples)
judge_b2_edge_correctness(edges_df, nodes_df)
judge_b3_pair_suitability()
```

**Why split?**

1. **Deterministic checks are fast and cheap**:
   - Run in seconds, no API cost
   - Should gate every pipeline run
   - Catch structural bugs before expensive LLM validation

2. **LLM checks are expensive but catch semantic issues**:
   - "Is this edge supported by the quoted evidence?" requires understanding
   - Regex can't judge if "Box 1b is the qualified portion" is a valid `defines` edge
   - LLM-as-judge fills the gap

3. **Phased rollout**:
   - Phase A implemented and running
   - Phase B is placeholder (will integrate Claude API)
   - Can ship with A-only validation, add B incrementally

**The split also reflects confidence**:
- A checks have confidence 1.0 (pass/fail is objective)
- B checks have confidence 0.7-0.95 (LLM judgment can be wrong)

---

### Q17: What does A7 (Hierarchy Integrity) actually check?

**Code Reference** (`validate_graph.py:683-922`):
```python
def audit_a7_hierarchy_integrity(edges_df, nodes_df):
    """
    Three independent checks:
    1. Parent cardinality: roots=0 parents, non-roots=exactly 1 parent
    2. Acyclicity: no cycles in parent_of subgraph (DFS with coloring)
    3. Depth bound: reachable nodes have depth <= MAX_HIERARCHY_DEPTH
    """
```

**Why these three checks?**

1. **Parent cardinality (single-parent tree)**:
   - Without this, the hierarchy is a DAG or graph, not a tree
   - Training pair generation assumes tree structure
   - Multi-parent nodes create ambiguous ancestry

2. **Acyclicity (no back-edges)**:
   - A cycle means A is ancestor of B, B is ancestor of A
   - This breaks depth computation, traversal termination
   - DFS coloring (WHITE→GRAY→BLACK) detects back-edges

3. **Depth bound (MAX_HIERARCHY_DEPTH=5)**:
   - doc_root(0) → section(1) → box(2) → concept(3) → paragraph(4)
   - Deeper than 5 suggests classification bug
   - Also catches infinite loops if cycle detection misses something

**The DFS coloring algorithm** (`validate_graph.py:794-824`):
```python
WHITE, GRAY, BLACK = 0, 1, 2

def dfs_cycle_detect(node, path):
    color[node] = GRAY  # Currently visiting
    for child in parent_to_children.get(node, []):
        if color[child] == GRAY:
            # Back edge = cycle!
            cycles_found.append(path + [child])
        elif color[child] == WHITE:
            dfs_cycle_detect(child, path + [child])
    color[node] = BLACK  # Done visiting
```

This is textbook cycle detection with O(V+E) complexity.

---

### Q18: Why does A8 fail when `references_box` exceeds 80%?

**Code Reference** (`validate_graph.py:925-1045`):
```python
REF_DOMINANCE_FAIL = 0.80  # FAIL if references_box > 80%
MIN_SEMANTIC_TYPES = 2     # Need at least 2 distinct types

if ref_ratio > REF_DOMINANCE_FAIL:
    findings.append(Finding(
        severity="error",
        message=f"references_box dominates non-structural edges ({ref_ratio:.0%} > 80%)",
        evidence=f"references_box={ref_box_n}, semantic={semantic_n}",
        recommendation="Increase typed edge recall or reduce low-value references_box edges"
    ))
```

**Why this threshold?**

1. **Reference-only graphs have limited training value**:
   - `references_box` tells you "Box X mentions Box Y"
   - It doesn't tell you: defines, excludes, includes, applies_if
   - Training pairs from references alone are shallow

2. **Semantic diversity drives pair quality**:
   - `excludes` edges give hard negatives (critical for contrastive learning)
   - `defines` edges give definition pairs (critical for concept queries)
   - 80% references = only 20% semantic = insufficient diversity

3. **The 80% threshold is empirical**:
   - Below 65%: healthy mix
   - 65-80%: warning (track improvement)
   - Above 80%: fail (semantic poverty)

**Current status**: 88% references_box, only 5 semantic edges → FAIL

**Fix path**: Improve typed edge patterns in `typed_edges.py` to capture more `defines`, `qualifies`, `applies_if`.

---

### Q19: What would B2 (Edge Correctness) check if implemented?

**Code Reference** (`validate_graph.py:1090-1119`):
```python
def judge_b2_edge_correctness(edges_df, nodes_df):
    """B2: Judge edge correctness for expansion-critical edges"""

    # Filter to expansion-critical edge types
    critical_types = {"includes", "references_box", "references_section", "defines"}
    critical_edges = edges_df[edges_df["edge_type"].isin(critical_types)]

    # Placeholder: Sample and have LLM judge:
    #   - Is the edge supported by the quoted evidence?
    #   - Is the direction correct?
    #   - Is the edge type correct?
    #   - Should confidence be adjusted?
```

**What LLM would check**:

1. **Evidence support**:
   - Edge evidence: "...Box 1a includes amounts in boxes 1b and 2e..."
   - LLM question: "Does this evidence support an 'includes' relationship from Box 1a to Box 1b?"
   - Expected: Yes (clear containment statement)

2. **Direction correctness**:
   - Some edges are directional (parent_of, includes)
   - LLM checks if source→target matches the semantic relationship
   - Common error: swapping source and target

3. **Type correctness**:
   - Did we label `excludes` vs `includes` correctly?
   - Edge evidence with negation should be `excludes`, not `includes`
   - LLM catches type misclassification

4. **Confidence adjustment**:
   - LLM might say "this is ambiguous, lower confidence to 0.7"
   - Or "this is definitively supported, raise to 0.95"
   - Updates `confidence` field based on semantic judgment

**Why not implemented yet?**: API integration, cost management, structured output parsing. These are Phase 2 tasks.

---

### Q20: How does the validation report enable CI integration?

**Code Reference** (`validate_graph.py:1148-1289, 1363-1383`):
```python
def generate_markdown_report(report):
    # Human-readable markdown with tables, findings, recommendations
    ...

def run_validation(output_dir):
    # Save markdown report
    report_path = output_path / "graph_quality_report.md"
    with open(report_path, "w") as f:
        f.write(md_report)

    # Save JSON for CI
    json_path = output_path / "graph_quality_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "summary": report.summary,
            "checks": [...]
        }, f, indent=2)
```

**Dual output for different consumers**:

1. **Markdown for humans**:
   - Tables showing check results
   - Findings with evidence snippets
   - Ranked fix recommendations
   - Readable in GitHub, Jupyter, or terminal

2. **JSON for CI/CD**:
   - Machine-parseable summary
   - `overall_status`: "PASS" or "FAIL"
   - Can gate deployment on `checks_passed >= N`

**CI integration pattern**:
```bash
# In CI pipeline:
python validate_graph.py
STATUS=$(jq -r '.summary.overall_status' output/graph_quality_report.json)
if [ "$STATUS" != "PASS" ]; then
  echo "Graph validation failed"
  exit 1
fi
```

---

## Summary: Key Design Principles

1. **Granularity hierarchy**: doc_root → section → box → concept → paragraph enables both coarse and fine retrieval

2. **Typed edges carry semantics**: Not just "related" but "how related" (includes, excludes, defines, applies_if)

3. **Template-first extraction**: Deterministic, auditable, fast. LLM fills gaps where patterns fail.

4. **Sentence gating**: Prevents cross-sentence bleed, provides precise evidence provenance

5. **Exclusion precedence**: Negative knowledge wins, enabling clean hard negative mining

6. **Two-phase validation**: Fast deterministic checks gate every run; expensive LLM checks validate semantics

7. **Dual-format output**: Markdown for humans, JSON for CI integration

---

## Current Gap: A8 Failure

**Problem**: references_box at 88% > 80% threshold, only 5 semantic edges

**Root cause**: Typed edge patterns aren't matching enough content

**Fix options**:
1. Add more patterns to `typed_edges.py` (covers more cases)
2. Loosen existing patterns (risks false positives)
3. Add LLM extraction for ambiguous cases (Phase 2)
4. Review false negatives in current extraction (debug mode)

The architecture is sound; the patterns need tuning.

---

## Quick Reference: Stage-by-Stage

| Stage | What | Role | Without It | How |
|-------|------|------|------------|-----|
| **8: Graph Construction** | Build nodes + edges | Create the knowledge graph | No graph, no pairs, no expansion | Layer by layer |
| **8-Nodes** | Build nodes at granularities | Retrieval units | Nothing to retrieve | sections→section nodes, elements→paragraph nodes |
| **8-Edges (Structural)** | parent_of, follows, in_section | Hierarchy + sequence | Flat graph | State machine + reading order |
| **8-Edges (Reference)** | references_box, same_group | Cross-connections | No horizontal links | From references_df |
| **8-Edges (Semantic)** | excludes, defines, applies_if | Deep semantics | Shallow training pairs | Regex patterns + sentence gating |
| **9: Validation** | Quality checks | Gate bad graphs | Silent failures | A1-A8 deterministic, B1-B3 LLM |
| **9-Phase A** | Deterministic audits | Fast, cheap, objective | Structural bugs slip through | Python checks |
| **9-Phase B** | LLM-as-judge | Semantic correctness | Wrong edge types undetected | Claude API (planned) |
