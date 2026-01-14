# Stages 5-6: Dialectical Analysis

A rigorous examination of anchor detection and content assignment - the structural backbone of the knowledge graph.

---

## Overview: From Elements to Sections

```
Classified Elements (from Stage 4)
         │
         ▼ [Stage 5: Anchor Detection]
Anchors (box_1a, sec_general_instructions, sub_qualified_dividends_a1b2c3d4)
         │
         ▼ [Stage 5: Timeline Building]
Anchor Timeline (each anchor owns reading_order range [start, end])
         │
         ▼ [Stage 6: Content Assignment]
Elements with anchor_id assignments
         │
         ▼ [Stage 6: Section Materialization]
Sections (aggregated text, metadata, bbox)
```

**The Core Insight**: Anchors are the structural skeleton. Content "flows" to the nearest anchor above it in reading order.

---

## Stage 5: Anchor Detection & Timeline Building

> **What it is**: Transform classified header elements into logical anchors with stable IDs, then build a timeline of reading order ranges that each anchor "owns."
>
> **Role in bigger picture**: Anchors become the section-level nodes in the knowledge graph. The timeline determines which content belongs to which anchor - the foundation of all content assignment.
>
> **What we'd lose without it**: No structural skeleton. Elements would have no parent. The graph would be flat (just paragraphs) with no hierarchy. Cross-references like "see Box 1a" couldn't resolve.
>
> **How we do it**: Parse box/section/subsection headers with ordered regex cascade, generate stable IDs, sort by reading order, compute ownership ranges via "next anchor starts where previous ends."

---

## Stage 5a: What Is an Anchor?

> **What it is**: A logical structural unit that "owns" content until the next anchor appears.
>
> **Role in bigger picture**: Anchors are the abstraction layer between physical headers and graph nodes. They handle one-to-many (grouped boxes), canonicalization, and ID stability.
>
> **What we'd lose without it**: Would have to use raw header elements as nodes. "Boxes 14-16" would be one node instead of three. IDs would be position-based (fragile) instead of semantic (stable).
>
> **How we do it**: Create Anchor dataclass with anchor_id, anchor_type, box_key, label, source_element_id, page, reading_order, and group info.

### Definition

An **anchor** is a structural header that "owns" the content that follows it until the next anchor. Anchors become nodes in the knowledge graph.

| Anchor Type | Source | Example | ID Format |
|-------------|--------|---------|-----------|
| `box` | BoxHeader elements | "Box 1a. Ordinary Dividends" | `box_1a` |
| `section` | SectionHeader elements | "General Instructions" | `sec_general_instructions` |
| `subsection` | SubsectionHeader elements | "Qualified Dividends" | `sub_qualified_dividends_a1b2c3d4` |

### Why Anchors Instead of Just Using Headers?

**Q: The headers are already classified. Why create a separate "anchor" concept?**

A: Headers are **physical** (text, position, role). Anchors are **logical** (semantic unit with ID, type, ownership range).

Consider "Boxes 14 through 16. State information":
- This is **one physical header** (one element)
- But **three logical anchors** (box_14, box_15, box_16)

The anchor abstraction handles:
1. **One-to-many**: One header → multiple anchors (grouped boxes)
2. **Canonicalization**: "Boxes 14-16" and "Boxes 14, 15, 16" → same anchor IDs
3. **ID stability**: Same anchor across document versions

---

## Stage 5b: Box Key Parsing

> **What it is**: Extract canonical box keys (like "1a", "14", "2e") from various header formats (single, range, double, "through").
>
> **Role in bigger picture**: Box keys are the primary identifier for cross-references. When text says "see Box 1a", we need to resolve "1a" to the anchor `box_1a`.
>
> **What we'd lose without it**: Couldn't parse "Boxes 14-16" into three separate anchors. Range references would fail to resolve. Training pairs would miss box relationships.
>
> **How we do it**: Ordered regex cascade - try range pattern first, then "through", then "and", then single. Expand numeric ranges (14-16 → [14,15,16]).

### The Problem

Box headers have multiple formats:
```
Box 1a. Ordinary dividends           → single: ["1a"]
Boxes 14-16. State information       → range:  ["14", "15", "16"]
Boxes 2a and 2b. Capital gains       → double: ["2a", "2b"]
Boxes 14 through 16. State info      → through: ["14", "15", "16"]
```

We need to extract canonical box keys from all formats.

### The Solution: Ordered Regex Cascade

```python
# anchors.py - Order matters: specific patterns first
BOX_RANGE_PARSE = r"^Box(?:es)?\s+(\d+[a-z]?)\s*[-–]\s*(\d+[a-z]?)\b"   # 14-16
BOX_THROUGH_PARSE = r"^Box(?:es)?\s+(\d+)\s+through\s+(\d+)\b"           # 14 through 16
BOX_DOUBLE_PARSE = r"^Box(?:es)?\s+(\d+[a-z]?)\s+and\s+(\d+[a-z]?)\b"   # 2a and 2b
BOX_SINGLE_PARSE = r"^Box(?:es)?\s+(\d+[a-z]?)\b"                        # 1a
```

**Q: Why check range before single?**

A: "Box 14-16" would match `BOX_SINGLE_PARSE` as "Box 14" if we checked it first. By checking range first, we capture the full intent.

**Q: Why both "range" (14-16) and "through" (14 through 16)?**

A: IRS documents use both formats. The regex handles the hyphen variants (`-` and `–`) and the word "through" separately because:
- Range: `14-16` → extracts both numbers from the pattern
- Through: `14 through 16` → extracts both numbers with "through" as separator

Both expand to the same result: `["14", "15", "16"]`.

### Range Expansion

```python
def _expand_numeric_range(lo: str, hi: str) -> List[str]:
    """Expand "14", "16" → ["14", "15", "16"]"""
    lo_num = int(re.sub(r"[a-z]", "", lo))  # Strip letters for range
    hi_num = int(re.sub(r"[a-z]", "", hi))
    return [str(k) for k in range(min(lo_num, hi_num), max(lo_num, hi_num) + 1)]
```

**Q: Why strip letters before expanding?**

A: Consider "Box 2a-2f". We can't enumerate "2a" to "2f" directly. By stripping to "2" and "2", we get `["2"]`. This is a **safe fallback** - we don't generate wrong keys.

**Q: What about "1a-1b"?**

A: This returns `["1"]` (just the number). The IRS format doesn't use letter ranges like "1a through 1b" - they use "Boxes 1a and 1b" for those cases.

---

## Stage 5c: Section and Subsection IDs

> **What it is**: Generate stable, human-readable IDs for section and subsection anchors.
>
> **Role in bigger picture**: IDs are how we reference nodes in edges, training pairs, and queries. Stable IDs enable cross-document alignment and debugging.
>
> **What we'd lose without it**: Position-based IDs like "el_2_3_seg0" that break when document changes. No way to say "this section in doc A is the same as this section in doc B."
>
> **How we do it**: Sections use canonical map lookup with slug fallback. Subsections use slug + position hash for uniqueness.

### Section IDs: Canonical Mapping

```python
DEFAULT_SECTION_ID_MAP = {
    "future developments": "sec_future_developments",
    "general instructions": "sec_general_instructions",
    "specific instructions": "sec_specific_instructions",
    # ...
}

def get_section_id(text: str) -> Tuple[str, str]:
    for pattern, sid in section_map.items():
        if text.lower().startswith(pattern):
            return sid, text
    # Fallback: auto-generate from text
    return "sec_" + slug(text[:30]), text
```

**Q: Why a hardcoded map instead of always auto-generating?**

A: **Stability across documents**. If 1099-DIV says "General Instructions" and 1099-INT says "General Instruction" (typo), both should map to `sec_general_instructions`. The canonical map handles variations.

**Q: What if a section doesn't match the map?**

A: Fallback to auto-generated ID: `sec_` + slugified first 30 chars. This handles document-specific sections gracefully.

### Subsection IDs: Slug + Position Hash

```python
def get_subsection_id(text: str, source_element_id: str) -> Tuple[str, str]:
    title_slug = slug_title(text, max_len=30)
    position_hash = stable_hash([source_element_id], length=8)
    return f"sub_{title_slug}_{position_hash}", text
```

**Q: Why include a position hash?**

A: **Uniqueness without content dependency**. Consider:
- "Qualified Dividends" appears on page 2
- "Qualified Dividends" appears on page 4 (different section!)

If we only used the slug, both would be `sub_qualified_dividends` → collision.

The position hash (`a1b2c3d4`) comes from the source element ID, which includes page and block:
```
1099div_filer:2:3:seg0 → hash = a1b2c3d4
1099div_filer:4:1:seg0 → hash = e5f6g7h8
```

Result: `sub_qualified_dividends_a1b2c3d4` and `sub_qualified_dividends_e5f6g7h8`.

**Q: Why not just use the element ID directly?**

A: **Human readability**. `sub_qualified_dividends_a1b2c3d4` tells you what the section is about. `sub_1099div_filer_2_3_seg0` is opaque.

---

## Stage 5d: Anchor Timeline

> **What it is**: Assign each anchor a reading order range [start, end] representing which elements it "owns."
>
> **Role in bigger picture**: The timeline is the mechanism for content assignment. Without it, we'd have no way to know which paragraphs belong to which boxes.
>
> **What we'd lose without it**: Content assignment would require spatial proximity heuristics that break on two-column layouts. No principled way to handle "anchor owns content until next anchor."
>
> **How we do it**: Sort anchors by (page, reading_order), then for each anchor: end_ro = next_anchor.start_ro - 1. Last anchor on page owns until max reading order.

### The Concept

Each anchor "owns" a range of reading orders on its page:

```
Page 2 Timeline:
  anchor_id          start_ro  end_ro
  ─────────────────  ────────  ──────
  sec_general_inst   0         4
  box_1a             5         12
  box_1b             13        18
  ...
```

Elements with `reading_order` in `[5, 12]` belong to `box_1a`.

### Building the Timeline

```python
def build_anchor_timeline(anchors_df, elements_df):
    # 1. Sort anchors by (page, reading_order)
    timeline = anchors_df.sort_values(["page", "start_reading_order"])

    # 2. For each anchor, end_ro = next_anchor.start_ro - 1
    for page in timeline["page"].unique():
        page_anchors = timeline[timeline["page"] == page]
        for i, anchor in enumerate(page_anchors):
            if i + 1 < len(page_anchors):
                next_start = page_anchors.iloc[i + 1]["start_reading_order"]
                anchor["end_reading_order"] = next_start - 1
            else:
                # Last anchor: owns until end of page
                anchor["end_reading_order"] = max_reading_order_on_page
```

**Q: Why reading order instead of spatial proximity (y-coordinate)?**

A: **Two-column layouts break spatial proximity**. Consider:

```
Page 2:
┌─────────────────────┬─────────────────────┐
│ Box 1a              │ Box 1b              │
│ Content for 1a...   │ Content for 1b...   │
│ More content...     │ More content...     │
└─────────────────────┴─────────────────────┘
```

Spatially, "Content for 1a" (left) and "Content for 1b" (right) have the same y-coordinate. But their reading orders are:
- Box 1a: ro=5, Content 1a: ro=6, ro=7
- Box 1b: ro=15, Content 1b: ro=16, ro=17

Reading order respects column flow. Spatial proximity doesn't.

**Q: What if an anchor has no content?**

A: It gets an empty range: `[start_ro, start_ro - 1]` which is invalid, so no elements match. This is intentional - header-only anchors exist (they'll have empty body text in the section).

---

## Stage 5e: Deduplication

> **What it is**: Remove near-duplicate anchors that arise from layout detection artifacts.
>
> **Role in bigger picture**: Clean anchors = clean graph. Duplicate anchors would create duplicate nodes, confusing edges, and wasted training pairs.
>
> **What we'd lose without it**: Same heading detected twice would create two nodes. Validation (A1) would fail on duplicate anchor_ids. Graph traversal would hit duplicates.
>
> **How we do it**: Check (page, normalized_label) for near-duplicates on subsections. Drop exact anchor_id duplicates. Keep="first" preserves reading order.

### Near-Duplicate Detection

```python
def deduplicate_anchors(anchors_df):
    # Check for near-duplicates: same page + same normalized label
    subsections["_norm_label"] = subsections["label"].str.lower().str.strip()
    dup_mask = subsections.duplicated(subset=["page", "_norm_label"], keep="first")

    # Also drop exact anchor_id duplicates
    anchors_df = anchors_df.drop_duplicates(subset=["anchor_id"], keep="first")
```

**Q: Why check for near-duplicates on subsections specifically?**

A: Box and section headers are rare and distinct. But subsection detection (layout-based) can produce duplicates:
- Same bold text appears twice (repeated heading in PDF)
- Slightly different formatting triggers double detection

By checking `(page, normalized_label)`, we keep only the first occurrence.

**Q: Why `keep="first"`?**

A: First in reading order = the "real" header. Later duplicates are artifacts or repeats.

---

## Stage 6: Content Assignment & Section Materialization

> **What it is**: Assign each element to its governing anchor using the timeline, then aggregate elements into sections with full text, metadata, and provenance.
>
> **Role in bigger picture**: Transforms the anchor skeleton + element content into complete sections ready for graph construction. This is where content meets structure.
>
> **What we'd lose without it**: Elements would have no parent. Couldn't build hierarchy edges. Couldn't generate training pairs from section content. Graph would be structure-only (no text).
>
> **How we do it**: IntervalIndex lookup for O(n log m) assignment, special handling for grouped anchors and preamble, then aggregate with header/body separation and hyphenation repair.

---

## Stage 6a: Content Assignment

> **What it is**: For each element, find which anchor's reading order range contains it and assign that anchor_id.
>
> **Role in bigger picture**: This is the actual "content to structure" binding. After this, every element knows its parent anchor.
>
> **What we'd lose without it**: Elements would be orphans. No way to know "this paragraph belongs to Box 1a."
>
> **How we do it**: Build IntervalIndex from anchor ranges, vectorized lookup O(n log m), handle grouped anchors by assigning to all members.

### The Algorithm

For each element, find which anchor's reading order range contains it:

```python
# Naive O(n*m) approach:
for element in elements:
    for anchor in page_anchors:
        if anchor.start_ro <= element.reading_order <= anchor.end_ro:
            element.anchor_id = anchor.anchor_id
            break
```

### Vectorized Assignment with IntervalIndex

```python
def _assign_page_elements_vectorized(elements_df, page_mask, page_anchors, grouped_map):
    # Build IntervalIndex from anchor ranges
    intervals = pd.IntervalIndex.from_arrays(
        page_anchors["start_reading_order"].values,
        page_anchors["end_reading_order"].values,
        closed="both",  # start <= ro <= end
    )

    # Vectorized lookup: O(n log m) instead of O(n*m)
    page_ro = elements_df.loc[page_mask, "reading_order"].values
    idx_matches = intervals.get_indexer(page_ro)  # Returns index or -1

    # Single vectorized assignment
    elements_df.loc[page_mask, "anchor_id"] = np.where(
        idx_matches >= 0,
        anchor_ids[idx_matches],
        existing_anchor_ids,
    )
```

**Q: Why IntervalIndex?**

A: **Performance**. For 1099-DIV:
- ~120 elements
- ~50 anchors per page (at most)

Naive: O(120 × 50) = 6,000 comparisons
IntervalIndex: O(120 × log(50)) ≈ 680 comparisons

For larger documents, the difference is dramatic:
- 1000 elements × 200 anchors = 200,000 comparisons (naive)
- 1000 × log(200) ≈ 7,600 comparisons (IntervalIndex)

**Q: Why keep the iterrows implementation?**

A: **Parity testing during rollout**. The `verify_vectorized_parity()` function runs both implementations and compares results:

```python
def verify_vectorized_parity(elements_df, timeline):
    # Run both implementations
    df_vec = vectorized_assign(elements_df)
    df_iter = iterrows_assign(elements_df)

    # Compare results
    mismatched = df_vec["anchor_id"] != df_iter["anchor_id"]
    return {"matched": len(mismatched) == 0, "speedup": iter_time / vec_time}
```

This ensures correctness before removing the legacy code.

### Grouped Anchor Assignment

```python
# For "Boxes 14-16", elements belong to ALL three anchors
if src_elem in grouped_map:
    elements_df.at[idx, "anchor_ids"] = grouped_map[src_elem]  # ["box_14", "box_15", "box_16"]
```

**Q: Why assign to ALL anchors in a group, not just one?**

A: **Semantic accuracy**. When IRS says "Boxes 14-16. State information", the content is about ALL three boxes. For the knowledge graph:
- "State ID number" → relates to box_14, box_15, AND box_16
- Training pairs should include all three relationships

If we only assigned to `box_14`, we'd miss edges to `box_15` and `box_16`.

### Preamble Assignment

```python
# Elements before first anchor go to "preamble"
first_anchor_start = page_anchors["start_reading_order"].min()
preamble_mask = elements["reading_order"] < first_anchor_start
elements.loc[preamble_mask, "anchor_id"] = "preamble"
```

**Q: Why have a preamble anchor at all?**

A: **Every element needs an anchor**. The document starts with:
- Title
- Date
- Revision info
- Introductory text

None of this belongs to a box or section. Without "preamble", these would be "unassigned" - a validation failure.

---

## Stage 6b: Section Materialization

> **What it is**: Aggregate all elements assigned to an anchor into a single section record with full text, metadata, and provenance.
>
> **Role in bigger picture**: Sections are the retrieval units. They become nodes in the graph with text for embedding, bbox for highlighting, element_ids for provenance.
>
> **What we'd lose without it**: Would have to re-aggregate on every query. No pre-computed text for embedding. Lost element-level provenance. No bbox for UI highlighting.
>
> **How we do it**: Group elements by anchor_id, separate header elements from body, join with appropriate newlines, repair hyphenation, compute bbox union.

### What Is Materialization?

Converting anchors + assigned elements into **sections** - the final retrieval units.

```python
Section:
    anchor_id: "box_1a"
    anchor_type: "box"
    box_key: "1a"
    label: "Ordinary Dividends"
    header_text: "Box 1a. Ordinary Dividends"
    body_text: "Enter the total ordinary dividends..."
    full_text: "Box 1a. Ordinary Dividends\n\nEnter the total..."
    element_count: 5
    element_ids: ["el_2_3_seg0", "el_2_3_seg1", ...]
    pages: [2]
    bbox: [72, 150, 300, 400]
```

### Header/Body Separation

```python
def split_header_body(header_elements_text, body_elements_text):
    header_text = "\n".join(header_elements_text)      # Single newline
    body_text = "\n\n".join(body_elements_text)        # Double newline (paragraphs)
    full_text = f"{header_text}\n\n{body_text}"
    return header_text, body_text, full_text
```

**Q: Why separate header and body?**

A: **Different semantic purposes**:
- `header_text`: For display, titles, navigation
- `body_text`: For retrieval, embedding, answering questions
- `full_text`: For complete context

The graph node uses `full_text` for embeddings, but UI might show `label` (header) separately.

**Q: Why different newline treatments?**

A: **Readability**:
- Headers are typically single lines or close together
- Body paragraphs need visual separation

```
Box 1a. Ordinary Dividends     ← header (single \n if multi-line)

Enter the total ordinary...    ← body paragraph 1

Include amounts from 1099...   ← body paragraph 2 (\n\n separates)
```

### Hyphenation Repair

```python
def repair_hyphenation(text: str) -> str:
    # "fur-\nnishing" → "furnishing"
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    return text
```

**Q: Why repair hyphenation?**

A: **PDF extraction artifact**. PDFs preserve line-break hyphens:
```
The fur-
nishing costs...
```
Extracted as: `"The fur-\nnishing costs..."`

This creates problems:
1. **Retrieval**: Query "furnishing" won't match "fur-\nnishing"
2. **Embedding**: The hyphenated form has different embedding
3. **Display**: Looks broken in UI

**Q: Why only join if continuation is lowercase?**

A: **Avoid false repairs**:
```
"Well-\nKnown exceptions..."  → Keep as-is (Known is capitalized)
"well-\nknown exceptions..."  → Repair to "wellknown" ✗

Actually:
"fur-\nnishing" → "furnishing" ✓ (lowercase continuation)
"Smith-\nJones" → Keep as-is (Jones is capitalized = proper noun)
```

The lowercase check prevents destroying intentional hyphenation.

### Bounding Box Union

```python
def compute_bbox_union(bboxes: List) -> List[float]:
    x0 = min(b[0] for b in valid_bboxes)
    y0 = min(b[1] for b in valid_bboxes)
    x1 = max(b[2] for b in valid_bboxes)
    y1 = max(b[3] for b in valid_bboxes)
    return [x0, y0, x1, y1]
```

**Q: Why compute bbox union?**

A: **Provenance for highlighting**. When a user clicks on a search result, we highlight the section in the PDF viewer. The union bbox encompasses all elements in the section.

**Q: What if elements span multiple pages?**

A: The `pages` field is a list: `[2, 3]`. The bbox union is computed across all elements regardless of page - it's used for single-page highlighting (show the relevant portion on each page).

---

## Architectural Questions

### Why Not Just Use Headers Directly as Graph Nodes?

**Argument**: Headers already have IDs (element_id), position, text. Why create anchors?

**Answer**: The abstraction handles complexity:

| Scenario | Headers Only | Anchors |
|----------|--------------|---------|
| "Boxes 14-16" | 1 header | 3 anchors |
| Same header twice | 2 nodes (duplicate) | 1 anchor (deduped) |
| ID stability | `el_2_3_seg0` (position-based) | `box_1a` (semantic) |
| Cross-document alignment | IDs differ | Same anchor_id |

### Why Reading Order Ranges Instead of Parent-Child?

**Alternative**: Make each element a child of its anchor in a tree:
```
box_1a
├── paragraph_1
├── paragraph_2
└── list_item_1
```

**Problem**: How do you determine which anchor is the parent without... reading order ranges?

The "parent-child" relationship IS the reading order range assignment. We just express it differently (range containment vs. tree edge). Ranges are:
- Easier to compute (sort + scan)
- Easier to query (IntervalIndex)
- More explicit (visible in data, not implicit in structure)

### Why Materialize Sections Instead of Computing On-Demand?

**Alternative**: Keep elements and anchors separate. Compute section text when needed:
```python
def get_section_text(anchor_id):
    elements = elements_df[elements_df["anchor_id"] == anchor_id]
    return "\n\n".join(elements["text"])
```

**Answer**: **Cost amortization**. Section text is needed for:
1. Embedding generation (offline, once)
2. Retrieval display (every query)
3. Training pair generation (offline, once)
4. Validation (offline, occasionally)

Pre-materializing means:
- Text concatenation: once
- Hyphenation repair: once
- Bbox union: once
- Storage: cheap (Delta parquet)

On-demand would repeat this work constantly.

---

## Summary: Why This Architecture

| Decision | Why |
|----------|-----|
| Anchors as abstraction | Handles grouped headers, provides stable IDs |
| Ordered regex cascade | Specific patterns first prevents false matches |
| Slug + hash for subsection IDs | Human-readable + unique |
| Reading order ranges | Works with two-column layouts |
| IntervalIndex assignment | O(n log m) vs O(n*m) |
| Grouped anchor assignment | All boxes in "14-16" get the content |
| Header/body separation | Different semantic purposes |
| Hyphenation repair | Fixes PDF extraction artifacts |
| Pre-materialized sections | Amortizes computation cost |

The architecture converts **physical document structure** (headers, paragraphs, positions) into **logical semantic structure** (anchors, sections, ownership) that the knowledge graph can traverse.

---

## Quick Reference: Stage-by-Stage

| Stage | What | Role | Without It | How |
|-------|------|------|------------|-----|
| **5: Anchor Detection** | Create logical anchors from headers | Structural skeleton | No hierarchy, flat graph | Parse headers, generate IDs |
| **5a: Anchor Concept** | Abstract headers into anchors | Handle one-to-many, stability | Position-based fragile IDs | Anchor dataclass |
| **5b: Box Parsing** | Extract box keys from headers | Enable cross-references | "Boxes 14-16" = 1 node | Ordered regex cascade |
| **5c: Section IDs** | Generate stable section/subsection IDs | Cross-document alignment | IDs break on doc changes | Canonical map + slug+hash |
| **5d: Timeline** | Assign ownership ranges | Content assignment basis | No principled assignment | Sort + next_start - 1 |
| **5e: Deduplication** | Remove duplicate anchors | Clean graph | Duplicate nodes, failed validation | (page, label) dedupe |
| **6: Content Assignment** | Bind elements to anchors | Content meets structure | Orphan elements | IntervalIndex lookup |
| **6a: Assignment** | Vectorized element→anchor | Performance + correctness | O(n*m) slow lookup | IntervalIndex O(n log m) |
| **6b: Materialization** | Aggregate into sections | Retrieval-ready units | Re-aggregate every time | Group, join, repair, union |
