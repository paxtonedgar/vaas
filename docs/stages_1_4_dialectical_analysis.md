# Stages 1-4: Dialectical Analysis

A rigorous examination of what we do, why we do it that way, and what alternatives we rejected.

---

## Overview: The Extraction Stack

```
PDF Binary
    │
    ▼ [Stage 1: PyMuPDF]
Spans (atomic text units with font/position)
    │
    ▼ [Stage 2: Font Analysis]
Body Size Anchor (9.0pt for 1099-DIV)
    │
    ▼ [Stage 3: Line Building + Structure]
Lines with structural flags
    │
    ▼ [Stage 4: Element Classification]
Classified elements (BoxHeader, BodyTextBlock, etc.)
```

Each stage has design decisions worth questioning.

---

## Stage 1: Span Extraction

> **What it is**: Extract atomic text units (spans) from PDF with font metadata and bounding boxes.
>
> **Role in bigger picture**: Creates the foundation layer - everything downstream depends on reliable spans with accurate font/position data.
>
> **What we'd lose without it**: No extraction at all. Raw PDF bytes are unusable without parsing to text units. Poor span extraction cascades into wrong font detection, broken line building, and misclassified elements.
>
> **How we do it**: PyMuPDF's `page.get_text("dict")` walks block→line→span hierarchy, capturing text, font, flags, size, and bbox for each span.

### What We Do

```python
# pdf.py:extract_spans_from_pdf
for block in page.get_text("dict")["blocks"]:
    for line in block["lines"]:
        for span in line["spans"]:
            # Extract: text, font, flags, size, bbox
```

We extract at the **span level** - the atomic unit of text with consistent font properties.

### Why Spans?

**Q: Why not extract at the block or paragraph level directly?**

A: PyMuPDF's "blocks" are **visual groupings based on whitespace**, not semantic units. A block might contain:
- A heading AND its body text
- Two unrelated paragraphs that happen to be close
- Half of a column (if the layout confuses the parser)

Spans are the only reliable atomic unit. Everything else (lines, blocks, paragraphs) must be reconstructed with domain knowledge.

**Q: Why keep block_id, line_id, span_id?**

A: **Provenance tracking**. When we later find an edge with `source_evidence: "see Box 1a"`, we can trace back:
```
edge → element_id → source_span_ids → original PDF coordinates
```
This enables:
1. Debugging extraction errors
2. PDF highlighting in UI
3. Validation that evidence actually exists in source

### What We Capture

| Field | Purpose |
|-------|---------|
| `text` | Content |
| `font` | Font name (for bold detection via "Bold" substring) |
| `flags` | PDF text flags (bit 4 = bold) |
| `size` | Font size in points |
| `bbox` | Bounding box [x0, y0, x1, y1] |

**Q: Why both font name AND flags for bold detection?**

A: **Belt and suspenders**. Some fonts encode bold in the name ("Helvetica-Bold"), others in flags. We check both:

```python
# geometry.py:is_bold
def is_bold(font, flags):
    if "bold" in str(font).lower():
        return True
    return bool(int(flags or 0) & 16)  # Bit 4
```

---

## Stage 2: Body Font Inference

> **What it is**: Determine the dominant "body text" font size by finding the mode across all spans.
>
> **Role in bigger picture**: The body size becomes the baseline for all font-relative decisions - "larger than body" signals headings, "smaller than body" might signal footnotes.
>
> **What we'd lose without it**: All emphasis detection (headers, bold headings, subsections) would fail. Without knowing what "normal" looks like, we can't identify what's "emphasized."
>
> **How we do it**: Round span sizes to 0.1pt precision, count frequencies, take the mode. For 1099-DIV, this yields 9.0pt.

### What We Do

```python
# pdf.py:infer_body_font_size
sizes = spans_df["size"].round(1)
body_size = sizes.value_counts().idxmax()  # Mode
# → 9.0 for 1099-DIV
```

We use the **mode (most frequent)** font size as the body text anchor.

### Why Mode?

**Q: Why not median or mean?**

A: Consider the distribution:
- Body text (9.0pt): 2000 spans
- Headings (10pt): 50 spans
- Box headers (10.5pt): 22 spans

**Median** would be 9.0 (correct, but by accident)
**Mean** would be ~9.05 (slightly inflated by headers)
**Mode** is 9.0 (correct by definition - most common = body)

Mode directly captures "what most text looks like" which is the definition of body text.

**Q: Why round to 0.1 before counting?**

A: PDF extraction produces slight variations:
```
9.0, 9.0, 9.001, 8.999, 9.0, 9.0
```
Without rounding, these would be counted separately. Rounding to 0.1 clusters them:
```
9.0, 9.0, 9.0, 9.0, 9.0, 9.0  → count = 6
```

**Q: What if the document has multiple body fonts?**

A: This is a **simplifying assumption** for IRS documents, which are standardized. A mixed-font document would need a more sophisticated approach (clustering, per-section analysis). We accept this limitation because:
1. IRS docs use one body font
2. The pipeline is explicitly for this domain
3. Generalization adds complexity without benefit here

---

## Stage 3: Line Building + Structure Detection

> **What it is**: Aggregate spans into lines, detect two-column layouts, identify structural patterns (boxes, sections, subsections), and mark split triggers.
>
> **Role in bigger picture**: Transforms flat span data into structured lines with semantic flags. This is where we first understand "this is a header" vs "this is body text."
>
> **What we'd lose without it**: Couldn't distinguish headers from body text. All text would be one undifferentiated blob. Column reading order would be wrong (reading across instead of down).
>
> **How we do it**: Multiple sub-systems: line aggregation via groupby, column detection via x0-peak analysis, subsection detection via structural heuristics, split trigger assignment via priority-ordered pattern matching.

This is the most complex stage. Multiple sub-systems interact.

### 3a: Line Aggregation

> **What it is**: Combine spans that share the same line into a single line record with aggregated properties.
>
> **Role in bigger picture**: Lines are the unit of reading - humans read line by line. All downstream structure detection operates on lines, not spans.
>
> **What we'd lose without it**: Would have to re-aggregate spans for every downstream operation. Properties like "line is bold" require aggregation.
>
> **How we do it**: GroupBy on (doc_id, page, block_id, line_id), aggregate text via join, bbox via min/max, size via median, bold via any().

```python
# lines.py:build_line_dataframe
line_df = spans.groupby(["doc_id", "page", "block_id", "line_id"]).agg(
    line_text=("text", lambda x: "".join(x)),
    geom_x0=("x0", "min"),
    geom_y0=("y0", "min"),
    geom_x1=("x1", "max"),
    geom_y1=("y1", "max"),
    line_size=("size", "median"),
    line_bold=("bold", lambda x: bool(np.any(x))),
)
```

**Q: Why median size for the line, not mode?**

A: A line might have mixed fonts:
```
"See " (9pt) + "Box 1a" (9pt bold) + " for details." (9pt)
```
If one span happens to be slightly different (8.9pt), mode might pick the wrong size. Median is robust to outliers.

**Q: Why "any bold" instead of "majority bold"?**

A: **Emphasis is semantically significant even if partial**. The line:
```
"See Box 1a for details."
```
Where only "Box 1a" is bold - this is still an emphasized line for our purposes. We want to flag it for potential header detection.

### 3b: Column Detection

> **What it is**: Determine if a page has one or two columns by analyzing the distribution of left-margin x-coordinates.
>
> **Role in bigger picture**: Enables correct reading order. Two-column pages must be read column-by-column (down left, then down right), not row-by-row across.
>
> **What we'd lose without it**: Reading order would interleave columns. "Box 1a content" would be followed by "Box 2a content" (next row) instead of continuing Box 1a's content in the same column.
>
> **How we do it**: Bucket x0 values into 4px bins, find peaks in the distribution, require peaks to be >25% page width apart and second peak to have >25% of first peak's count.

```python
# columns.py:detect_columns_for_page
x0_rounded = (filtered["geom_x0"] / 4.0).round() * 4.0  # 4px buckets
x0_counts = x0_rounded.value_counts()
# Find two peaks separated by >= 25% page width
```

**Q: Why detect columns at all? Why not just use page_mid_x = 306?**

A: **Columns vary by page**. The 1099-DIV has:
- Page 1: Single column (title, intro)
- Pages 2-5: Two columns (instructions)

Hardcoding would misclassify page 1. Dynamic detection handles this:
```
Page 1: 1 column (no second peak)
Page 2: 2 columns (peaks at ~72 and ~318)
```

**Q: Why exclude list items from column detection?**

A: Bullets and enumerations are **indented**:
```
• Item 1          ← x0 = 84 (indented)
• Item 2          ← x0 = 84 (indented)
Regular text      ← x0 = 72 (margin)
```

If we include list items, we might detect x0=84 as a "peak" and incorrectly identify it as a column boundary. By excluding `_is_list_item`, we only count margin-aligned text.

**Q: Why min_peak_ratio = 0.25?**

A: The second column must have at least 25% as many margin-aligned lines as the first. This prevents noise:
- 100 lines at x0=72 (left margin)
- 80 lines at x0=318 (right margin) → ratio = 0.8 ✓
- 3 lines at x0=150 (random indent) → ratio = 0.03 ✗

**Q: Why min_distance_pct = 0.25?**

A: Columns must be at least 25% of page width apart. For 612pt page width, that's 153pt. This prevents:
- Misidentifying slight indentation as a second column
- Detecting intra-column structure as columns

### 3c: Subsection Detection

> **What it is**: Identify bold, margin-aligned lines that are likely subsection headers (not box headers or section headers, but lower-level structure).
>
> **Role in bigger picture**: Subsections become "concept" nodes in the graph. They're the intermediate granularity between box (coarse) and paragraph (fine).
>
> **What we'd lose without it**: No concept-level nodes. Boxes would contain undifferentiated text. Queries like "what are qualified dividends?" would have no concept node to return.
>
> **How we do it**: Multi-factor heuristics: must be bold, 12-60 chars, multiple words, no trailing period, column-aligned, not already classified. Then require 2-of-3 structural confirmations (not in header/footer band, gap above, followed by body).

This is the most nuanced part.

```python
# layout_detection.py:detect_subsection_candidates

# PRIMARY CRITERIA (all required):
is_base_candidate = (
    line_bold &                    # Must be bold
    is_heading_length &            # 12-60 chars
    has_multiple_words &           # ≥2 words
    ~ends_with_period &            # Not ending with period
    col_left_aligned &             # At column margin
    ~is_box_strong &               # Not already a box
    ~is_section &                  # Not already a section
    ~is_page_marker                # Not a page marker
)

# STRUCTURAL CONFIRMATIONS (require 2 of 3):
confirm_not_in_band = ~(in_header_band | in_footer_band)
confirm_gap_or_early = has_large_gap_above
confirm_followed_by_body = followed_by_body

is_subsection_candidate = (
    is_base_candidate &
    (structural_confirms >= 2)
)
```

**Q: Why "12-60 chars" for heading length?**

A: Empirically derived from IRS documents:
- Too short (<12): Likely abbreviations, list markers, numbers
- Too long (>60): Likely body text that happens to be bold
- Sweet spot (12-60): "Qualified Dividends", "Section 404(k) Dividends", etc.

**Q: Why require "not ending with period"?**

A: **Sentences end with periods; headings don't**. This simple heuristic has high precision:
- "Qualified Dividends" → no period → likely heading ✓
- "Report this amount." → period → likely body ✗

**Q: Why col_left_aligned instead of block-relative left_aligned?**

A: **Block alignment is relative to potentially indented blocks**. Consider:

```
[Block: indented content]
    Heading Text        ← left_aligned=True (within block)
    Body paragraph...
```

The heading is left-aligned within its block, but it's NOT at the column margin. Using `col_left_aligned` catches this:

```
    Heading Text        ← col_left_aligned=False (not at ~72 or ~318)
```

Real section headers are at the column margin, not indented.

**Q: Why require 2 of 3 structural confirmations?**

A: **Reduces false positives while allowing flexibility**:

| Scenario | not_in_band | gap_above | followed_by_body | Confirms | Pass? |
|----------|-------------|-----------|------------------|----------|-------|
| Normal heading | ✓ | ✓ | ✓ | 3 | ✓ |
| End-of-block heading | ✓ | ✓ | ✗ (no next line) | 2 | ✓ |
| Footer artifact | ✗ | ✓ | ✓ | 2 | ✓ (borderline) |
| Random bold text | ✓ | ✗ | ✗ | 1 | ✗ |

The "2 of 3" rule handles edge cases like end-of-block headings (no following line to check) while rejecting truly random bold text.

**Q: Why is followed_by_body relaxed for missing next lines?**

```python
followed_by_body = (
    (~next_line_exists) |  # No next line = OK
    (~next_bold) |
    (next_char_count > line_char_count * 1.2)
)
```

A: **Don't penalize end-of-block headings**. A heading at the end of a block has no "next line" to check. Without this relaxation, all such headings would fail the `followed_by_body` check.

### 3d: Split Triggers

> **What it is**: Mark lines that should start new elements (structural boundaries between content units).
>
> **Role in bigger picture**: Creates the segmentation points for Stage 4. Split triggers determine where one element ends and another begins.
>
> **What we'd lose without it**: No element boundaries. Entire blocks would become single elements, mixing headers with body text.
>
> **How we do it**: Priority-ordered pattern matching. Check page_marker first, then box_strong, box_weak (with gates), section, subsection. First match wins, preventing double-classification.

```python
# layout_detection.py:assign_split_triggers
# Priority order:
1. page_marker     → split_kind = "page_artifact"
2. box_strong      → split_kind = "box"
3. box_weak + gate → split_kind = "box"
4. section + emph  → split_kind = "section"
5. subsection_cand → split_kind = "subsection"
```

**Q: Why priority order?**

A: **Prevents double-classification**. A line matching "Box 1a." might also match the subsection criteria (bold, 12-60 chars, etc.). By checking box patterns FIRST, we classify it correctly.

**Q: Why do weak box patterns need additional gates?**

```python
weak_box_trigger = weak_box & (early | left_aligned | has_emphasis)
```

A: `BOX_WEAK_RX` (`^Box 1a.`) is looser than `BOX_STRONG_RX` (`^Box 1a. Description`). The weak pattern might match false positives like:
```
"Box 1a is reported separately."  ← Not a header, just mentions box
```

The gates require:
- `early`: Near start of block (headers are typically first)
- `left_aligned`: At margin (headers are margin-aligned)
- `has_emphasis`: Bold/larger (headers are emphasized)

---

## Stage 4: Element Splitting + Classification

> **What it is**: Split blocks at trigger points into discrete elements, then classify each element's role (BoxHeader, SectionHeader, SubsectionHeader, ListBlock, BodyTextBlock, PageArtifact).
>
> **Role in bigger picture**: Elements are the content units that become either anchors (headers) or content (paragraphs). This is the final extraction output before graph construction.
>
> **What we'd lose without it**: No role classification. Couldn't distinguish headers from body text. All content would be undifferentiated.
>
> **How we do it**: For each block, find split points from triggers, create segments between them, classify each segment by its split_kind (from Stage 3) or pattern matching.

### 4a: Block Splitting

> **What it is**: Take blocks (PDF's visual groupings) and split them into semantic segments based on split triggers.
>
> **Role in bigger picture**: Transforms visual structure into semantic structure. A PDF block might contain header + body; splitting separates them.
>
> **What we'd lose without it**: Headers and body would be merged. BoxHeader elements would include their body text.
>
> **How we do it**: Iterate through lines, mark split points at triggers, create segments between consecutive split points, aggregate each segment's lines into one element.

```python
# elements.py:split_blocks_into_elements
for (doc_id, page, block_id), group in line_df.groupby(...):
    # Find split points
    for i, row in group.iterrows():
        if row["split_trigger"]:
            starts.append(i)

    # Create segments between split points
    for start, end in segment_bounds:
        seg = group.iloc[start:end]
        element = aggregate_segment(seg)
```

**Q: Why segment-based instead of line-based elements?**

A: **A structural unit can span multiple lines**:

```
Box 1a. Ordinary             ← Line 1 (split_trigger)
dividends.                   ← Line 2 (continuation)
Enter the total amount...    ← Line 3 (body, new segment)
```

If we made each line an element, we'd split "Box 1a. Ordinary dividends." into two elements. Segment-based splitting keeps related lines together.

**Q: Why does split_trigger start a new segment AND potentially end the previous?**

A: Look at the segment creation logic:
```python
if split_trigger and i not in starts:
    starts.append(i)
    if i + 1 < len(g):
        starts.append(i + 1)
```

This creates segments where:
1. The trigger line is its own segment (or start of one)
2. The following line starts a new segment

So `Box 1a. Title` becomes its own element, and the body text starts fresh.

### 4b: Role Classification

> **What it is**: Assign a semantic role to each element: BoxHeader, SectionHeader, SubsectionHeader, ListBlock, BodyTextBlock, or PageArtifact.
>
> **Role in bigger picture**: Roles determine what becomes an anchor (headers) vs content (body, lists). Anchors create structure nodes; content creates paragraph nodes.
>
> **What we'd lose without it**: Couldn't build anchor hierarchy. All elements would be treated as body text. No boxes, no sections, no structure.
>
> **How we do it**: Default to BodyTextBlock, then override based on split_kind (from Stage 3). Pattern fallbacks catch edge cases.

```python
# elements.py:classify_elements
df["role"] = ROLE_BODY_TEXT  # Default

# Box headers (from split_kind)
df.loc[split_kind == "box", "role"] = ROLE_BOX_HEADER

# Section headers (from split_kind)
df.loc[split_kind == "section", "role"] = ROLE_SECTION_HEADER

# List items (pattern-based)
is_list = text.str.match(BULLET_RX) | text.str.match(ENUM_RX)
df.loc[is_list, "role"] = ROLE_LIST_BLOCK

# Page artifacts (split_kind + pattern fallback)
df.loc[split_kind == "page_artifact", "role"] = ROLE_PAGE_ARTIFACT
df.loc[text.str.contains(PAGE_ARTIFACT_RX), "role"] = ROLE_PAGE_ARTIFACT
```

**Q: Why is classification based on split_kind instead of re-running patterns?**

A: **Single source of truth**. The split triggers already determined what each line IS. Re-running patterns would:
1. Duplicate logic
2. Risk inconsistency (different regex in different places)
3. Waste computation

The split_kind IS the classification; we just map it to role names.

**Q: Why does PageArtifact have a pattern fallback?**

```python
is_artifact = text.str.contains(PAGE_ARTIFACT_RX) & is_short & (role == BODY_TEXT)
df.loc[is_artifact, "role"] = ROLE_PAGE_ARTIFACT
```

A: **Belt and suspenders**. Some artifacts might not be detected by split triggers:
- Artifacts in the middle of a block (no visual separation)
- Header/footer text that doesn't match isolation patterns

The pattern fallback catches these stragglers.

**Q: Why confidence scores (0.7-0.99)?**

```python
df.loc[is_box_split & box_pattern_match, "role_conf"] = 0.95
df.loc[is_box_split & ~box_pattern_match, "role_conf"] = 0.75
df.loc[is_list, "role_conf"] = 0.85
```

A: **Downstream weighting**. Not all classifications are equally confident:
- `BoxHeader` with pattern match: 0.95 (high confidence)
- `BoxHeader` without pattern (just split_kind): 0.75 (medium)
- `ListBlock` by pattern: 0.85 (reliable but not certain)

This allows downstream systems (training pair generation, validation) to weight decisions by confidence.

---

## Architectural Questions

### Why Not Use ML for Classification?

**Argument for ML**: Neural networks are better at pattern recognition. A BERT classifier could learn header patterns from examples.

**Argument against (and why we chose regex)**:

1. **Domain is highly constrained**: IRS documents are standardized. Box headers follow `^Box \d+[a-z]?\.` without exception. Regex achieves 100% recall on this pattern.

2. **No training data needed**: Regex works day one. ML requires labeled examples, training infrastructure, model updates.

3. **Interpretability**: When a classification is wrong, we can inspect the regex and fix it. ML models are opaque.

4. **Stability**: Regex produces identical results every run. ML has stochastic elements.

5. **Efficiency**: Regex is O(n) string matching. ML requires model inference.

For a **general-purpose** PDF extractor, ML might be better. For **IRS 1099-DIV specifically**, regex is optimal.

### Why Bottom-Up (Span→Line→Element) Instead of Top-Down?

**Top-down approach**: Use PDF structure (outline, headings) to segment document, then fill in details.

**Problem**: PDF structure is often wrong or absent:
- No semantic outline
- Visual "blocks" don't match content boundaries
- Headers aren't tagged as headers

**Our bottom-up approach**:
1. Start with reliable atoms (spans)
2. Apply domain knowledge at each aggregation step
3. Build structure from patterns, not PDF metadata

This works because we have **strong domain knowledge** (box patterns, font conventions) that we can apply programmatically.

### Why Detect Columns Instead of Using OCR Layout?

**Alternative**: Use advanced OCR (Tesseract, Azure Form Recognizer) which detects columns automatically.

**Problems**:
1. **Adds dependency**: External service or heavy library
2. **Accuracy varies**: OCR column detection isn't perfect
3. **Overkill**: We're not doing OCR (PDF has text), just layout analysis
4. **Control**: Our algorithm is tuned for this specific document type

Our x0-peak algorithm is:
- Simple (50 lines of code)
- Deterministic
- Tuned for two-column IRS layouts
- Fast (O(n) where n = lines per page)

---

## Summary: Why This Architecture

| Decision | Why |
|----------|-----|
| Span-level extraction | Atomic unit with reliable font data |
| Mode for body size | Most frequent = body by definition |
| Dynamic column detection | Pages vary (1 vs 2 columns) |
| 2-of-3 structural confirms | Reduces FP while handling edge cases |
| Segment-based elements | Multi-line headers stay together |
| Regex classification | Perfect for constrained domain |
| Bottom-up processing | PDF structure unreliable; domain knowledge reliable |

The architecture is **domain-optimized**. It would NOT generalize to arbitrary PDFs, but it achieves high accuracy on IRS instruction documents with minimal complexity.

---

## Quick Reference: Stage-by-Stage

| Stage | What | Role | Without It | How |
|-------|------|------|------------|-----|
| **1: Span Extraction** | Extract atomic text+font units | Foundation layer | No extraction possible | PyMuPDF dict traversal |
| **2: Body Font** | Find dominant font size | Baseline for emphasis | Can't detect headers | Mode of rounded sizes |
| **3a: Line Agg** | Combine spans → lines | Unit of reading | Re-aggregate constantly | GroupBy + agg |
| **3b: Columns** | Detect 1 vs 2 columns | Correct reading order | Wrong row-wise reading | X0 peak analysis |
| **3c: Subsections** | Find concept headers | Intermediate granularity | No concept nodes | Multi-factor heuristics |
| **3d: Split Triggers** | Mark element boundaries | Segmentation points | Headers merged with body | Priority pattern matching |
| **4a: Splitting** | Segment blocks at triggers | Semantic structure | Visual = semantic conflation | Iterate + slice |
| **4b: Classification** | Assign roles | Anchor vs content | Everything is body text | split_kind mapping |
