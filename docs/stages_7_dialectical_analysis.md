# Stage 7: Dialectical Analysis

A rigorous examination of section refinement, reference extraction, and concept role classification.

---

## Overview: Refining Sections for the Knowledge Graph

> **What Stage 7 is**: Quality refinement - taking raw sections from Stage 6 and preparing them for graph construction through merging, reference extraction, and semantic labeling.
>
> **Role in bigger picture**: Bridge between raw extraction and graph construction. Ensures sections are clean (merged), connected (references extracted), and semantically labeled (roles assigned).
>
> **What we'd lose without it**: Over-split sections with orphan headings. No cross-reference edges. No semantic role labels for training pair categorization.
>
> **How we do it**: Three sub-stages: merge-forward thin subsections, extract references per-element with evidence, classify concept roles via regex patterns.

Stage 7 is about **quality refinement** - taking the raw sections from Stage 6 and preparing them for graph construction:

```
Raw Sections (from Stage 6)
         │
         ▼ [7a: Merge Forward]
Consolidated Sections (thin fragments merged)
         │
         ▼ [7b: Reference Extraction]
References DataFrame (cross-references extracted)
         │
         ▼ [7c: Concept Role Classification]
Sections with semantic role labels
```

Each step addresses a specific quality problem.

---

## Stage 7a: Merge-Forward Thin Subsections

> **What it is**: Detect subsections that are "too thin" (likely over-split) and merge them into the following anchor.
>
> **Role in bigger picture**: Fixes extraction artifacts where a heading got separated from its body. Ensures sections have meaningful content.
>
> **What we'd lose without it**: Many subsections would be header-only (empty body). Their real content would be in the next anchor with a fragment header. Training pairs would be low quality.
>
> **How we do it**: Identify thin subsections (short total chars, few elements, lowercase continuation). Merge forward within same page/column. Iterate until stable.

### The Problem

PDF extraction + layout detection can **over-split** content:

```
Original PDF layout:
┌──────────────────────────────┐
│ Qualified Dividends          │  ← Bold heading
│                              │
│ These are dividends that     │  ← Body text starts here
│ meet the holding period...   │
└──────────────────────────────┘

Extraction result:
  Anchor 1: "sub_qualified_dividends_abc123"
            header: "Qualified Dividends"
            body: ""                         ← Empty! Just the heading

  Anchor 2: "sub_these_are_dividends_def456"
            header: "These are dividends..."  ← Body became its own anchor!
            body: "meet the holding period..."
```

The heading got extracted as its own anchor, but its real body content ended up in the next anchor.

### The Solution: Merge Forward

```python
# merge.py
def merge_forward_thin_subsections(sections_df, col_info_df, config):
    while iteration < max_iterations:
        # 1. Sort by reading order (page, column, y-position)
        df = df.sort_values(["_page", "_col", "_y0"])

        # 2. Identify "thin" subsections
        thin_mask = identify_thin_subsections(df, config)

        # 3. For each thin subsection, try to merge into next anchor
        for i in range(len(df) - 1):
            if thin_mask[i]:
                can_merge, reason = can_merge_into(df.iloc[i], df.iloc[i + 1])
                if can_merge:
                    perform_single_merge(df, i, i + 1)

        # 4. Repeat until stable
```

### What Makes a Subsection "Thin"?

```python
def identify_thin_subsections(df, config):
    is_subsection = df["anchor_type"] == "subsection"
    is_short_total = df["char_count"] < config.thin_char_thresh      # < 160 chars
    is_few_elements = df["element_count"] <= config.thin_elem_thresh  # <= 2 elements
    is_fragment_body = (body_len <= config.body_char_thresh) |        # < 120 chars body
                       starts_lower                                    # OR starts lowercase

    return is_subsection & is_short_total & is_few_elements & is_fragment_body
```

**Q: Why check if body starts with lowercase?**

A: **Continuation detection**. If body starts with lowercase, it's likely a sentence fragment:
```
Anchor 1: header="Qualified Dividends", body=""
Anchor 2: header="that meet the holding", body="period requirements..."
          ^-- lowercase = continuation, not new thought
```

**Q: Why the 160/120 char thresholds?**

A: **Empirically tuned**. A real subsection typically has:
- Header: 20-60 chars
- Body: 200-2000 chars
- Total: 220-2060 chars

A thin fragment is usually:
- Header (promoted from body): 50-150 chars
- Body: 0-50 chars (remainder of the sentence)

The thresholds capture this pattern with margin for error.

### Merge Rules

```python
def can_merge_into(source_row, target_row, body_char_thresh):
    # Must be same page
    if source_row["_page"] != target_row["_page"]:
        return False, "different_page"

    # Must be same column
    if source_row["_col"] != target_row["_col"]:
        return False, "different_column"

    # Don't merge into box_section
    if target_type in ("box", "box_section"):
        return False, "target_is_box"

    # Don't merge into section unless source has minimal body
    if target_type == "section":
        if len(source_body) > body_char_thresh:
            return False, "source_has_substantial_body"

    # Don't merge into subsection that has a header
    if target_type == "subsection":
        if target_header:
            return False, "target_has_header"

    return True, "ok"
```

**Q: Why only merge forward, not backward?**

A: **Reading order semantics**. A heading "owns" content that follows it, not precedes it. If we merged backward:
```
Anchor 1: "Box 1a content..."      ← Would absorb the orphan heading
Anchor 2: "Qualified Dividends"    ← Orphan heading
Anchor 3: "These are dividends..." ← Real body for "Qualified Dividends"
```
Merging backward would put "Qualified Dividends" into Box 1a - wrong!

**Q: Why only within same column?**

A: **Column boundaries are content boundaries**. In a two-column layout:
```
┌────────────────┬────────────────┐
│ Left column    │ Right column   │
│ ends here.     │ starts here.   │
│                │                │
│ [Thin anchor]  │ [Next anchor]  │  ← Different columns!
└────────────────┴────────────────┘
```
Merging across columns would corrupt content assignment.

**Q: Why don't merge into box_section?**

A: **Keep box content clean**. Box sections are the primary retrieval units. Merging orphan subsections into them would add noise:
```
Box 1a content: "Enter total dividends..."
+ orphan heading: "Qualified Dividends"  ← This belongs to Box 1b!
```

**Q: Why iterate to stabilization?**

A: **Cascading merges**. Consider:
```
A (thin) → B (thin) → C (normal)
```
First pass: A merges into B
Second pass: (A+B) still thin, merges into C

Without iteration, we'd leave B as a thin fragment.

### What Gets Merged

```python
def perform_single_merge(df, source_idx, target_idx):
    # Prepend source label to target header
    merged_header = f"{source_label}\n{target_header}"

    # Prepend source body to target body
    merged_body = f"{source_body}\n\n{target_body}"

    # Union element_ids (dedupe preserving order)
    merged_eids = list(dict.fromkeys(source_eids + target_eids))

    # Union pages, bbox
    merged_pages = sorted(set(source_pages + target_pages))
    merged_bbox = [min(x0s), min(y0s), max(x1s), max(y1s)]
```

**Q: Why prepend source, not append?**

A: **Preserve reading order**. The source (thin anchor) comes BEFORE the target in reading order. Its content should come first in the merged text.

---

## Stage 7b: Reference Extraction

> **What it is**: Extract cross-references from element text (box refs, publication refs, IRC section refs, form refs) with evidence quotes.
>
> **Role in bigger picture**: References become `references_box` edges in the graph. They connect sections that mention each other, enabling graph expansion during retrieval.
>
> **What we'd lose without it**: No cross-reference edges. "See Box 1a" in Box 2b wouldn't create a link. Graph would be hierarchy-only (no horizontal connections).
>
> **How we do it**: Regex patterns per reference type, extract per-element for clean provenance, validate target_exists, deduplicate by (source_anchor, target_anchor).

### The Problem

Documents contain cross-references that create edges in the knowledge graph:
- "See Box 1a for more information"
- "Refer to Pub. 550"
- "As defined in section 301(c)(1)"

We need to extract these with clean provenance.

### Regex Patterns

```python
# Box reference: "box 1a", "boxes 1a and 1b", "boxes 14-16"
BOX_REF_RX = re.compile(
    r"[Bb]ox(?:es)?\s+(\d+[a-z]?(?:[^\n]*?(?:,|and|through|[-–]|to)\s*\d+[a-z]?)*)",
    re.IGNORECASE
)

# Publication reference: "Pub. 550", "Publication 17"
PUB_REF_RX = re.compile(r"[Pp]ub(?:lication)?\.?\s*(\d+)")

# IRC section reference: "section 301(c)(1)", "§ 301"
IRC_REF_RX = re.compile(r"(?:[Ss]ection|§)\s*(\d+[A-Za-z]?(?:\([a-z0-9]+\))*)")

# Form reference: "Form 1099-DIV", "Form W-2"
FORM_REF_RX = re.compile(r"[Ff]orm\s+(\d+[A-Z\-]*|[A-Z]-?\d+)")
```

**Q: Why `[^\n]` in the box pattern?**

A: **Newline boundary**. Without it, the pattern might capture across element boundaries:
```
"See box 1a for details.\n\nBox 2a. Capital gains distribution."
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          Would capture "1a for details.\n\nBox 2a" !
```

The `[^\n]` stops at the newline, giving us just "1a for details."

**Q: Why allow multiple levels of parentheses in IRC pattern?**

A: **IRC citations are nested**: `section 301(c)(1)(A)(ii)`. The pattern `(?:\([a-z0-9]+\))*` captures zero or more parenthetical levels.

### Per-Element Extraction

```python
def extract_references_from_element(element_id, text, anchor_id, page, valid_box_keys):
    references = []

    for match in BOX_REF_RX.finditer(text):
        target_keys = parse_box_ref_keys(match.group(1))
        evidence = extract_evidence_quote(text, match, context_chars=50)

        for key in target_keys:
            # Skip self-references
            if f"box_{key}" == anchor_id:
                continue

            references.append(Reference(
                source_element_id=element_id,
                source_anchor_id=anchor_id,
                target_anchor_id=f"box_{key}",
                target_exists=key in valid_box_keys,
                evidence_text=evidence,
                confidence=0.95 if target_exists else 0.70,
            ))

    return references
```

**Q: Why per-element instead of per-section?**

A: **Clean provenance**. Each reference needs an evidence quote. Per-section extraction would give:
```
Section text: "See box 1a... [500 chars]... and box 2a..."
Evidence for box_1a: "See box 1a... [500 chars]... and box 2a..."  ← Too long!
```

Per-element extraction gives:
```
Element 1: "See box 1a for details."
Evidence for box_1a: "See box 1a for details."  ← Clean!

Element 5: "Also check box 2a for capital gains."
Evidence for box_2a: "...check box 2a for capital gains."  ← Clean!
```

**Q: Why skip self-references?**

A: **No information value**. "Box 1a" mentioning "box 1a" doesn't create a useful edge:
```
Box 1a text: "Box 1a. Enter the amount from box 1a of Form 1099."
                                        ^^^^^^^^
                                        Self-reference - skip!
```

**Q: Why different confidence for target_exists?**

```python
confidence = 0.95 if target_exists else 0.70
```

A: **Validity signal**. If target_exists=True, we KNOW the target anchor exists in this document - high confidence. If target_exists=False, the reference might be:
1. To a box that doesn't exist in this form (error in document)
2. To a box in a different form (cross-form reference)
3. An OCR/extraction error

Lower confidence flags these for review.

### Evidence Quotes

```python
def extract_evidence_quote(text, match, context_chars=50):
    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)

    quote = text[start:end].strip()
    quote = " ".join(quote.split())  # Collapse whitespace

    if start > 0:
        quote = "..." + quote
    if end < len(text):
        quote = quote + "..."

    return quote
```

**Q: Why 50 chars of context?**

A: **Balance**. Too little context loses meaning:
```
"...box 1a..."  ← What about it?
```

Too much context adds noise:
```
"The total ordinary dividends shown in box 1a includes amounts shown in boxes 1b..."
                                                      ^^^ 80 chars of context
```

50 chars typically captures the sentence fragment:
```
"...ordinary dividends shown in box 1a includes amounts..."  ← Clear context
```

### Deduplication

```python
# Deduplicate by (source_anchor_id, target_anchor_id, ref_type)
references_df = references_df.drop_duplicates(
    subset=["source_anchor_id", "target_anchor_id", "ref_type"],
    keep="first",
)
```

**Q: Why deduplicate by anchor, not element?**

A: **Graph edges are anchor-to-anchor**. If Box 1a mentions "box 2a" three times in different elements, we only need ONE edge from box_1a to box_2a. Multiple evidence quotes don't add value to the graph structure.

---

## Stage 7c: Concept Role Classification

> **What it is**: Assign semantic role labels (definition, condition, exception, procedure, qualification) to subsections.
>
> **Role in bigger picture**: Roles enable semantic training pair generation. "Exception" sections become hard negatives. "Definition" sections answer "what is X?" queries.
>
> **What we'd lose without it**: All subsections would be undifferentiated. Couldn't generate role-aware training pairs. Couldn't filter by "show me exceptions."
>
> **How we do it**: Regex patterns with priority ordering (exception highest). Check body for action patterns, header for definition patterns. Return NULL on low confidence.

### The Problem

Subsections have different **semantic purposes**:
- "Qualified Dividends" → defines a term
- "If the recipient held the stock..." → states a condition
- "Enter the amount on line 1" → gives a procedure
- "Does NOT include nondividend..." → states an exception

These roles matter for:
1. **Training pairs**: Definition + exception = hard negative
2. **Retrieval**: Procedure queries should find procedure sections
3. **UI**: Can filter by "show me all exceptions"

### The Role Set

```python
CONCEPT_ROLES = frozenset({
    "definition",     # Defines a term or concept
    "qualification",  # Constrains applicability
    "condition",      # Introduces conditional logic (if/when)
    "exception",      # Excludes or negates applicability
    "procedure",      # Describes actions to take
})
```

**Q: Why this closed set?**

A: **Coverage vs. precision tradeoff**. These five roles cover ~90% of IRS subsection content:

| Role | Frequency | Example |
|------|-----------|---------|
| definition | 35% | "Qualified dividends are..." |
| procedure | 25% | "Enter the amount..." |
| condition | 20% | "If the recipient..." |
| qualification | 12% | "This applies to..." |
| exception | 8% | "Does NOT include..." |

Adding more roles would fragment the distribution and reduce classifier confidence.

**Q: Why not use ML classification?**

A: Same reasoning as Stage 4. IRS language is formulaic:
- Exceptions almost always use "does not include", "except", "not reported"
- Procedures almost always start with "Enter", "Report", "File"
- Conditions almost always use "if", "when", "only if"

Regex captures these patterns with near-perfect precision. ML would need training data and might not improve.

### Pattern Priority

```python
_ROLE_PATTERNS = [
    # 1. Exception patterns (highest priority)
    ("negation_phrase", r"does\s+not\s+include|except\s+|excluding\s+", "exception", 0.9, "body"),

    # 2. Condition patterns
    ("early_conditional", r"if\s+(?:you|the)|when\s+(?:you|the)|only\s+if", "condition", 0.9, "body"),

    # 3. Procedure patterns
    ("imperative_verb", r"^(?:Enter|Report|File|Include|Use)", "procedure", 0.85, "body"),

    # 4. Qualification patterns
    ("qualifier_phrase", r"the\s+following|these\s+dividends|applies\s+to", "qualification", 0.75, "body"),

    # 5. Definition patterns (lowest - fallback)
    ("noun_phrase_header", r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$", "definition", 0.8, "header"),
]
```

**Q: Why exception highest priority?**

A: **Negation overrides everything**. Consider:
```
"These dividends do NOT include amounts from..."
```
This could match:
- `qualification`: "These dividends"
- `exception`: "do NOT include"

The exception meaning dominates - this section is about what's EXCLUDED, not what's included.

**Q: Why check body for most patterns but header for definition?**

A: **Pattern location**:
- Exception/condition/procedure → appear in body text (the explanation)
- Definition → appears in header (the term being defined)

```
Header: "Qualified Dividends"        ← Noun phrase = definition
Body: "These are dividends that..."  ← Explanation
```

vs.

```
Header: "Reporting Requirements"
Body: "If you received..."           ← "If" in body = condition
```

### Position-Based Tiebreakers

```python
def classify_concept_role(header_text, body_text, position_info=None):
    # Try regex first
    for pattern in _ROLE_PATTERNS:
        if pattern.search(text):
            return RoleClassification(role=..., method="regex")

    # Position fallbacks (only when regex fails)
    if position_info:
        # Before first box → likely definition
        if position_info.get("is_before_first_box"):
            return RoleClassification(role="definition", confidence=0.6)

        # After box header with short body → likely qualification
        if position_info.get("is_after_box_header") and body_char_count < 200:
            return RoleClassification(role="qualification", confidence=0.5)

    # No confident classification
    return RoleClassification(role=None, confidence=0.0, method="null")
```

**Q: Why return NULL instead of best guess?**

A: **Explicit uncertainty is better than wrong labels**. If we guessed:
```
role="definition" (guessed, confidence=0.3)
```
Downstream systems might use this for training pairs or retrieval filtering. A wrong label causes:
- Training on incorrect pairs
- Missing relevant results in filtered queries
- Validation failures

With NULL:
```
role=None (unknown)
```
Downstream systems know to:
- Skip for training pairs
- Include in unfiltered queries
- Flag for human review

**Q: Why lower confidence for position-based?**

```python
# Regex match: confidence=0.85-0.90
# Position match: confidence=0.50-0.60
```

A: **Position is weak signal**. "Before first box" doesn't MEAN definition - it's just correlated. Many preamble sections are:
- Reminders (not definition)
- "What's New" (not definition)
- Future developments (not definition)

The 0.5-0.6 confidence warns: "This might be a definition, but verify."

---

## Putting It All Together

Stage 7 transforms raw sections into graph-ready data:

| Input | Problem | Solution | Output |
|-------|---------|----------|--------|
| Over-split sections | Heading separated from body | Merge forward | Consolidated sections |
| Unextracted references | Cross-refs not captured | Regex extraction | References DataFrame |
| Unlabeled subsections | No semantic meaning | Role classification | Labeled sections |

### Data Flow

```
sections_df (from Stage 6)
    │
    ├─→ merge_forward_thin_subsections()
    │       │
    │       └─→ sections_df (consolidated)
    │
    ├─→ extract_references(elements_df)
    │       │
    │       └─→ references_df
    │
    └─→ classify_section_roles()
            │
            └─→ sections_df (with concept_role columns)
```

### Quality Metrics

After Stage 7, we can measure:

| Metric | Target | Purpose |
|--------|--------|---------|
| Thin sections remaining | 0 | Merge completeness |
| Box refs with target_exists=True | >90% | Reference validity |
| Subsections with role != NULL | >70% | Classification coverage |
| Exception roles detected | >0 | Hard negative availability |

---

## Summary: Why This Architecture

| Decision | Why |
|----------|-----|
| Merge forward only | Reading order semantics (heading owns following content) |
| Same-column constraint | Column boundaries are content boundaries |
| Iterate to stabilization | Handles cascading thin fragments |
| Per-element reference extraction | Clean evidence quotes with exact provenance |
| Skip self-references | No information value |
| Confidence by target_exists | Flags potentially invalid references |
| Closed role set | Coverage/precision balance |
| Regex-first classification | IRS language is formulaic |
| Return NULL on low confidence | Explicit uncertainty beats wrong labels |

Stage 7 ensures the sections are **clean** (merged), **connected** (references extracted), and **semantically labeled** (roles assigned) before graph construction.

---

## Quick Reference: Stage-by-Stage

| Stage | What | Role | Without It | How |
|-------|------|------|------------|-----|
| **7: Section Refinement** | Clean, connect, label sections | Bridge to graph construction | Low-quality sections | Three sub-stages |
| **7a: Merge Forward** | Combine over-split subsections | Fix extraction artifacts | Header-only orphan sections | Thin detection + iterative merge |
| **7b: Reference Extraction** | Extract cross-references | Create horizontal graph edges | No references_box edges | Per-element regex + evidence |
| **7c: Role Classification** | Label subsection semantics | Enable role-aware pairs | Undifferentiated subsections | Priority-ordered regex patterns |
