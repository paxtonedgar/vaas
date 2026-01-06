# Tax Document RAG Pipeline: Implementation Reference

**Version:** 1.0  
**Last Updated:** 2026-01-05  
**Status:** Cells 1-4 implemented, Cells 5-10 in progress  
**Epic Reference:** UTILITIESPLATFORM-5327

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Cell-by-Cell Reference](#2-cell-by-cell-reference)
3. [Current Implementation Status](#3-current-implementation-status)
4. [Key Functions](#4-key-functions)
5. [Configuration & Constants](#5-configuration--constants)
6. [Known Issues & Workarounds](#6-known-issues--workarounds)
7. [Testing & Validation](#7-testing--validation)

---

## 1. Pipeline Overview

### 1.1 Pipeline Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTRACTION PIPELINE                                  │
│                                                                              │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐            │
│  │ Cell 1 │──►│ Cell 2 │──►│ Cell 3 │──►│ Cell 4 │──►│ Cell 5 │            │
│  │        │   │        │   │        │   │        │   │        │            │
│  │Extract │   │ Infer  │   │ Build  │   │Detect  │   │ Build  │            │
│  │ Spans  │   │Body Sz │   │Elements│   │Anchors │   │Timeline│            │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘            │
│      │            │            │            │            │                  │
│      ▼            ▼            ▼            ▼            ▼                  │
│  spans_df    body_size    elements_df  anchors_df  anchor_events_df        │
│                                                                              │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐            │
│  │ Cell 6 │──►│ Cell 7 │──►│ Cell 8 │──►│ Cell 9 │──►│Cell 10 │            │
│  │        │   │        │   │        │   │        │   │        │            │
│  │ Merge  │   │ Assign │   │ Build  │   │Extract │   │ Emit   │            │
│  │Elements│   │Content │   │Sections│   │  Refs  │   │to Delta│            │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘            │
│      │            │            │            │            │                  │
│      ▼            ▼            ▼            ▼            ▼                  │
│  merged_df   assigned_df  sections_df  refs_df    graph_nodes              │
│                                                        graph_edges          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 File Locations

```
/Volumes/112557_prefetch_ctg_prd_exp/112557_prefetch_raw/irs_raw/
├── i1099div.pdf          # Filer instructions (primary target)
├── f1099div.pdf          # Form PDF
└── [other forms]

/catalog/112557_prefetch_ctg_prd_exp/112557_prefetch_raw/
├── box_registry          # Box metadata table
├── graph_nodes           # KG nodes table
└── graph_edges           # KG edges table
```

---

## 2. Cell-by-Cell Reference

### Cell 1: Span Extraction

**Purpose:** Extract raw text spans with font/position metadata from PDF.

**Input:** PDF file path  
**Output:** `spans_df`

**Key Code:**
```python
import fitz  # PyMuPDF

BASE = Path("/Volumes/112557_prefetch_ctg_prd_exp/112557_prefetch_raw/irs_raw/")
INSTRUCTIONS_PATH = BASE / "i1099div.pdf"

def extract_spans(pdf_path):
    """Extract all text spans with metadata."""
    doc = fitz.open(pdf_path)
    spans = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
            for line_idx, line in enumerate(block["lines"]):
                for span_idx, span in enumerate(line["spans"]):
                    spans.append({
                        "page": page_num,
                        "block_id": block_idx,
                        "line_id": line_idx,
                        "span_id": span_idx,
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "flags": span["flags"],
                        "x0": span["bbox"][0],
                        "y0": span["bbox"][1],
                        "x1": span["bbox"][2],
                        "y1": span["bbox"][3],
                    })
    
    doc.close()
    return pd.DataFrame(spans)

# Execute
doc, spans_df, profile = extract_spans(INSTRUCTIONS_PATH)
```

**Output Schema:**

| Column | Type | Description |
|--------|------|-------------|
| page | int | Page number (0-indexed) |
| block_id | int | Block index |
| line_id | int | Line index within block |
| span_id | int | Span index within line |
| text | string | Span text |
| font | string | Font name |
| size | float | Font size (points) |
| flags | int | Font flags (bold=16, italic=2) |
| x0, y0, x1, y1 | float | Bounding box |

---

### Cell 2: Body Size Inference

**Purpose:** Determine modal font size for body text.

**Input:** `spans_df`  
**Output:** `body_size` (scalar)

**Key Code:**
```python
def infer_body_font_size(spans_df):
    """Mode of rounded sizes tends to be body text in IRS PDFs."""
    if spans_df is None or spans_df.empty:
        return 10.0  # Default
    return float(spans_df["size"].round(1).value_counts().idxmax())

body_size = infer_body_font_size(spans_df)
print(f"Body font size estimate: {body_size}")
```

**Typical Result:** `body_size = 9.0` for IRS instruction documents.

---

### Cell 3: Element Construction

**Purpose:** Build line-level elements with role classification.

**Input:** `spans_df`, `body_size`  
**Output:** `elements_df`

**Key Code:**
```python
# Role constants
ROLE_DOC_TITLE = "DocTitle"
ROLE_SECTION_HEADER = "SectionHeader"
ROLE_BOX_HEADER = "BoxHeader"
ROLE_BODY_TEXT = "BodyText"
ROLE_LIST_BLOCK = "ListBlock"
ROLE_TABLE_BLOCK = "TableBlock"
ROLE_PAGE_ARTIFACT = "PageArtifact"

# Regex patterns
SECTION_HEADER_RX = re.compile(
    r"^(What|Who|When|How|General|Specific|Instructions|Future Developments)\s*$",
    re.IGNORECASE
)

BOX_RX_SINGLE = re.compile(r"^Box\s*(\d+[a-z]?)\.?\s+", re.IGNORECASE)
BOX_RX_RANGE = re.compile(r"^Boxes?\s*(\d+[a-z]?)\s*(through|[-–])\s*(\d+[a-z]?)", re.IGNORECASE)

def is_bold(fonts_joined: str, flags_any_bold: bool) -> bool:
    """Conservative: either font name contains 'bold' or flags indicate bold."""
    f = (fonts_joined or "").lower()
    return ("bold" in f) or bool(flags_any_bold)

def build_line_elements(spans_df, body_size):
    """Build line-level elements from spans grouped by (page, block_id, line_id)."""
    
    elements_raw = build_layout_elements_from_blocks(spans_df, body_size=body_size)
    lines_df = build_line_elements(spans_df, body_size=body_size)
    
    # Split blocks on header-like lines (including Box headings)
    elements_df = split_blocks_on_header_lines(
        elements_df=elements_raw,
        lines_df=lines_df,
        body_size=body_size,
        stable_hash_fn=stable_hash,
    )
    
    return elements_df

elements_df = build_line_elements(spans_df, body_size)
print(f"After header splitting: {len(elements_df)}")
```

**Output Schema:**

| Column | Type | Description |
|--------|------|-------------|
| doc_id | string | Document identifier |
| page | int | Page number |
| element_id | string | Unique element ID |
| text | string | Normalized text |
| role | string | Classified role |
| role_conf | float | Classification confidence |
| size_mode | string | "body", "header", "subheader" |
| is_bold | bool | Bold flag |
| x_min, y_min, x_max, y_max | float | Bounding box |
| reading_order | int | Sequential position |

---

### Cell 4: Anchor Detection

**Purpose:** Identify box headers and section headers as anchors.

**Input:** `elements_df`  
**Output:** `anchors_df`

**Key Code:**
```python
def anchors_from_elements(elements_df):
    """Extract anchors from elements with BoxHeader or SectionHeader role."""
    
    # Get BoxHeadingCandidate elements
    box_cands = elements_df[
        elements_df.get("role_hint", "").str.contains("BoxHeadingCandidate", na=False)
    ]
    
    anchors = []
    for _, row in box_cands.iterrows():
        text = row["text"]
        
        # Try single box pattern
        match = BOX_RX_SINGLE.match(text)
        if match:
            box_num = match.group(1).lower()
            anchors.append({
                "anchor_id": f"{row['doc_id']}_{row['page']}_{box_num}",
                "key": f"box_{box_num}",
                "title_text": text,
                "page": row["page"],
                "y_position": row["y_min"],
                "parent_anchor_id": None,
                "role_conf": row.get("role_conf", 0.9)
            })
            continue
        
        # Try range pattern (Boxes 14-16)
        match = BOX_RX_RANGE.match(text)
        if match:
            start, end = match.group(1).lower(), match.group(3).lower()
            group_key = f"box_{start}_{end}_group"
            
            # Create group anchor
            anchors.append({
                "anchor_id": f"{row['doc_id']}_{row['page']}_{group_key}",
                "key": group_key,
                "title_text": text,
                "page": row["page"],
                "y_position": row["y_min"],
                "parent_anchor_id": None,
                "role_conf": row.get("role_conf", 0.9)
            })
            
            # Create child anchors
            for box_num in expand_box_range(start, end):
                anchors.append({
                    "anchor_id": f"{row['doc_id']}_{row['page']}_{box_num}",
                    "key": f"box_{box_num}",
                    "title_text": f"Box {box_num}",
                    "page": row["page"],
                    "y_position": row["y_min"],
                    "parent_anchor_id": f"{row['doc_id']}_{row['page']}_{group_key}",
                    "role_conf": row.get("role_conf", 0.85)
                })
    
    return pd.DataFrame(anchors)

anchors_df = anchors_from_elements(elements_df)
anchors_df = normalize_anchors(anchors_df)  # Standardize keys

print(f"Anchors detected: {len(anchors_df)}")
display(anchors_df[["anchor_id", "key", "parent_anchor_id", "title_text", "page", "role_conf"]].head(50))
```

**Coverage Check:**
```python
# Verify all expected boxes found
expected = set(["1a", "1b", "2a", "2b", "2c", "2d", "2e", "2f", "3", "4", "5", 
                "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])
found = set(anchors_df["key"].str.replace("box_", "").tolist())

print(f"Expected: {len(expected)}, Found: {len(found)}")
print(f"Missing: {sorted(expected - found)}")
print(f"Extra: {sorted(found - expected)}")
```

---

### Cell 5: Anchor Timeline

**Purpose:** Build reading order timeline of anchors.

**Input:** `anchors_df`  
**Output:** `anchor_events_df`

**Key Code:**
```python
def build_anchor_timeline(anchors_df):
    """Sort anchors by (page, y_position) to create timeline."""
    
    anchor_events_df = (
        anchors_df
        .sort_values(["doc_id", "page", "y_position"])
        .reset_index(drop=True)
    )
    
    # Assign reading order
    anchor_events_df["anchor_reading_order"] = (
        anchor_events_df
        .groupby("doc_id")
        .cumcount()
    )
    
    return anchor_events_df

anchor_events_df = build_anchor_timeline(anchors_df)
```

---

### Cell 6: Element-Anchor Merge

**Purpose:** Join elements with anchor timeline.

**Input:** `elements_df`, `anchor_events_df`  
**Output:** `merged_df`

**Known Issue:** Merge collision if element IDs not unique.

**Key Code:**
```python
def merge_elements_with_anchors(elements_df, anchor_events_df):
    """
    Join elements with anchor events for assignment.
    
    IMPORTANT: Ensure unique (doc_id, page, element_id) before merge.
    """
    
    # Validate uniqueness
    dup_check = elements_df.groupby(["doc_id", "page", "element_id"]).size()
    dups = dup_check[dup_check > 1]
    if len(dups) > 0:
        raise ValueError(f"Duplicate element IDs found: {dups.head()}")
    
    # Prepare anchor events for merge
    anchor_events_df = (
        anchors_df
        .merge(
            elements_df[["doc_id", "page", "element_id", "reading_order"]] + 
            [["bbox"] if "bbox" in elements_df.columns else []],
            on=["doc_id", "page", "element_id"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_elem")
        )
        .rename(columns={"reading_order": "anchor_reading_order", "bbox": "anchor_bbox"})
    )
    
    # Handle case where bbox already on anchors_df
    if "bbox" not in anchors_df.columns:
        anchor_events_df["anchor_bbox"] = anchor_events_df.get("anchor_bbox", 
            pd.Series([None] * len(anchor_events_df)))
    
    return anchor_events_df

anchor_events_df = merge_elements_with_anchors(elements_df, anchors_df)
```

**Debug if merge fails:**
```python
# Check for duplicates
dups = elements_df.groupby(["doc_id", "page", "element_id"]).size()
print(dups[dups > 1])

# Check merge keys exist in both
print("Elements columns:", elements_df.columns.tolist())
print("Anchors columns:", anchors_df.columns.tolist())
```

---

### Cell 7: Content Assignment

**Purpose:** Assign each element to its owning anchor.

**Input:** `merged_df`  
**Output:** `assigned_df` (elements with `parent_anchor_id`)

**Key Code:**
```python
def assign_content_to_anchors(elements_df, anchor_events_df):
    """
    For each element, find most recent anchor in reading order.
    """
    
    # Sort both by reading order
    elements_sorted = elements_df.sort_values(["doc_id", "reading_order"])
    anchors_sorted = anchor_events_df.sort_values(["doc_id", "anchor_reading_order"])
    
    assigned = []
    
    for doc_id in elements_sorted["doc_id"].unique():
        doc_elements = elements_sorted[elements_sorted["doc_id"] == doc_id]
        doc_anchors = anchors_sorted[anchors_sorted["doc_id"] == doc_id]
        
        anchor_positions = doc_anchors[["anchor_id", "anchor_reading_order"]].values.tolist()
        
        for _, elem in doc_elements.iterrows():
            elem_order = elem["reading_order"]
            
            # Find most recent anchor (largest anchor_reading_order <= elem_order)
            parent_anchor = None
            for anchor_id, anchor_order in anchor_positions:
                if anchor_order <= elem_order:
                    parent_anchor = anchor_id
                else:
                    break
            
            elem_dict = elem.to_dict()
            elem_dict["parent_anchor_id"] = parent_anchor
            assigned.append(elem_dict)
    
    return pd.DataFrame(assigned)

assigned_df = assign_content_to_anchors(elements_df, anchor_events_df)
```

---

### Cell 8: Section Assembly

**Purpose:** Group elements by anchor into coherent sections.

**Input:** `assigned_df`  
**Output:** `sections_df`

**Key Code:**
```python
def build_sections(assigned_df):
    """Concatenate elements under each anchor into sections."""
    
    sections = []
    
    for anchor_id, group in assigned_df.groupby("parent_anchor_id"):
        if anchor_id is None:
            continue  # Skip unassigned elements
        
        # Sort by reading order
        group_sorted = group.sort_values("reading_order")
        
        # Concatenate text
        section_text = "\n".join(group_sorted["text"].tolist())
        
        sections.append({
            "anchor_id": anchor_id,
            "section_text": section_text,
            "element_ids": group_sorted["element_id"].tolist(),
            "element_count": len(group_sorted),
            "page_start": group_sorted["page"].min(),
            "page_end": group_sorted["page"].max(),
        })
    
    return pd.DataFrame(sections)

sections_df = build_sections(assigned_df)
```

---

### Cell 9: Reference Extraction

**Purpose:** Extract cross-references from section text.

**Input:** `sections_df`  
**Output:** `refs_df`

**Key Code:**
```python
# Reference patterns
BOX_REF_RX = re.compile(r"[Bb]ox(?:es)?\s*(\d+[a-z]?(?:\s*(?:,|and|through|[-–])\s*\d+[a-z]?)*)")
SECTION_REF_RX = re.compile(r"[Ss]ee\s+([A-Z][a-zA-Z\s]+?)(?:,\s*(earlier|later|above|below))?")
PUB_REF_RX = re.compile(r"[Pp]ub(?:lication)?\.?\s*(\d+)")
IRC_REF_RX = re.compile(r"[Ss]ection\s+(\d+[A-Za-z]?(?:\([a-z]\))?)")

def extract_references(sections_df, registry_df=None):
    """Extract all references from section text."""
    
    refs = []
    
    for _, section in sections_df.iterrows():
        text = section["section_text"]
        anchor_id = section["anchor_id"]
        
        # Box references
        for match in BOX_REF_RX.finditer(text):
            box_refs = parse_box_reference(match.group(0))
            for box_key in box_refs:
                refs.append({
                    "source_anchor_id": anchor_id,
                    "reference_text": match.group(0),
                    "reference_type": "box",
                    "target_key": f"box_{box_key}",
                    "confidence": 0.95,
                    "extraction_method": "regex"
                })
        
        # Section references
        for match in SECTION_REF_RX.finditer(text):
            section_name = match.group(1).strip()
            refs.append({
                "source_anchor_id": anchor_id,
                "reference_text": match.group(0),
                "reference_type": "section",
                "target_key": normalize_section_name(section_name),
                "confidence": 0.85,
                "extraction_method": "regex"
            })
        
        # Publication references
        for match in PUB_REF_RX.finditer(text):
            refs.append({
                "source_anchor_id": anchor_id,
                "reference_text": match.group(0),
                "reference_type": "publication",
                "target_key": f"pub_{match.group(1)}",
                "confidence": 0.95,
                "extraction_method": "regex"
            })
        
        # IRC references
        for match in IRC_REF_RX.finditer(text):
            refs.append({
                "source_anchor_id": anchor_id,
                "reference_text": match.group(0),
                "reference_type": "irc",
                "target_key": f"irc_{match.group(1)}",
                "confidence": 0.90,
                "extraction_method": "regex"
            })
    
    return pd.DataFrame(refs)

refs_df = extract_references(sections_df)
```

---

### Cell 10: Emit to Delta

**Purpose:** Write graph nodes and edges to Delta tables.

**Input:** `sections_df`, `anchors_df`, `refs_df`, `registry_df`  
**Output:** `graph_nodes`, `graph_edges` (Delta tables)

**Key Code:**
```python
def emit_to_graph_nodes(sections_df, anchors_df, registry_df):
    """Build and emit graph_nodes table."""
    
    nodes = []
    
    for _, anchor in anchors_df.iterrows():
        # Get section text if available
        section = sections_df[sections_df["anchor_id"] == anchor["anchor_id"]]
        text = section["section_text"].iloc[0] if len(section) > 0 else anchor["title_text"]
        
        # Registry lookup for canonical_id
        canonical_id = lookup_canonical_id(anchor["key"], registry_df)
        
        nodes.append({
            "node_id": anchor["anchor_id"],
            "doc_id": anchor["doc_id"],
            "doc_type": "filer_instructions",  # Hardcoded for now
            "chunk_type": "anchor",
            "canonical_id": canonical_id,
            "level": "anchor",
            "parent_node_id": anchor.get("parent_anchor_id"),
            "depth": 1 if anchor.get("parent_anchor_id") is None else 2,
            "text": text,
            "text_raw": text,
            "page_start": anchor["page"],
            "page_end": section["page_end"].iloc[0] if len(section) > 0 else anchor["page"],
            "doc_version": "Rev. January 2024",
            "extraction_confidence": anchor.get("role_conf", 0.9),
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "extracted_at": datetime.now(),
            "extraction_version": "1.0.0",
            "concepts": [],
            "source_node_ids": []
        })
    
    return spark.createDataFrame(nodes)

def emit_to_graph_edges(anchors_df, refs_df, registry_df):
    """Build and emit graph_edges table."""
    
    edges = []
    
    # Structural edges: parent_of from anchor hierarchy
    for _, anchor in anchors_df.iterrows():
        if anchor.get("parent_anchor_id"):
            edges.append({
                "edge_id": f"edge_parent_{anchor['anchor_id']}",
                "doc_id": anchor["doc_id"],
                "source_node_id": anchor["parent_anchor_id"],
                "target_node_id": anchor["anchor_id"],
                "edge_type": "parent_of",
                "direction": "directed",
                "confidence": 1.0,
                "source_evidence": f"Hierarchy: {anchor['parent_anchor_id']} contains {anchor['anchor_id']}",
                "created_by": "structural",
                "created_at": datetime.now(),
                "metadata": {}
            })
    
    # Reference edges from refs_df
    for _, ref in refs_df.iterrows():
        target_node = resolve_reference_target(ref["target_key"], anchors_df, registry_df)
        if target_node:
            edge_type = map_reference_type_to_edge_type(ref["reference_type"])
            edges.append({
                "edge_id": f"edge_ref_{ref['source_anchor_id']}_{target_node}",
                "doc_id": anchors_df[anchors_df["anchor_id"] == ref["source_anchor_id"]]["doc_id"].iloc[0],
                "source_node_id": ref["source_anchor_id"],
                "target_node_id": target_node,
                "edge_type": edge_type,
                "direction": "directed",
                "confidence": ref["confidence"],
                "source_evidence": ref["reference_text"],
                "created_by": ref["extraction_method"],
                "created_at": datetime.now(),
                "metadata": {"reference_type": ref["reference_type"]}
            })
    
    return spark.createDataFrame(edges)

# Execute
graph_nodes_df = emit_to_graph_nodes(sections_df, anchors_df, registry_df)
graph_edges_df = emit_to_graph_edges(anchors_df, refs_df, registry_df)

# Write to Delta
graph_nodes_df.write.mode("overwrite").saveAsTable("catalog.schema.graph_nodes")
graph_edges_df.write.mode("overwrite").saveAsTable("catalog.schema.graph_edges")
```

---

## 3. Current Implementation Status

### 3.1 Status by Cell

| Cell | Status | Notes |
|------|--------|-------|
| Cell 1 | ✅ Complete | Span extraction working |
| Cell 2 | ✅ Complete | Body size = 9.0 for 1099-DIV |
| Cell 3 | ✅ Complete | Role classification working |
| Cell 4 | ✅ Complete | 22 anchors detected, groups handled |
| Cell 5 | ✅ Complete | Timeline built |
| Cell 6 | ⚠️ In Progress | Merge collision issue identified |
| Cell 7 | 📋 Specced | Waiting on Cell 6 |
| Cell 8 | 📋 Specced | Waiting on Cell 7 |
| Cell 9 | 📋 Specced | Waiting on Cell 8 |
| Cell 10 | 📋 Specced | Waiting on Cell 9 |

### 3.2 Validation Results

**Anchor Coverage (Cell 4):**
```
Expected: 16 boxes + groups
Found: 22 anchors

Box coverage: 100%
Groups detected: box_14_16_group

Missing concept sections: Qualified Dividends, RICs and REITs, Section 897 Gain
(These are SectionHeaders without box patterns - need separate detection)
```

**Role Distribution (Cell 3):**
```
Role counts:
- BodyText: 487
- ListBlock: 156
- BoxHeader: 22
- SectionHeader: 15
- PageArtifact: 12
- DocTitle: 1
```

---

## 4. Key Functions

### 4.1 Utility Functions

```python
def stable_hash(parts: str, length: int = 16) -> str:
    """Generate stable short hash for IDs."""
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:length]

def normalize_whitespace(s: str) -> str:
    """Collapse whitespace, strip."""
    return re.sub(r"\s+", " ", (s or "").strip())

def bbox_union(bboxes: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """Compute union of bounding boxes."""
    if not bboxes:
        return None
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (float(x0), float(y0), float(x1), float(y1))
```

### 4.2 Registry Functions

```python
def lookup_canonical_id(box_key: str, registry_df: pd.DataFrame) -> Optional[str]:
    """Look up canonical_id from registry by box_key or alias."""
    
    # Direct key match
    match = registry_df[registry_df["box_key"] == box_key]
    if len(match) > 0:
        return match["canonical_id"].iloc[0]
    
    # Alias match
    for _, row in registry_df.iterrows():
        aliases = row.get("aliases", [])
        if box_key in aliases or box_key.replace("box_", "") in aliases:
            return row["canonical_id"]
    
    return None

def resolve_reference_target(target_key: str, anchors_df: pd.DataFrame, registry_df: pd.DataFrame) -> Optional[str]:
    """Resolve reference target_key to actual node_id."""
    
    # Direct anchor match
    match = anchors_df[anchors_df["key"] == target_key]
    if len(match) > 0:
        return match["anchor_id"].iloc[0]
    
    # Registry alias lookup
    canonical = lookup_canonical_id(target_key, registry_df)
    if canonical:
        match = anchors_df[anchors_df["key"].apply(
            lambda k: lookup_canonical_id(k, registry_df) == canonical
        )]
        if len(match) > 0:
            return match["anchor_id"].iloc[0]
    
    return None  # External or unresolved reference
```

---

## 5. Configuration & Constants

### 5.1 Regex Patterns

```python
# Section header detection
SECTION_HEADER_RX = re.compile(
    r"^(What|Who|When|How|General|Specific|Instructions|Future Developments|"
    r"Qualified Dividends|RICs and REITs|Section \d+|Nondividend)\s*",
    re.IGNORECASE
)

# Boilerplate detection
BOILERPLATE_RX = re.compile(
    r"Instructions for Form|Department of the Treasury|Internal Revenue Service|"
    r"Rev\.\s*\w+\s*\d{4}|www\.irs\.gov",
    re.IGNORECASE
)

# Box patterns
BOX_RX_SINGLE = re.compile(r"^[Bb]ox\s*(\d+[a-z]?)\.?\s+(\w+)", re.IGNORECASE)
BOX_RX_RANGE = re.compile(r"^[Bb]oxes?\s*(\d+[a-z]?)\s*(through|[-–])\s*(\d+[a-z]?)", re.IGNORECASE)
BOX_RX_DOUBLE = re.compile(r"^[Bb]oxes?\s*(\d+[a-z]?)\s*and\s*(\d+[a-z]?)", re.IGNORECASE)
BOX_LINE_PREFIX_RX = re.compile(r"^\s*[Bb]oxes?\s*\d+", re.IGNORECASE)
BOX_PREFIX_RX = re.compile(r"^\s*[Bb]oxes?\s*\d+", re.IGNORECASE)

# Bullet and list detection
BULLET_RX = re.compile(r"^\s*[•\-\*]\s*")
ENUM_RX = re.compile(r"^\s*\(?[0-9]{1,2}[).]?\s*[A-Za-z]")
```

### 5.2 Role Constants

```python
ROLE_DOC_TITLE = "DocTitle"
ROLE_SECTION_HEADER = "SectionHeader"
ROLE_BOX_HEADER = "BoxHeader"
ROLE_BODY_TEXT = "BodyText"
ROLE_LIST_BLOCK = "ListBlock"
ROLE_TABLE_BLOCK = "TableBlock"
ROLE_PAGE_ARTIFACT = "PageArtifact"

ROLE_CONFIDENCE_THRESHOLDS = {
    ROLE_DOC_TITLE: 0.99,
    ROLE_SECTION_HEADER: 0.90,
    ROLE_BOX_HEADER: 0.95,
    ROLE_BODY_TEXT: 0.80,
    ROLE_LIST_BLOCK: 0.85,
    ROLE_TABLE_BLOCK: 0.85,
    ROLE_PAGE_ARTIFACT: 0.95,
}
```

---

## 6. Known Issues & Workarounds

### 6.1 Cell 6 Merge Collision

**Issue:** "Merge keys are not unique in left dataset" error when merging elements with anchors.

**Root Cause:** Duplicate `(doc_id, page, element_id)` combinations from split operation.

**Diagnosis:**
```python
# Find duplicates
dups = elements_df.groupby(["doc_id", "page", "element_id"]).size()
print(dups[dups > 1])
```

**Workaround:**
```python
# Deduplicate before merge
elements_df = elements_df.drop_duplicates(subset=["doc_id", "page", "element_id"], keep="first")
```

**Proper Fix:** Ensure unique IDs during split operation in Cell 3.

### 6.2 Concept Sections Not Detected as Anchors

**Issue:** "Qualified Dividends", "RICs and REITs" etc. are SectionHeaders but don't match box pattern.

**Impact:** These sections won't become anchors, so their content won't be properly attributed.

**Workaround (Phase A):** Manually add section anchors after Cell 4:
```python
concept_sections = [
    {"key": "section_qualified_dividends", "title_text": "Qualified Dividends", "page": 2},
    {"key": "section_rics_reits", "title_text": "RICs and REITs", "page": 2},
    {"key": "section_897_gain", "title_text": "Section 897 Gain", "page": 3},
]
# Add to anchors_df
```

**Proper Fix (Phase B):** Extend Cell 4 to detect SectionHeaders that match known concept names from registry.

### 6.3 Cross-Page Section Continuity

**Issue:** Sections that span multiple pages may have content incorrectly split.

**Impact:** Section text may be incomplete or content assigned to wrong anchor.

**Workaround:** Post-processing to merge sections with same anchor across pages.

**Proper Fix:** Handle page continuity in Cell 7 assignment logic.

---

## 7. Testing & Validation

### 7.1 Unit Tests

```python
def test_box_pattern_detection():
    """Test box regex patterns."""
    test_cases = [
        ("Box 1a. Total Ordinary Dividends", "1a"),
        ("Box 2e. Section 897 Ordinary Dividends", "2e"),
        ("Boxes 14-16. State Tax Information", ["14", "15", "16"]),
        ("Boxes 2a, 2b, and 2c.", ["2a", "2b", "2c"]),
    ]
    
    for text, expected in test_cases:
        result = extract_box_numbers(text)
        assert result == expected, f"Failed: {text} -> {result} (expected {expected})"

def test_anchor_coverage():
    """Verify all expected boxes found."""
    expected_boxes = set(["1a", "1b", "2a", "2b", "2c", "2d", "2e", "2f", 
                          "3", "4", "5", "6", "7", "8", "9", "10", 
                          "11", "12", "13", "14", "15", "16"])
    
    found_boxes = set(anchors_df["key"].str.replace("box_", "").tolist())
    
    missing = expected_boxes - found_boxes
    assert len(missing) == 0, f"Missing boxes: {missing}"
```

### 7.2 Integration Tests

```python
def test_end_to_end_1099div():
    """Test full pipeline on 1099-DIV."""
    
    # Run pipeline
    spans_df = extract_spans(INSTRUCTIONS_PATH)
    body_size = infer_body_font_size(spans_df)
    elements_df = build_line_elements(spans_df, body_size)
    anchors_df = anchors_from_elements(elements_df)
    
    # Validate outputs
    assert len(spans_df) > 1000, "Too few spans extracted"
    assert body_size > 0, "Invalid body size"
    assert len(elements_df) > 100, "Too few elements"
    assert len(anchors_df) >= 16, "Too few anchors"
    
    # Validate anchor coverage
    box_keys = anchors_df["key"].str.extract(r"box_(\d+[a-z]?)")[0].dropna()
    assert len(box_keys) >= 16, f"Only {len(box_keys)} boxes found"
```

### 7.3 Manual Validation Checklist

- [ ] All 16 boxes detected in anchors_df
- [ ] Box groups (14-16) have correct parent_anchor_id
- [ ] Role classification matches visual inspection
- [ ] Reading order follows document flow
- [ ] No duplicate element IDs
- [ ] Reference extraction catches "Box 1a includes..."
- [ ] Graph edges form valid DAG

---

*Document generated: 2026-01-05*  
*Next update: After Cell 6 fix*
