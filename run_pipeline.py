#!/usr/bin/env python3
"""
Run the extraction pipeline on 1099-DIV and assess graph quality.
"""

import sys
sys.path.insert(0, 'src')

import re
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

from vaas.semantic.concept_roles import classify_concept_role
from vaas.semantic.typed_edges import extract_typed_edges_from_section
from vaas.extraction import detect_subsection_candidates, assign_split_triggers

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install PyMuPDF")
    sys.exit(1)

# =============================================================================
# CELL 1: Extract Spans
# =============================================================================
print("=" * 70)
print("CELL 1: Extracting spans from PDF")
print("=" * 70)

def stable_hash(parts, length=16):
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:length]

def repair_hyphenation(text):
    """Fix hyphenated line breaks in extracted PDF text.

    Handles cases like 'fur-\\nnishing' -> 'furnishing'
    Only joins if next word starts with lowercase (avoids proper nouns).
    """
    # Pattern: word ending with hyphen, newline, then lowercase continuation
    text = re.sub(r'(\w)-\n([a-z])', r'\1\2', text)
    # Also handle with space after newline
    text = re.sub(r'(\w)-\n\s+([a-z])', r'\1\2', text)
    return text

pdf_path = "data/i1099div.pdf"
doc = fitz.open(pdf_path)
doc_id = "1099div_filer"

span_rows = []
for pno in range(doc.page_count):
    page = doc.load_page(pno)
    d = page.get_text("dict")
    for b_idx, block in enumerate(d.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for l_idx, line in enumerate(block.get("lines", [])):
            for s_idx, span in enumerate(line.get("spans", [])):
                text = span.get("text", "")
                if not text or not text.strip():
                    continue
                bbox = span.get("bbox", None)
                if bbox is None:
                    continue
                span_rows.append({
                    "doc_id": doc_id,
                    "page": int(pno + 1),
                    "block_id": int(b_idx),
                    "line_id": int(l_idx),
                    "span_id": int(s_idx),
                    "text": text,
                    "font": span.get("font", ""),
                    "flags": int(span.get("flags", 0)),
                    "size": float(span.get("size", 0.0)),
                    "bbox": tuple(map(float, bbox)),
                })

spans_df = pd.DataFrame(span_rows)
print(f"Pages: {doc.page_count}")
print(f"Spans extracted: {len(spans_df)}")

# =============================================================================
# CELL 2: Infer Body Font Size
# =============================================================================
print("\n" + "=" * 70)
print("CELL 2: Inferring body font size")
print("=" * 70)

s = spans_df["size"].astype(float).round(1)
body_size = float(s.value_counts().idxmax())
print(f"Body font size: {body_size}")
print(f"Font size distribution (top 5):")
print(s.value_counts().head(5).to_string())

# =============================================================================
# CELL 3: Split Blocks on Header-Like Lines
# =============================================================================
print("\n" + "=" * 70)
print("CELL 3: Split blocks on header-like lines")
print("=" * 70)

# Regexes - NOTE: Box(?:es)? matches "Box" or "Boxes" (NOT "Boxes?" which matches "Boxe" or "Boxes")
# STRONG: Requires period/colon + title text (unambiguous header)
BOX_STRONG_RX = re.compile(
    r"^Box(?:es)?\s+\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?"
    r"(?:\.\s+[A-Z]|:\s+[A-Z]|\s+[-–]\s+[A-Z])",  # Title must start with capital
    re.IGNORECASE
)
# WEAK: Requires period or colon after box key (prevents "box 2a may qualify...")
BOX_WEAK_RX = re.compile(
    r"^Box(?:es)?\s+\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?[.:]",
    re.IGNORECASE
)
# Main section headers (top-level document structure)
SECTION_HDR_RX = re.compile(
    r"^(Specific Instructions|General Instructions|Definitions|"
    r"Future Developments|Reminders|What's New|How To|Where To|"
    r"Paperwork Reduction Act Notice|Additional Information)\b",
    re.IGNORECASE
)

# Subsection headers are detected by LAYOUT, not by enumerated text patterns
# This generalizes across IRS instruction PDFs without maintaining brittle lists
# Criteria: bold, short, left-aligned, title-case-ish, not ending in period
# (The actual detection happens below after line geometry is computed)

# Fix A: Page marker detection - isolated centered markers like "-2-", "-3-", "-4-"
PAGE_MARKER_RX = re.compile(r"^\s*-\s*\d{1,3}\s*-\s*$")

# Bullets and list items (used to exclude from column detection)
BULLET_RX = re.compile(r"^\s*[•\-\*]\s*")
# ENUM_RX: require punctuation after number to avoid false matches like "10 years..."
# Enum pattern: requires . or ) terminator (not optional) to avoid false positives on numeric-leading text
ENUM_RX = re.compile(r"^\s*\(?\d{1,2}\s*[.)]\s+[A-Za-z]")

def _safe_bbox(b):
    try:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        return np.nan, np.nan, np.nan, np.nan
    except:
        return np.nan, np.nan, np.nan, np.nan

def _is_bold(font, flags):
    font_str = str(font or "").lower()
    if "bold" in font_str:
        return True
    return bool(int(flags or 0) & 16)

# Build line elements
tmp = spans_df.copy()
tmp[["x0", "y0", "x1", "y1"]] = pd.DataFrame(
    tmp["bbox"].apply(_safe_bbox).tolist(), index=tmp.index
)
tmp["bold"] = tmp.apply(lambda r: _is_bold(r.get("font", ""), r.get("flags", 0)), axis=1)
tmp["text"] = tmp["text"].fillna("").astype(str)

line_keys = ["doc_id", "page", "block_id", "line_id"]
tmp_sorted = tmp.sort_values(line_keys + ["x0", "span_id"], kind="mergesort")

line_df = tmp_sorted.groupby(line_keys, as_index=False).agg(
    line_text=("text", lambda x: "".join(x).strip()),
    geom_x0=("x0", "min"),
    geom_y0=("y0", "min"),
    geom_x1=("x1", "max"),
    geom_y1=("y1", "max"),
    line_size=("size", "median"),
    line_bold=("bold", lambda x: bool(np.any(x))),
    span_ids=("span_id", lambda x: list(map(str, x))),
)

line_df = line_df.sort_values(
    ["doc_id", "page", "block_id", "geom_y0", "geom_x0", "line_id"], kind="mergesort"
)
line_df["line_idx_in_block"] = line_df.groupby(["doc_id", "page", "block_id"]).cumcount()

# Detect header-like lines
def _margin_tol(block_width):
    return float(max(2.0, 0.02 * (block_width if pd.notna(block_width) else 0.0)))

df = line_df.copy()
block_geom = df.groupby(["doc_id", "page", "block_id"], as_index=False).agg(
    block_x0=("geom_x0", "min"), block_x1=("geom_x1", "max")
)
df = df.merge(block_geom, on=["doc_id", "page", "block_id"], how="left")
df["block_width"] = df["block_x1"] - df["block_x0"]
df["margin_tol"] = df["block_width"].apply(_margin_tol)
df["left_aligned"] = (df["geom_x0"] - df["block_x0"]).abs() <= df["margin_tol"]
df["has_emphasis"] = (df["line_size"] > float(body_size) + 0.5) | (df["line_bold"] == True)

# Fix A: Compute page geometry for page marker detection
page_geom = df.groupby(["doc_id", "page"], as_index=False).agg(
    page_x0=("geom_x0", "min"), page_x1=("geom_x1", "max")
)
page_geom["page_mid_x"] = (page_geom["page_x0"] + page_geom["page_x1"]) / 2
page_geom["page_width"] = page_geom["page_x1"] - page_geom["page_x0"]
df = df.merge(page_geom[["doc_id", "page", "page_mid_x", "page_width"]], on=["doc_id", "page"], how="left")

# =============================================================================
# COLUMN-AWARE GEOMETRY: Detect columns per page via dominant x0 peaks
# =============================================================================
# Two-column PDFs have content at two distinct x0 positions. We detect this by
# finding dominant peaks in the rounded x0 distribution (more robust than gap detection).

DEBUG_COLUMNS = False  # Set True to see per-page column detection details

# First, detect bullets/enumerations to exclude from column detection
_t_col = df["line_text"].fillna("").astype(str).str.strip()
df["_is_bullet"] = _t_col.str.match(BULLET_RX, na=False)
df["_is_enum"] = _t_col.str.match(ENUM_RX, na=False)
df["_is_list_item"] = df["_is_bullet"] | df["_is_enum"]

def detect_columns_for_page(page_df, min_peak_ratio=0.25):
    """Detect 1 or 2 columns for a page based on x0 peak analysis.

    Uses rounded x0 values and frequency counts to find dominant column margins.
    Excludes bullets/enumerations to avoid locking onto indent levels.

    Returns dict: {num_columns, col_0_x0, col_1_x0 (if 2 cols), split_x (if 2 cols)}
    """
    page_width = page_df["page_width"].iloc[0] if len(page_df) > 0 else 500.0

    # Compute text length locally (line_char_count may not exist yet)
    text_len = page_df["line_text"].fillna("").astype(str).str.len()

    # Only use non-list-item lines with sufficient text for column detection
    mask = (~page_df["_is_list_item"]) & (text_len >= 20)
    filtered = page_df.loc[mask, "geom_x0"].dropna()

    if len(filtered) < 5:
        # Not enough data; fall back to all lines
        filtered = page_df["geom_x0"].dropna()

    if len(filtered) < 2:
        return {"num_columns": 1, "col_0_x0": float(filtered.min()) if len(filtered) else 0.0}

    # Round x0 to nearest 4 pixels to cluster nearby values
    x0_rounded = (filtered / 4).round() * 4

    # Count frequencies of rounded x0
    x0_counts = x0_rounded.value_counts().sort_index()

    # Get significant peaks (at least 3 occurrences)
    significant = x0_counts[x0_counts >= 3].sort_values(ascending=False)

    if len(significant) == 0:
        return {"num_columns": 1, "col_0_x0": float(x0_rounded.min())}

    peak1_x0 = significant.index[0]
    peak1_count = significant.iloc[0]

    # Look for second column: needs sufficient count AND distance from peak1
    if len(significant) >= 2:
        for i in range(1, len(significant)):
            peak2_x0 = significant.index[i]
            peak2_count = significant.iloc[i]
            distance = abs(peak2_x0 - peak1_x0)

            # Second peak must be: 25%+ of first peak AND far enough apart (25% page width)
            if peak2_count >= peak1_count * min_peak_ratio and distance > page_width * 0.25:
                left_x0 = min(peak1_x0, peak2_x0)
                right_x0 = max(peak1_x0, peak2_x0)
                split_x = (left_x0 + right_x0) / 2
                return {
                    "num_columns": 2,
                    "col_0_x0": float(left_x0),
                    "col_1_x0": float(right_x0),
                    "split_x": float(split_x),
                }

    return {"num_columns": 1, "col_0_x0": float(peak1_x0)}

# Detect columns per page (column geometry only - page geometry is separate)
col_info_rows = []
for (doc_id_c, page_c), page_df in df.groupby(["doc_id", "page"]):
    col_info = detect_columns_for_page(page_df)
    col_info_rows.append({
        "doc_id": doc_id_c,
        "page": page_c,
        "num_columns": col_info["num_columns"],
        "col_0_x0": col_info["col_0_x0"],
        "col_1_x0": col_info.get("col_1_x0", np.nan),
        "col_split_x": col_info.get("split_x", np.nan),
    })
col_info_df = pd.DataFrame(col_info_rows)
df = df.merge(col_info_df, on=["doc_id", "page"], how="left")

# Vectorized column assignment
df["line_column"] = 0
is_two_col = df["num_columns"] == 2
df.loc[is_two_col & (df["geom_x0"] >= df["col_split_x"]), "line_column"] = 1

# Vectorized column left margin lookup
df["col_left_x0"] = df["col_0_x0"]
df.loc[df["line_column"] == 1, "col_left_x0"] = df["col_1_x0"]

# Adaptive tolerance: 2% of page width (minimum 6px)
df["col_margin_tol"] = np.maximum(6.0, df["page_width"] * 0.02)

# COLUMN-AWARE left alignment: line x0 is at column's left margin
df["col_left_aligned"] = (df["geom_x0"] - df["col_left_x0"]).abs() <= df["col_margin_tol"]

# Summary output
two_col_pages = col_info_df[col_info_df["num_columns"] == 2]
print(f"Column detection: {len(two_col_pages)}/{len(col_info_df)} pages have 2 columns")

if DEBUG_COLUMNS:
    for _, ci in col_info_df.iterrows():
        if ci["num_columns"] == 2:
            print(f"  p{ci['page']}: 2 cols (L={ci['col_0_x0']:.1f}, R={ci['col_1_x0']:.1f}, split={ci['col_split_x']:.1f})")
        else:
            print(f"  p{ci['page']}: 1 col (L={ci['col_0_x0']:.1f})")

# Cleanup temporary columns (errors="ignore" for robustness)
df.drop(columns=["_is_bullet", "_is_enum", "_is_list_item"], inplace=True, errors="ignore")

# Line center for centering check
df["line_center_x"] = (df["geom_x0"] + df["geom_x1"]) / 2
df["line_width"] = df["geom_x1"] - df["geom_x0"]

# Page marker detection: regex + geometry (centered + narrow)
t = df["line_text"].fillna("").astype(str).str.strip()
df["is_page_marker_text"] = t.str.match(PAGE_MARKER_RX, na=False)
df["is_centered"] = (df["line_center_x"] - df["page_mid_x"]).abs() <= df["page_width"] * 0.15
df["is_narrow"] = df["line_width"] <= df["page_width"] * 0.15
df["is_page_marker"] = df["is_page_marker_text"] & df["is_centered"] & df["is_narrow"]

page_markers_found = df[df["is_page_marker"]]
print(f"Page markers detected: {len(page_markers_found)}")
for _, pm in page_markers_found.iterrows():
    print(f"  p{pm['page']}: '{pm['line_text']}' (center_x={pm['line_center_x']:.1f}, width={pm['line_width']:.1f})")

# Note: 't' was already computed above for page marker detection
df["is_box_strong"] = t.str.match(BOX_STRONG_RX, na=False)
df["is_box_weak"] = t.str.match(BOX_WEAK_RX, na=False)
df["is_section"] = t.str.match(SECTION_HDR_RX, na=False)

# LAYOUT-DRIVEN subsection detection (geometry + structure, no enumerated patterns)
# =============================================================================

# Basic text properties
df["line_char_count"] = t.str.len()
df["is_heading_length"] = df["line_char_count"].between(12, 60)  # 12-60 chars
df["ends_with_period"] = t.str.endswith(".")
df["has_multiple_words"] = t.str.count(r"\s+") >= 1
df["is_title_case_ish"] = t.str.match(r"^[A-Z][A-Za-z0-9\s\-\(\)\']+$", na=False)
df["looks_like_date"] = t.str.match(r"^[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}$", na=False)

# FIX: Per-doc early-in-doc check
df["min_page_doc"] = df.groupby("doc_id")["page"].transform("min")
df["is_early_in_doc"] = (df["page"] == df["min_page_doc"]) & (df["line_idx_in_block"] < 8)

# STRUCTURAL FILTER 1: Header/footer band exclusion (top/bottom 10% of page)
page_y_bounds = df.groupby(["doc_id", "page"], as_index=False).agg(
    page_y_min=("geom_y0", "min"),
    page_y_max=("geom_y1", "max")
)
page_y_bounds["page_h"] = page_y_bounds["page_y_max"] - page_y_bounds["page_y_min"]
df = df.merge(page_y_bounds[["doc_id", "page", "page_y_min", "page_y_max", "page_h"]], on=["doc_id", "page"], how="left")
df["in_header_band"] = df["geom_y0"] <= df["page_y_min"] + 0.10 * df["page_h"]
df["in_footer_band"] = df["geom_y1"] >= df["page_y_max"] - 0.10 * df["page_h"]

# STRUCTURAL FILTER 2: Repeated-line artifact detection (appears on 2+ pages = likely header/footer)
# FIX: Only treat repeated lines as artifacts if ALSO in header/footer band
# This prevents real headings that appear on multiple pages (e.g., continued sections) from being excluded
df["_norm_text"] = t.str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True).str.strip()  # Keep digits for rev/date distinction
page_count_per_text = df.groupby(["doc_id", "_norm_text"])["page"].transform("nunique")
df["is_repeated_across_pages"] = page_count_per_text >= 2
# FIX: Repeated text is only artifact if ALSO in header/footer band
df["is_artifact_repeat"] = df["is_repeated_across_pages"] & (df["in_header_band"] | df["in_footer_band"])

# STRUCTURAL FILTER 3: Whitespace gap above (heading isolation)
# Compute gap to previous line within same block
df = df.sort_values(["doc_id", "page", "block_id", "geom_y0"]).reset_index(drop=True)
df["prev_y1"] = df.groupby(["doc_id", "page", "block_id"])["geom_y1"].shift(1)
df["gap_above"] = df["geom_y0"] - df["prev_y1"].fillna(df["geom_y0"])
median_gap = df.groupby(["doc_id", "page", "block_id"])["gap_above"].transform("median")
df["has_large_gap_above"] = (df["gap_above"] >= median_gap * 1.3) | (df["line_idx_in_block"] <= 1)

# LAYOUT-DRIVEN SUBSECTION DETECTION (extracted to vaas.extraction.layout_detection)
df = detect_subsection_candidates(df)
df = assign_split_triggers(df)

# Print subsection candidates for debugging
subsec_found = df[df["split_kind"] == "subsection"]
print(f"Subsection candidates (layout-driven): {len(subsec_found)}")
for _, ss in subsec_found.head(10).iterrows():
    print(f"  p{ss['page']}: '{ss['line_text'][:50]}...' (bold={ss['line_bold']}, len={ss['line_char_count']})")

line_df_flags = df.reset_index(drop=True)

# Split blocks into elements
rows = []
for (doc_id_g, page, block_id), g in line_df_flags.groupby(
    ["doc_id", "page", "block_id"], sort=False
):
    g = g.sort_values(["geom_y0", "geom_x0", "line_id"], kind="mergesort").reset_index(drop=True)

    starts = [0]
    for i in range(len(g)):
        if g.loc[i, "split_trigger"] and i not in starts:
            starts.append(i)
            if i + 1 < len(g):
                starts.append(i + 1)

    starts = sorted(set(starts))
    seg_bounds = []
    for si, st in enumerate(starts):
        en = starts[si + 1] if si + 1 < len(starts) else len(g)
        if st < en:
            seg_bounds.append((st, en))

    seg_idx = 0
    for st, en in seg_bounds:
        seg = g.iloc[st:en].copy()
        first_is_trigger = bool(seg.iloc[0]["split_trigger"])
        kind = seg.iloc[0]["split_kind"] if first_is_trigger else "body"

        seg_text = "\n".join([str(x).strip() for x in seg["line_text"].tolist() if str(x).strip()]).strip()

        x0 = float(seg["geom_x0"].min())
        y0 = float(seg["geom_y0"].min())
        x1 = float(seg["geom_x1"].max())
        y1 = float(seg["geom_y1"].max())
        bbox = [x0, y0, x1, y1]

        source_line_ids = list(map(str, seg["line_id"].tolist()))
        span_ids = []
        for sids in seg["span_ids"].tolist():
            span_ids.extend(list(map(str, sids)))
        seen = set()
        span_ids = [s for s in span_ids if not (s in seen or seen.add(s))]

        element_id = f"{doc_id_g}:{int(page)}:{block_id}:seg{seg_idx}"
        seg_idx += 1

        rows.append({
            "doc_id": str(doc_id_g),
            "page": int(page),
            "block_id": str(block_id),
            "element_id": element_id,
            "text": seg_text,
            "bbox": bbox,
            "geom_x0": x0, "geom_y0": y0, "geom_x1": x1, "geom_y1": y1,
            "source_line_ids": source_line_ids,
            "source_span_ids": span_ids,
            "split_kind": kind,
        })

elements_df = pd.DataFrame(rows)

# Detect columns: x < 300 is left column (0), x >= 300 is right column (1)
# This handles typical 2-column IRS form layouts
PAGE_MID_X = 300.0
elements_df["x_column"] = (elements_df["geom_x0"] >= PAGE_MID_X).astype(int)

# Reading order: left column first (all of it), then right column
# Within each column: top to bottom
elements_df = elements_df.sort_values(
    ["doc_id", "page", "x_column", "geom_y0", "geom_x0", "element_id"], kind="mergesort"
)
elements_df["reading_order"] = elements_df.groupby(["doc_id", "page"]).cumcount()
elements_df = elements_df.reset_index(drop=True)

print(f"Lines: {len(line_df_flags)}")
print(f"Elements after split: {len(elements_df)}")
print(f"Split triggers found: {line_df_flags['split_trigger'].sum()}")

triggered = line_df_flags[line_df_flags["split_trigger"]]
print(f"\n--- Header-like lines ({len(triggered)}) ---")
for _, r in triggered.head(15).iterrows():
    print(f"  p{r['page']} [{r['split_kind']}] {r['line_text'][:60]}...")

# =============================================================================
# CELL 4: Classify Layout Elements
# =============================================================================
print("\n" + "=" * 70)
print("CELL 4: Classify layout elements")
print("=" * 70)

ROLE_BOX_HEADER = "BoxHeader"
ROLE_SECTION_HEADER = "SectionHeader"
ROLE_SUBSECTION_HEADER = "SubsectionHeader"
ROLE_LIST_BLOCK = "ListBlock"
ROLE_PAGE_ARTIFACT = "PageArtifact"
ROLE_BODY_TEXT = "BodyTextBlock"

# BULLET_RX and ENUM_RX defined earlier (line ~129)
PAGE_ARTIFACT_RX = re.compile(
    r"(?:^Page\s+\d+|Instructions for Form|Department of the Treasury|"
    r"Internal Revenue Service|\(Rev\.\s*\w*\s*\d{4}\)|www\.irs\.gov|"
    r"^-\d+-$|Cat\.\s*No\.\s*\d+)",
    re.IGNORECASE
)

elements_df["role"] = ROLE_BODY_TEXT
elements_df["role_conf"] = 0.7

text = elements_df["text"].fillna("").astype(str).str.strip()

is_box_split = elements_df["split_kind"] == "box"
box_pattern_match = text.str.match(BOX_WEAK_RX, na=False)
elements_df.loc[is_box_split & box_pattern_match, "role"] = ROLE_BOX_HEADER
elements_df.loc[is_box_split & box_pattern_match, "role_conf"] = 0.95
elements_df.loc[is_box_split & ~box_pattern_match, "role"] = ROLE_BOX_HEADER
elements_df.loc[is_box_split & ~box_pattern_match, "role_conf"] = 0.75

is_section_split = elements_df["split_kind"] == "section"
elements_df.loc[is_section_split, "role"] = ROLE_SECTION_HEADER
elements_df.loc[is_section_split, "role_conf"] = 0.90

# Subsection headers
is_subsection_split = elements_df["split_kind"] == "subsection"
elements_df.loc[is_subsection_split, "role"] = ROLE_SUBSECTION_HEADER
elements_df.loc[is_subsection_split, "role_conf"] = 0.85

is_list = text.str.match(BULLET_RX, na=False) | text.str.match(ENUM_RX, na=False)
is_list = is_list & (elements_df["role"] == ROLE_BODY_TEXT)
elements_df.loc[is_list, "role"] = ROLE_LIST_BLOCK
elements_df.loc[is_list, "role_conf"] = 0.85

# Fix A: Elements with split_kind="page_artifact" are definitively artifacts (geometry + regex confirmed)
is_split_artifact = elements_df["split_kind"] == "page_artifact"
elements_df.loc[is_split_artifact, "role"] = ROLE_PAGE_ARTIFACT
elements_df.loc[is_split_artifact, "role_conf"] = 0.99  # High confidence - geometry confirmed

# Also catch other artifacts by content pattern
is_artifact = text.str.contains(PAGE_ARTIFACT_RX, na=False, regex=True)
is_short = text.str.len() < 100
is_artifact = is_artifact & is_short & (elements_df["role"] == ROLE_BODY_TEXT)
elements_df.loc[is_artifact, "role"] = ROLE_PAGE_ARTIFACT
elements_df.loc[is_artifact, "role_conf"] = 0.90

print("Role distribution:")
print(elements_df["role"].value_counts().to_string())

box_headers = elements_df[elements_df["role"] == ROLE_BOX_HEADER]
print(f"\n--- Box Headers Found ({len(box_headers)}) ---")
for _, r in box_headers.head(25).iterrows():
    print(f"  p{r['page']}: {r['text'][:70]}...")

# =============================================================================
# CELL 5: Anchor Extraction + Normalization
# =============================================================================
print("\n" + "=" * 70)
print("CELL 5: Anchor extraction + normalization")
print("=" * 70)

EXPECTED_BOXES_1099DIV = {
    "1a", "1b", "2a", "2b", "2c", "2d", "2e", "2f",
    "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16"
}

BOX_SINGLE_PARSE = re.compile(r"^Box(?:es)?\s+(\d+[a-z]?)\b", re.IGNORECASE)
BOX_RANGE_PARSE = re.compile(r"^Box(?:es)?\s+(\d+[a-z]?)\s*[-–]\s*(\d+[a-z]?)\b", re.IGNORECASE)
BOX_DOUBLE_PARSE = re.compile(r"^Box(?:es)?\s+(\d+[a-z]?)\s+and\s+(\d+[a-z]?)\b", re.IGNORECASE)
BOX_THROUGH_PARSE = re.compile(r"^Box(?:es)?\s+(\d+)\s+through\s+(\d+)\b", re.IGNORECASE)

def parse_box_keys(text):
    t = (text or "").strip()

    m = BOX_RANGE_PARSE.match(t)
    if m:
        lo, hi = m.group(1).lower(), m.group(2).lower()
        try:
            lo_num = int(re.sub(r'[a-z]', '', lo))
            hi_num = int(re.sub(r'[a-z]', '', hi))
            keys = [str(k) for k in range(min(lo_num, hi_num), max(lo_num, hi_num) + 1)]
        except:
            keys = [lo, hi]
        label = BOX_RANGE_PARSE.sub("", t).strip().lstrip(".-–: ")
        return {"kind": "range", "keys": keys, "label": label}

    m = BOX_THROUGH_PARSE.match(t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        keys = [str(k) for k in range(min(lo, hi), max(lo, hi) + 1)]
        label = BOX_THROUGH_PARSE.sub("", t).strip().lstrip(".-–: ")
        return {"kind": "through", "keys": keys, "label": label}

    m = BOX_DOUBLE_PARSE.match(t)
    if m:
        keys = [m.group(1).lower(), m.group(2).lower()]
        label = BOX_DOUBLE_PARSE.sub("", t).strip().lstrip(".-–: ")
        return {"kind": "double", "keys": keys, "label": label}

    m = BOX_SINGLE_PARSE.match(t)
    if m:
        keys = [m.group(1).lower()]
        label = BOX_SINGLE_PARSE.sub("", t).strip().lstrip(".-–: ")
        return {"kind": "single", "keys": keys, "label": label}

    return {"kind": "unknown", "keys": [], "label": t}

anchor_rows = []

# Extract BOX anchors
for _, row in box_headers.iterrows():
    parsed = parse_box_keys(row["text"])
    if parsed["kind"] == "unknown" or not parsed["keys"]:
        print(f"  WARNING: Could not parse: {row['text'][:60]}")
        continue

    is_grouped = len(parsed["keys"]) > 1
    group_id = f"group_{row['element_id']}" if is_grouped else None

    for box_key in parsed["keys"]:
        anchor_id = f"box_{box_key}"
        anchor_rows.append({
            "anchor_id": anchor_id,
            "box_key": box_key,
            "anchor_type": "box",
            "label": parsed["label"],
            "source_element_id": row["element_id"],
            "source_text": row["text"],
            "parse_kind": parsed["kind"],
            "is_grouped": is_grouped,
            "group_id": group_id,
            "page": row["page"],
            "reading_order": row["reading_order"],
            "geom_y0": row["geom_y0"],
            "geom_x0": row["geom_x0"],
        })

# Fix B: Extract SECTION anchors (for preamble structure)
section_headers = elements_df[elements_df["role"] == ROLE_SECTION_HEADER]

# Map section header text to canonical section IDs
SECTION_ID_MAP = {
    "future developments": "sec_future_developments",
    "reminders": "sec_reminders",
    "general instructions": "sec_general_instructions",
    "specific instructions": "sec_specific_instructions",
    "what's new": "sec_whats_new",
    "definitions": "sec_definitions",
    "how to": "sec_how_to",
    "where to": "sec_where_to",
    "paperwork reduction act notice": "sec_paperwork_reduction",
    "additional information": "sec_additional_info",
}

for _, row in section_headers.iterrows():
    section_text = row["text"].strip()
    # Find matching section ID
    section_id = None
    section_label = section_text
    for pattern, sid in SECTION_ID_MAP.items():
        if section_text.lower().startswith(pattern):
            section_id = sid
            section_label = section_text
            break

    if section_id is None:
        # Fallback: create ID from text
        section_id = "sec_" + re.sub(r"[^a-z0-9]+", "_", section_text.lower()[:30]).strip("_")

    anchor_rows.append({
        "anchor_id": section_id,
        "box_key": "",  # Sections don't have box keys
        "anchor_type": "section",
        "label": section_label,
        "source_element_id": row["element_id"],
        "source_text": row["text"],
        "parse_kind": "section",
        "is_grouped": False,
        "group_id": None,
        "page": row["page"],
        "reading_order": row["reading_order"],
        "geom_y0": row["geom_y0"],
        "geom_x0": row["geom_x0"],
    })

print(f"Section anchors extracted: {len(section_headers)}")

# Extract SUBSECTION anchors (content-level structure)
# LAYOUT-DRIVEN: No enumerated ID map - IDs are generated from content + position hash
subsection_headers = elements_df[elements_df["role"] == ROLE_SUBSECTION_HEADER]

def slug_title(text, max_len=30):
    """Convert title text to URL-safe slug."""
    # Take first line only
    first_line = text.strip().split('\n')[0]
    # Normalize: lowercase, replace non-alphanum with underscore
    slug = re.sub(r"[^a-z0-9]+", "_", first_line.lower())
    # Strip leading/trailing underscores, truncate
    return slug.strip("_")[:max_len]

for _, row in subsection_headers.iterrows():
    subsection_text = row["text"].strip()
    subsection_label = subsection_text.split('\n')[0][:60]  # First line, truncated

    # Generate stable ID from content slug + position hash
    # This ensures: (1) human-readable prefix, (2) uniqueness via hash
    title_slug = slug_title(subsection_text)
    source_elem_id = row["element_id"]
    position_hash = stable_hash([source_elem_id], length=8)
    subsection_id = f"sub_{title_slug}_{position_hash}"

    anchor_rows.append({
        "anchor_id": subsection_id,
        "box_key": "",
        "anchor_type": "subsection",
        "label": subsection_label,
        "source_element_id": row["element_id"],
        "source_text": row["text"],
        "parse_kind": "subsection",
        "is_grouped": False,
        "group_id": None,
        "page": row["page"],
        "reading_order": row["reading_order"],
        "geom_y0": row["geom_y0"],
        "geom_x0": row["geom_x0"],
    })

print(f"Subsection anchors extracted: {len(subsection_headers)}")

anchors_df = pd.DataFrame(anchor_rows)
if not anchors_df.empty:
    anchors_df = anchors_df.sort_values(["page", "reading_order"])

    # INVARIANT CHECK: Detect near-duplicate subsections (same page + similar position + similar text)
    # This catches layout detection false positives
    subsections = anchors_df[anchors_df["anchor_type"] == "subsection"].copy()
    if len(subsections) > 0:
        # Normalize labels for comparison
        subsections["_norm_label"] = subsections["label"].str.lower().str.strip()
        # Check for same (page, norm_label) - these are true duplicates
        dup_mask = subsections.duplicated(subset=["page", "_norm_label"], keep="first")
        if dup_mask.any():
            dup_count = dup_mask.sum()
            print(f"  WARNING: Dropping {dup_count} near-duplicate subsection anchors")
            dup_ids = subsections[dup_mask]["anchor_id"].tolist()
            anchors_df = anchors_df[~anchors_df["anchor_id"].isin(dup_ids)]

    # Drop exact duplicates on anchor_id (shouldn't happen with hashed IDs, but safe)
    anchors_df = anchors_df.drop_duplicates(subset=["anchor_id"], keep="first")
    anchors_df = anchors_df.reset_index(drop=True)

# Validate BOX coverage (section anchors don't have box_key)
box_anchors = anchors_df[anchors_df["anchor_type"] == "box"] if not anchors_df.empty else pd.DataFrame()
found = set(box_anchors["box_key"].tolist()) if not box_anchors.empty else set()
missing = EXPECTED_BOXES_1099DIV - found
extras = found - EXPECTED_BOXES_1099DIV

print(f"\n--- Box Anchor Validation ---")
print(f"Expected boxes: {len(EXPECTED_BOXES_1099DIV)}")
print(f"Found boxes: {len(found)}")
print(f"Missing: {sorted(missing) if missing else 'None'}")
print(f"Extras: {sorted(extras) if extras else 'None'}")

if not missing:
    print(f"\n✓ VALIDATION PASSED - All {len(EXPECTED_BOXES_1099DIV)} boxes found!")
else:
    print(f"\n❌ VALIDATION FAILED - Missing boxes: {sorted(missing)}")

# Show section anchors
section_anchors = anchors_df[anchors_df["anchor_type"] == "section"] if not anchors_df.empty else pd.DataFrame()
print(f"\n--- Section Anchors ({len(section_anchors)}) ---")
for _, r in section_anchors.iterrows():
    print(f"  {r['anchor_id']}: {r['label'][:50]}...")

# Show subsection anchors
subsection_anchors = anchors_df[anchors_df["anchor_type"] == "subsection"] if not anchors_df.empty else pd.DataFrame()
print(f"\n--- Subsection Anchors ({len(subsection_anchors)}) ---")
for _, r in subsection_anchors.head(15).iterrows():
    print(f"  {r['anchor_id']}: {r['label'][:50]}...")
if len(subsection_anchors) > 15:
    print(f"  ... and {len(subsection_anchors) - 15} more")

print(f"\n--- Box Anchors ({len(box_anchors)}) ---")
for _, r in box_anchors.head(10).iterrows():
    print(f"  {r['anchor_id']}: [{r['parse_kind']}] {r['label'][:40]}...")

# =============================================================================
# CELL 6: Build Anchor Timeline
# =============================================================================
print("\n" + "=" * 70)
print("CELL 6: Build anchor timeline")
print("=" * 70)

timeline = anchors_df.merge(
    elements_df[["element_id", "reading_order", "page"]].rename(
        columns={"element_id": "source_element_id"}
    ),
    on="source_element_id",
    how="left",
    suffixes=("", "_elem")
)
timeline["start_reading_order"] = timeline["reading_order_elem"].fillna(timeline["reading_order"])
timeline = timeline.sort_values(["page", "start_reading_order"]).reset_index(drop=True)

timeline["end_reading_order"] = None
for page in timeline["page"].unique():
    page_mask = timeline["page"] == page
    page_indices = timeline[page_mask].index.tolist()
    for i, idx in enumerate(page_indices):
        if i + 1 < len(page_indices):
            next_idx = page_indices[i + 1]
            next_start = timeline.loc[next_idx, "start_reading_order"]
            timeline.loc[idx, "end_reading_order"] = next_start - 1
        else:
            max_ro = elements_df[elements_df["page"] == page]["reading_order"].max()
            timeline.loc[idx, "end_reading_order"] = max_ro

anchor_timeline = timeline

print(f"Anchor timeline entries: {len(anchor_timeline)}")
print(f"\n--- Timeline (first 15) ---")
for _, r in anchor_timeline.head(15).iterrows():
    print(f"  {r['anchor_id']}: p{r['page']} ro[{r['start_reading_order']}-{r['end_reading_order']}]")

# =============================================================================
# CELL 7: Assign Content to Anchors
# =============================================================================
print("\n" + "=" * 70)
print("CELL 7: Assign content to anchors")
print("=" * 70)

# Build a map of grouped anchors (anchors that share the same source element)
grouped_anchor_map = {}  # source_element_id -> [anchor_ids]
if "source_element_id" in anchor_timeline.columns:
    for src_elem, grp in anchor_timeline.groupby("source_element_id"):
        if len(grp) > 1:  # This is a grouped header
            grouped_anchor_map[src_elem] = grp["anchor_id"].tolist()

elements_df["anchor_id"] = "unassigned"
# For grouped anchors, we'll store multiple anchor assignments
elements_df["anchor_ids"] = None  # Will hold list for grouped cases

for page in elements_df["page"].unique():
    page_elements = elements_df[elements_df["page"] == page].copy()
    page_anchors = anchor_timeline[anchor_timeline["page"] == page].copy()

    if page_anchors.empty:
        elements_df.loc[elements_df["page"] == page, "anchor_id"] = "preamble"
        continue

    page_anchors = page_anchors.sort_values("start_reading_order")
    first_anchor_start = page_anchors["start_reading_order"].min()

    preamble_mask = (elements_df["page"] == page) & (elements_df["reading_order"] < first_anchor_start)
    elements_df.loc[preamble_mask, "anchor_id"] = "preamble"

    for idx, row in page_elements.iterrows():
        ro = row["reading_order"]
        for _, anchor in page_anchors.iterrows():
            start = anchor["start_reading_order"]
            end = anchor["end_reading_order"]
            if start is not None and end is not None:
                if start <= ro <= end:
                    anchor_id = anchor["anchor_id"]
                    elements_df.loc[idx, "anchor_id"] = anchor_id

                    # Check if this anchor is part of a group
                    src_elem = anchor.get("source_element_id")
                    if src_elem in grouped_anchor_map:
                        # Assign to ALL anchors in the group
                        elements_df.at[idx, "anchor_ids"] = grouped_anchor_map[src_elem]
                    break

print("Anchor assignment distribution:")
print(elements_df["anchor_id"].value_counts().head(20).to_string())

# =============================================================================
# CELL 8: Materialize Sections
# =============================================================================
print("\n" + "=" * 70)
print("CELL 8: Materialize sections")
print("=" * 70)

# First, collect all anchors that need sections (including grouped ones)
all_anchor_ids = set(elements_df["anchor_id"].unique())

# Add any grouped anchor members that might not have direct assignments
# (they share content via anchor_ids column)
for anchor_ids_list in elements_df["anchor_ids"].dropna():
    if isinstance(anchor_ids_list, list):
        all_anchor_ids.update(anchor_ids_list)

# Build a reverse map: anchor_id -> elements (including grouped sharing)
anchor_to_elements = {aid: [] for aid in all_anchor_ids}

for idx, row in elements_df.iterrows():
    primary = row["anchor_id"]
    grouped = row.get("anchor_ids")

    if grouped and isinstance(grouped, list):
        # This element belongs to ALL anchors in the group
        for aid in grouped:
            anchor_to_elements[aid].append(idx)
    elif primary in anchor_to_elements:
        anchor_to_elements[primary].append(idx)

sections = []
for anchor_id in all_anchor_ids:
    element_indices = anchor_to_elements.get(anchor_id, [])
    if not element_indices:
        continue

    anchor_elements = elements_df.loc[element_indices].copy()
    anchor_elements = anchor_elements.sort_values(["page", "reading_order"])

    anchor_meta = anchor_timeline[anchor_timeline["anchor_id"] == anchor_id]
    if not anchor_meta.empty:
        meta = anchor_meta.iloc[0]
        box_key = meta.get("box_key", "")
        label = meta.get("label", "")
        anchor_type = meta.get("anchor_type", "box")
        is_grouped = meta.get("is_grouped", False)
        group_id = meta.get("group_id", None)
    else:
        box_key = ""
        label = anchor_id
        anchor_type = "preamble" if anchor_id == "preamble" else "other"
        is_grouped = False
        group_id = None

    # Include box, section, AND subsection headers as header elements
    header_elements = anchor_elements[
        (anchor_elements["role"] == ROLE_BOX_HEADER) |
        (anchor_elements["role"] == ROLE_SECTION_HEADER) |
        (anchor_elements["role"] == ROLE_SUBSECTION_HEADER)
    ]
    # Exclude headers AND page artifacts from body
    body_elements = anchor_elements[
        (anchor_elements["role"] != ROLE_BOX_HEADER) &
        (anchor_elements["role"] != ROLE_SECTION_HEADER) &
        (anchor_elements["role"] != ROLE_SUBSECTION_HEADER) &
        (anchor_elements["role"] != ROLE_PAGE_ARTIFACT)  # Fix A: exclude artifacts
    ]

    header_text = "\n".join(header_elements["text"].tolist()).strip()
    body_text = "\n\n".join(body_elements["text"].tolist()).strip()
    full_text = f"{header_text}\n\n{body_text}".strip() if header_text else body_text

    # Apply hyphenation repair to fix PDF line-break artifacts like "fur-\nnishing"
    header_text = repair_hyphenation(header_text)
    body_text = repair_hyphenation(body_text)
    full_text = repair_hyphenation(full_text)

    all_bboxes = anchor_elements["bbox"].tolist()
    x0 = min(b[0] for b in all_bboxes if b) if all_bboxes else 0
    y0 = min(b[1] for b in all_bboxes if b) if all_bboxes else 0
    x1 = max(b[2] for b in all_bboxes if b) if all_bboxes else 0
    y1 = max(b[3] for b in all_bboxes if b) if all_bboxes else 0

    pages = sorted(anchor_elements["page"].unique().tolist())
    element_ids = anchor_elements["element_id"].tolist()

    sections.append({
        "anchor_id": anchor_id,
        "box_key": box_key,
        "anchor_type": anchor_type,
        "label": label,
        "header_text": header_text,
        "body_text": body_text,
        "full_text": full_text,
        "pages": pages,
        "bbox": [x0, y0, x1, y1],
        "element_count": len(anchor_elements),
        "element_ids": element_ids,
        "char_count": len(full_text),
        "is_grouped": is_grouped,
        "group_id": group_id,
    })

sections_df = pd.DataFrame(sections)

def sort_key(row):
    if row["anchor_id"] == "preamble":
        return (0, 0, "")
    elif row["anchor_id"] == "unassigned":
        return (2, 0, "")
    else:
        key = row["box_key"]
        match = re.match(r"(\d+)([a-z]?)", key)
        if match:
            num = int(match.group(1))
            letter = match.group(2) or ""
            return (1, num, letter)
        return (1, 999, key)

sections_df["_sort"] = sections_df.apply(sort_key, axis=1)
sections_df = sections_df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

# =============================================================================
# MERGE-FORWARD: Fold thin subsection anchors into the next anchor
# =============================================================================
# Instead of pruning thin anchors (which loses text), merge them forward into
# the next anchor. This preserves heading text for retrieval while fixing over-splits.

def _bbox_y0(b):
    """Extract y0 from bbox for sorting."""
    try:
        if isinstance(b, (list, tuple, np.ndarray)) and len(b) >= 2:
            return float(b[1])
        return float("inf")
    except Exception:
        return float("inf")

def merge_forward_thin_subsections(sdf: pd.DataFrame,
                                   col_info: pd.DataFrame = None,
                                   page_geom: pd.DataFrame = None,
                                   thin_char_thresh: int = 160,
                                   thin_elem_thresh: int = 2,
                                   body_char_thresh: int = 120) -> pd.DataFrame:
    """
    Merge thin subsection anchors into the next anchor on the same page/column.
    This fixes over-splitting where a heading gets extracted but its real body
    starts in the next anchor.

    Args:
        sdf: Sections DataFrame
        col_info: DataFrame with column geometry (col_split_x per page)
        page_geom: DataFrame with page geometry (page_mid_x per page)
        thin_char_thresh: Max char count to be considered thin
        thin_elem_thresh: Max element count to be considered thin
        body_char_thresh: Max body length to be considered a fragment

    Returns: DataFrame with thin nodes merged into their successors.
    """
    if sdf.empty:
        return sdf

    s = sdf.copy()

    # Build (doc_id, page) -> page_mid_x map from page_geom
    page_mid_map = {}
    if page_geom is not None and not page_geom.empty:
        for _, row in page_geom.iterrows():
            doc_id = row.get("doc_id", "unknown")
            page = int(row["page"])
            page_mid_map[(doc_id, page)] = float(row.get("page_mid_x", 306.0))

    # Build (doc_id, page) -> split_x map from col_info
    # Use page_mid_x as fallback for single-column pages
    split_map = {}
    if col_info is not None and not col_info.empty:
        for _, row in col_info.iterrows():
            doc_id = row.get("doc_id", "unknown")
            page = int(row["page"])
            num_cols = row.get("num_columns", 1)
            split_x = row.get("col_split_x", np.nan)
            page_mid = page_mid_map.get((doc_id, page), 306.0)

            # Use col_split_x for 2-column pages, page_mid_x for single-column
            if num_cols == 2 and pd.notna(split_x):
                split_map[(doc_id, page)] = float(split_x)
            else:
                split_map[(doc_id, page)] = float(page_mid)

    # Helper: extract page number from pages array
    def _page_num(p):
        if isinstance(p, (list, tuple, np.ndarray)) and len(p) > 0:
            return int(p[0])
        if isinstance(p, str):
            digits = "".join(ch for ch in p if ch.isdigit())
            return int(digits) if digits else 0
        return int(p) if pd.notna(p) else 0

    s["_page"] = s["pages"].apply(_page_num)

    # Get doc_id from sections_df (assume single doc or use first)
    # If doc_id column exists, use it; otherwise default to first doc in col_info
    default_doc_id = col_info["doc_id"].iloc[0] if col_info is not None and not col_info.empty else "unknown"
    if "doc_id" not in s.columns:
        s["_doc_id"] = default_doc_id
    else:
        s["_doc_id"] = s["doc_id"]

    # Helper: extract column from bbox using per-page split_x
    def _col_from_bbox(row):
        b = row["bbox"]
        doc_id = row["_doc_id"]
        page = row["_page"]
        split_x = split_map.get((doc_id, page), 306.0)  # Fallback to 306 if not found
        try:
            if isinstance(b, (list, tuple, np.ndarray)) and len(b) >= 1:
                return 0 if float(b[0]) < split_x else 1
            return 0
        except Exception:
            return 0

    s["_col"] = s.apply(_col_from_bbox, axis=1)
    s["_y0"] = s["bbox"].apply(_bbox_y0)

    # Sort in reading order: page, then column (left before right), then y0
    s = s.sort_values(["_page", "_col", "_y0"]).reset_index(drop=True)

    # Identify thin node candidates
    body_len = s["body_text"].fillna("").astype(str).str.strip().str.len()
    starts_lower = s["body_text"].fillna("").astype(str).str.strip().str[:1].str.islower()

    thin = (
        (s["anchor_type"] == "subsection") &
        (s["char_count"] < thin_char_thresh) &
        (s["element_count"] <= thin_elem_thresh) &
        ((body_len <= body_char_thresh) | starts_lower.fillna(False))
    )

    to_drop = set()
    merged_ids = []

    for i in range(len(s) - 1):
        # Skip if already dropped or if target is already dropped
        if i in to_drop or (i + 1) in to_drop:
            continue

        if not bool(thin.iloc[i]):
            continue

        # Only merge forward to next anchor on same page AND same column
        if s.loc[i, "_page"] != s.loc[i + 1, "_page"]:
            continue
        if s.loc[i, "_col"] != s.loc[i + 1, "_col"]:
            continue

        # Guard: only merge if thin node is ABOVE target in y (sorted order sanity)
        # After bbox unions this can get messy, so enforce explicitly
        if s.loc[i, "_y0"] > s.loc[i + 1, "_y0"] + 2:
            continue

        # Don't merge into box_section (keep box sections clean)
        if s.loc[i + 1, "anchor_type"] in ("box", "box_section"):
            continue

        # Don't merge into section anchors UNLESS thin has minimal body (header fragment case)
        # This allows doc titles like "Dividends and Distributions" to merge into first section
        if s.loc[i + 1, "anchor_type"] == "section":
            if body_len.iloc[i] > body_char_thresh:  # Has substantial body - don't merge into section
                continue

        # Don't merge into another subsection that has a header (prevents subsections eating each other)
        tgt_type = s.loc[i + 1, "anchor_type"]
        tgt_header = (s.loc[i + 1, "header_text"] or "").strip()
        if tgt_type == "subsection" and tgt_header:
            continue

        # Perform merge
        prefix = (s.loc[i, "label"] or "").strip()
        frag_body = (s.loc[i, "body_text"] or "").strip()
        next_header = (s.loc[i + 1, "header_text"] or "").strip()
        next_body = (s.loc[i + 1, "body_text"] or "").strip()

        # Prepend heading to next header_text
        merged_header = (prefix if not next_header else f"{prefix}\n{next_header}").strip()

        # Prepend fragment body to next body
        merged_body = next_body
        if frag_body:
            merged_body = f"{frag_body}\n\n{next_body}".strip() if next_body else frag_body

        s.loc[i + 1, "header_text"] = merged_header
        s.loc[i + 1, "body_text"] = merged_body
        s.loc[i + 1, "full_text"] = f"{merged_header}\n\n{merged_body}".strip() if merged_header else merged_body

        # Merge element provenance
        eids_i = s.loc[i, "element_ids"]
        eids_n = s.loc[i + 1, "element_ids"]
        if isinstance(eids_i, np.ndarray):
            eids_i = eids_i.tolist()
        if isinstance(eids_n, np.ndarray):
            eids_n = eids_n.tolist()
        eids_i = eids_i if eids_i else []
        eids_n = eids_n if eids_n else []
        merged_eids = list(dict.fromkeys(list(eids_i) + list(eids_n)))
        s.at[i + 1, "element_ids"] = merged_eids
        s.loc[i + 1, "element_count"] = len(merged_eids)

        # Union pages
        p_i = s.loc[i, "pages"]
        p_n = s.loc[i + 1, "pages"]
        if isinstance(p_i, np.ndarray):
            p_i = p_i.tolist()
        if isinstance(p_n, np.ndarray):
            p_n = p_n.tolist()
        p_i = p_i if p_i else []
        p_n = p_n if p_n else []
        merged_pages = sorted(set(list(p_i) + list(p_n)))
        s.at[i + 1, "pages"] = np.array(merged_pages, dtype=int)

        # Union bbox (use list for consistent serialization)
        b1, b2 = s.loc[i, "bbox"], s.loc[i + 1, "bbox"]
        if isinstance(b1, np.ndarray):
            b1 = b1.tolist()
        if isinstance(b2, np.ndarray):
            b2 = b2.tolist()
        if b1 and b2 and len(b1) >= 4 and len(b2) >= 4:
            s.at[i + 1, "bbox"] = [
                float(min(b1[0], b2[0])),
                float(min(b1[1], b2[1])),
                float(max(b1[2], b2[2])),
                float(max(b1[3], b2[3])),
            ]

        # Recompute char_count
        s.loc[i + 1, "char_count"] = len(s.loc[i + 1, "full_text"] or "")

        merged_ids.append((s.loc[i, "anchor_id"], s.loc[i + 1, "anchor_id"]))
        to_drop.add(i)

    if merged_ids:
        print(f"\n--- Merge-forward: {len(merged_ids)} thin subsections merged ---")
        for src, tgt in merged_ids:
            print(f"  {src} -> {tgt}")

    if to_drop:
        s = s.drop(index=list(to_drop)).reset_index(drop=True)

    # Normalize element_ids to list for consistent serialization
    s["element_ids"] = s["element_ids"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else (x or [])
    )

    # Normalize bbox to list for consistent serialization
    s["bbox"] = s["bbox"].apply(
        lambda b: b.tolist() if isinstance(b, np.ndarray) else (list(b) if b else [])
    )

    # Normalize pages to list for consistent serialization
    s["pages"] = s["pages"].apply(
        lambda p: p.tolist() if isinstance(p, np.ndarray) else (list(p) if p else [])
    )

    return s.drop(columns=["_page", "_col", "_y0"], errors="ignore")

# Apply merge-forward (pass col_info_df for column geometry, page_geom for page geometry)
sections_before = len(sections_df)
sections_df = merge_forward_thin_subsections(sections_df, col_info=col_info_df, page_geom=page_geom)
sections_after = len(sections_df)
if sections_before != sections_after:
    print(f"Sections: {sections_before} -> {sections_after} (merged {sections_before - sections_after})")

# =============================================================================
# CONCEPT ROLE CLASSIFICATION (Phase 1a)
# =============================================================================
# Classify subsection anchors into semantic roles using regex-first heuristics.
# Roles: definition, qualification, condition, exception, procedure
# Returns NULL when confidence is low (unknown > wrong).

def classify_section_roles(sdf: pd.DataFrame) -> pd.DataFrame:
    """Apply concept role classification to subsection anchors."""
    sdf = sdf.copy()
    sdf["concept_role"] = None
    sdf["concept_role_confidence"] = 0.0
    sdf["concept_role_method"] = "null"

    subsection_mask = sdf["anchor_type"] == "subsection"
    for idx in sdf[subsection_mask].index:
        header = sdf.loc[idx, "label"] or ""
        body = sdf.loc[idx, "body_text"] or ""

        result = classify_concept_role(header, body)

        sdf.loc[idx, "concept_role"] = result.role
        sdf.loc[idx, "concept_role_confidence"] = result.confidence
        sdf.loc[idx, "concept_role_method"] = result.method

    return sdf

sections_df = classify_section_roles(sections_df)

# Report concept role distribution
role_counts = sections_df[sections_df["anchor_type"] == "subsection"]["concept_role"].value_counts(dropna=False)
print(f"\n--- Concept Role Classification ---")
for role, count in role_counts.items():
    role_label = role if role else "NULL (low confidence)"
    print(f"  {role_label}: {count}")

print(f"\nSections created: {len(sections_df)}")
print(f"\n--- Section Summary ---")
for _, r in sections_df.iterrows():
    print(f"  {r['anchor_id']}: {r['element_count']} elements, {r['char_count']} chars - {r['label'][:30]}...")

# =============================================================================
# CELL 9: Extract References
# =============================================================================
print("\n" + "=" * 70)
print("CELL 9: Extract references")
print("=" * 70)

# Fix C: Extract references per-element for clean evidence (no cross-newline capture)
# Regex that stops at newlines - use [^\n] instead of . where needed
BOX_REF_RX = re.compile(
    r"[Bb]ox(?:es)?\s+(\d+[a-z]?(?:[^\n]*?(?:,|and|through|[-–]|to)\s*\d+[a-z]?)*)",
    re.IGNORECASE
)
PUB_REF_RX = re.compile(r"[Pp]ub(?:lication)?\.?\s*(\d+)")
IRC_REF_RX = re.compile(r"[Ss]ection\s+(\d+[A-Za-z]?(?:\([a-z0-9]\))?)")
FORM_REF_RX = re.compile(r"[Ff]orm\s+(\d+[A-Z\-]*)")

def parse_box_ref_keys(ref_text):
    keys = []
    range_match = re.search(r"(\d+)\s*(?:[-–]|through|to)\s*(\d+)", ref_text, re.IGNORECASE)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        keys.extend([str(k) for k in range(min(lo, hi), max(lo, hi) + 1)])
    individual = re.findall(r"\b(\d+[a-z]?)\b", ref_text, re.IGNORECASE)
    for k in individual:
        k_lower = k.lower()
        if k_lower not in keys:
            keys.append(k_lower)
    return keys

def extract_evidence_quote(text, match, context_chars=50):
    """Extract clean evidence quote with context around match."""
    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)
    quote = text[start:end].strip()
    # Clean up any remaining newlines in quote
    quote = " ".join(quote.split())
    if start > 0:
        quote = "..." + quote
    if end < len(text):
        quote = quote + "..."
    return quote

# Fix C: Extract per-element, not per-section
# This avoids cross-element contamination and provides element_id provenance
references = []

# Build element -> anchor mapping for reference extraction
element_to_anchor = dict(zip(elements_df["element_id"], elements_df["anchor_id"]))

for _, row in elements_df.iterrows():
    # Skip artifacts and unassigned elements
    if row["role"] == ROLE_PAGE_ARTIFACT:
        continue
    if row["anchor_id"] in ["unassigned", ""]:
        continue

    source_anchor = row["anchor_id"]
    source_element = row["element_id"]
    text = row["text"]
    if not text or not text.strip():
        continue

    # Normalize text for matching (collapse whitespace, but preserve for quote extraction)
    clean_text = " ".join(text.split())  # Collapse all whitespace

    for match in BOX_REF_RX.finditer(clean_text):
        ref_text = match.group(0).strip()
        ref_keys = parse_box_ref_keys(match.group(1))
        evidence_quote = extract_evidence_quote(clean_text, match)

        for key in ref_keys:
            target_anchor = f"box_{key}"
            if target_anchor == source_anchor:
                continue
            target_exists = target_anchor in sections_df["anchor_id"].values
            references.append({
                "source_anchor_id": source_anchor,
                "source_element_id": source_element,  # Fix C: element provenance
                "target_anchor_id": target_anchor,
                "ref_type": "box_reference",
                "ref_text": ref_text,
                "evidence_quote": evidence_quote,  # Fix C: clean context
                "target_key": key,
                "target_exists": target_exists,
                "confidence": 0.95 if target_exists else 0.7,
                "created_by": "regex",
            })

    for match in PUB_REF_RX.finditer(clean_text):
        evidence_quote = extract_evidence_quote(clean_text, match)
        references.append({
            "source_anchor_id": source_anchor,
            "source_element_id": source_element,
            "target_anchor_id": f"pub_{match.group(1)}",
            "ref_type": "publication_reference",
            "ref_text": match.group(0),
            "evidence_quote": evidence_quote,
            "target_key": match.group(1),
            "target_exists": False,
            "confidence": 0.90,
            "created_by": "regex",
        })

references_df = pd.DataFrame(references)
if not references_df.empty:
    references_df = references_df.drop_duplicates(
        subset=["source_anchor_id", "target_anchor_id", "ref_type"], keep="first"
    )

print(f"References extracted: {len(references_df)}")
if not references_df.empty:
    print(f"\nReference types:")
    print(references_df["ref_type"].value_counts().to_string())

    internal_refs = references_df[
        (references_df["ref_type"] == "box_reference") &
        (references_df["target_exists"] == True)
    ]
    print(f"\n--- Internal Box References ({len(internal_refs)}) ---")
    for _, r in internal_refs.head(15).iterrows():
        print(f"  {r['source_anchor_id']} -> {r['target_anchor_id']}: \"{r['ref_text'][:40]}\"")

# =============================================================================
# CELL 10: Emit Graph
# =============================================================================
print("\n" + "=" * 70)
print("CELL 10: Emit graph nodes + edges")
print("=" * 70)

DOC_ID = "1099div_filer"

# Build nodes
nodes = []
for _, section in sections_df.iterrows():
    node_id = f"{DOC_ID}:{section['anchor_id']}"
    if section["anchor_id"] == "preamble":
        node_type = "preamble"
    elif section["anchor_id"] == "unassigned":
        node_type = "unassigned"
    elif section["anchor_type"] == "box":
        node_type = "box_section"
    elif section["anchor_type"] == "subsection":
        node_type = "concept"  # Promoted from subsection to concept
    else:
        node_type = "section"

    # Get concept_role for subsection/concept nodes
    concept_role = section.get("concept_role") if section["anchor_type"] == "subsection" else None

    nodes.append({
        "node_id": node_id,
        "doc_id": DOC_ID,
        "node_type": node_type,
        "anchor_id": section["anchor_id"],
        "box_key": section["box_key"],
        "label": section["label"],
        "text": section["full_text"],
        "pages": section["pages"],
        "element_count": section["element_count"],
        "char_count": section["char_count"],
        "concept_role": concept_role,
    })

graph_nodes = pd.DataFrame(nodes)

# Build edges
edges = []
edge_id = 0

for _, ref in references_df.iterrows():
    if ref["ref_type"] == "box_reference" and ref["target_exists"]:
        edges.append({
            "edge_id": f"e_{edge_id}",
            "source_node_id": f"{DOC_ID}:{ref['source_anchor_id']}",
            "target_node_id": f"{DOC_ID}:{ref['target_anchor_id']}",
            "edge_type": "references_box",
            "direction": "directed",
            "confidence": ref["confidence"],
            "source_evidence": ref.get("evidence_quote", ref["ref_text"]),  # Fix C: clean evidence
            "created_by": ref["created_by"],
        })
        edge_id += 1

# Same-group edges
if not anchors_df.empty and "group_id" in anchors_df.columns:
    grouped = anchors_df[anchors_df["is_grouped"] == True]
    for group_id in grouped["group_id"].dropna().unique():
        group_members = grouped[grouped["group_id"] == group_id]["anchor_id"].tolist()
        for i, a1 in enumerate(group_members):
            for a2 in group_members[i+1:]:
                edges.append({
                    "edge_id": f"e_{edge_id}",
                    "source_node_id": f"{DOC_ID}:{a1}",
                    "target_node_id": f"{DOC_ID}:{a2}",
                    "edge_type": "same_group",
                    "direction": "bidirectional",
                    "confidence": 1.0,
                    "source_evidence": f"Grouped header: {group_id}",
                    "created_by": "structural",
                })
                edge_id += 1

# =============================================================================
# TYPED EDGE EXTRACTION (Phase 1b) - Negative Knowledge
# =============================================================================
# Extract excludes edges from sections that contain negation patterns
# near box references. This captures "does not include", "except", etc.

# Build set of valid box keys for target validation
valid_box_keys = set()
if not anchors_df.empty:
    box_anchors = anchors_df[anchors_df["anchor_type"] == "box"]
    valid_box_keys = set(box_anchors["box_key"].str.lower().dropna())

typed_edge_counts = {"excludes": 0, "applies_if": 0, "defines": 0, "qualifies": 0, "requires": 0}
for _, section in sections_df.iterrows():
    anchor_id = section["anchor_id"]
    body_text = section.get("body_text", "") or ""

    # Get source_box_key for requires edges (box → box dependency)
    source_box_key = None
    if section.get("anchor_type") == "box":
        source_box_key = section.get("box_key", "").lower() or None

    # Extract typed edges (Phase 1b + 2a)
    typed_edges = extract_typed_edges_from_section(
        anchor_id=anchor_id,
        body_text=body_text,
        valid_box_keys=valid_box_keys,
        source_box_key=source_box_key,
    )

    for te in typed_edges:
        # Map box_key to anchor_id
        target_anchor_id = f"box_{te.target_box_key}"

        edges.append({
            "edge_id": f"e_{edge_id}",
            "source_node_id": f"{DOC_ID}:{anchor_id}",
            "target_node_id": f"{DOC_ID}:{target_anchor_id}",
            "edge_type": te.edge_type,
            "direction": "directed",
            "confidence": te.confidence,
            "source_evidence": te.evidence_text,
            "created_by": "regex",
            "pattern_matched": te.pattern_matched,
            "polarity": te.polarity,
        })
        edge_id += 1
        if te.edge_type in typed_edge_counts:
            typed_edge_counts[te.edge_type] += 1

total_typed = sum(typed_edge_counts.values())
if total_typed > 0:
    print(f"\n--- Typed Edges (Phase 1b + 2a) ---")
    for etype, count in typed_edge_counts.items():
        if count > 0:
            print(f"  {etype}: {count}")

graph_edges = pd.DataFrame(edges)

# Filter edges by active nodes (handles pruned nodes cleanly)
if not graph_edges.empty:
    active_node_ids = set(graph_nodes["node_id"].astype(str))
    edges_before = len(graph_edges)
    graph_edges = graph_edges[
        graph_edges["source_node_id"].isin(active_node_ids) &
        graph_edges["target_node_id"].isin(active_node_ids)
    ].reset_index(drop=True)
    edges_dropped = edges_before - len(graph_edges)
    if edges_dropped > 0:
        print(f"Dropped {edges_dropped} edges referencing pruned/missing nodes")

print(f"Graph nodes: {len(graph_nodes)}")
print(f"Graph edges: {len(graph_edges)}")

print(f"\n--- Node Types ---")
print(graph_nodes["node_type"].value_counts().to_string())

if not graph_edges.empty:
    print(f"\n--- Edge Types ---")
    print(graph_edges["edge_type"].value_counts().to_string())

# =============================================================================
# VALIDATION
# =============================================================================
print("\n" + "=" * 70)
print("KNOWLEDGE GRAPH QUALITY ASSESSMENT")
print("=" * 70)

# 1. Coverage check
box_nodes = graph_nodes[graph_nodes["node_type"] == "box_section"]
found_boxes = set(box_nodes["box_key"].tolist())
missing_boxes = EXPECTED_BOXES_1099DIV - found_boxes

print(f"\n1. BOX COVERAGE")
print(f"   Expected: {len(EXPECTED_BOXES_1099DIV)}")
print(f"   Found: {len(found_boxes)}")
if missing_boxes:
    print(f"   ❌ Missing: {sorted(missing_boxes)}")
else:
    print(f"   ✓ All boxes found")

# 2. Content quality
print(f"\n2. CONTENT QUALITY")
avg_chars = sections_df[sections_df["anchor_type"] == "box"]["char_count"].mean()
min_chars = sections_df[sections_df["anchor_type"] == "box"]["char_count"].min()
max_chars = sections_df[sections_df["anchor_type"] == "box"]["char_count"].max()
print(f"   Avg chars per box section: {avg_chars:.0f}")
print(f"   Min: {min_chars}, Max: {max_chars}")

empty_sections = sections_df[sections_df["char_count"] < 50]
if len(empty_sections) > 0:
    print(f"   ⚠️ {len(empty_sections)} sections with <50 chars")
else:
    print(f"   ✓ All sections have substantial content")

# 3. Reference density
print(f"\n3. REFERENCE DENSITY")
internal_refs = references_df[
    (references_df["ref_type"] == "box_reference") &
    (references_df["target_exists"] == True)
] if not references_df.empty else pd.DataFrame()
print(f"   Internal box references: {len(internal_refs)}")
print(f"   Avg refs per box: {len(internal_refs) / len(box_nodes):.1f}" if len(box_nodes) > 0 else "   N/A")

# 4. Graph connectivity
print(f"\n4. GRAPH STRUCTURE")
if not graph_edges.empty:
    sources = set(graph_edges["source_node_id"])
    targets = set(graph_edges["target_node_id"])
    connected_nodes = sources | targets
    box_node_ids = set(f"{DOC_ID}:{a}" for a in found_boxes.union({"box_" + b for b in found_boxes}))
    box_node_ids = set(f"{DOC_ID}:box_{b}" for b in found_boxes)
    connected_boxes = box_node_ids & connected_nodes
    print(f"   Boxes with edges: {len(connected_boxes)} / {len(box_nodes)}")
else:
    print(f"   No edges")

# 5. Sample content verification
print(f"\n5. SAMPLE CONTENT VERIFICATION")
sample_boxes = ["1a", "2a", "1b"]
for bk in sample_boxes:
    section = sections_df[sections_df["box_key"] == bk]
    if not section.empty:
        s = section.iloc[0]
        print(f"\n   Box {bk}: {s['label'][:40]}...")
        print(f"   Elements: {s['element_count']}, Chars: {s['char_count']}")
        preview = s["body_text"][:200].replace("\n", " ")
        print(f"   Preview: {preview}...")

print("\n" + "=" * 70)
print("ASSESSMENT COMPLETE")
print("=" * 70)

# Save outputs
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Centralized normalization for list columns (prevents parquet serialization errors)
def normalize_cell_list(x):
    """Normalize list-like values to plain Python lists for parquet serialization."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (int, float, np.integer)):
        return [int(x)]
    return []

def normalize_bbox(b):
    """Normalize bbox to list of 4 floats, or empty list if invalid."""
    b = normalize_cell_list(b)
    if len(b) == 4:
        return [float(x) for x in b]
    return []  # Invalid bbox - return empty

# Apply to all list columns in sections_df
sections_df["pages"] = sections_df["pages"].apply(normalize_cell_list)
sections_df["bbox"] = sections_df["bbox"].apply(normalize_bbox)
sections_df["element_ids"] = sections_df["element_ids"].apply(normalize_cell_list)

# Apply to all list columns in graph_nodes
graph_nodes["pages"] = graph_nodes["pages"].apply(normalize_cell_list)

graph_nodes.to_parquet(output_dir / "graph_nodes.parquet", index=False)
graph_edges.to_parquet(output_dir / "graph_edges.parquet", index=False)
sections_df.to_parquet(output_dir / "sections.parquet", index=False)
print(f"\nOutputs saved to {output_dir.absolute()}")
