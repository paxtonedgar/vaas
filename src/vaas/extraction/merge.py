"""
Merge-forward logic for thin subsection anchors.

This module handles the case where PDF extraction creates over-split sections:
a heading gets extracted as its own anchor, but its actual body content starts
in the next anchor. We fix this by merging thin fragments forward.

Algorithm Overview:
1. Sort sections by reading order (page, column, y-position)
2. Identify "thin" subsections (low char count, few elements, short body)
3. For each thin subsection, check if it can merge into the next anchor
4. If valid, concatenate text, union metadata, drop the thin anchor
5. Repeat until no more merges possible (iterative stabilization)

Merge Rules:
- Only merge subsections (not boxes or sections)
- Only merge within same page AND same column
- Only merge forward (into next anchor in reading order)
- Don't merge into box_section (keep box content clean)
- Don't merge into section unless thin has minimal body
- Don't merge into subsection that already has a header
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from vaas.utils.serialize import normalize_cell_list, normalize_bbox, safe_extract_page


logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """
    Configuration for merge-forward operation.

    Attributes:
        thin_char_thresh: Max char count to be considered thin.
        thin_elem_thresh: Max element count to be considered thin.
        body_char_thresh: Max body length to be considered a fragment.
        max_iterations: Safety limit on merge iterations.
        default_split_x: Fallback column split x-coordinate.
    """

    thin_char_thresh: int = 160
    thin_elem_thresh: int = 2
    body_char_thresh: int = 120
    max_iterations: int = 10
    default_split_x: float = 306.0


@dataclass
class MergeResult:
    """
    Result of merge-forward operation.

    Attributes:
        sections_df: The resulting sections DataFrame after merges.
        merges_performed: Total number of merges performed.
        iterations: Number of iterations until stabilization.
        merged_pairs: List of (source_anchor_id, target_anchor_id) pairs.
        anchors_dropped: Set of anchor_ids that were merged away.
    """

    sections_df: pd.DataFrame
    merges_performed: int = 0
    iterations: int = 0
    merged_pairs: List[Tuple[str, str]] = field(default_factory=list)
    anchors_dropped: Set[str] = field(default_factory=set)

    def __repr__(self) -> str:
        return (
            f"MergeResult(merges={self.merges_performed}, "
            f"iterations={self.iterations}, "
            f"dropped={len(self.anchors_dropped)})"
        )


def build_column_split_map(
    col_info_df: Optional[pd.DataFrame],
    page_geom_df: Optional[pd.DataFrame],
    default_split_x: float = 306.0,
) -> Dict[Tuple[str, int], float]:
    """
    Build a lookup map from (doc_id, page) to column split x-coordinate.

    For two-column pages, uses col_split_x from col_info_df.
    For single-column pages, uses page_mid_x from page_geom_df.
    Falls back to default_split_x if neither available.

    Args:
        col_info_df: DataFrame with columns: doc_id, page, num_columns, col_split_x
        page_geom_df: DataFrame with columns: doc_id, page, page_mid_x
        default_split_x: Fallback value when no geometry available.

    Returns:
        Dict mapping (doc_id, page) to split x-coordinate.
    """
    # Build page_mid_x lookup first
    page_mid_map: Dict[Tuple[str, int], float] = {}
    if page_geom_df is not None and not page_geom_df.empty:
        for _, row in page_geom_df.iterrows():
            doc_id = row.get("doc_id", "unknown")
            page = int(row["page"])
            page_mid_map[(doc_id, page)] = float(row.get("page_mid_x", default_split_x))

    # Build split_x map using col_info with page_mid fallback
    split_map: Dict[Tuple[str, int], float] = {}
    if col_info_df is not None and not col_info_df.empty:
        for _, row in col_info_df.iterrows():
            doc_id = row.get("doc_id", "unknown")
            page = int(row["page"])
            num_cols = row.get("num_columns", 1)
            split_x = row.get("col_split_x", np.nan)
            page_mid = page_mid_map.get((doc_id, page), default_split_x)

            # Use col_split_x for 2-column pages, page_mid_x for single-column
            if num_cols == 2 and pd.notna(split_x):
                split_map[(doc_id, page)] = float(split_x)
            else:
                split_map[(doc_id, page)] = float(page_mid)

    return split_map


def add_merge_sort_columns(
    df: pd.DataFrame,
    split_map: Dict[Tuple[str, int], float],
    default_doc_id: str = "unknown",
    default_split_x: float = 306.0,
) -> pd.DataFrame:
    """
    Add temporary columns for merge sorting: _page, _col, _y0, _doc_id.

    Args:
        df: Sections DataFrame.
        split_map: Map from (doc_id, page) to column split x.
        default_doc_id: Default doc_id if not in DataFrame.
        default_split_x: Fallback split x if not in map.

    Returns:
        DataFrame with added columns, sorted by reading order.
    """
    df = df.copy()

    # Extract page number
    df["_page"] = df["pages"].apply(safe_extract_page)

    # Set doc_id
    if "doc_id" not in df.columns:
        df["_doc_id"] = default_doc_id
    else:
        df["_doc_id"] = df["doc_id"]

    # Determine column from bbox x0
    def get_column(row: pd.Series) -> int:
        bbox = row.get("bbox")
        doc_id = row["_doc_id"]
        page = row["_page"]
        split_x = split_map.get((doc_id, page), default_split_x)

        if bbox is None:
            return 0

        try:
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 1:
                x0 = float(bbox[0])
                return 0 if x0 < split_x else 1
            return 0
        except (TypeError, ValueError, IndexError):
            return 0

    df["_col"] = df.apply(get_column, axis=1)

    # Extract y0 from bbox
    def get_y0(bbox: Any) -> float:
        if bbox is None:
            return float("inf")
        try:
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 2:
                return float(bbox[1])
            return float("inf")
        except (TypeError, ValueError, IndexError):
            return float("inf")

    df["_y0"] = df["bbox"].apply(get_y0)

    # Sort by reading order
    df = df.sort_values(["_page", "_col", "_y0"]).reset_index(drop=True)

    return df


def identify_thin_subsections(
    df: pd.DataFrame,
    config: MergeConfig,
) -> pd.Series:
    """
    Identify subsections that are candidates for merging (thin fragments).

    A subsection is "thin" if ALL of these conditions are met:
    - anchor_type == "subsection"
    - char_count < thin_char_thresh
    - element_count <= thin_elem_thresh
    - body_text is short OR starts with lowercase (continuation fragment)

    Args:
        df: Sections DataFrame with merge sort columns.
        config: MergeConfig with threshold values.

    Returns:
        Boolean Series where True indicates a thin subsection.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    # Compute body text length
    body_text = df["body_text"].fillna("").astype(str).str.strip()
    body_len = body_text.str.len()

    # Check if body starts with lowercase (indicates continuation)
    starts_lower = body_text.str[:1].str.islower().fillna(False)

    # Build thin mask
    is_subsection = df["anchor_type"] == "subsection"
    is_short_total = df["char_count"] < config.thin_char_thresh
    is_few_elements = df["element_count"] <= config.thin_elem_thresh
    is_fragment_body = (body_len <= config.body_char_thresh) | starts_lower

    thin = is_subsection & is_short_total & is_few_elements & is_fragment_body

    return thin


def can_merge_into(
    source_row: pd.Series,
    target_row: pd.Series,
    body_char_thresh: int,
) -> Tuple[bool, str]:
    """
    Check if source row can be merged into target row.

    Merge is allowed if:
    - Same page AND same column
    - Source is above target in y-position (reading order)
    - Target is not a box_section (keep boxes clean)
    - Target is not a section (unless source has minimal body)
    - Target subsection doesn't already have a header

    Args:
        source_row: The thin subsection to potentially merge.
        target_row: The next anchor in reading order.
        body_char_thresh: Max body length for merging into sections.

    Returns:
        Tuple of (can_merge: bool, reason: str).
    """
    # Must be same page
    if source_row["_page"] != target_row["_page"]:
        return False, "different_page"

    # Must be same column
    if source_row["_col"] != target_row["_col"]:
        return False, "different_column"

    # Source must be above target (sanity check after sorting)
    if source_row["_y0"] > target_row["_y0"] + 2:
        return False, "source_below_target"

    # Don't merge into box_section
    target_type = target_row.get("anchor_type", "")
    if target_type in ("box", "box_section"):
        return False, "target_is_box"

    # Don't merge into section unless source has minimal body
    if target_type == "section":
        source_body = (source_row.get("body_text") or "").strip()
        if len(source_body) > body_char_thresh:
            return False, "source_has_substantial_body"

    # Don't merge into subsection that has a header
    if target_type == "subsection":
        target_header = (target_row.get("header_text") or "").strip()
        if target_header:
            return False, "target_has_header"

    return True, "ok"


def perform_single_merge(
    df: pd.DataFrame,
    source_idx: int,
    target_idx: int,
) -> pd.DataFrame:
    """
    Merge source row into target row, updating all fields.

    Updates:
    - header_text: prepend source label
    - body_text: prepend source body
    - full_text: recompute from header + body
    - element_ids: concatenate and dedupe
    - element_count: recompute
    - pages: union
    - bbox: union (min x0/y0, max x1/y1)
    - char_count: recompute

    Args:
        df: Sections DataFrame.
        source_idx: Index of thin subsection to merge.
        target_idx: Index of target anchor.

    Returns:
        Updated DataFrame (modified in place and returned).
    """
    # Extract text components
    source_label = (df.loc[source_idx, "label"] or "").strip()
    source_body = (df.loc[source_idx, "body_text"] or "").strip()
    target_header = (df.loc[target_idx, "header_text"] or "").strip()
    target_body = (df.loc[target_idx, "body_text"] or "").strip()

    # Merge header: prepend source label
    if target_header:
        merged_header = f"{source_label}\n{target_header}".strip()
    else:
        merged_header = source_label

    # Merge body: prepend source body
    if source_body and target_body:
        merged_body = f"{source_body}\n\n{target_body}"
    elif source_body:
        merged_body = source_body
    else:
        merged_body = target_body

    # Update text fields
    df.loc[target_idx, "header_text"] = merged_header
    df.loc[target_idx, "body_text"] = merged_body
    df.loc[target_idx, "full_text"] = (
        f"{merged_header}\n\n{merged_body}".strip() if merged_header else merged_body
    )

    # Merge element_ids
    source_eids = normalize_cell_list(df.loc[source_idx, "element_ids"])
    target_eids = normalize_cell_list(df.loc[target_idx, "element_ids"])
    merged_eids = list(dict.fromkeys(source_eids + target_eids))  # Dedupe preserving order
    df.at[target_idx, "element_ids"] = merged_eids
    df.loc[target_idx, "element_count"] = len(merged_eids)

    # Merge pages
    source_pages = normalize_cell_list(df.loc[source_idx, "pages"])
    target_pages = normalize_cell_list(df.loc[target_idx, "pages"])
    merged_pages = sorted(set(source_pages + target_pages))
    df.at[target_idx, "pages"] = merged_pages

    # Merge bbox (union)
    source_bbox = normalize_bbox(df.loc[source_idx, "bbox"])
    target_bbox = normalize_bbox(df.loc[target_idx, "bbox"])

    if source_bbox and target_bbox:
        merged_bbox = [
            min(source_bbox[0], target_bbox[0]),  # x0
            min(source_bbox[1], target_bbox[1]),  # y0
            max(source_bbox[2], target_bbox[2]),  # x1
            max(source_bbox[3], target_bbox[3]),  # y1
        ]
        df.at[target_idx, "bbox"] = merged_bbox

    # Recompute char_count
    df.loc[target_idx, "char_count"] = len(df.loc[target_idx, "full_text"] or "")

    return df


def normalize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize list columns and remove temporary columns before output.

    Args:
        df: Sections DataFrame with merge artifacts.

    Returns:
        Cleaned DataFrame ready for serialization.
    """
    df = df.copy()

    # Normalize list columns for parquet serialization
    if "element_ids" in df.columns:
        df["element_ids"] = df["element_ids"].apply(normalize_cell_list)

    if "bbox" in df.columns:
        df["bbox"] = df["bbox"].apply(
            lambda b: normalize_bbox(b) if b is not None else []
        )

    if "pages" in df.columns:
        df["pages"] = df["pages"].apply(normalize_cell_list)

    # Drop temporary columns
    temp_cols = ["_page", "_col", "_y0", "_doc_id"]
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors="ignore")

    return df


def merge_forward_thin_subsections(
    sections_df: pd.DataFrame,
    col_info_df: Optional[pd.DataFrame] = None,
    page_geom_df: Optional[pd.DataFrame] = None,
    config: Optional[MergeConfig] = None,
) -> MergeResult:
    """
    Merge thin subsection anchors into the next anchor on same page/column.

    This fixes over-splitting where a heading gets extracted but its real body
    starts in the next anchor. Preserves heading text for retrieval.

    The function iterates until no more merges are possible, up to max_iterations.

    Args:
        sections_df: Sections DataFrame to process.
        col_info_df: Column info DataFrame (from detect_columns_for_document).
        page_geom_df: Page geometry DataFrame with page_mid_x.
        config: MergeConfig with thresholds. Uses defaults if None.

    Returns:
        MergeResult with processed DataFrame and statistics.

    Example:
        >>> result = merge_forward_thin_subsections(sections_df, col_info_df)
        >>> print(f"Merged {result.merges_performed} sections")
        >>> processed_df = result.sections_df
    """
    if config is None:
        config = MergeConfig()

    # Handle empty input
    if sections_df.empty:
        return MergeResult(sections_df=sections_df)

    # Build column split map
    split_map = build_column_split_map(
        col_info_df, page_geom_df, config.default_split_x
    )

    # Get default doc_id
    default_doc_id = "unknown"
    if col_info_df is not None and not col_info_df.empty and "doc_id" in col_info_df.columns:
        default_doc_id = col_info_df["doc_id"].iloc[0]

    # Track statistics
    all_merged_pairs: List[Tuple[str, str]] = []
    all_dropped: Set[str] = set()
    iteration = 0

    df = sections_df.copy()

    # Iterate until stable (no merges) or max iterations
    while iteration < config.max_iterations:
        iteration += 1

        # Add sort columns and sort
        df = add_merge_sort_columns(df, split_map, default_doc_id, config.default_split_x)

        # Identify thin subsections
        thin_mask = identify_thin_subsections(df, config)

        if not thin_mask.any():
            logger.debug(f"Iteration {iteration}: No thin subsections found")
            break

        # Track merges this iteration
        to_drop: Set[int] = set()
        merged_this_iter: List[Tuple[str, str]] = []

        # Process each potential merge
        for i in range(len(df) - 1):
            # Skip if already marked for drop
            if i in to_drop or (i + 1) in to_drop:
                continue

            # Skip if not thin
            if not thin_mask.iloc[i]:
                continue

            # Check if can merge
            can_merge, reason = can_merge_into(
                df.iloc[i], df.iloc[i + 1], config.body_char_thresh
            )

            if not can_merge:
                logger.debug(
                    f"Cannot merge {df.iloc[i]['anchor_id']} -> "
                    f"{df.iloc[i + 1]['anchor_id']}: {reason}"
                )
                continue

            # Perform merge
            source_id = df.iloc[i]["anchor_id"]
            target_id = df.iloc[i + 1]["anchor_id"]

            logger.debug(f"Merging {source_id} -> {target_id}")

            df = perform_single_merge(df, i, i + 1)
            to_drop.add(i)
            merged_this_iter.append((source_id, target_id))
            all_dropped.add(source_id)

        # If no merges this iteration, we're stable
        if not merged_this_iter:
            logger.debug(f"Iteration {iteration}: No merges performed, stable")
            break

        # Drop merged rows
        df = df.drop(index=list(to_drop)).reset_index(drop=True)
        all_merged_pairs.extend(merged_this_iter)

        logger.info(
            f"Iteration {iteration}: Merged {len(merged_this_iter)} subsections"
        )

    # Log if we hit max iterations
    if iteration >= config.max_iterations:
        logger.warning(
            f"Merge-forward hit max iterations ({config.max_iterations}). "
            f"Consider increasing max_iterations if sections remain over-split."
        )

    # Normalize output
    df = normalize_output_columns(df)

    return MergeResult(
        sections_df=df,
        merges_performed=len(all_merged_pairs),
        iterations=iteration,
        merged_pairs=all_merged_pairs,
        anchors_dropped=all_dropped,
    )


# Convenience function matching original signature
def merge_thin_subsections(
    sections_df: pd.DataFrame,
    col_info: Optional[pd.DataFrame] = None,
    page_geom: Optional[pd.DataFrame] = None,
    thin_char_thresh: int = 160,
    thin_elem_thresh: int = 2,
    body_char_thresh: int = 120,
) -> pd.DataFrame:
    """
    Legacy-compatible wrapper for merge_forward_thin_subsections.

    Returns just the DataFrame (not MergeResult) for drop-in compatibility.
    For full statistics, use merge_forward_thin_subsections() directly.

    Args:
        sections_df: Sections DataFrame.
        col_info: Column info DataFrame.
        page_geom: Page geometry DataFrame.
        thin_char_thresh: Max char count for thin classification.
        thin_elem_thresh: Max element count for thin classification.
        body_char_thresh: Max body length for thin classification.

    Returns:
        Processed sections DataFrame.
    """
    config = MergeConfig(
        thin_char_thresh=thin_char_thresh,
        thin_elem_thresh=thin_elem_thresh,
        body_char_thresh=body_char_thresh,
    )

    result = merge_forward_thin_subsections(
        sections_df=sections_df,
        col_info_df=col_info,
        page_geom_df=page_geom,
        config=config,
    )

    # Print merge info (matching original behavior)
    if result.merged_pairs:
        print(f"\n--- Merge-forward: {result.merges_performed} thin subsections merged ---")
        for src, tgt in result.merged_pairs:
            print(f"  {src} -> {tgt}")

    return result.sections_df
