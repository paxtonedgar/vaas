"""PDF extraction and structural analysis modules."""

from .pdf import (
    PDFExtractionResult,
    extract_spans_from_pdf,
    infer_body_font_size,
    get_font_size_distribution,
    extract_spans_legacy,
)

from .lines import (
    LineBuildResult,
    build_line_dataframe,
    add_block_geometry,
    add_page_geometry,
    add_emphasis_flags,
    add_list_item_flags,
    add_text_properties,
    add_header_pattern_flags,
    add_page_marker_flags,
    add_structural_filters,
    build_lines_with_features,
    build_lines_legacy,
    # Regex patterns
    BOX_STRONG_RX,
    BOX_WEAK_RX,
    SECTION_HDR_RX,
    PAGE_MARKER_RX,
    BULLET_RX,
    ENUM_RX,
)

from .elements import (
    ElementSplitResult,
    ElementClassificationResult,
    split_blocks_into_elements,
    split_elements_with_result,
    classify_elements,
    classify_elements_with_result,
    split_and_classify_legacy,
    # Constants
    ROLE_BOX_HEADER,
    ROLE_SECTION_HEADER,
    ROLE_SUBSECTION_HEADER,
    ROLE_LIST_BLOCK,
    ROLE_PAGE_ARTIFACT,
    ROLE_BODY_TEXT,
)

from .layout_detection import (
    detect_subsection_candidates,
    assign_split_triggers,
    detect_and_assign_structure,
)

from .geometry import (
    safe_bbox,
    is_bold,
    margin_tolerance,
    bbox_y0,
    bbox_x0,
    reading_order_sort_key,
    compute_line_geometry,
    is_centered,
)

from .columns import (
    ColumnInfo,
    detect_columns_for_page,
    detect_columns_for_document,
    assign_line_columns,
    mark_list_items,
    cleanup_list_item_columns,
)

from .merge import (
    MergeConfig,
    MergeResult,
    build_column_split_map,
    add_merge_sort_columns,
    identify_thin_subsections,
    can_merge_into,
    perform_single_merge,
    normalize_output_columns,
    merge_forward_thin_subsections,
    merge_thin_subsections,
)

from .anchors import (
    # Data structures
    BoxParseResult,
    AnchorRecord,
    AnchorExtractionResult,
    BoxValidationResult,
    AnchorTimeline,
    # Box parsing
    parse_box_keys,
    # Section/subsection ID generation
    get_section_id,
    get_subsection_id,
    DEFAULT_SECTION_ID_MAP,
    # Anchor extraction
    extract_box_anchors,
    extract_section_anchors,
    extract_subsection_anchors,
    extract_anchors,
    deduplicate_anchors,
    # Validation
    validate_box_coverage,
    EXPECTED_BOXES_1099DIV,
    EXPECTED_BOXES_1099INT,
    # Timeline
    build_anchor_timeline,
    assign_elements_to_anchors,
    # Convenience
    extract_and_assign_anchors,
)

from .references import (
    # Data structures
    Reference,
    ReferenceExtractionResult,
    # Regex patterns
    BOX_REF_RX,
    PUB_REF_RX,
    IRC_REF_RX,
    FORM_REF_RX,
    # Parsing helpers
    parse_box_ref_keys,
    extract_evidence_quote,
    # Main extraction
    extract_references_from_element,
    extract_references,
    # Filtering
    filter_internal_box_references,
    filter_external_references,
    get_reference_summary,
    # Legacy
    extract_references_legacy,
)

from .sections import (
    # Data structures
    Section,
    SectionMaterializationResult,
    # Helper functions
    compute_bbox_union,
    split_header_body,
    get_section_sort_key,
    build_anchor_to_elements_map,
    # Main materialization
    materialize_section,
    materialize_sections,
    materialize_sections_legacy,
    # Utilities
    filter_sections_by_type,
    get_section_by_anchor_id,
    get_sections_summary,
)

__all__ = [
    # PDF extraction
    "PDFExtractionResult",
    "extract_spans_from_pdf",
    "infer_body_font_size",
    "get_font_size_distribution",
    "extract_spans_legacy",
    # Line building
    "LineBuildResult",
    "build_line_dataframe",
    "add_block_geometry",
    "add_page_geometry",
    "add_emphasis_flags",
    "add_list_item_flags",
    "add_text_properties",
    "add_header_pattern_flags",
    "add_page_marker_flags",
    "add_structural_filters",
    "build_lines_with_features",
    "build_lines_legacy",
    "BOX_STRONG_RX",
    "BOX_WEAK_RX",
    "SECTION_HDR_RX",
    "PAGE_MARKER_RX",
    "BULLET_RX",
    "ENUM_RX",
    # Element splitting and classification
    "ElementSplitResult",
    "ElementClassificationResult",
    "split_blocks_into_elements",
    "split_elements_with_result",
    "classify_elements",
    "classify_elements_with_result",
    "split_and_classify_legacy",
    "ROLE_BOX_HEADER",
    "ROLE_SECTION_HEADER",
    "ROLE_SUBSECTION_HEADER",
    "ROLE_LIST_BLOCK",
    "ROLE_PAGE_ARTIFACT",
    "ROLE_BODY_TEXT",
    # Layout detection
    "detect_subsection_candidates",
    "assign_split_triggers",
    "detect_and_assign_structure",
    # Geometry utilities
    "safe_bbox",
    "is_bold",
    "margin_tolerance",
    "bbox_y0",
    "bbox_x0",
    "reading_order_sort_key",
    "compute_line_geometry",
    "is_centered",
    # Column detection
    "ColumnInfo",
    "detect_columns_for_page",
    "detect_columns_for_document",
    "assign_line_columns",
    "mark_list_items",
    "cleanup_list_item_columns",
    # Merge-forward
    "MergeConfig",
    "MergeResult",
    "build_column_split_map",
    "add_merge_sort_columns",
    "identify_thin_subsections",
    "can_merge_into",
    "perform_single_merge",
    "normalize_output_columns",
    "merge_forward_thin_subsections",
    "merge_thin_subsections",
    # Anchor detection
    "BoxParseResult",
    "AnchorRecord",
    "AnchorExtractionResult",
    "BoxValidationResult",
    "AnchorTimeline",
    "parse_box_keys",
    "get_section_id",
    "get_subsection_id",
    "DEFAULT_SECTION_ID_MAP",
    "extract_box_anchors",
    "extract_section_anchors",
    "extract_subsection_anchors",
    "extract_anchors",
    "deduplicate_anchors",
    "validate_box_coverage",
    "EXPECTED_BOXES_1099DIV",
    "EXPECTED_BOXES_1099INT",
    "build_anchor_timeline",
    "assign_elements_to_anchors",
    "extract_and_assign_anchors",
    # Reference extraction
    "Reference",
    "ReferenceExtractionResult",
    "BOX_REF_RX",
    "PUB_REF_RX",
    "IRC_REF_RX",
    "FORM_REF_RX",
    "parse_box_ref_keys",
    "extract_evidence_quote",
    "extract_references_from_element",
    "extract_references",
    "filter_internal_box_references",
    "filter_external_references",
    "get_reference_summary",
    "extract_references_legacy",
    # Section materialization
    "Section",
    "SectionMaterializationResult",
    "compute_bbox_union",
    "split_header_body",
    "get_section_sort_key",
    "build_anchor_to_elements_map",
    "materialize_section",
    "materialize_sections",
    "materialize_sections_legacy",
    "filter_sections_by_type",
    "get_section_by_anchor_id",
    "get_sections_summary",
]
