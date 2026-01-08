"""
Shared constants across vaas modules.

This module is the single source of truth for:
- Role constants (element classification)
- Anchor type constants
- Derived sets (header roles, skip roles)
"""

# =============================================================================
# ROLE CONSTANTS
# =============================================================================
# Element role classifications from extraction pipeline

ROLE_BOX_HEADER = "BoxHeader"
ROLE_SECTION_HEADER = "SectionHeader"
ROLE_SUBSECTION_HEADER = "SubsectionHeader"
ROLE_LIST_BLOCK = "ListBlock"
ROLE_PAGE_ARTIFACT = "PageArtifact"
ROLE_BODY_TEXT = "BodyTextBlock"


# =============================================================================
# ANCHOR TYPE CONSTANTS
# =============================================================================
# Anchor classifications for document structure

ANCHOR_TYPE_BOX = "box"
ANCHOR_TYPE_SECTION = "section"
ANCHOR_TYPE_SUBSECTION = "subsection"
ANCHOR_TYPE_PREAMBLE = "preamble"


# =============================================================================
# DERIVED SETS
# =============================================================================

# Header roles - elements represented as anchor nodes, not paragraph nodes
HEADER_ROLES = {ROLE_BOX_HEADER, ROLE_SECTION_HEADER, ROLE_SUBSECTION_HEADER}

# Roles to skip when creating paragraph nodes
DEFAULT_SKIP_ROLES = {ROLE_PAGE_ARTIFACT}


# =============================================================================
# NODE TYPE MAPPING
# =============================================================================

# Anchor type to graph node type mapping
ANCHOR_TYPE_TO_NODE_TYPE = {
    ANCHOR_TYPE_BOX: "box_section",
    ANCHOR_TYPE_SECTION: "section",
    ANCHOR_TYPE_SUBSECTION: "concept",
    ANCHOR_TYPE_PREAMBLE: "preamble",
}
