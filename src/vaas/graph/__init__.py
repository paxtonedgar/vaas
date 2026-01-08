"""Graph construction modules for knowledge graph."""

from .edges import (
    # Data structures
    Edge,
    EdgeBuildResult,
    # Edge ID generation
    generate_edge_id,
    # Structural edge builders
    build_section_hierarchy_edges,
    build_section_follows_edges,
    build_anchor_paragraph_edges,
    build_paragraph_follows_edges,
    build_in_section_edges,
    # Reference edge builders
    build_box_reference_edges,
    build_same_group_edges,
    # Typed edge builders
    build_typed_edges,
    # Orchestrator
    build_all_edges,
    # Legacy compatibility
    build_edges_legacy,
)

from .nodes import (
    # Data structures
    Node,
    NodeBuildResult,
    # Constants
    ROLE_BOX_HEADER,
    ROLE_SECTION_HEADER,
    ROLE_SUBSECTION_HEADER,
    ROLE_LIST_BLOCK,
    ROLE_PAGE_ARTIFACT,
    ROLE_BODY_TEXT,
    DEFAULT_SKIP_ROLES,
    HEADER_ROLES,
    ANCHOR_TYPE_TO_NODE_TYPE,
    # Node ID generation
    generate_node_id,
    generate_paragraph_node_id,
    # Node type determination
    get_node_type_for_section,
    # Node builders
    build_doc_root_node,
    build_section_nodes,
    build_paragraph_nodes,
    # Orchestrator
    build_all_nodes,
    # Legacy compatibility
    build_nodes_legacy,
)

__all__ = [
    # Edge data structures
    "Edge",
    "EdgeBuildResult",
    # Edge ID generation
    "generate_edge_id",
    # Structural edge builders
    "build_section_hierarchy_edges",
    "build_section_follows_edges",
    "build_anchor_paragraph_edges",
    "build_paragraph_follows_edges",
    "build_in_section_edges",
    # Reference edge builders
    "build_box_reference_edges",
    "build_same_group_edges",
    # Typed edge builders
    "build_typed_edges",
    # Edge orchestrator
    "build_all_edges",
    # Edge legacy compatibility
    "build_edges_legacy",
    # Node data structures
    "Node",
    "NodeBuildResult",
    # Node constants
    "ROLE_BOX_HEADER",
    "ROLE_SECTION_HEADER",
    "ROLE_SUBSECTION_HEADER",
    "ROLE_LIST_BLOCK",
    "ROLE_PAGE_ARTIFACT",
    "ROLE_BODY_TEXT",
    "DEFAULT_SKIP_ROLES",
    "HEADER_ROLES",
    "ANCHOR_TYPE_TO_NODE_TYPE",
    # Node ID generation
    "generate_node_id",
    "generate_paragraph_node_id",
    # Node type determination
    "get_node_type_for_section",
    # Node builders
    "build_doc_root_node",
    "build_section_nodes",
    "build_paragraph_nodes",
    # Node orchestrator
    "build_all_nodes",
    # Node legacy compatibility
    "build_nodes_legacy",
]
