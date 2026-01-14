"""
Reference extraction from PDF document elements.

This module extracts cross-references from document text, including:
- Box references: "See box 1a", "Boxes 14 through 16"
- Publication references: "Pub. 550", "Publication 17"
- IRC section references: "section 301(c)(1)", "§ 1202"
- Form references: "Form 1099-DIV", "Form W-2"

References are extracted per-element (not per-section) to provide clean
evidence quotes and accurate element-level provenance.

Key Design Decisions:
1. Regex patterns stop at newlines to avoid cross-element contamination
2. Box keys are normalized to lowercase for consistent matching
3. Evidence quotes include context but collapse whitespace
4. Each reference gets a stable ID based on element_id + position
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Match, Optional, Set, Tuple

import pandas as pd

from vaas.utils.text import stable_hash


logger = logging.getLogger(__name__)


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Box reference: "box 1a", "boxes 1a and 1b", "boxes 14-16", "boxes 14 through 16"
# Uses [^\n] to stop at newlines and avoid cross-element capture
BOX_REF_RX = re.compile(
    r"[Bb]ox(?:es)?\s+(\d+[a-z]?(?:[^\n]*?(?:,|and|through|[-–]|to)\s*\d+[a-z]?)*)",
    re.IGNORECASE
)

# Publication reference: "Pub. 550", "Publication 17", "Pub 550"
PUB_REF_RX = re.compile(r"[Pp]ub(?:lication)?\.?\s*(\d+)")

# IRC section reference: "section 301(c)(1)", "section 1202", "§ 301"
# Handles multiple levels of parentheses like 301(c)(1)(A)
IRC_REF_RX = re.compile(r"(?:[Ss]ection|§)\s*(\d+[A-Za-z]?(?:\([a-z0-9]+\))*)")

# Form reference: "Form 1099-DIV", "Form W-2", "Form 8949"
FORM_REF_RX = re.compile(r"[Ff]orm\s+(\d+[A-Z\-]*|[A-Z]-?\d+)")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Reference:
    """
    A single reference extracted from document text.

    Attributes:
        ref_id: Unique identifier for this reference.
        doc_id: Document identifier for provenance.
        source_element_id: Element where reference was found.
        source_anchor_id: Anchor containing the source element.
        ref_type: Type of reference (box_reference, publication, irc_section, form_reference).
        ref_text: The matched text (e.g., "box 1a", "Pub. 550").
        target_keys: Extracted keys (e.g., ["1a"], ["550"], ["301(c)(1)"]).
        target_anchor_id: For box refs, the target anchor ID (e.g., "box_1a").
        target_exists: For box refs, whether target anchor was found.
        evidence_text: Contextual quote around the reference.
        confidence: Confidence score (0.0-1.0).
        page: Page number where reference appears.
        position: Character position in element text.
        created_by: How reference was created ("regex", "llm", etc.).
    """

    ref_id: str
    doc_id: Optional[str]
    source_element_id: str
    source_anchor_id: Optional[str]
    ref_type: str
    ref_text: str
    target_keys: List[str]
    target_anchor_id: Optional[str] = None
    target_exists: Optional[bool] = None
    evidence_text: str = ""
    confidence: float = 0.9
    page: int = 0
    position: int = 0
    created_by: str = "regex"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "ref_id": self.ref_id,
            "doc_id": self.doc_id,
            "source_element_id": self.source_element_id,
            "source_anchor_id": self.source_anchor_id,
            "ref_type": self.ref_type,
            "ref_text": self.ref_text,
            "target_keys": self.target_keys,
            "target_anchor_id": self.target_anchor_id,
            "target_exists": self.target_exists,
            "evidence_text": self.evidence_text,
            "confidence": self.confidence,
            "page": self.page,
            "position": self.position,
            "created_by": self.created_by,
        }


@dataclass
class ReferenceExtractionResult:
    """
    Result of reference extraction from document elements.

    Attributes:
        references_df: DataFrame with all extracted references.
        box_refs: Count of box references.
        pub_refs: Count of publication references.
        irc_refs: Count of IRC section references.
        form_refs: Count of form references.
        elements_processed: Number of elements processed.
        elements_with_refs: Number of elements containing references.
    """

    references_df: pd.DataFrame
    box_refs: int = 0
    pub_refs: int = 0
    irc_refs: int = 0
    form_refs: int = 0
    elements_processed: int = 0
    elements_with_refs: int = 0

    @property
    def total(self) -> int:
        """Total number of references extracted."""
        return self.box_refs + self.pub_refs + self.irc_refs + self.form_refs

    def __repr__(self) -> str:
        return (
            f"ReferenceExtractionResult(total={self.total}, "
            f"box={self.box_refs}, pub={self.pub_refs}, "
            f"irc={self.irc_refs}, form={self.form_refs})"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_box_ref_keys(ref_text: str) -> List[str]:
    """
    Parse box reference text into individual box keys.

    Handles various formats:
    - Single: "1a" → ["1a"]
    - List: "1a and 1b" → ["1a", "1b"]
    - Range: "14-16" or "14 through 16" → ["14", "15", "16"]
    - Mixed: "1a, 1b, and 2a" → ["1a", "1b", "2a"]

    Args:
        ref_text: The captured group from BOX_REF_RX (e.g., "1a and 1b").

    Returns:
        List of box keys, normalized to lowercase.

    Examples:
        >>> parse_box_ref_keys("1a")
        ['1a']
        >>> parse_box_ref_keys("14-16")
        ['14', '15', '16']
        >>> parse_box_ref_keys("1a and 1b")
        ['1a', '1b']
    """
    keys: List[str] = []

    # Check for range pattern first (14-16, 14 through 16, 14 to 16)
    range_match = re.search(r"(\d+)\s*(?:[-–]|through|to)\s*(\d+)", ref_text, re.IGNORECASE)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        keys.extend([str(k) for k in range(min(lo, hi), max(lo, hi) + 1)])

    # Find all individual box keys (handles "1a", "1b", etc.)
    individual = re.findall(r"\b(\d+[a-z]?)\b", ref_text, re.IGNORECASE)
    for k in individual:
        k_lower = k.lower()
        if k_lower not in keys:
            keys.append(k_lower)

    return keys


def extract_evidence_quote(
    text: str,
    match: Match[str],
    context_chars: int = 50,
) -> str:
    """
    Extract clean evidence quote with context around match.

    Creates a quote that includes text before and after the match,
    cleans up whitespace, and adds ellipsis markers when truncated.

    Args:
        text: Full text being searched.
        match: Regex match object.
        context_chars: Number of characters of context on each side.

    Returns:
        Clean evidence quote with ellipsis markers.

    Example:
        >>> text = "For more information, see Pub. 550 for details on investment income."
        >>> match = PUB_REF_RX.search(text)
        >>> extract_evidence_quote(text, match, 20)
        '...information, see Pub. 550 for details on...'
    """
    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)

    quote = text[start:end].strip()

    # Clean up whitespace (collapse multiple spaces/newlines)
    quote = " ".join(quote.split())

    # Add ellipsis markers
    if start > 0:
        quote = "..." + quote
    if end < len(text):
        quote = quote + "..."

    return quote


def generate_ref_id(
    element_id: str,
    ref_type: str,
    position: int,
) -> str:
    """
    Generate stable reference ID.

    Args:
        element_id: Source element ID.
        ref_type: Type of reference.
        position: Character position in text.

    Returns:
        Stable reference ID.
    """
    parts = [element_id, ref_type, str(position)]
    hash_suffix = stable_hash(parts, length=8)
    return f"ref_{ref_type[:3]}_{hash_suffix}"


def generate_ref_occurrence_id(
    doc_id: Optional[str],
    source_element_id: Optional[str],
    sentence_idx: Optional[int],
    char_start: Optional[int],
    char_end: Optional[int],
    ref_text: Optional[str],
    ref_type: Optional[str],
    target_anchor_id: Optional[str] = None,
) -> str:
    """Create stable surrogate key for an individual reference occurrence."""
    parts = [
        doc_id or "",
        source_element_id or "",
        "" if sentence_idx is None else str(sentence_idx),
        "" if char_start is None else str(char_start),
        "" if char_end is None else str(char_end),
        ref_text or "",
        ref_type or "",
        target_anchor_id or "",
    ]
    return f"refocc_{stable_hash(parts, length=12)}"


# =============================================================================
# REFERENCE EXTRACTION
# =============================================================================

def extract_references_from_element(
    doc_id: Optional[str],
    element_id: str,
    text: str,
    anchor_id: Optional[str],
    page: int,
    valid_box_keys: Optional[Set[str]] = None,
    context_chars: int = 50,
) -> List[Reference]:
    """
    Extract all references from a single element.

    Searches text for box references, publication references,
    IRC section references, and form references.

    Args:
        element_id: ID of the source element.
        text: Element text to search.
        anchor_id: Anchor containing this element.
        page: Page number.
        valid_box_keys: Set of known box keys for target_exists check.
        context_chars: Characters of context for evidence quotes.

    Returns:
        List of Reference objects found in the text.

    Example:
        >>> refs = extract_references_from_element(
        ...     "elem_1",
        ...     "See box 1a and Pub. 550 for details.",
        ...     "box_2a",
        ...     page=1,
        ...     valid_box_keys={"1a", "1b", "2a"}
        ... )
        >>> len(refs)
        2
    """
    if not text or not text.strip():
        return []

    valid_box_keys = valid_box_keys or set()
    references: List[Reference] = []

    # Normalize text for matching (collapse whitespace)
    clean_text = " ".join(text.split())

    # Extract box references
    for match in BOX_REF_RX.finditer(clean_text):
        ref_text = match.group(0).strip()
        target_keys = parse_box_ref_keys(match.group(1))
        evidence = extract_evidence_quote(clean_text, match, context_chars)
        position = match.start()

        # Create one reference record per target key
        for key in target_keys:
            target_anchor = f"box_{key}"

            # Skip self-references
            if target_anchor == anchor_id:
                continue

            target_exists = key in valid_box_keys

            references.append(
                Reference(
                    ref_id=generate_ref_id(element_id, "box", position),
                    doc_id=doc_id,
                    source_element_id=element_id,
                    source_anchor_id=anchor_id,
                    ref_type="box_reference",
                    ref_text=ref_text,
                    target_keys=[key],
                    target_anchor_id=target_anchor,
                    target_exists=target_exists,
                    evidence_text=evidence,
                    confidence=0.95 if target_exists else 0.70,
                    page=page,
                    position=position,
                    created_by="regex",
                )
            )

    # Extract publication references
    for match in PUB_REF_RX.finditer(clean_text):
        pub_num = match.group(1)
        evidence = extract_evidence_quote(clean_text, match, context_chars)
        position = match.start()

        references.append(
            Reference(
                ref_id=generate_ref_id(element_id, "pub", position),
                doc_id=doc_id,
                source_element_id=element_id,
                source_anchor_id=anchor_id,
                ref_type="publication",
                ref_text=match.group(0),
                target_keys=[pub_num],
                target_anchor_id=f"pub_{pub_num}",
                target_exists=None,  # External reference
                evidence_text=evidence,
                confidence=0.90,
                page=page,
                position=position,
                created_by="regex",
            )
        )

    # Extract IRC section references
    for match in IRC_REF_RX.finditer(clean_text):
        section = match.group(1)
        evidence = extract_evidence_quote(clean_text, match, context_chars)
        position = match.start()

        references.append(
            Reference(
                ref_id=generate_ref_id(element_id, "irc", position),
                doc_id=doc_id,
                source_element_id=element_id,
                source_anchor_id=anchor_id,
                ref_type="irc_section",
                ref_text=match.group(0),
                target_keys=[section],
                target_anchor_id=f"irc_{section}",
                target_exists=None,  # External reference
                evidence_text=evidence,
                confidence=0.90,
                page=page,
                position=position,
                created_by="regex",
            )
        )

    # Extract form references
    for match in FORM_REF_RX.finditer(clean_text):
        form = match.group(1)
        evidence = extract_evidence_quote(clean_text, match, context_chars)
        position = match.start()

        references.append(
            Reference(
                ref_id=generate_ref_id(element_id, "form", position),
                doc_id=doc_id,
                source_element_id=element_id,
                source_anchor_id=anchor_id,
                ref_type="form_reference",
                ref_text=match.group(0),
                target_keys=[form],
                target_anchor_id=f"form_{form}",
                target_exists=None,  # External reference
                evidence_text=evidence,
                confidence=0.90,
                page=page,
                position=position,
                created_by="regex",
            )
        )

    return references


def extract_references(
    elements_df: pd.DataFrame,
    doc_id: str,
    valid_box_keys: Optional[Set[str]] = None,
    context_chars: int = 50,
    skip_roles: Optional[Set[str]] = None,
    skip_anchor_ids: Optional[Set[str]] = None,
    element_id_col: str = "element_id",
    text_col: str = "text",
    anchor_id_col: str = "anchor_id",
    role_col: str = "role",
    page_col: str = "page",
    deduplicate: bool = False,
) -> ReferenceExtractionResult:
    """
    Extract references from all elements in a DataFrame.

    Processes each element, extracts references, and returns
    a consolidated DataFrame with deduplication.

    Args:
        elements_df: DataFrame with element data.
        doc_id: Document identifier for stable IDs.
        valid_box_keys: Set of known box keys for target_exists check.
        context_chars: Characters of context for evidence quotes.
        skip_roles: Roles to skip (e.g., {"PageArtifact"}).
        skip_anchor_ids: Anchor IDs to skip (e.g., {"unassigned"}).
        element_id_col: Column name for element ID.
        text_col: Column name for text.
        anchor_id_col: Column name for anchor ID.
        role_col: Column name for role.
        page_col: Column name for page.
        deduplicate: Whether to deduplicate by (source, target, type).

    Returns:
        ReferenceExtractionResult with references DataFrame and statistics.

    Example:
        >>> result = extract_references(elements_df, valid_box_keys={"1a", "1b"})
        >>> print(f"Found {result.total} references")
        >>> box_refs = result.references_df[result.references_df['ref_type'] == 'box_reference']
    """
    skip_roles = skip_roles or {"PageArtifact", "page_artifact"}
    skip_anchor_ids = skip_anchor_ids or {"unassigned", ""}
    valid_box_keys = valid_box_keys or set()

    all_references: List[Reference] = []
    elements_processed = 0
    elements_with_refs = 0

    for _, row in elements_df.iterrows():
        # Skip based on role
        role = row.get(role_col)
        if role in skip_roles:
            continue

        # Skip based on anchor_id
        anchor_id = row.get(anchor_id_col)
        if anchor_id in skip_anchor_ids:
            continue

        element_id = row[element_id_col]
        text = row.get(text_col, "")
        page = int(row.get(page_col, 0))

        if not text or not str(text).strip():
            continue

        elements_processed += 1

        # Extract references from this element
        refs = extract_references_from_element(
            doc_id=doc_id,
            element_id=element_id,
            text=str(text),
            anchor_id=anchor_id,
            page=page,
            valid_box_keys=valid_box_keys,
            context_chars=context_chars,
        )

        if refs:
            elements_with_refs += 1
            all_references.extend(refs)

    # Convert to DataFrame
    if all_references:
        references_df = pd.DataFrame([r.to_dict() for r in all_references])
    else:
        references_df = pd.DataFrame()

    # Deduplicate by (source_anchor_id, target_anchor_id, ref_type)
    if deduplicate and not references_df.empty:
        before = len(references_df)
        references_df = references_df.drop_duplicates(
            subset=["source_anchor_id", "target_anchor_id", "ref_type"],
            keep="first",
        )
        after = len(references_df)
        if before != after:
            logger.debug(f"Deduplicated references: {before} -> {after}")

    # Count by type
    box_refs = 0
    pub_refs = 0
    irc_refs = 0
    form_refs = 0

    if not references_df.empty:
        type_counts = references_df["ref_type"].value_counts()
        box_refs = int(type_counts.get("box_reference", 0))
        pub_refs = int(type_counts.get("publication", 0))
        irc_refs = int(type_counts.get("irc_section", 0))
        form_refs = int(type_counts.get("form_reference", 0))

    logger.info(
        f"Extracted {len(references_df)} references from {elements_processed} elements "
        f"({elements_with_refs} with refs)"
    )

    return ReferenceExtractionResult(
        references_df=references_df,
        box_refs=box_refs,
        pub_refs=pub_refs,
        irc_refs=irc_refs,
        form_refs=form_refs,
        elements_processed=elements_processed,
        elements_with_refs=elements_with_refs,
    )


# =============================================================================
# REFERENCE FILTERING & ANALYSIS
# =============================================================================

def filter_internal_box_references(
    references_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter to only internal box references (where target exists).

    Args:
        references_df: DataFrame with all references.

    Returns:
        DataFrame with only box references where target_exists=True.
    """
    if references_df.empty:
        return references_df

    mask = (
        (references_df["ref_type"] == "box_reference") &
        (references_df["target_exists"] == True)
    )
    return references_df[mask].copy()


def filter_external_references(
    references_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter to only external references (publications, IRC, forms).

    Args:
        references_df: DataFrame with all references.

    Returns:
        DataFrame with only external references.
    """
    if references_df.empty:
        return references_df

    external_types = {"publication", "irc_section", "form_reference"}
    return references_df[references_df["ref_type"].isin(external_types)].copy()


def get_reference_summary(references_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for references.

    Args:
        references_df: DataFrame with references.

    Returns:
        Dictionary with summary statistics.
    """
    if references_df.empty:
        return {
            "total": 0,
            "by_type": {},
            "internal_box_refs": 0,
            "external_refs": 0,
            "unique_targets": 0,
        }

    type_counts = references_df["ref_type"].value_counts().to_dict()

    internal = filter_internal_box_references(references_df)
    external = filter_external_references(references_df)

    unique_targets = references_df["target_anchor_id"].nunique()

    return {
        "total": len(references_df),
        "by_type": type_counts,
        "internal_box_refs": len(internal),
        "external_refs": len(external),
        "unique_targets": unique_targets,
    }


