"""
Typed edge extraction for knowledge graph.

Extracts semantic edge types from text using template-first matching.
Edges point from rule to thing being ruled on.

Edge types (Phase 1b + 2a):
- excludes: Negation/exception relationship (concept → box)
- applies_if: Conditional applicability (concept → box)
- defines: Semantic meaning (concept → box)
- qualifies: Scope/constraint (concept → box)
- requires: Computational dependency (box → box)
- portion_of: Subset relationship (concept → box)
  e.g., "If any part of box 1a is qualified dividends..."
  The concept describes a portion/subset of the box's contents.
"""

import re
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class TypedEdgeCandidate:
    """A candidate typed edge extracted from text."""
    edge_type: str
    source_anchor_id: str
    target_box_key: str
    confidence: float
    evidence_text: str
    pattern_matched: str
    polarity: str  # "positive" or "negative"
    # Sentence-level provenance (for sentence-gated extraction)
    sentence_idx: Optional[int] = None
    sentence_char_start: Optional[int] = None
    sentence_char_end: Optional[int] = None


# =============================================================================
# SENTENCE SPLITTING (IRS-tuned, precision-first, offset-safe)
# =============================================================================

# Common abbreviations that shouldn't trigger sentence breaks
ABBREV = {
    "u.s.", "no.", "sec.", "pub.", "rev.", "i.e.", "e.g.", "etc.", "fig.", "vol.",
    "mr.", "mrs.", "ms.", "dr.", "jr.", "sr.", "inc.", "ltd.", "corp.", "cat."
}

# Bullet/numbered list boundary pattern
BULLET_BREAK = re.compile(r"(?:\n\s*(?:[-•]|(?:\(\d+\))|(?:\d+\.))\s+)")


def split_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences with character offsets.

    IRS-tuned, precision-first splitter that:
    - Handles common abbreviations (U.S., No., Sec., Pub., etc.)
    - Treats bullet/numbered lists as sentence boundaries
    - Under-splits rather than over-splits (safe for extraction)

    Args:
        text: The text to split.

    Returns:
        List of (sentence_text, char_start, char_end) tuples.
        Offsets are relative to input text.
    """
    if not text or not text.strip():
        return []

    sents: List[Tuple[str, int, int]] = []
    n = len(text)
    start = 0
    i = 0

    def emit(end: int):
        nonlocal start
        seg = text[start:end].strip()
        if seg:
            # Compute trimmed offsets
            left = text[start:end].find(seg)
            s = start + left
            e = s + len(seg)
            sents.append((seg, s, e))
        start = end

    while i < n:
        # Bullet/newline break => treat as boundary
        m = BULLET_BREAK.match(text, i)
        if m:
            # End current sentence at bullet start
            if i > start:
                emit(i)
            # Skip the bullet marker itself but keep it in next sentence
            i = m.end() - 1
            start = i + 1
            i += 1
            continue

        ch = text[i]
        if ch in ".!?":
            # Look back for abbreviation / decimal
            window_start = max(start, i - 12)
            prev = text[window_start:i + 1].strip().lower()

            # Decimal number like 1.5
            if i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
                i += 1
                continue

            # Abbreviation check (endswith any abbrev)
            if any(prev.endswith(a) for a in ABBREV):
                i += 1
                continue

            # Candidate boundary: require whitespace then new-ish sentence token
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j >= n:
                emit(n)
                break
            nxt = text[j]
            if nxt.isupper() or nxt.isdigit() or nxt in "([\"":
                emit(i + 1)
                i = j
                continue

        i += 1

    # Tail
    if start < n:
        emit(n)

    # If we somehow produced nothing, return whole text as one sentence
    if not sents:
        stripped = text.strip()
        if stripped:
            start_idx = text.find(stripped)
            return [(stripped, start_idx, start_idx + len(stripped))]
        return []

    return sents


# =============================================================================
# SUBSET CUE GATING (for portion_of precision)
# =============================================================================

# Subset cues - tight list for precision (required for portion_of)
SUBSET_CUES = [
    re.compile(r'(?i)\bany\s+part\s+of\b'),
    re.compile(r'(?i)\bpart\s+of\b'),
    re.compile(r'(?i)\bportion\s+of\b'),
    re.compile(r'(?i)\bsome\s+of\b'),
    re.compile(r'(?i)\bto\s+the\s+extent\b'),
    re.compile(r'(?i)\battributable\s+to\b'),
    re.compile(r'(?i)\bthe\s+qualified\s+portion\b'),
    re.compile(r'(?i)\bonly\s+the\s+portion\b'),
    re.compile(r'(?i)\bsubset\s+of\b'),
]

# Narration filters - phrases that indicate descriptive text, not rules
# Only applied to portion_of to suppress false positives
NARRATION_FILTERS = [
    re.compile(r'(?i)\bwas\s+determined\b'),
    re.compile(r'(?i)\bdetermined\s+by\b'),
    re.compile(r'(?i)\bwas\s+calculated\b'),
    re.compile(r'(?i)\bhas\s+been\s+allocated\b'),
    re.compile(r'(?i)\bwas\s+allocated\b'),
]


def has_subset_cue(sentence: str) -> bool:
    """Check if sentence contains a subset cue phrase."""
    return any(cue.search(sentence) for cue in SUBSET_CUES)


def has_narration_filter(sentence: str) -> bool:
    """Check if sentence contains narration phrases (FP indicator for portion_of)."""
    return any(filt.search(sentence) for filt in NARRATION_FILTERS)


# =============================================================================
# PATTERN TABLES (Phase 1b + 2a)
# =============================================================================
# Format: (pattern_name, compiled_regex, confidence)
# Patterns are evaluated in priority order within each type.

# --- EXCLUDES (Phase 1b) - Negation patterns ---
EXCLUDES_PATTERNS = [
    ("does_not_include", re.compile(r"(?i)does\s+not\s+include"), 0.95),
    ("do_not_include", re.compile(r"(?i)do\s+not\s+include"), 0.95),
    ("do_not_report", re.compile(r"(?i)do\s+not\s+report"), 0.95),
    ("not_reported", re.compile(r"(?i)(?:is|are)\s+not\s+reported"), 0.90),
    ("except", re.compile(r"(?i)\bexcept\s+(?:for\s+)?(?:the\s+)?(?:amounts?|dividends?|distributions?)?"), 0.85),
    ("excluding", re.compile(r"(?i)\bexcluding\b"), 0.90),
    ("other_than", re.compile(r"(?i)\bother\s+than\b"), 0.85),
    ("not_qualified", re.compile(r"(?i)(?:is|are)\s+not\s+qualified"), 0.90),
    ("not_eligible", re.compile(r"(?i)(?:is|are)\s+not\s+eligible"), 0.85),
]

# --- APPLIES_IF (Phase 2a) - Conditional applicability ---
# Direction: concept → box
# NOTE: "see instructions for box X" removed - it's a reference, not a conditional.
# That pattern should emit references_box at the structural layer, not applies_if.
APPLIES_IF_PATTERNS = [
    ("report_if", re.compile(r"(?i)report\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)\s+if\b"), 0.90),
    ("include_when", re.compile(r"(?i)include\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)\s+when\b"), 0.90),
    ("if_report", re.compile(r"(?i)if\s+.{1,60}?,\s*report\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)"), 0.85),
    ("only_if", re.compile(r"(?i)only\s+if\s+.{1,60}?,\s*include\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)"), 0.85),
    # Holding period conditions - "holding period requirement has been met"
    ("holding_period_box", re.compile(r"(?i)holding\s+period\s+requirement.{0,40}?box\s+(\d+[a-z]?)"), 0.90),
    # "qualifies for" patterns with box reference
    ("qualifies_for_box", re.compile(r"(?i)qualifies\s+for.{0,60}?box\s+(\d+[a-z]?)"), 0.85),
]

# --- DEFINES (Phase 2a) - Semantic meaning ---
# Direction: concept → box
# NOTE: Patterns tightened to avoid "term refers to" garbage captures.
DEFINES_PATTERNS = [
    # "Box X is/means..." - direct definition
    ("box_is", re.compile(r"(?i)box\s+(\d+[a-z]?)\s+(?:is|means)\b"), 0.95),
    # 'The term "X" refers to...' - formal definition frame only
    ("term_refers", re.compile(
        r'(?i)\bthe\s+term\s+"?([A-Za-z][A-Za-z\s\-]{2,40})"?\s+refers\s+to\b'
    ), 0.90),
    # "X are dividends that..." - sentence-start anchored
    ("noun_phrase_are", re.compile(
        r"(?i)^([A-Z][A-Za-z][A-Za-z\s\-]{2,40})\s+are\s+dividends\s+that\b"
    ), 0.85),
    # "X includes dividends that..." - sentence-start anchored
    ("includes_definition", re.compile(
        r"(?i)^([A-Z][A-Za-z][A-Za-z\s\-]{2,40})\s+includes?\s+dividends\s+that\b"
    ), 0.85),
]

# --- QUALIFIES (Phase 2a) - Scope/constraint ---
# Direction: concept → box
QUALIFIES_PATTERNS = [
    ("includes_only", re.compile(r"(?i)box\s+(\d+[a-z]?)\s+includes\s+only\b"), 0.95),
    ("applies_to_box", re.compile(r"(?i)for\s+box\s+(\d+[a-z]?),\s+this\s+applies\s+to\b"), 0.90),
    ("only_portion", re.compile(r"(?i)only\s+the\s+.+?\s+portion\s+.{0,40}?\s+reported\s+in\s+box\s+(\d+[a-z]?)"), 0.85),
    ("limited_to", re.compile(r"(?i)limited\s+to\s+amounts\s+reported\s+in\s+box\s+(\d+[a-z]?)"), 0.85),
]

# --- REQUIRES (Phase 2a) - Computational dependency ---
# Direction: box → box (source = current box anchor, target = captured boxes)
REQUIRES_PATTERNS = [
    ("include_from", re.compile(r"(?i)include\s+amounts?\s+from\s+box(?:es)?\s+((?:\d+[a-z]?(?:,?\s*(?:and\s+)?)?)+)"), 0.95),
    ("also_report", re.compile(r"(?i)also\s+report\s+.{0,40}?\s+in\s+box\s+(\d+[a-z]?)"), 0.85),
    ("see_box", re.compile(r"(?i)see\s+box\s+(\d+[a-z]?)"), 0.80),
    ("combine_with", re.compile(r"(?i)combine\s+with\s+amounts?\s+in\s+box\s+(\d+[a-z]?)"), 0.85),
]

# --- INCLUDES (Phase 2a) - Containment relationship ---
# Direction: box → box (e.g., "Box 1a includes amounts in boxes 1b and 2e")
INCLUDES_PATTERNS = [
    # "Box 1a includes amounts entered in boxes 1b and 2e"
    ("includes_amounts", re.compile(r"(?i)box\s+(\d+[a-z]?)\s+includes\s+(?:the\s+)?amounts?\s+(?:in|from|entered\s+in|reported\s+in)\s+box(?:es)?\s+((?:\d+[a-z]?(?:,?\s*(?:and\s+)?)?)+)"), 0.95),
    # "includes amounts in box 1b" (without explicit source box)
    ("includes_box", re.compile(r"(?i)includes?\s+(?:the\s+)?(?:amounts?\s+)?(?:in|from|entered\s+in)\s+box(?:es)?\s+((?:\d+[a-z]?(?:,?\s*(?:and\s+)?)?)+)"), 0.90),
    # "reported in both box 1a and 1b"
    ("reported_in_both", re.compile(r"(?i)reported\s+in\s+both\s+box(?:es)?\s+(\d+[a-z]?)\s+and\s+(\d+[a-z]?)"), 0.85),
    # "also includes" pattern for secondary inclusions
    ("also_includes", re.compile(r"(?i)also\s+includes?\s+(?:the\s+)?(?:amounts?\s+)?(?:in|from|entered\s+in)?\s*box(?:es)?\s+((?:\d+[a-z]?(?:,?\s*(?:and\s+)?)?)+)"), 0.85),
]

# --- PORTION_OF (Phase 2a) - Subset relationship ---
# Direction: concept → box (the concept describes a portion/subset of the box)
# This captures "If any part of box 1a is qualified dividends..."
# NOTE: Patterns precision-tightened for sentence gating.
# Dropped: attributable_to_box (overlaps box_attributable), subset_of_box (rare/loose)
PORTION_OF_PATTERNS = [
    # "any part of [the amount/total/dividends] [reported in] box X" - strongest cue
    ("any_part_of_box", re.compile(
        r"(?i)(?:if\s+)?any\s+part\s+of\s+(?:the\s+)?"
        r"(?:amount|total|dividends?|distribution|gain|income)?\s*"
        r"(?:reported\s+in\s+)?box\s+(\d+[a-z]?)"
    ), 0.95),
    # "some of [the amount/total/dividends] [reported in] box X"
    ("some_of_box", re.compile(
        r"(?i)some\s+of\s+(?:the\s+)?"
        r"(?:amount|total|dividends?|distribution|gain|income)?\s*"
        r"(?:reported\s+in\s+)?box\s+(\d+[a-z]?)"
    ), 0.90),
    # "portion of [the amount/distribution] reported in box X" - noun-anchored
    ("portion_of_reported", re.compile(
        r"(?i)(?:the\s+)?portion\s+of\s+(?:the\s+)?"
        r"(?:amount|total|distribution|dividends?|gain|income)\s+"
        r"reported\s+in\s+box\s+(\d+[a-z]?)"
    ), 0.90),
    # "[amount] in box X is attributable to [concept]" - implies subset attribution
    ("box_attributable", re.compile(
        r"(?i)(?:reported|entered|shown)\s+in\s+box\s+(\d+[a-z]?)\s+"
        r"(?:is|are)\s+attributable\s+to"
    ), 0.90),
]

# Box reference pattern for general use
BOX_REF_PATTERN = re.compile(
    r"[Bb]ox(?:es)?\s+(\d+[a-z]?(?:\s*(?:,|and|or)\s*\d+[a-z]?)*)",
    re.IGNORECASE
)


def _clean_evidence(text: str, ref_pos: int, match_len: int) -> str:
    """Extract and clean evidence snippet around a match."""
    start = max(0, ref_pos - 40)
    end = min(len(text), ref_pos + match_len + 40)
    evidence = text[start:end].strip()
    # Clean newlines (prevents validation warnings)
    evidence = " ".join(evidence.split())
    if start > 0:
        evidence = "..." + evidence
    if end < len(text):
        evidence = evidence + "..."
    return evidence


def _extract_box_keys(ref_text: str) -> List[str]:
    """Extract individual box keys from a reference string."""
    individual = re.findall(r"\b(\d+[a-z]?)\b", ref_text, re.IGNORECASE)
    return [k.lower() for k in individual]


def _has_negation_context(text: str, pos: int, window: int = 80) -> bool:
    """Check if position has negation context (excludes should win)."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    context = text[start:end]

    for _, pattern, _ in EXCLUDES_PATTERNS:
        if pattern.search(context):
            return True
    return False


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_excludes_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract excludes edges from text.
    Direction: concept → box
    """
    if not text:
        return []

    edges = []
    seen = set()  # (box_key) - one edge per box per anchor

    for match in BOX_REF_PATTERN.finditer(text):
        ref_text = match.group(1)
        ref_pos = match.start()

        # Check for negation context
        start = max(0, ref_pos - 80)
        end = min(len(text), ref_pos + 80)
        context = text[start:end]

        negation_match = None
        for pattern_name, pattern, confidence in EXCLUDES_PATTERNS:
            m = pattern.search(context)
            if m:
                negation_match = (pattern_name, confidence)
                break

        if not negation_match:
            continue

        pattern_name, confidence = negation_match
        box_keys = _extract_box_keys(ref_text)

        for box_key in box_keys:
            if box_key not in valid_box_keys:
                continue
            if box_key in seen:
                continue
            seen.add(box_key)

            evidence = _clean_evidence(text, ref_pos, len(match.group(0)))

            edges.append(TypedEdgeCandidate(
                edge_type="excludes",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="negative",
            ))

    return edges


def extract_applies_if_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract applies_if edges from text.
    Direction: concept → box
    """
    if not text:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in APPLIES_IF_PATTERNS:
        for match in pattern.finditer(text):
            box_key = match.group(1).lower()

            if box_key not in valid_box_keys:
                continue
            if box_key in excluded_boxes:
                continue  # excludes wins
            if box_key in seen:
                continue

            # Check for negation context (excludes wins)
            if _has_negation_context(text, match.start()):
                continue

            seen.add(box_key)
            evidence = _clean_evidence(text, match.start(), len(match.group(0)))

            edges.append(TypedEdgeCandidate(
                edge_type="applies_if",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="positive",
            ))

    return edges


def extract_defines_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract defines edges from text.
    Direction: concept → box

    Note: Some patterns capture terms, not boxes. Those need box resolution
    via nearby references. For now, only emit when box is directly captured.
    """
    if not text:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in DEFINES_PATTERNS:
        for match in pattern.finditer(text):
            captured = match.group(1)

            # Check if captured is a box key or a term
            if re.match(r"^\d+[a-z]?$", captured, re.IGNORECASE):
                # Direct box reference
                box_key = captured.lower()
            else:
                # Term captured - need to find nearby box reference
                # For now, skip (Phase 2b can add term→box resolution)
                continue

            if box_key not in valid_box_keys:
                continue
            if box_key in excluded_boxes:
                continue
            if box_key in seen:
                continue

            if _has_negation_context(text, match.start()):
                continue

            seen.add(box_key)
            evidence = _clean_evidence(text, match.start(), len(match.group(0)))

            edges.append(TypedEdgeCandidate(
                edge_type="defines",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="positive",
            ))

    return edges


def extract_qualifies_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract qualifies edges from text.
    Direction: concept → box
    """
    if not text:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in QUALIFIES_PATTERNS:
        for match in pattern.finditer(text):
            box_key = match.group(1).lower()

            if box_key not in valid_box_keys:
                continue
            if box_key in excluded_boxes:
                continue
            if box_key in seen:
                continue

            if _has_negation_context(text, match.start()):
                continue

            seen.add(box_key)
            evidence = _clean_evidence(text, match.start(), len(match.group(0)))

            edges.append(TypedEdgeCandidate(
                edge_type="qualifies",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="positive",
            ))

    return edges


def extract_requires_edges(
    anchor_id: str,
    source_box_key: Optional[str],
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract requires edges from text.
    Direction: box → box (source = current section's box, target = referenced boxes)

    Only emits edges when source is a box anchor.
    """
    if not text or not source_box_key:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in REQUIRES_PATTERNS:
        for match in pattern.finditer(text):
            ref_text = match.group(1)
            target_keys = _extract_box_keys(ref_text)

            for box_key in target_keys:
                if box_key not in valid_box_keys:
                    continue
                if box_key in excluded_boxes:
                    continue
                if box_key == source_box_key:
                    continue  # No self-edges for requires
                if box_key in seen:
                    continue

                if _has_negation_context(text, match.start()):
                    continue

                seen.add(box_key)
                evidence = _clean_evidence(text, match.start(), len(match.group(0)))

                edges.append(TypedEdgeCandidate(
                    edge_type="requires",
                    source_anchor_id=anchor_id,
                    target_box_key=box_key,
                    confidence=confidence,
                    evidence_text=evidence,
                    pattern_matched=pattern_name,
                    polarity="positive",
                ))

    return edges


def extract_includes_edges(
    anchor_id: str,
    source_box_key: Optional[str],
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract includes edges from text.
    Direction: box → box (e.g., "Box 1a includes amounts in boxes 1b and 2e")

    Only emits edges when source is a box anchor.
    """
    if not text or not source_box_key:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in INCLUDES_PATTERNS:
        for match in pattern.finditer(text):
            # Different patterns capture differently
            if pattern_name == "includes_amounts":
                # Group 1 is source box, group 2 is target boxes
                src_box = match.group(1).lower()
                if src_box != source_box_key:
                    continue  # Only emit if matches current box
                ref_text = match.group(2)
            elif pattern_name == "reported_in_both":
                # Both groups are boxes in a relationship
                ref_text = f"{match.group(1)}, {match.group(2)}"
            else:
                ref_text = match.group(1)

            target_keys = _extract_box_keys(ref_text)

            for box_key in target_keys:
                if box_key not in valid_box_keys:
                    continue
                if box_key in excluded_boxes:
                    continue
                if box_key == source_box_key:
                    continue  # No self-edges
                if box_key in seen:
                    continue

                if _has_negation_context(text, match.start()):
                    continue

                seen.add(box_key)
                evidence = _clean_evidence(text, match.start(), len(match.group(0)))

                edges.append(TypedEdgeCandidate(
                    edge_type="includes",
                    source_anchor_id=anchor_id,
                    target_box_key=box_key,
                    confidence=confidence,
                    evidence_text=evidence,
                    pattern_matched=pattern_name,
                    polarity="positive",
                ))

    return edges


def extract_portion_of_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract portion_of edges from text.
    Direction: concept → box

    Captures subset/portion relationships like:
    - "If any part of box 1a is qualified dividends..."
    - "portion of the amount reported in box 2a"
    - "some of box 1a is attributable to..."

    NOTE: This function is now intended to be called on a single sentence.
    Sentence gating + subset cue + narration filter enforced here.
    """
    if not text or not text.strip():
        return []

    # SENTENCE-GATED FIXES (precision-first)
    # Require a subset cue AND reject narration patterns.
    if not has_subset_cue(text):
        return []
    if has_narration_filter(text):
        return []

    edges: List[TypedEdgeCandidate] = []
    seen: Set[str] = set()

    for pattern_name, pattern, confidence in PORTION_OF_PATTERNS:
        for match in pattern.finditer(text):
            box_key = match.group(1).lower()

            if box_key not in valid_box_keys:
                continue
            if box_key in excluded_boxes:
                continue  # excludes wins
            if box_key in seen:
                continue

            # Check for negation context (excludes wins)
            if _has_negation_context(text, match.start()):
                continue

            seen.add(box_key)

            # Since we're sentence-gated, evidence should be the whole sentence.
            evidence = " ".join(text.split())

            edges.append(TypedEdgeCandidate(
                edge_type="portion_of",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="positive",
            ))

    return edges


def _attach_sentence_provenance(
    edges: List[TypedEdgeCandidate],
    sentence_idx: int,
    char_start: int,
    char_end: int,
) -> List[TypedEdgeCandidate]:
    """Attach sentence-level provenance to edge candidates."""
    for e in edges:
        e.sentence_idx = sentence_idx
        e.sentence_char_start = char_start
        e.sentence_char_end = char_end
    return edges


def extract_typed_edges_from_section(
    anchor_id: str,
    body_text: str,
    valid_box_keys: Set[str],
    source_box_key: Optional[str] = None,
) -> List[TypedEdgeCandidate]:
    """
    Extract all typed edges from a section's body text.

    Precedence (unchanged intent):
    1) excludes wins for targets (suppresses other edge types targeting same box)
    2) if box-section has excludes, suppress includes/requires from that source
    3) portion_of before defines (more specific wins)

    Sentence gating:
    - excludes / applies_if / portion_of / defines / qualifies: sentence-level
    - requires / includes: section-level (box→box context)

    Args:
        anchor_id: The source anchor ID
        body_text: Body text of the section
        valid_box_keys: Set of valid box keys
        source_box_key: If section is a box, its key (for requires/includes edges + self-edge prevention)

    Returns:
        List of TypedEdgeCandidate objects
    """
    if not body_text or not body_text.strip():
        return []

    all_edges: List[TypedEdgeCandidate] = []

    # Split once; keep offsets stable vs full_text
    sents = split_sentences_with_offsets(body_text)

    # -------------------------------------------------------------------------
    # PASS 1: EXCLUDES (global precedence) across all sentences
    # -------------------------------------------------------------------------
    excludes_edges: List[TypedEdgeCandidate] = []
    for i, (sent, s, e) in enumerate(sents):
        edges_i = extract_excludes_edges(anchor_id, sent, valid_box_keys)
        edges_i = _attach_sentence_provenance(edges_i, i, s, e)
        excludes_edges.extend(edges_i)

    # Filter out self-edges for excludes (if source is a box section)
    if source_box_key:
        excludes_edges = [ed for ed in excludes_edges if ed.target_box_key != source_box_key]

    all_edges.extend(excludes_edges)

    excluded_boxes = {ed.target_box_key for ed in excludes_edges}
    has_excludes_from_source = source_box_key is not None and len(excludes_edges) > 0

    # -------------------------------------------------------------------------
    # PASS 2: Other sentence-gated semantic edges
    # -------------------------------------------------------------------------
    applies_if_edges: List[TypedEdgeCandidate] = []
    portion_of_edges: List[TypedEdgeCandidate] = []
    defines_edges: List[TypedEdgeCandidate] = []
    qualifies_edges: List[TypedEdgeCandidate] = []

    # applies_if + portion_of first (portion_of precedes defines later)
    for i, (sent, s, e) in enumerate(sents):
        a = extract_applies_if_edges(anchor_id, sent, valid_box_keys, excluded_boxes)
        applies_if_edges.extend(_attach_sentence_provenance(a, i, s, e))

        p = extract_portion_of_edges(anchor_id, sent, valid_box_keys, excluded_boxes)
        portion_of_edges.extend(_attach_sentence_provenance(p, i, s, e))

    all_edges.extend(applies_if_edges)
    all_edges.extend(portion_of_edges)

    portion_of_boxes = {ed.target_box_key for ed in portion_of_edges}
    excluded_from_defines = excluded_boxes | portion_of_boxes

    # defines + qualifies after we know portion_of targets
    for i, (sent, s, e) in enumerate(sents):
        d = extract_defines_edges(anchor_id, sent, valid_box_keys, excluded_from_defines)
        defines_edges.extend(_attach_sentence_provenance(d, i, s, e))

        q = extract_qualifies_edges(anchor_id, sent, valid_box_keys, excluded_boxes)
        qualifies_edges.extend(_attach_sentence_provenance(q, i, s, e))

    all_edges.extend(defines_edges)
    all_edges.extend(qualifies_edges)

    # -------------------------------------------------------------------------
    # PASS 3: Box→box edges remain section-level
    # -------------------------------------------------------------------------
    if not has_excludes_from_source:
        all_edges.extend(extract_requires_edges(
            anchor_id, source_box_key, body_text, valid_box_keys, excluded_boxes
        ))
        all_edges.extend(extract_includes_edges(
            anchor_id, source_box_key, body_text, valid_box_keys, excluded_boxes
        ))

    return all_edges


# =============================================================================
# PUBLIC APIs (Phase B)
# =============================================================================
# These are the only functions edges.py should import.


def extract_concept_to_box_edges(
    source_node_id: str,
    text: str,
    valid_box_keys: Set[str],
    parent_box_key: Optional[str] = None,
) -> List[TypedEdgeCandidate]:
    """
    Extract concept→box semantic edges from text (paragraph or concept section).

    This is the Phase B "correct" extraction: the source is the rule-holder
    (paragraph node), not the parent anchor.

    Extracts: excludes, applies_if, portion_of, defines, qualifies
    Does NOT extract: requires, includes (those are box→box)

    Sentence gating is applied for precision.

    Args:
        source_node_id: Full node ID of the source (e.g., "doc_id:el_123").
                        This becomes source_anchor_id in the candidates.
        text: Text content to extract from
        valid_box_keys: Set of valid box keys
        parent_box_key: If source is under a box, its key (to skip same-box semantics)

    Returns:
        List of TypedEdgeCandidate objects
    """
    if not text or not text.strip():
        return []

    all_edges: List[TypedEdgeCandidate] = []

    # Split into sentences for gating
    sents = split_sentences_with_offsets(text)

    # -------------------------------------------------------------------------
    # PASS 1: EXCLUDES (global precedence) across all sentences
    # -------------------------------------------------------------------------
    excludes_edges: List[TypedEdgeCandidate] = []
    for i, (sent, s, e) in enumerate(sents):
        edges_i = extract_excludes_edges(source_node_id, sent, valid_box_keys)
        edges_i = _attach_sentence_provenance(edges_i, i, s, e)
        excludes_edges.extend(edges_i)

    all_edges.extend(excludes_edges)
    excluded_boxes = {ed.target_box_key for ed in excludes_edges}

    # -------------------------------------------------------------------------
    # PASS 2: Other concept→box semantic edges (sentence-gated)
    # -------------------------------------------------------------------------
    applies_if_edges: List[TypedEdgeCandidate] = []
    portion_of_edges: List[TypedEdgeCandidate] = []
    defines_edges: List[TypedEdgeCandidate] = []
    qualifies_edges: List[TypedEdgeCandidate] = []

    for i, (sent, s, e) in enumerate(sents):
        a = extract_applies_if_edges(source_node_id, sent, valid_box_keys, excluded_boxes)
        applies_if_edges.extend(_attach_sentence_provenance(a, i, s, e))

        p = extract_portion_of_edges(source_node_id, sent, valid_box_keys, excluded_boxes)
        portion_of_edges.extend(_attach_sentence_provenance(p, i, s, e))

    all_edges.extend(applies_if_edges)
    all_edges.extend(portion_of_edges)

    portion_of_boxes = {ed.target_box_key for ed in portion_of_edges}
    excluded_from_defines = excluded_boxes | portion_of_boxes

    for i, (sent, s, e) in enumerate(sents):
        d = extract_defines_edges(source_node_id, sent, valid_box_keys, excluded_from_defines)
        defines_edges.extend(_attach_sentence_provenance(d, i, s, e))

        q = extract_qualifies_edges(source_node_id, sent, valid_box_keys, excluded_boxes)
        qualifies_edges.extend(_attach_sentence_provenance(q, i, s, e))

    all_edges.extend(defines_edges)
    all_edges.extend(qualifies_edges)

    # -------------------------------------------------------------------------
    # GUARD: Skip edges targeting same box as parent (semantic self-reference)
    # -------------------------------------------------------------------------
    if parent_box_key:
        all_edges = [e for e in all_edges if e.target_box_key != parent_box_key]

    return all_edges


def extract_box_to_box_edges(
    source_node_id: str,
    source_box_key: str,
    text: str,
    valid_box_keys: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract box→box dependency edges from text (box section).

    Extracts: requires, includes
    These describe relationships between boxes, not rules.

    Args:
        source_node_id: Full node ID of the source box (e.g., "doc_id:box_1a")
        source_box_key: Box key of the source (e.g., "1a")
        text: Full text of the box section
        valid_box_keys: Set of valid box keys

    Returns:
        List of TypedEdgeCandidate objects
    """
    if not text or not text.strip() or not source_box_key:
        return []

    all_edges: List[TypedEdgeCandidate] = []

    # Extract anchor_id from node_id for internal functions
    # (they expect anchor_id like "box_1a", not full node_id)
    anchor_id = source_node_id.split(":")[-1] if ":" in source_node_id else source_node_id

    # Extract requires edges
    requires = extract_requires_edges(
        anchor_id=anchor_id,
        source_box_key=source_box_key,
        text=text,
        valid_box_keys=valid_box_keys,
        excluded_boxes=set(),
    )
    all_edges.extend(requires)

    # Extract includes edges
    includes = extract_includes_edges(
        anchor_id=anchor_id,
        source_box_key=source_box_key,
        text=text,
        valid_box_keys=valid_box_keys,
        excluded_boxes=set(),
    )
    all_edges.extend(includes)

    return all_edges
