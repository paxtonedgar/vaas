"""
Canonical ID helpers for semantic KG artifacts.

These utilities normalize raw instruction text into canonical identifiers that
remain stable across documents (authority citations, box references, subjects,
etc.). Canonical IDs are the backbone for deduplicating claims, linking to
external ontologies, and enabling downstream training data generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Optional

AuthorityId = str
CanonicalId = str


@dataclass(frozen=True)
class Scope:
    """
    Minimal context describing the current extraction scope.

    Attributes:
        doc_id: Document identifier (e.g., "1099div_filer").
        form_id: Canonical form identifier (e.g., "form:1099-div").
        anchor_id: Current anchor/section identifier if available.
    """

    doc_id: str
    form_id: Optional[str] = None
    anchor_id: Optional[str] = None


NOTICE_RX = re.compile(r"(?i)notice\s+(\d{4}-\d+)")
REV_RUL_RX = re.compile(r"(?i)rev\.\s*rul\.?\s*(\d{4}-\d+)")
REV_PROC_RX = re.compile(r"(?i)rev\.\s*proc\.?\s*(\d{4}-\d+)")
TREAS_REG_RX = re.compile(r"(?i)(?:treasury\s+reg(?:ulation)?|treas\.\s*reg\.?|reg\.)\s*([\d\.]+-[\w\(\)\.-]+)")
PUB_RX = re.compile(r"(?i)(?:pub\.?|publication)\s*(\d+)")
FORM_RX = re.compile(r"(?i)form\s+([0-9][\w\-]+)")
IRC_RX = re.compile(r"(?i)(?:section|§)\s*(\d+[A-Za-z]?(?:\([a-z0-9]+\))*)")


def canonicalize_authority(raw: str) -> Optional[AuthorityId]:
    """Normalize raw authority strings to canonical IDs."""
    if not raw:
        return None
    text = raw.strip()

    # IRC sections
    m = IRC_RX.search(text)
    if m:
        section = m.group(1)
        normalized = section.replace(" ", "").lower()
        return f"authority:irc:{normalized}"

    # Treasury regulations
    m = TREAS_REG_RX.search(text)
    if m:
        reg = m.group(1).lower()
        reg = reg.replace(" ", "")
        return f"authority:treas_reg:{reg}"

    # IRS Notices
    m = NOTICE_RX.search(text)
    if m:
        return f"authority:notice:{m.group(1)}"

    # Revenue rulings / procedures
    m = REV_RUL_RX.search(text)
    if m:
        return f"authority:rev_rul:{m.group(1)}"

    m = REV_PROC_RX.search(text)
    if m:
        return f"authority:rev_proc:{m.group(1)}"

    # Publications
    m = PUB_RX.search(text)
    if m:
        return f"authority:pub:{m.group(1)}"

    # Forms (treated as authority for provenance)
    m = FORM_RX.search(text)
    if m:
        form = canonicalize_form(f"Form {m.group(1)}")
        if form:
            return form

    return None


def canonicalize_subject_object(
    node_id: str,
    canonical_map: Optional[Mapping[str, str]],
    scope: Scope,
) -> Optional[CanonicalId]:
    """Canonicalize node identifiers using precomputed maps and fallbacks."""
    if not node_id:
        return None
    if canonical_map:
        canonical = canonical_map.get(node_id)
        if canonical:
            return canonical
    if ":box_" in node_id:
        key = node_id.split(":")[-1].replace("box_", "")
        return canonicalize_box(f"Box {key}", form_id=scope.form_id or "form:unknown")
    return node_id


def canonicalize_box(raw: str, form_id: Optional[str] = None) -> Optional[str]:
    """
    Normalize box identifiers to canonical `form:<slug>:box:<key>` representation.
    """
    if not raw:
        return None
    match = re.search(r"(?i)box\s+(\d+[a-z]?)", raw)
    if not match:
        return None
    key = match.group(1).lower()
    form = form_id or "form:1099-div"
    return f"{form}:box:{key}"


def canonicalize_form(raw: str) -> Optional[str]:
    """Normalize form references to `form:<slug>` identifiers."""
    if not raw:
        return None
    match = FORM_RX.search(raw)
    if not match:
        return None
    form = match.group(1).lower()
    form = form.replace(" ", "-")
    return f"form:{form}"


def fingerprint_constraints(values: Mapping[str, Any]) -> str:
    """Create a deterministic fingerprint for normalized constraint fields."""
    items = sorted((k, str(v)) for k, v in values.items())
    payload = "|".join(f"{k}={v}" for k, v in items)
    digest = sha256(payload.encode("utf-8")).hexdigest()
    return f"cns_{digest[:16]}"


__all__ = [
    "AuthorityId",
    "CanonicalId",
    "Scope",
    "canonicalize_authority",
    "canonicalize_subject_object",
    "canonicalize_box",
    "canonicalize_form",
    "fingerprint_constraints",
]
