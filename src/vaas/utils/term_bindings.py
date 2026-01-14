"""Term-to-binding helpers for semantic extraction."""

from __future__ import annotations

import re
from typing import Optional

from vaas.core.bindings import BINDING_REGISTRY
from vaas.utils.text import slug_title

BOX_KEY_RX = re.compile(r"(?i)box\s+(\d+[a-z]?)")
EXCEPTION_CONTEXT_RX = re.compile(
    r"(?i)not\s+qualified\s+(dividends?(?:\s+income)?)"
)
TERM_SYNONYMS = {
    "qualified dividend": "qualified dividends",
    "qualified dividend income": "qualified dividends",
}
_REGISTRY_READY = False


def _ensure_bindings_loaded() -> None:
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return
    from vaas.core import register_canonical_bindings

    register_canonical_bindings()
    _REGISTRY_READY = True


def form_type_from_form_id(form_id: Optional[str]) -> Optional[str]:
    if not form_id:
        return None
    text = str(form_id)
    if text.startswith("form:"):
        text = text[len("form:"):]
    return text.upper()


def semantic_id_from_term(term: str) -> Optional[str]:
    if not term:
        return None
    _ensure_bindings_loaded()
    normalized = str(term).strip().lower()
    if normalized in TERM_SYNONYMS:
        term = TERM_SYNONYMS[normalized]
    candidate = slug_title(term, max_len=64)
    if candidate in BINDING_REGISTRY.all_ids():
        return candidate
    return None


def box_key_for_term(term: str, form_id: Optional[str]) -> Optional[str]:
    semantic_id = semantic_id_from_term(term)
    if not semantic_id:
        return None
    form_type = form_type_from_form_id(form_id)
    if not form_type:
        return None
    physical_id = BINDING_REGISTRY.resolve(semantic_id, form_type)
    if not physical_id:
        return None
    match = BOX_KEY_RX.search(physical_id)
    if not match:
        return None
    return match.group(1).lower()


def exception_term_from_context(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = EXCEPTION_CONTEXT_RX.search(str(text))
    if not match:
        return None
    return f"qualified {match.group(1)}"


__all__ = [
    "box_key_for_term",
    "exception_term_from_context",
    "form_type_from_form_id",
    "semantic_id_from_term",
]
