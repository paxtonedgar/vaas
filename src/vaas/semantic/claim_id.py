"""Stable claim ID generation utilities."""

from hashlib import sha256

CLAIM_ID_PREFIX = "clm_"


def make_claim_id(
    doc_id: str,
    source_element_id: str,
    sentence_idx: int,
    char_start: int,
    char_end: int,
    predicate: str,
    subject_canonical_id: str,
    object_canonical_id: str,
    polarity: str,
) -> str:
    """
    Build a deterministic claim identifier.

    The ID is a sha256 hash over the key structural components that define a
    claim. Only a short prefix is kept for readability, but the probability of
    collision remains negligible for our scale.
    """
    payload = "|".join(
        [
            doc_id,
            source_element_id,
            str(sentence_idx),
            str(char_start),
            str(char_end),
            predicate,
            subject_canonical_id or "",
            object_canonical_id or "",
            polarity,
        ]
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()
    return f"{CLAIM_ID_PREFIX}{digest[:16]}"


__all__ = ["make_claim_id", "CLAIM_ID_PREFIX"]
