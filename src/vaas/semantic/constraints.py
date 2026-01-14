"""Typed constraint extraction."""

from __future__ import annotations

import re
from typing import List, Optional

from vaas.schemas.semantic_contract import ConstraintRow
from vaas.semantic.canonical_ids import fingerprint_constraints

AMOUNT_REGEX = re.compile(r"\$?(?P<value>\d{1,3}(?:,\d{3})*)(?:\s*(?:dollars|USD))?", re.IGNORECASE)
VERBAL_AMOUNT_REGEX = re.compile(r"(?P<value>\d+)\s+dollars", re.IGNORECASE)
PERCENT_REGEX = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
DAY_REGEX = re.compile(r"(?P<value>\d{1,3})\s+day(?:s)?", re.IGNORECASE)
MONTH_REGEX = re.compile(r"(?P<value>\d{1,2})\s+month(?:s)?", re.IGNORECASE)
BETWEEN_REGEX = re.compile(r"between\s+(?P<min>\d+)\s+and\s+(?P<max>\d+)", re.IGNORECASE)
DATE_REGEX = re.compile(
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})",
    re.IGNORECASE,
)
ON_OR_AFTER_REGEX = re.compile(r"on\s+or\s+after", re.IGNORECASE)
ON_OR_BEFORE_REGEX = re.compile(r"on\s+or\s+before", re.IGNORECASE)
AT_LEAST_REGEX = re.compile(r"at\s+least", re.IGNORECASE)
AT_MOST_REGEX = re.compile(r"at\s+most|no\s+more\s+than", re.IGNORECASE)


def _strip_commas(value: str) -> int:
    return int(value.replace(",", ""))


def extract_constraints(
    doc_id: str,
    source_element_id: str,
    sentence_idx: int,
    sentence: str,
) -> List[ConstraintRow]:
    constraints: List[ConstraintRow] = []
    if not sentence:
        return constraints

    text = sentence.strip()

    def append_row(**kwargs) -> None:
        base = {
            "doc_id": doc_id,
            "source_element_id": source_element_id,
            "sentence_idx": sentence_idx,
            "raw_text": text,
        }
        payload = {**base, **kwargs}
        fingerprint = fingerprint_constraints({k: str(v) for k, v in payload.items() if k not in {"doc_id", "source_element_id", "sentence_idx", "raw_text"}})
        payload["constraint_id"] = fingerprint
        constraints.append(ConstraintRow(**payload))

    for match in BETWEEN_REGEX.finditer(text):
        append_row(
            constraint_type="range",
            min_value=int(match.group("min")),
            max_value=int(match.group("max")),
            inclusive_min=True,
            inclusive_max=True,
        )

    for match in DAY_REGEX.finditer(text):
        append_row(
            constraint_type="duration",
            unit="days",
            value_int=int(match.group("value")),
        )

    for match in MONTH_REGEX.finditer(text):
        append_row(
            constraint_type="duration",
            unit="months",
            value_int=int(match.group("value")),
        )

    for match in AMOUNT_REGEX.finditer(text):
        append_row(
            constraint_type="amount",
            amount_currency="USD",
            amount_value=_strip_commas(match.group("value")),
        )

    for match in VERBAL_AMOUNT_REGEX.finditer(text):
        append_row(
            constraint_type="amount",
            amount_currency="USD",
            amount_value=int(match.group("value")),
        )

    for match in PERCENT_REGEX.finditer(text):
        append_row(
            constraint_type="percentage",
            percent_value=float(match.group("value")),
        )

    for match in DATE_REGEX.finditer(text):
        append_row(
            constraint_type="date",
            date_month=match.group("month").title(),
            date_day=int(match.group("day")),
            date_year=int(match.group("year")),
        )

    comparator = None
    if ON_OR_AFTER_REGEX.search(text) or AT_LEAST_REGEX.search(text):
        comparator = "gte"
    elif ON_OR_BEFORE_REGEX.search(text) or AT_MOST_REGEX.search(text):
        comparator = "lte"
    if comparator:
        append_row(
            constraint_type="comparator",
            comparator=comparator,
        )

    return constraints


__all__ = ["extract_constraints"]
