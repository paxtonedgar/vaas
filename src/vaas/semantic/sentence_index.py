"""
Sentence index utilities: build canonical sentence spans per element.
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd

from vaas.semantic.typed_edges import split_sentences_with_offsets


def build_sentence_index(elements_df: pd.DataFrame, doc_id: str) -> pd.DataFrame:
    """
    Build a canonical list of sentences for each element.

    Returns DataFrame with columns:
        doc_id, source_element_id, sentence_idx, sentence_text,
        sentence_char_start, sentence_char_end
    """
    rows: List[Dict[str, object]] = []

    if elements_df is None or elements_df.empty:
        return pd.DataFrame(columns=[
            "doc_id",
            "source_element_id",
            "sentence_idx",
            "sentence_text",
            "sentence_char_start",
            "sentence_char_end",
        ])

    for _, elem in elements_df.iterrows():
        element_id = elem.get("element_id")
        if element_id is None:
            continue
        element_id_str = str(element_id)
        text = elem.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        sentences = split_sentences_with_offsets(text)
        for idx, (sentence, start, end) in enumerate(sentences):
            rows.append({
                "doc_id": doc_id,
                "source_element_id": element_id_str,
                "sentence_idx": idx,
                "sentence_text": sentence,
                "sentence_char_start": start,
                "sentence_char_end": end,
            })

    return pd.DataFrame(rows, columns=[
        "doc_id",
        "source_element_id",
        "sentence_idx",
        "sentence_text",
        "sentence_char_start",
        "sentence_char_end",
    ])


__all__ = ["build_sentence_index"]
