"""
PDF extraction utilities.

This module provides functions for extracting spans from PDF documents
and inferring document properties like body font size.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionResult:
    """Result of PDF span extraction."""
    spans_df: pd.DataFrame
    page_count: int
    doc_id: str


def extract_spans_from_pdf(
    pdf_path: str,
    doc_id: str = "document",
) -> PDFExtractionResult:
    """
    Extract text spans from PDF with font/position metadata.

    Uses PyMuPDF (fitz) to extract spans with:
    - Text content
    - Font name and flags (bold, italic)
    - Font size
    - Bounding box coordinates

    Args:
        pdf_path: Path to PDF file.
        doc_id: Document identifier for span IDs.

    Returns:
        PDFExtractionResult with spans DataFrame and metadata.

    Raises:
        ImportError: If PyMuPDF is not installed.
        FileNotFoundError: If PDF file doesn't exist.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Run: pip install PyMuPDF"
        )

    doc = fitz.open(pdf_path)

    span_rows = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        d = page.get_text("dict")

        for b_idx, block in enumerate(d.get("blocks", [])):
            # Skip non-text blocks (images, etc.)
            if block.get("type") != 0:
                continue

            for l_idx, line in enumerate(block.get("lines", [])):
                for s_idx, span in enumerate(line.get("spans", [])):
                    text = span.get("text", "")
                    if not text or not text.strip():
                        continue

                    bbox = span.get("bbox", None)
                    if bbox is None:
                        continue

                    span_rows.append({
                        "doc_id": doc_id,
                        "page": int(pno + 1),  # 1-indexed
                        "block_id": int(b_idx),
                        "line_id": int(l_idx),
                        "span_id": int(s_idx),
                        "text": text,
                        "font": span.get("font", ""),
                        "flags": int(span.get("flags", 0)),
                        "size": float(span.get("size", 0.0)),
                        "bbox": tuple(map(float, bbox)),
                    })

    spans_df = pd.DataFrame(span_rows)

    logger.info(f"Extracted {len(spans_df)} spans from {doc.page_count} pages")

    return PDFExtractionResult(
        spans_df=spans_df,
        page_count=doc.page_count,
        doc_id=doc_id,
    )


def infer_body_font_size(spans_df: pd.DataFrame) -> float:
    """
    Infer body text font size from span distribution.

    Uses mode (most frequent) of rounded font sizes as the body size.

    Args:
        spans_df: Spans DataFrame with 'size' column.

    Returns:
        Body font size (rounded to 0.1).
    """
    if spans_df.empty or "size" not in spans_df.columns:
        logger.warning("No spans or size column, defaulting to 9.0")
        return 9.0

    sizes = spans_df["size"].astype(float).round(1)
    body_size = float(sizes.value_counts().idxmax())

    logger.info(f"Inferred body font size: {body_size}")
    return body_size


def get_font_size_distribution(spans_df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """
    Get font size distribution for debugging/analysis.

    Args:
        spans_df: Spans DataFrame with 'size' column.
        top_n: Number of top sizes to return.

    Returns:
        Series of size counts.
    """
    if spans_df.empty or "size" not in spans_df.columns:
        return pd.Series(dtype=int)

    sizes = spans_df["size"].astype(float).round(1)
    return sizes.value_counts().head(top_n)


# Legacy wrapper for backward compatibility
def extract_spans_legacy(
    pdf_path: str,
    doc_id: str = "1099div_filer",
) -> Tuple[pd.DataFrame, int]:
    """
    Legacy wrapper returning (spans_df, page_count).

    For drop-in replacement in run_pipeline.py.
    """
    result = extract_spans_from_pdf(pdf_path, doc_id)

    print(f"Pages: {result.page_count}")
    print(f"Spans extracted: {len(result.spans_df)}")

    return result.spans_df, result.page_count
