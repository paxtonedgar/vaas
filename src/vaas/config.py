"""
Pipeline configuration for VaaS extraction.

This module defines the PipelineConfig dataclass that captures all configurable
parameters for the extraction pipeline, replacing hardcoded values scattered
throughout run_pipeline.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set


@dataclass
class PipelineConfig:
    """
    Configuration for the VaaS extraction pipeline.

    Attributes:
        pdf_path: Path to the input PDF file.
        doc_id: Document identifier. If None, auto-derived from pdf_path stem.
        output_dir: Directory for output parquet files.

        header_size_delta: Font size delta above body_size to classify as header.
        column_min_peak_ratio: Minimum ratio for second column peak detection.
        column_min_distance_pct: Minimum distance between columns as % of page width.
        page_mid_x_fallback: Fallback x-coordinate for column split when detection fails.

        thin_char_thresh: Max char count for a section to be considered "thin".
        thin_elem_thresh: Max element count for a section to be considered "thin".
        body_char_thresh: Max body length for merge-forward eligibility.

        evidence_context_chars: Characters of context around reference matches.

        expected_boxes: Set of expected box keys for validation.
        section_id_map: Mapping from section header text to canonical IDs.
    """

    # Input/Output
    pdf_path: Path
    doc_id: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Font detection
    header_size_delta: float = 0.5

    # Column detection
    column_min_peak_ratio: float = 0.25
    column_min_distance_pct: float = 0.25
    page_mid_x_fallback: float = 306.0

    # Merge-forward thresholds
    thin_char_thresh: int = 160
    thin_elem_thresh: int = 2
    body_char_thresh: int = 120

    # Reference extraction
    evidence_context_chars: int = 50

    # Validation schema
    expected_boxes: Set[str] = field(default_factory=set)
    section_id_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-derive doc_id from pdf_path if not provided."""
        # Ensure pdf_path is a Path object
        if isinstance(self.pdf_path, str):
            self.pdf_path = Path(self.pdf_path)

        # Ensure output_dir is a Path object
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Auto-derive doc_id from filename
        if self.doc_id is None:
            # e.g., "i1099div.pdf" -> "1099div_filer"
            stem = self.pdf_path.stem  # "i1099div"
            # Remove leading 'i' (instructions) or 'f' (form) prefix
            if stem.startswith(("i", "f")) and stem[1:2].isdigit():
                stem = stem[1:]
            # Append document type suffix
            if self.pdf_path.stem.startswith("i"):
                self.doc_id = f"{stem}_filer"
            elif self.pdf_path.stem.startswith("f"):
                self.doc_id = f"{stem}_form"
            else:
                self.doc_id = stem

    @classmethod
    def for_1099div(cls, pdf_path: str = "data/i1099div.pdf") -> "PipelineConfig":
        """
        Create configuration for 1099-DIV filer instructions.

        Args:
            pdf_path: Path to the 1099-DIV instructions PDF.

        Returns:
            PipelineConfig with 1099-DIV specific settings.
        """
        return cls(
            pdf_path=Path(pdf_path),
            doc_id="1099div_filer",
            expected_boxes={
                "1a", "1b", "2a", "2b", "2c", "2d", "2e", "2f",
                "3", "4", "5", "6", "7", "8", "9", "10",
                "11", "12", "13", "14", "15", "16"
            },
            section_id_map={
                "future developments": "sec_future_developments",
                "reminders": "sec_reminders",
                "general instructions": "sec_general_instructions",
                "specific instructions": "sec_specific_instructions",
                "what's new": "sec_whats_new",
                "definitions": "sec_definitions",
                "how to": "sec_how_to",
                "where to": "sec_where_to",
                "paperwork reduction act notice": "sec_paperwork_reduction",
                "additional information": "sec_additional_info",
            },
        )

    @classmethod
    def for_1099int(cls, pdf_path: str = "data/i1099int.pdf") -> "PipelineConfig":
        """
        Create configuration for 1099-INT filer instructions.

        Args:
            pdf_path: Path to the 1099-INT instructions PDF.

        Returns:
            PipelineConfig with 1099-INT specific settings.
        """
        return cls(
            pdf_path=Path(pdf_path),
            doc_id="1099int_filer",
            expected_boxes={
                "1", "2", "3", "4", "5", "6", "7", "8",
                "9", "10", "11", "12", "13", "14", "15", "16", "17"
            },
            section_id_map={
                "future developments": "sec_future_developments",
                "reminders": "sec_reminders",
                "general instructions": "sec_general_instructions",
                "specific instructions": "sec_specific_instructions",
                "what's new": "sec_whats_new",
                "definitions": "sec_definitions",
            },
        )
