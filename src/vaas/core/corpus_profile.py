"""
Corpus Profile: Static Configuration for Document Extraction.

This replaces the over-engineered AuthorityAdapter system.
A CorpusProfile describes a document set and drives extraction + interpretation.

For IRS-only scope, this is all we need:
- Document metadata (form, revision, effective years)
- Citation scheme for normalization
- Surface model (what "box" means)
- Optional precedence weights for citation types

NO routing, NO adapters, NO multi-jurisdiction abstraction.

Usage:
    profile = IRS_1099DIV_2024
    # Use profile.citation_scheme to normalize references
    # Use profile.surface_model to understand box structure
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Set


class CitationType(Enum):
    """Types of citations in IRS documents."""
    IRC = "irc"                     # Internal Revenue Code section
    TREAS_REG = "treas_reg"         # Treasury Regulation
    NOTICE = "notice"               # IRS Notice
    REV_RUL = "rev_rul"             # Revenue Ruling
    REV_PROC = "rev_proc"           # Revenue Procedure
    PUB = "pub"                     # IRS Publication
    FORM_INST = "form_inst"         # Form Instructions reference
    PLR = "plr"                     # Private Letter Ruling


class SurfaceType(Enum):
    """Types of reporting surfaces in IRS forms."""
    BOX = "box"                     # Numbered box (Box 1a, Box 2e)
    LINE = "line"                   # Schedule line
    STATEMENT = "statement"         # Free-text statement requirement
    FLAG = "flag"                   # Checkbox/boolean
    AMOUNT = "amount"               # Dollar amount field


@dataclass(frozen=True)
class CitationScheme:
    """
    How to parse and normalize citations in this corpus.

    Patterns map citation types to regex patterns for extraction.
    Weights provide relative precedence (higher = more authoritative).
    """
    patterns: Dict[CitationType, str] = field(default_factory=dict)
    weights: Dict[CitationType, int] = field(default_factory=dict)

    def weight_for(self, citation_type: CitationType) -> int:
        """Get precedence weight for a citation type."""
        return self.weights.get(citation_type, 0)


# Standard IRS citation scheme
IRS_CITATION_SCHEME = CitationScheme(
    patterns={
        CitationType.IRC: r"(?:IRC\s*)?(?:section|§)\s*(\d+[A-Za-z]?(?:\([a-z]\))?)",
        CitationType.TREAS_REG: r"(?:Treas\.?\s*)?(?:Reg\.?\s*)?§?\s*1\.(\d+-\d+)",
        CitationType.NOTICE: r"Notice\s+(\d{4}-\d+)",
        CitationType.REV_RUL: r"Rev\.?\s*Rul\.?\s+(\d{4}-\d+)",
        CitationType.REV_PROC: r"Rev\.?\s*Proc\.?\s+(\d{4}-\d+)",
        CitationType.PUB: r"(?:Pub\.?|Publication)\s+(\d+)",
    },
    weights={
        # Higher = more authoritative
        CitationType.IRC: 100,          # Statute - binding
        CitationType.TREAS_REG: 90,     # Regulations - binding
        CitationType.REV_RUL: 70,       # Revenue rulings - IRS position
        CitationType.REV_PROC: 70,      # Revenue procedures
        CitationType.NOTICE: 60,        # Notices
        CitationType.PUB: 30,           # Publications - informal guidance
        CitationType.FORM_INST: 50,     # Instructions themselves
        CitationType.PLR: 20,           # PLRs - not precedential
    },
)


@dataclass(frozen=True)
class SurfaceModel:
    """
    What "surface" means for this form.

    Defines the structure of reporting targets.
    """
    primary_type: SurfaceType
    box_pattern: str = r"Box\s*(\d+[a-z]?)"
    supports_statements: bool = True
    max_boxes: int = 50


# Standard IRS form surface model
IRS_FORM_SURFACE = SurfaceModel(
    primary_type=SurfaceType.BOX,
    box_pattern=r"Box\s*(\d+[a-z]?)",
    supports_statements=True,
    max_boxes=50,
)


@dataclass(frozen=True)
class CorpusProfile:
    """
    Static configuration for a document corpus.

    This is all you need to drive extraction and interpretation.
    No routing, no adapters.
    """
    # Identity
    issuer: str                              # "IRS", "Treasury"
    doc_type: str                            # "instructions", "form", "publication"
    form_id: str                             # "1099-DIV", "1099-INT"
    revision: str                            # "Rev. 01-2024"
    revision_date: Optional[date] = None

    # Effective scope
    effective_tax_years: Set[int] = field(default_factory=set)
    supersedes: Optional[str] = None         # Previous revision this replaces

    # Interpretation config
    citation_scheme: CitationScheme = field(default_factory=lambda: IRS_CITATION_SCHEME)
    surface_model: SurfaceModel = field(default_factory=lambda: IRS_FORM_SURFACE)

    @property
    def corpus_id(self) -> str:
        """Unique identifier for this corpus."""
        return f"{self.issuer.lower()}:{self.form_id.lower()}:{self.revision.replace(' ', '_').lower()}"


# =============================================================================
# CANONICAL PROFILES
# =============================================================================

IRS_1099DIV_2024 = CorpusProfile(
    issuer="IRS",
    doc_type="instructions",
    form_id="1099-DIV",
    revision="Rev. 01-2024",
    revision_date=date(2024, 1, 1),
    effective_tax_years={2024, 2025},
    citation_scheme=IRS_CITATION_SCHEME,
    surface_model=IRS_FORM_SURFACE,
)

IRS_1099INT_2024 = CorpusProfile(
    issuer="IRS",
    doc_type="instructions",
    form_id="1099-INT",
    revision="Rev. 01-2024",
    revision_date=date(2024, 1, 1),
    effective_tax_years={2024, 2025},
    citation_scheme=IRS_CITATION_SCHEME,
    surface_model=IRS_FORM_SURFACE,
)


# =============================================================================
# AUTHORITY AS PROVENANCE (not routing)
# =============================================================================

@dataclass(frozen=True)
class AuthorityProvenance:
    """
    Authority as a metadata dimension for KG nodes.

    NOT a routing architecture. Just provenance tracking.
    """
    issuer: str                     # "IRS"
    citation_type: CitationType     # IRC, TREAS_REG, etc.
    citation_id: str                # "section 1(h)(11)"
    weight: int = 0                 # Precedence weight from citation scheme

    @classmethod
    def from_citation(
        cls,
        citation_str: str,
        citation_type: CitationType,
        scheme: CitationScheme,
    ) -> "AuthorityProvenance":
        """Create provenance from a parsed citation."""
        return cls(
            issuer="IRS",
            citation_type=citation_type,
            citation_id=citation_str,
            weight=scheme.weight_for(citation_type),
        )
