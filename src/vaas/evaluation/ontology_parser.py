"""
Ontology Parser for Form-Specific Semantic Targets

Parses markdown ontology files to extract:
- Entities (nodes in the semantic model)
- Rules (conditional logic with when/effect/governed_by)
- Windows (temporal constructs with offsets/thresholds)
- Relationships (predicates between entities)

This is form-agnostic: the parser extracts structure from ANY ontology
following the established markdown format.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum


class OntologyElementType(Enum):
    """Types of elements in the ontology."""
    ENTITY = "entity"
    RULE = "rule"
    WINDOW = "window"
    THRESHOLD = "threshold"
    GUIDELINE = "guideline"


@dataclass
class Relationship:
    """A semantic relationship between entities."""
    source: str                    # Source entity/rule/window name
    predicate: str                 # Relationship type (e.g., "governed_by", "when", "effect")
    target: str                    # Target entity or value
    is_negated: bool = False       # True if relationship is negative (e.g., "does_not_apply")
    context: Optional[str] = None  # Additional context from the relationship line

    def __hash__(self):
        return hash((self.source, self.predicate, self.target, self.is_negated))

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return (self.source == other.source and
                self.predicate == other.predicate and
                self.target == other.target and
                self.is_negated == other.is_negated)


@dataclass
class OntologyElement:
    """An element (entity/rule/window) in the ontology."""
    name: str
    element_type: OntologyElementType
    section_id: int                          # Which numbered section it came from
    section_title: str                       # Title of the section
    relationships: List[Relationship] = field(default_factory=list)
    description: Optional[str] = None        # Any description text

    @property
    def predicates(self) -> Set[str]:
        """Get all unique predicates used by this element."""
        return {r.predicate for r in self.relationships}

    @property
    def targets(self) -> Set[str]:
        """Get all unique targets referenced by this element."""
        return {r.target for r in self.relationships}


@dataclass
class ParsedOntology:
    """Complete parsed ontology with query capabilities."""
    source_file: str
    form_id: str                             # e.g., "1099-DIV"
    elements: List[OntologyElement] = field(default_factory=list)

    @property
    def entities(self) -> List[OntologyElement]:
        return [e for e in self.elements if e.element_type == OntologyElementType.ENTITY]

    @property
    def rules(self) -> List[OntologyElement]:
        return [e for e in self.elements if e.element_type == OntologyElementType.RULE]

    @property
    def windows(self) -> List[OntologyElement]:
        return [e for e in self.elements if e.element_type == OntologyElementType.WINDOW]

    @property
    def thresholds(self) -> List[OntologyElement]:
        return [e for e in self.elements if e.element_type == OntologyElementType.THRESHOLD]

    @property
    def all_predicates(self) -> Set[str]:
        """Get all unique predicates across the ontology."""
        preds = set()
        for elem in self.elements:
            preds.update(elem.predicates)
        return preds

    @property
    def all_relationships(self) -> List[Relationship]:
        """Get all relationships across the ontology."""
        rels = []
        for elem in self.elements:
            rels.extend(elem.relationships)
        return rels

    def get_relationships_by_predicate(self, predicate: str) -> List[Relationship]:
        """Get all relationships with a specific predicate."""
        return [r for r in self.all_relationships if r.predicate == predicate]

    def get_elements_by_section(self, section_id: int) -> List[OntologyElement]:
        """Get all elements from a specific section."""
        return [e for e in self.elements if e.section_id == section_id]

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "form_id": self.form_id,
            "total_elements": len(self.elements),
            "entities": len(self.entities),
            "rules": len(self.rules),
            "windows": len(self.windows),
            "thresholds": len(self.thresholds),
            "unique_predicates": len(self.all_predicates),
            "total_relationships": len(self.all_relationships),
        }


# =============================================================================
# PARSING PATTERNS
# =============================================================================

# Section header pattern: "## 1. Default Reporting Rules"
SECTION_HEADER_RX = re.compile(r'^##\s+(\d+)\.\s+(.+)$')

# Subsection header pattern: "### Rule: DefaultDividendOnUncertainty"
SUBSECTION_HEADER_RX = re.compile(r'^###\s+(?:(Rule|Window|Threshold|Guideline|Entity):\s+)?(.+)$')

# Entity list item: "* **Entity Name**" or "- Entity Name"
ENTITY_RX = re.compile(r'^[\*\-]\s+\*?\*?([A-Z][A-Za-z0-9\s\-\(\)]+?)\*?\*?\s*(?:\(|$)')

# Relationship pattern: "* **Source** —(predicate)—> **Target**"
# Also handles: "* Source —(predicate)—> Target" without bold
RELATIONSHIP_RX = re.compile(
    r'^\s*[\*\-]\s+'
    r'\*?\*?([^*\n]+?)\*?\*?\s*'         # Source (with optional bold)
    r'—\(([a-z_]+)\)—>\s*'               # Predicate in —()—>
    r'\*?\*?([^*\n]+?)\*?\*?\s*$'        # Target (with optional bold)
)

# Simple relationship: "—(predicate)—>" anywhere on line
SIMPLE_REL_RX = re.compile(r'—\(([a-z_]+)\)—>')

# Form ID from title: "# Form 1099-DIV Ontology"
FORM_ID_RX = re.compile(r'Form\s+(\d+[A-Z\-]+)', re.IGNORECASE)

# Inline entity in relationships section
INLINE_ENTITY_RX = re.compile(r'\*\*([A-Z][A-Za-z0-9\s\-\(\)]+?)\*\*')


def _classify_element_type(name: str, header_type: Optional[str]) -> OntologyElementType:
    """Classify an element based on its name and header context."""
    name_lower = name.lower()

    # Explicit header type
    if header_type:
        header_lower = header_type.lower()
        if "rule" in header_lower:
            return OntologyElementType.RULE
        if "window" in header_lower:
            return OntologyElementType.WINDOW
        if "threshold" in header_lower:
            return OntologyElementType.THRESHOLD
        if "guideline" in header_lower:
            return OntologyElementType.GUIDELINE

    # Infer from name
    if name_lower.startswith("rule:") or "rule" in name_lower:
        return OntologyElementType.RULE
    if name_lower.startswith("window:") or "window" in name_lower:
        return OntologyElementType.WINDOW
    if name_lower.startswith("threshold:") or "threshold" in name_lower:
        return OntologyElementType.THRESHOLD
    if name_lower.startswith("guideline:"):
        return OntologyElementType.GUIDELINE

    return OntologyElementType.ENTITY


def _extract_relationships_from_block(lines: List[str], source_default: str) -> List[Relationship]:
    """Extract relationships from a block of lines."""
    relationships = []
    current_source = source_default

    for line in lines:
        # Try full relationship pattern
        match = RELATIONSHIP_RX.match(line)
        if match:
            source = match.group(1).strip()
            predicate = match.group(2).strip()
            target = match.group(3).strip()

            # Update current source if specified
            if source and not source.startswith("—"):
                current_source = source

            # Check for negation in predicate
            is_negated = predicate.startswith("not_") or predicate.startswith("does_not_")

            relationships.append(Relationship(
                source=current_source,
                predicate=predicate,
                target=target,
                is_negated=is_negated,
            ))
            continue

        # Try simple relationship (continuation with same source)
        simple_match = SIMPLE_REL_RX.search(line)
        if simple_match:
            predicate = simple_match.group(1)
            # Extract target after the arrow
            after_arrow = line[simple_match.end():].strip()
            target = INLINE_ENTITY_RX.search(after_arrow)
            if target:
                target = target.group(1)
            else:
                # Take everything after arrow up to newline or parenthesis
                target = re.split(r'[(\n]', after_arrow)[0].strip().strip('*')

            if target:
                is_negated = predicate.startswith("not_") or predicate.startswith("does_not_")
                relationships.append(Relationship(
                    source=current_source,
                    predicate=predicate,
                    target=target,
                    is_negated=is_negated,
                ))

    return relationships


def parse_ontology(file_path: str) -> ParsedOntology:
    """
    Parse an ontology markdown file into structured elements.

    Args:
        file_path: Path to the ontology markdown file

    Returns:
        ParsedOntology with all extracted elements and relationships
    """
    path = Path(file_path)
    content = path.read_text(encoding='utf-8')
    lines = content.split('\n')

    # Extract form ID from title
    form_id = "unknown"
    for line in lines[:10]:
        match = FORM_ID_RX.search(line)
        if match:
            form_id = match.group(1)
            break

    ontology = ParsedOntology(
        source_file=str(path),
        form_id=form_id,
    )

    current_section_id = 0
    current_section_title = ""
    current_element: Optional[OntologyElement] = None
    current_block_lines: List[str] = []
    in_relationships_block = False

    def flush_element():
        """Flush current element with accumulated relationships."""
        nonlocal current_element, current_block_lines
        if current_element:
            rels = _extract_relationships_from_block(current_block_lines, current_element.name)
            current_element.relationships.extend(rels)
            ontology.elements.append(current_element)
        current_element = None
        current_block_lines = []

    for i, line in enumerate(lines):
        # Section header
        section_match = SECTION_HEADER_RX.match(line)
        if section_match:
            flush_element()
            current_section_id = int(section_match.group(1))
            current_section_title = section_match.group(2).strip()
            in_relationships_block = False
            continue

        # Subsection header (element definition)
        subsection_match = SUBSECTION_HEADER_RX.match(line)
        if subsection_match:
            flush_element()
            header_type = subsection_match.group(1)  # Rule/Window/Threshold/etc.
            element_name = subsection_match.group(2).strip()

            current_element = OntologyElement(
                name=element_name,
                element_type=_classify_element_type(element_name, header_type),
                section_id=current_section_id,
                section_title=current_section_title,
            )
            in_relationships_block = False
            continue

        # Check for "Relationships" or "Entities" subheader
        if line.strip().lower() in ("**relationships**", "**entities**", "relationships", "entities"):
            in_relationships_block = True
            continue

        # Collect lines for relationship extraction
        if current_element and (in_relationships_block or SIMPLE_REL_RX.search(line)):
            current_block_lines.append(line)
            continue

        # Entity list items (standalone entities)
        entity_match = ENTITY_RX.match(line)
        if entity_match and not current_element:
            entity_name = entity_match.group(1).strip()
            # Skip common non-entity headers
            if entity_name.lower() not in ("relationships", "entities", "core concept node"):
                ontology.elements.append(OntologyElement(
                    name=entity_name,
                    element_type=OntologyElementType.ENTITY,
                    section_id=current_section_id,
                    section_title=current_section_title,
                ))

    # Flush final element
    flush_element()

    return ontology


def extract_predicate_vocabulary(ontology: ParsedOntology) -> Dict[str, List[str]]:
    """
    Extract a vocabulary of predicates organized by semantic category.

    Returns:
        Dict mapping category to list of predicates
    """
    categories = {
        "temporal": [],        # when, effect, length_days, starts_offset_from
        "regulatory": [],      # governed_by, defined_by, subject_to
        "conditional": [],     # when, if_*, requires, applies_*
        "compositional": [],   # includes, excludes, contains, portion_of
        "reporting": [],       # reported_on, reported_in, required_when
        "classification": [],  # classified_as, is_a, type_of
        "other": [],
    }

    temporal_keywords = {"when", "effect", "length", "offset", "days", "period", "window"}
    regulatory_keywords = {"governed", "defined", "subject", "section", "irc", "treas"}
    conditional_keywords = {"if", "when", "requires", "applies", "triggers", "condition"}
    compositional_keywords = {"includes", "excludes", "contains", "portion", "part", "subset"}
    reporting_keywords = {"reported", "required", "filed", "furnished", "statement"}
    classification_keywords = {"classified", "is_a", "type", "category", "treated"}

    for pred in ontology.all_predicates:
        pred_lower = pred.lower()

        if any(kw in pred_lower for kw in temporal_keywords):
            categories["temporal"].append(pred)
        elif any(kw in pred_lower for kw in regulatory_keywords):
            categories["regulatory"].append(pred)
        elif any(kw in pred_lower for kw in conditional_keywords):
            categories["conditional"].append(pred)
        elif any(kw in pred_lower for kw in compositional_keywords):
            categories["compositional"].append(pred)
        elif any(kw in pred_lower for kw in reporting_keywords):
            categories["reporting"].append(pred)
        elif any(kw in pred_lower for kw in classification_keywords):
            categories["classification"].append(pred)
        else:
            categories["other"].append(pred)

    return categories


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ontology_parser.py <ontology_file.md>")
        sys.exit(1)

    ontology = parse_ontology(sys.argv[1])

    print(f"\n=== Ontology Summary ===")
    for k, v in ontology.summary().items():
        print(f"  {k}: {v}")

    print(f"\n=== Elements by Type ===")
    print(f"  Rules ({len(ontology.rules)}):")
    for r in ontology.rules[:10]:
        print(f"    - {r.name} ({len(r.relationships)} relationships)")

    print(f"\n  Windows ({len(ontology.windows)}):")
    for w in ontology.windows:
        print(f"    - {w.name}")

    print(f"\n=== Predicate Vocabulary ===")
    vocab = extract_predicate_vocabulary(ontology)
    for cat, preds in vocab.items():
        if preds:
            print(f"  {cat}: {preds}")

    print(f"\n=== Sample Relationships ===")
    for rel in ontology.all_relationships[:20]:
        neg = " (NEGATED)" if rel.is_negated else ""
        print(f"  {rel.source} —({rel.predicate})—> {rel.target}{neg}")
