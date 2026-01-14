# VaaS - Tax Document Intelligence Pipeline

Graph-centered RAG system for IRS tax documents (currently focused on 1099-DIV). The project extracts structured nodes/edges from PDF guidance, aligns them with semantic primitives, and emits artifacts that power retrieval and downstream QA flows. Notebook prototyping is complete—the repository now contains the canonical Python implementation and orchestration script.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install the package in editable mode (installs runtime deps)
pip install -e .

# 3. Add sample PDFs
mkdir -p data
cp /path/to/i1099div.pdf data/

# 4. Run the pipeline
python -m vaas.run_pipeline_v2 --pdf data/i1099div.pdf --output output --validate
```

## Running the Pipeline

### Basic Usage

```bash
# Run extraction pipeline on a PDF
python -m vaas.run_pipeline_v2 --pdf data/i1099div.pdf --output output

# Run with validation checks
python -m vaas.run_pipeline_v2 --pdf data/i1099div.pdf --output output --validate
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pdf` | Path to input PDF file | Required |
| `--output` | Output directory for artifacts | `output/` |
| `--validate` | Run validation checks after extraction | Off |

### Pipeline Stages

The pipeline runs through these stages:

1. **Spans** - Extract text spans from PDF with font/position metadata
2. **Lines** - Aggregate spans into lines with layout features
3. **Layout Detection** - Identify subsection candidates and split triggers
4. **Elements** - Split blocks into semantic elements with role classification
5. **Anchors** - Extract box/section/subsection anchors from headers
6. **Timeline** - Build reading order ranges for content assignment
7. **Sections** - Materialize sections with header/body text
8. **References** - Extract cross-references (Box X, Pub Y, IRC §Z)
9. **Graph** - Build nodes and edges for knowledge graph
10. **Validation** - Run quality checks (if `--validate` flag set)

### Output Artifacts

After running, the output directory contains:

```
output/
├── graph_nodes.parquet      # All graph nodes (doc_root, sections, boxes, paragraphs)
├── graph_edges.parquet      # All graph edges (parent_of, follows, in_section, etc.)
├── sections.parquet         # Materialized sections with text content
├── anchors.parquet          # Extracted anchors (boxes, sections, subsections)
├── references.parquet       # Cross-references found in text
├── semkg_manifest.json      # Pipeline metadata and schema version
└── graph_quality_report.md  # Validation results (if --validate)
```

## Validation & Evaluation

### Running Validation

```bash
# Validate existing output (after running pipeline)
python -m vaas.validate_graph

# Validate a specific output directory
python -m vaas.validate_graph --output output_v2
```

### Validation Checks

The validator runs these check categories:

**Phase A: Anchor Coverage**
| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| A1 | Box coverage | All 22 expected boxes found |
| A2 | Artifact contamination | No page artifacts in sections |
| A3 | Monolith detection | No section > 4000 chars |
| A4 | Edge integrity | All edge endpoints exist |
| A5 | Skeleton coverage | >95% elements assigned |
| A6 | Provenance | All edges have source_evidence |
| A7 | Hierarchy integrity | No cycles, single root |
| A8 | Edge distribution | Reasonable edge type mix |

**Phase C: Semantic Checks** (when ontology available)
| Check | Description |
|-------|-------------|
| C1 | Ontology coverage | Claims map to primitives |
| C2 | Constraint validation | No conflicting rules |
| C3 | Authority chain | All claims have sources |

### Expected Validation Output

```
============================================================
VALIDATION RESULTS
============================================================

✅ A1: Anchor Coverage - 0 errors, 0 warnings
✅ A2: Artifact Contamination - 0 errors, 0 warnings
✅ A3: Monolith Detection - 0 errors, 1 warnings
✅ A4: Edge Integrity - 0 errors, 0 warnings
✅ A5: Skeleton Coverage - 0 errors, 0 warnings
✅ A6: Provenance - 0 errors, 0 warnings
✅ A7: Hierarchy Integrity - 0 errors, 0 warnings
✅ A8: Edge Distribution - 0 errors, 0 warnings

Overall Status: PASSED
Checks: 8/8 passed
```

### Quick Validation Script

For a quick structural validation without the full validator:

```python
import sys
sys.path.insert(0, 'src')

from vaas.extraction.anchors import validate_box_coverage, EXPECTED_BOXES_1099DIV
import pandas as pd

# Load anchors from output
anchors_df = pd.read_parquet('output/anchors.parquet')

# Validate box coverage
result = validate_box_coverage(anchors_df, EXPECTED_BOXES_1099DIV)
print(f"Boxes: {len(result.found)}/22 - {'PASS' if result.passed else 'FAIL'}")
if result.missing:
    print(f"Missing: {result.missing}")
```

## Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/extraction/test_anchors.py -v

# Run with coverage
pytest --cov=vaas tests/
```

## Project Structure

```
vaas/
├── src/vaas/
│   ├── run_pipeline_v2.py  # CLI entry point (python -m vaas.run_pipeline_v2)
│   ├── validate_graph.py   # Validation CLI (python -m vaas.validate_graph)
│   ├── __init__.py
│   ├── extraction/         # PDF parsing, layout analysis, anchors, sections
│   ├── graph/              # Node/edge builders
│   ├── semantic/           # Regime/role detection, concept mapping
│   ├── core/               # Semantic primitives, bindings, operators
│   └── utils/              # Shared constants and helpers
├── tests/                  # Pytest suites mirroring src layout
├── docs/                   # Design references (see docs/INDEX.md)
├── output/                 # Generated artifacts (gitignored)
├── output_v2/              # Experimental outputs (gitignored)
├── AGENTS.md               # Contributor workflow guide
├── CLAUDE.md               # Extended architecture & instructions
└── requirements.txt        # Base + dev dependencies
```

## Development Workflow

Key commands (also exposed through the `Makefile`):

```bash
# Install editable package with dev extras + spaCy model
make install-dev

# Run tests / lint / formatting / type checks
make test
make lint
make format
make typecheck

# Clean build artifacts
make clean
```

`make format` runs Black over `src/` and `tests/`; `make lint` uses flake8 with a 100-character budget and `W503` ignored.

## Documentation Map

- `docs/INDEX.md` – navigation hub and implementation status
- `docs/SEMANTIC_CORE_V3.md` – current 18-primitive ontology (supersedes earlier versions)
- `docs/Form_1099DIV_Ontology.md` – form-specific schema
- `docs/tax_rag_*` – legacy technical references, embedding plans, and schema catalogs
- `CLAUDE.md` – in-repo playbook for architecture, workflows, and outstanding tasks

Outputs such as `output/graph_quality_report.md` and `output/ontology_coverage_report.md` summarize pipeline validation runs. Update those files whenever the pipeline or schemas change materially.
