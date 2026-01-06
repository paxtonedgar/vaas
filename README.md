# VaaS - Tax Document Intelligence Pipeline

Graph-based RAG system for IRS tax documents (1099-DIV and related forms) using:
- Knowledge graph extraction from PDF documents
- Fine-tuned embeddings via contrastive learning on graph-derived training pairs
- Hybrid retrieval (BM25 + dense vector + cross-encoder reranking + graph expansion)

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Download spaCy model (optional, for NER)
python -m spacy download en_core_web_sm

# 4. Place PDF files in data/ directory
mkdir -p data
# Copy i1099div.pdf and f1099div.pdf to data/

# 5. Run the extraction pipeline
jupyter lab notebooks/01_pdf_extraction_pipeline.ipynb
```

## Project Structure

```
vaas/
├── src/vaas/              # Core Python modules
│   ├── extraction/        # PDF extraction pipeline
│   ├── utils/             # Shared utilities and regex patterns
│   └── __init__.py
├── notebooks/             # Jupyter notebooks (Databricks compatible)
├── scripts/               # Deployment and utility scripts
├── data/                  # Local PDF files (gitignored)
├── output/                # Generated outputs (gitignored)
├── requirements.txt       # All dependencies
└── setup.py               # Package installation
```

## Development

```bash
# Run tests
pytest

# Format code
black src/

# Type checking
mypy src/

# Lint
flake8 src/
```

## Databricks Deployment

Export notebooks and build wheel for Databricks:

```bash
# Convert all notebooks to Databricks format
python scripts/export_to_databricks.py --all

# Build wheel for cluster installation
python scripts/export_to_databricks.py --build-wheel

# Outputs in:
# - output/databricks/*.ipynb  (notebooks)
# - dist/*.whl                  (wheel for cluster)
```

## Documentation

See `CLAUDE.md` for detailed architecture, schema definitions, and implementation status.

Key design documents:
- `tax_rag_technical_overview.md` - Problem statement and architecture
- `tax_embedding_technical_overview.md` - Contrastive learning approach
- `tax_rag_implementation_reference.md` - Cell-by-cell implementation details
- `tax_rag_schema_catalog.md` - Delta table schemas

## Current Status

**Phase:** Implementation in progress via Databricks notebooks

- Cells 1-4: PDF extraction and element construction
- Cells 5-10: Pending - anchor timeline, section assembly, reference extraction
- Training pipeline: Not started
- Evaluation: Not started

See `CLAUDE.md` for full implementation status.
