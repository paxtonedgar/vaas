# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VaaS (Value-as-a-Service) Tax Document Intelligence Pipeline**

This project builds a graph-based RAG system for IRS tax documents (1099-DIV and related forms) using:
- Knowledge graph extraction from PDF documents
- Fine-tuned embeddings via contrastive learning on graph-derived training pairs
- Hybrid retrieval (BM25 + dense vector + cross-encoder reranking + graph expansion)

**Current Status:** Documentation phase complete. Implementation now advancing through the CLI pipeline (`run_pipeline_v2.py`), replacing the original notebook workflow.

**Epic References:** UTILITIESPLATFORM-5327, UTILITIESPLATFORM-5326

## Architecture Overview

### The Core Insight

Tax documents fail standard RAG because:
1. **Cross-referential structure**: "Box 1a includes amounts in boxes 1b and 2e" - embeddings don't capture these dependencies
2. **Multi-perspective content**: Same box described in filer vs. recipient instructions needs alignment
3. **Precise terminology**: "Box 2e" vs "Box 2f" are one character apart but completely different semantically

**Solution:** Build a knowledge graph that captures document structure, then use it for both training pair generation and retrieval augmentation.

### Pipeline Architecture

```
PDF → Extraction → Knowledge Graph → Training Pairs → Fine-tuned Embeddings
                         ↓
                   Retrieval (hybrid + graph expansion)
```

**Extraction Pipeline Stages (mirrors former notebook cells):**
- Cell 1-2: PDF → spans with font/position metadata → body size inference
- Cell 3-4: Spans → elements with role classification → anchor detection (boxes/sections)
- Cell 5-7: Anchor timeline → content assignment → section assembly
- Cell 8-9: Reference extraction (regex + LLM) → cross-reference edges
- Cell 10: Emit to Delta tables (`graph_nodes`, `graph_edges`, `box_registry`)
- Cell 11: Semantic claims → precedence → resolution outputs (`resolved_claims`, `resolution_groups`, `compiled_directives`, `constraints_resolved`)

**Knowledge Graph Schema:**
- **Nodes:** Chunks at various granularities (anchor, paragraph, sentence) with canonical IDs
- **Edges:** Typed relationships (`parent_of`, `includes`, `same_field`, `references_box`, etc.)
- **Registry:** Maps box keys to canonical IDs and aliases for cross-document alignment

**Training Strategy:**
- Generate pairs by traversing graph edges (hierarchical, cross-reference, same-field, hard negatives)
- Validate pair quality with LLM-as-judge (>90% valid required)
- Fine-tune BERT-class model (sentence-transformers) with contrastive loss

**Retrieval Pipeline:**
- Hybrid: BM25 + dense vector search with reciprocal rank fusion
- Cross-encoder reranking on top-100
- Graph expansion on top-20 reranked results (follow edges for context)
- Contextual prefixes: "Form 1099-DIV, Qualified Dividends (filer instructions): {text}"

## Infrastructure & Environment

**Platform:** Local Python 3.10 CLI (`run_pipeline_v2.py`) with git-tracked modules. Databricks remains the target runtime once pipeline parity is achieved.
**Storage:** Local `data/` directory for PDFs and `output/` / `output_v2/` for generated parquet/csv artifacts. Copy results to Unity Catalog volumes when syncing with Databricks.
**Vector Store:** Databricks Vector Search (planned deployment target)
**LLM:** Amazon Bedrock (Claude 3.5 Sonnet) for extraction assists, validation, pair generation

**Local File Locations:**
```
data/
├── i1099div.pdf          # Filer instructions (primary target)
├── f1099div.pdf          # Form PDF
└── ...

output/                   # graph_nodes.parquet, graph_edges.parquet, resolved_claims.parquet, compiled_directives.parquet, quality reports
output_v2/                # experimental runs
```

**Databricks Volume (when exporting artifacts):**
```
/Volumes/112557_prefetch_ctg_prd_exp/112557_prefetch_raw/irs_raw/
└── [IRS PDFs + derived tables]

Unity Catalog Tables:
catalog.schema.box_registry
catalog.schema.graph_nodes
catalog.schema.graph_edges
catalog.schema.training_pairs
catalog.schema.eval_queries
```

## Key Schemas

### graph_nodes
```
node_id, doc_id, doc_type (filer_instructions/recipient_instructions/form),
chunk_type (anchor/paragraph/conceptual), canonical_id (→ box_registry),
parent_node_id, depth, text, extraction_confidence, concepts[]
```

### graph_edges
```
edge_id, source_node_id, target_node_id,
edge_type (parent_of/includes/same_field/references_box/...),
direction (directed/bidirectional), confidence,
source_evidence, created_by (structural/regex/llm)
```

### box_registry
```
box_key (e.g., "box_1b"), canonical_id (e.g., "qualified_dividends"),
label, aliases[] (for natural language lookup)
```

## Critical Principles

### 1. LLM Provenance Framework

**Every LLM decision must cite evidence pointers back to source PDFs.**

LLMs are supervised operators, not sources of truth. They propose structure/edges/judgments, but must emit:
```json
{
  "decision": "...",
  "confidence": 0.0-1.0,
  "evidence": [
    {"doc_id": "...", "page": N, "element_id": "...", "quote": "..."}
  ],
  "unsupported_claims": []  // Must be empty to pass
}
```

Tier structure:
- **Tier 1 (deterministic):** Regex patterns, layout rules, registry lookup (confidence ~1.0)
- **Tier 2 (LLM proposes):** Ambiguous references, implicit relationships (must cite evidence)
- **Tier 3 (validation):** Checks LLM proposals against cited evidence, downgrades if unsupported

### 2. Edge Confidence & Gated Usage

All edges have `confidence`, `source_evidence`, `created_by` fields.

**Training:** Weight pairs by edge confidence. High-confidence edges contribute more to loss.

**Retrieval ranking:**
- Phase A: Use only structural edges (1.0 confidence) for ranking
- Phase B: After validating >90% precision on reference edges, add to ranking

### 3. Validation Checkpoints

Pipeline has gates at each stage. If checkpoint fails, stop and fix before proceeding:

| Stage | Check | Pass Criteria |
|-------|-------|---------------|
| Extraction | Role classification accuracy | >95% on sample |
| Anchor detection | Box coverage | 100% of expected boxes |
| Reference extraction | Valid references | >90% precision |
| Graph integrity | DAG, connectivity | No cycles, >95% connected |
| Pair quality | Valid pairs (LLM-as-judge) | >90% |
| Hard negatives | Not false negatives | <10% false negative rate |

### 4. Hard Negative Mining Strategy

**Problem:** Naive BM25 mining yields ~70% false negatives (Box 1b appears in Box 1a text because 1a references it).

**Solution:** BM25 + graph distance + positive-aware filtering
1. Find BM25 candidates (high lexical similarity)
2. Filter to graph-distant nodes (no path within 3 hops)
3. Check embedding similarity to known positives
4. If similarity to any positive > 0.85, discard (likely false negative)

## Current Implementation Status

**Completed (Cells 1-4):**
- ✅ Span extraction from PDF with PyMuPDF
- ✅ Body font size inference (9.0 for 1099-DIV)
- ✅ Element construction with role classification
- ✅ Anchor detection: 22 anchors found including box groups (14-16)

**In Progress (Cell 6):**
- ⚠️ Merge collision bug: duplicate element IDs from split operation
- Fix: Deduplicate before merge or ensure unique IDs during split

**Pending (Cells 7-10):**
- Content assignment to anchors
- Section assembly
- Reference extraction (regex + LLM)
- Graph emission to Delta

**Not Started:**
- Training pair generation
- Hard negative mining with validation
- Fine-tuning experiments
- Baseline retrieval metrics

## Evaluation Framework

**75-query evaluation set across 6 types:**
- Exact anchor (15): "What goes in Box 2e?"
- Concept (20): "What are qualified dividends?"
- Procedural (10): "How do I report foreign tax paid?"
- Scenario (15): "I received REIT dividends. Are they qualified?" (multi-hop)
- Comparative (5): "Difference between Box 1a and Box 1b?"
- Edge case (10): "60-day holding period for qualified dividends?"

**Metrics:**
- Recall@5 (primary): Target >0.85
- Concept coverage (multi-hop): Target >0.80
- MRR: How quickly first relevant result appears

**Baseline:** Cohere Embed v3 + contextual prefixes + BM25 hybrid + cross-encoder rerank

Fine-tuning is only justified if it beats this strong baseline by >10% on Recall@5.

## Phased Delivery

**Phase A (Weeks 1-4):** Foundation
- Complete Cells 1-10, emit 1099-DIV graph to Delta
- Measure baseline retrieval metrics
- Exit: Graph integrity passes, baseline measured

**Phase B (Weeks 5-8):** Training
- Pair generation, hard negative mining, LLM validation
- Fine-tuning experiments
- Add graph ranking signal (after edge validation)
- Exit: Fine-tuned Recall@5 > baseline + 10%

**Phase C (Weeks 9-12):** Multi-form expansion
- 1099-INT, 1099-MISC extraction
- Cross-form concept nodes
- Exit: No regression on 1099-DIV metrics

## Open Questions (Answer with Numbers)

These dialectical challenges question whether we're overbuilding:

**Q: Do we need a KG at all?**
Contextual retrieval (Anthropic) shows that prefixes + hybrid + rerank gets substantial gains without graph. Experiment: Measure baseline with contextual chunks. If Recall@5 > 0.80, KG value is "incremental" not "necessary."

**Q: ColBERT/SPLADE vs. single-vector fine-tuning?**
ColBERT (token-level late interaction) and SPLADE (learned sparse) address "Box 2e" vs "Box 2f" precision without custom training. Benchmark both before committing to fine-tuning.

**Q: Structured routing vs. retrieval?**
Many queries are deterministic: "Explain Box 2a" → registry lookup → return section. Measure: What % of eval queries resolve via registry/alias alone? If >50%, build router first, fall back to retrieval for ambiguous queries.

**Q: Does complex extraction beat cheap baseline?**
Compare sophisticated pipeline (spans → elements → anchors with font heuristics) against cheap baseline (plain text + regex anchors + page order). If cheap achieves >90% accuracy, keep complex as fallback only.

## Regex Patterns Reference

```python
# Box detection
BOX_RX_SINGLE = re.compile(r"^Box\s*(\d+[a-z]?)\.?\s+", re.IGNORECASE)
BOX_RX_RANGE = re.compile(r"^Boxes?\s*(\d+[a-z]?)\s*(through|[-–])\s*(\d+[a-z]?)", re.IGNORECASE)

# Section headers
SECTION_HEADER_RX = re.compile(
    r"^(What|Who|When|How|General|Specific|Instructions|Qualified Dividends|RICs and REITs)",
    re.IGNORECASE
)

# References
BOX_REF_RX = re.compile(r"[Bb]ox(?:es)?\s*(\d+[a-z]?(?:\s*(?:,|and|through|[-–])\s*\d+[a-z]?)*)")
SECTION_REF_RX = re.compile(r"[Ss]ee\s+([A-Z][a-zA-Z\s]+?)(?:,\s*(earlier|later|above|below))?")
PUB_REF_RX = re.compile(r"[Pp]ub(?:lication)?\.?\s*(\d+)")
IRC_REF_RX = re.compile(r"[Ss]ection\s+(\d+[A-Za-z]?(?:\([a-z]\))?)")
```

## Documentation Structure

### Semantic Architecture (Primary - in `docs/`)

Start here for understanding the meta-model:

- **[docs/INDEX.md](docs/INDEX.md)**: Navigation index for all documentation
- **[docs/SEMANTIC_CORE.md](docs/SEMANTIC_CORE.md)**: **12 Primitives** - The minimal semantic core from which all tax reporting logic composes
- **[docs/TAX_METAMODEL_DESIGN.md](docs/TAX_METAMODEL_DESIGN.md)**: Integration architecture - how primitives map to graph schema, atoms, bindings
- **[docs/Form_1099DIV_Ontology.md](docs/Form_1099DIV_Ontology.md)**: 1099-DIV semantic model - entities, rules, relationships

### Key Design Principles

1. **IRS concepts are instances, not types** - "Qualified Dividend" is a Qualification instance, not a Classification enum
2. **Semantic Atoms prevent diffusion** - "61-day holding" defined once, referenced everywhere
3. **Form bindings decouple semantics** - `ordinary_dividends_total` binds to "1099-DIV:Box 1a"
4. **Exclusions are explicit** - Dividend ⊕ Interest is asserted, not just avoided by precedence

### Legacy Docs (in repository root)

- **tax_rag_technical_overview.md**: Problem statement, solution architecture, high-level design
- **tax_embedding_technical_overview.md**: Deep dive on contrastive learning and graph-based pair generation
- **tax_rag_implementation_reference.md**: Cell-by-cell implementation status, code snippets, known issues
- **tax_rag_schema_catalog.md**: Complete schema definitions with ER diagrams and SQL examples
- **tax_embedding_implementation_plan.md**: 10-phase implementation plan with function specs
- **tax_rag_living_strategy.md**: Risk analysis, LLM provenance principles, dialectical challenges

When implementing features, start with the semantic architecture docs, then cross-reference legacy docs for context.

## Development Workflow

### CLI-Based Development

Primary development now happens in standard Python modules plus the orchestration script. The workflow:

1. **Local Development**: Edit modules under `src/vaas/`, keep tests in `tests/`, and expose new knobs through `PipelineConfig` inside `run_pipeline_v2.py`.
2. **Validation**: Run `python -m vaas.run_pipeline_v2 --pdf data/i1099div.pdf --output output --validate` to regenerate outputs and reports.
3. **Graph Checks**: Inspect `output/graph_quality_report.md` / `.json` and run `python validate_graph.py` before sharing artifacts or syncing with Databricks.

### Local Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies + tooling
make install-dev

# Place PDFs in data/ directory
mkdir -p data
# Copy i1099div.pdf, f1099div.pdf to data/
```

### Project Structure

```
vaas/
├── run_pipeline_v2.py     # CLI orchestrator (replaces notebooks)
├── validate_graph.py      # Consistency checker for emitted graph
├── src/vaas/
│   ├── extraction/        # PDF parsing, layout, anchors, sections
│   ├── semantic/          # Regime detection, concept roles
│   ├── graph/             # Node/edge builders
│   ├── core/              # Primitives, atoms, bindings
│   └── utils/             # Constants, regexes, shared helpers
├── tests/                 # Pytest suites mirroring src layout
├── docs/                  # Architectural references
├── output/, output_v2/    # Generated artifacts (gitignored)
├── AGENTS.md, README.md   # Contributor docs
└── requirements.txt       # Runtime dependencies
```

### Common Tasks

```bash
make install-dev    # Install runtime + tooling dependencies
make pipeline       # Run run_pipeline_v2.py with default arguments
make test           # Run pytest
make lint           # flake8 (100-char lines, ignore W503)
make format         # black across src/, tests/, run_pipeline_v2.py, validate_graph.py
make typecheck      # mypy src/
make clean          # Remove build artifacts and caches
```

## Git Status Note

Current branch has uncommitted changes in sibling property-service project. This is expected - vaas is early-stage documentation while property-service is active development.
