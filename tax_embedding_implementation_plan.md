# Tax Document Embedding Pipeline: Implementation Plan

## Overview

Fine-tune BERT-class embedding model for semantic retrieval over IRS tax forms and instructions. Handle cross-referential document structure through graph-based chunking and hierarchical pair generation.

### Document Types (per form)
1. **Form PDF** - Box labels, field structure, embedded recipient instructions
2. **Filer Instructions PDF** - Detailed guidance for form preparers
3. **Recipient Instructions** - Embedded in form PDF (Copy B), condensed guidance for taxpayers

### Expansion Pattern
1. 1099-DIV only → validate pipeline
2. Expand to 3 forms (1099-DIV, 1099-INT, 1099-MISC)
3. Expand to full corpus

---

## Infrastructure & Tools

### Available
| Tool | Purpose |
|------|---------|
| Databricks notebooks | Compute, orchestration |
| Databricks Delta Lake | Intermediate storage |
| Databricks Vector Search | Vector store (preferred over external OpenSearch) |
| Unity Catalog | Data governance |
| MLflow | Experiment tracking |
| Amazon Bedrock | LLM access (Claude 3.5 Sonnet) |
| Azure OpenAI | Backup LLM access |

### OSS Libraries
| Library | Purpose |
|---------|---------|
| PyMuPDF (fitz) | PDF text extraction with font/position metadata |
| pdfplumber | Table extraction from forms |
| spaCy | NER, custom entity patterns |
| NetworkX | Graph construction and analysis |
| sentence-transformers | Embedding model training |
| FAISS | Local vector search (fallback) |

### Not Available (requires approval)
- Amazon Textract
- Amazon Comprehend
- Amazon OpenSearch Service

---

## Phase 1: PDF Extraction & Normalization

### File: `01_pdf_extraction.py`

### Purpose
Extract raw text and structural metadata from IRS PDFs. Normalize into consistent chunk schema. Handle the messiness of real PDF parsing.

### Pre-Processing Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `download_irs_pdf(form_id, doc_type)` | Fetch PDFs from IRS | URLs: `irs.gov/pub/irs-pdf/f{form_id}.pdf`, `i{form_id}.pdf` |
| `compute_content_hash(pdf_bytes)` | SHA256 hash for incremental processing | Skip re-extraction if hash unchanged |
| `check_existing_extraction(hash)` | Query Delta for existing extraction | Returns chunks if already processed |

### Text Normalization Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `normalize_unicode(text)` | Normalize unicode characters | NFKC normalization, replace smart quotes/dashes |
| `remove_page_artifacts(text, page_num)` | Strip headers, footers, page numbers | IRS has consistent header patterns |
| `normalize_whitespace(text)` | Collapse whitespace, preserve paragraph breaks | Regex-based |
| `detect_and_remove_duplicates(pages)` | Dedupe repeated content across form copies | Copy A, B, 1, 2 have identical text |

### Form Extraction Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `extract_form_tables(pdf_path)` | Extract box structure using pdfplumber | Returns box labels, positions, relationships |
| `extract_box_metadata(table_cells)` | Parse box numbers, labels from table cells | Handle "1a", "2b", etc. patterns |
| `extract_recipient_instructions(pdf_path)` | Find and extract "Instructions for Recipient" block | Located after Copy B, distinct formatting |
| `segment_recipient_instructions(text)` | Split recipient instructions by box reference | Each "Box X." starts a segment |

### Instructions Extraction Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `extract_with_formatting(pdf_path)` | Extract text with font size, weight, position | PyMuPDF `get_text("dict")` |
| `detect_section_boundaries(blocks)` | Identify section/subsection breaks | Use font size changes, all-caps patterns |
| `build_section_hierarchy(sections)` | Construct parent-child relationships | Track nesting depth |
| `extract_special_blocks(blocks)` | Identify TIP, CAUTION, NOTE boxes | These have distinct formatting |
| `handle_page_continuations(sections)` | Merge sections split across pages | Match partial sentences |

### Chunk Assembly Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `create_chunk(text, metadata)` | Assemble chunk object with full metadata | Standard schema |
| `assign_canonical_id(chunk)` | Generate canonical ID for cross-doc linking | "box_1a", "section_qualified_dividends" |
| `validate_chunk(chunk)` | Check chunk meets quality thresholds | Non-empty, reasonable length, valid metadata |
| `compute_extraction_confidence(chunk)` | Score extraction quality | Based on formatting clarity, completeness |

### Chunk Schema
```python
{
    "chunk_id": str,              # Unique: "1099div_filer_box_1a_instructions"
    "canonical_id": str,          # Shared: "box_1a"
    "doc_id": str,                # "1099-DIV"
    "doc_type": str,              # "form" | "filer_instructions" | "recipient_instructions"
    "doc_version": str,           # "Rev. January 2024"
    "level": str,                 # "box" | "section" | "subsection" | "paragraph" | "special_block"
    "parent_chunk_id": str,       # Hierarchical parent
    "text": str,                  # Normalized extracted text
    "text_raw": str,              # Original text before normalization
    "page": int,
    "position": {                 # If available
        "x0": float, "y0": float,
        "x1": float, "y1": float
    },
    "formatting": {               # If available
        "font_size": float,
        "is_bold": bool,
        "is_header": bool
    },
    "extraction_confidence": float,
    "content_hash": str,          # For deduplication
    "extracted_at": timestamp,
    "extraction_version": str     # Pipeline version
}
```

### Outputs
- Delta table: `catalog.tax_embeddings.raw_chunks`
- Delta table: `catalog.tax_embeddings.extraction_log` (run metadata, errors)
- Partitioned by: `doc_id`, `doc_type`

### Validation Checks
- [ ] All 16 boxes extracted from form
- [ ] Recipient instructions segmented correctly (one chunk per box reference)
- [ ] Filer instructions hierarchy depth is reasonable (2-4 levels)
- [ ] No duplicate chunks (by content_hash)
- [ ] Extraction confidence > 0.8 for all chunks
- [ ] Parent-child relationships form valid tree

### Known Edge Cases
1. **Multi-line box labels** - Box 2e/2f have long labels that wrap
2. **Nested tables** - Form has tables within tables
3. **Section continuations** - "Qualified Dividends" spans multiple pages
4. **Embedded tips** - TIP/CAUTION boxes interrupt flow
5. **IRC references** - "section 1202" is a reference, not a section header

---

## Phase 2: Entity & Reference Extraction

### File: `02_reference_extraction.py`

### Purpose
Extract all references (box numbers, section names, external documents) from chunk text. Use hybrid approach: spaCy patterns for high-confidence extraction, LLM for comprehensive coverage.

### Why Not Pure Regex
IRS documents have inconsistent reference patterns:
- "Box 1a" vs "box 1a" vs "Box 1a."
- "boxes 2b, 2c, 2d, and 2f" (comma-separated list)
- "boxes 2b through 2f" (implied range)
- "(see box 4)" (parenthetical)
- "the amount in box 1a that is section 1202 gain" (nested)
- "See Qualified Dividends, earlier" (section reference by name)

Regex catches ~60% cleanly. LLM catches the rest.

### Entity Types
```python
ENTITY_TYPES = {
    "box_reference": "Box 1a, boxes 2b-2f",
    "section_reference": "See Qualified Dividends, earlier",
    "form_reference": "Form 1099-R, Form W-9",
    "publication_reference": "Pub. 550, Publication 1179",
    "irc_reference": "section 1202, section 199A",
    "regulation_reference": "Regulations section 1.199A-3",
    "notice_reference": "Notice 2003-71"
}
```

### spaCy Pattern Setup

| Function | Purpose | Notes |
|----------|---------|-------|
| `create_box_patterns()` | EntityRuler patterns for box references | Handle variations, lists, ranges |
| `create_section_patterns()` | Patterns for "See X, earlier/later" | Named section references |
| `create_external_patterns()` | Patterns for forms, pubs, IRC sections | Structured external references |
| `build_custom_nlp()` | Assemble spaCy pipeline with custom patterns | Return configured nlp object |

### spaCy Extraction Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `extract_entities_spacy(chunk, nlp)` | Run spaCy NER on chunk text | Returns entity spans with types |
| `resolve_box_lists(entities)` | Expand "boxes 2b, 2c, 2d" to individual refs | Handle comma lists |
| `resolve_box_ranges(entities)` | Expand "boxes 2b through 2f" to individual refs | Handle range expressions |
| `normalize_entity(entity)` | Standardize entity format | "box 1a" → "box_1a" |

### LLM Extraction Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `extract_entities_llm(chunk, existing_entities)` | LLM pass for entities spaCy missed | Pass existing to avoid duplicates |
| `batch_chunks_for_llm(chunks, max_tokens)` | Group chunks to fit context window | ~3000 tokens per batch |
| `parse_llm_entity_response(response)` | Parse structured JSON from LLM | Handle malformed responses |
| `validate_entity_exists(entity, all_chunks)` | Verify referenced target exists | Catch hallucinated references |

### LLM Prompt (Entity Extraction)
```
You are extracting references from IRS tax document text.

Text:
{chunk_text}

Already extracted (do not duplicate):
{existing_entities}

Extract ALL references to:
1. Box numbers (e.g., "Box 1a", "boxes 2b through 2f")
2. Document sections (e.g., "See Qualified Dividends, earlier")
3. Other IRS forms (e.g., "Form 1099-R")
4. IRS publications (e.g., "Pub. 550")
5. IRC sections (e.g., "section 1202")
6. Regulations (e.g., "Regulations section 1.199A-3")

Return JSON:
{
  "entities": [
    {
      "text": "original text as it appears",
      "type": "box_reference|section_reference|form_reference|...",
      "normalized": "standardized form (e.g., box_1a)",
      "start_char": int,
      "end_char": int
    }
  ]
}

If no additional entities found, return {"entities": []}
```

### Merge & Deduplication Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `merge_entity_sources(spacy_entities, llm_entities)` | Combine extractions, dedupe by span | Prefer spaCy for overlaps (higher precision) |
| `flag_conflicts(merged_entities)` | Identify contradictory extractions | For manual review |
| `compute_entity_confidence(entity, source)` | Score confidence by extraction source | spaCy patterns > LLM |

### Entity Schema
```python
{
    "entity_id": str,
    "chunk_id": str,              # Source chunk
    "text_original": str,         # As extracted
    "text_normalized": str,       # Standardized (e.g., "box_1a")
    "entity_type": str,
    "start_char": int,
    "end_char": int,
    "extraction_source": str,     # "spacy" | "llm"
    "confidence": float,
    "resolved_target": str,       # Target chunk_id if internal reference
    "is_external": bool,          # True for forms, pubs, IRC refs
    "extracted_at": timestamp
}
```

### Outputs
- Delta table: `catalog.tax_embeddings.extracted_entities`
- Delta table: `catalog.tax_embeddings.extraction_conflicts` (for review)

### Validation Checks
- [ ] Box 1a chunk has entities referencing 1b, 2e, 6 (per "includes" statement)
- [ ] All box_reference entities resolve to existing chunks
- [ ] Entity extraction coverage > 95% of known references (manual sample)
- [ ] Conflict rate < 5%

---

## Phase 3: Cross-Document Alignment

### File: `03_cross_doc_alignment.py`

### Purpose
Link chunks across document types (form, filer instructions, recipient instructions) that discuss the same field or concept.

### Alignment Types
1. **Box alignment** - Same box number across all three doc types
2. **Section-to-box alignment** - Filer instruction section → related boxes
3. **Concept alignment** - Same concept discussed differently in filer vs recipient

### String-Based Alignment Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `align_by_canonical_id(chunks)` | Group chunks with matching canonical_id | Direct match: "box_1a" in all doc_types |
| `validate_box_alignment(alignment)` | Check aligned chunks discuss same topic | Quick semantic check |
| `find_missing_alignments(chunks)` | Identify boxes present in one doc_type but not others | Gap detection |

### Embedding-Based Alignment Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `load_alignment_model()` | Load off-the-shelf embedding model | Use `bge-base-en-v1.5` or similar |
| `embed_chunks_batch(chunks, model)` | Generate embeddings for alignment | Batch for efficiency |
| `find_similar_chunks(chunk, candidates, threshold)` | Cosine similarity for fuzzy matching | threshold ~0.75 |
| `align_sections_to_boxes(section_chunks, box_chunks)` | Find which sections elaborate which boxes | Section may cover multiple boxes |

### LLM Validation Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `validate_alignment_llm(chunk_a, chunk_b)` | Confirm fuzzy match is valid | Binary yes/no with reasoning |
| `classify_alignment_type(chunk_a, chunk_b)` | Determine relationship type | same_field, elaborates, etc. |
| `batch_validation_requests(alignments)` | Group for efficient LLM calls | ~10 alignments per call |

### LLM Prompt (Alignment Validation)
```
Do these two chunks from IRS 1099-DIV documents discuss the same tax concept or form field?

Chunk A ({doc_type_a}):
{text_a}

Chunk B ({doc_type_b}):
{text_b}

Return JSON:
{
  "same_topic": bool,
  "relationship": "same_field|elaborates|related|unrelated",
  "confidence": float (0-1),
  "reasoning": "brief explanation"
}
```

### Edge Creation Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `create_alignment_edges(validated_alignments)` | Generate edge records | Include confidence, source |
| `determine_edge_direction(chunk_a, chunk_b)` | Set source/target based on doc_type | filer elaborates recipient |

### Alignment Edge Types
- `same_field` - Same box/field across doc types (bidirectional)
- `elaborates` - Filer instructions elaborate recipient instructions (directed)
- `summarizes` - Recipient instructions summarize filer (directed)
- `related` - Different aspects of same concept (bidirectional)

### Outputs
- Delta table: `catalog.tax_embeddings.alignment_edges`
- Report: Alignment coverage by box number

### Validation Checks
- [ ] All 16 boxes have alignment across doc types (where applicable)
- [ ] No orphan recipient instruction chunks (all should align to filer)
- [ ] LLM validation agrees with string matching > 90% of time
- [ ] Manual review of 10 random alignments

---

## Phase 4: Intra-Document Reference Graph

### File: `04_reference_graph.py`

### Purpose
Build edges for references within documents: "see also", "includes", "exception to", hierarchical relationships.

### Edge Classification

| Reference Pattern | Edge Type | Direction |
|-------------------|-----------|-----------|
| "Box 1a includes amounts in boxes 1b and 2e" | `includes` | 1a → 1b, 1a → 2e |
| "See Qualified Dividends, earlier" | `see_also` | source → Qualified Dividends |
| "The following are not qualified dividends" | `exception_to` | exceptions → Qualified Dividends |
| Section contains subsection | `parent_of` | section → subsection |
| Definition used in box instructions | `defines` | definition → usage |

### Reference-to-Edge Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `load_extracted_entities(doc_id)` | Load entities from Phase 2 | Filter to internal references |
| `resolve_entity_to_chunk(entity, chunks)` | Map entity to target chunk_id | Handle ambiguous targets |
| `classify_reference_relationship(context, entity)` | Determine edge type from surrounding text | LLM-assisted classification |
| `create_reference_edge(source_chunk, target_chunk, edge_type)` | Generate edge record | Include context snippet |

### LLM Prompt (Relationship Classification)
```
Given this reference in an IRS document:

Source chunk: {source_text}
Reference: "{entity_text}"
Target chunk: {target_text}

What is the relationship between source and target?

Options:
- includes: source includes/contains amounts from target
- see_also: source points to target for additional information
- exception_to: source describes exceptions to rule in target
- defines: source defines a term used in target
- requires: source requires information from target
- other: describe the relationship

Return JSON:
{
  "relationship": "includes|see_also|exception_to|defines|requires|other",
  "description": "if other, describe",
  "confidence": float
}
```

### Hierarchy Edge Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `build_hierarchy_edges(chunks)` | Create parent_of edges from doc structure | Use parent_chunk_id from extraction |
| `validate_hierarchy(edges)` | Check for cycles, orphans | Should be DAG |
| `compute_hierarchy_depth(chunk, edges)` | Calculate nesting level | For weighting |

### Graph Assembly Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `merge_all_edges(reference_edges, hierarchy_edges, alignment_edges)` | Combine edge sources | Dedupe, handle conflicts |
| `build_networkx_graph(chunks, edges)` | Construct graph object | For analysis |
| `add_node_attributes(G, chunks)` | Attach chunk metadata to nodes | text, doc_type, level |
| `add_edge_attributes(G, edges)` | Attach edge metadata | type, confidence, source |

### Edge Schema
```python
{
    "edge_id": str,
    "source_chunk_id": str,
    "target_chunk_id": str,
    "edge_type": str,           # includes, see_also, exception_to, parent_of, same_field, elaborates, defines
    "direction": str,           # "directed" | "bidirectional"
    "confidence": float,
    "source_evidence": str,     # Text snippet that indicates relationship
    "created_by": str,          # "structure" | "spacy" | "llm" | "embedding_match"
    "doc_id": str,
    "created_at": timestamp
}
```

### Outputs
- Delta table: `catalog.tax_embeddings.graph_edges` (all edge types unified)
- NetworkX graph pickle: `dbfs:/graphs/{doc_id}_graph.pkl`

### Validation Checks
- [ ] Graph is connected (or < 5% isolated nodes)
- [ ] Box 1a has outgoing edges to 1b, 2e, 6
- [ ] Qualified Dividends section has edges to Box 1b, exceptions, RICs/REITs
- [ ] Hierarchy edges form valid DAG (no cycles)
- [ ] All edges have confidence > 0.5

---

## Phase 5: Conceptual Chunk Construction

### File: `05_conceptual_chunks.py`

### Purpose
Build LLM-synthesized conceptual chunks for major tax concepts. Conservative approach: only named sections/concepts.

### Concept Identification (Conservative)

For 1099-DIV, concepts are explicitly named sections:
- Qualified Dividends
- Section 897 gain
- Section 199A dividends
- Section 404(k) dividends
- Nondividend distributions
- Backup withholding (Box 4)
- Foreign tax credit (Box 7)
- Liquidation distributions (Boxes 9-10)
- FATCA reporting (Box 11)
- Exempt-interest dividends (Box 12-13)

### Concept Candidate Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `identify_named_concepts(chunks)` | Find chunks that define/name concepts | Section headers, definition patterns |
| `filter_to_conservative_candidates(concepts)` | Keep only explicitly named concepts | No inferred concepts |
| `validate_concept_has_substance(concept, chunks)` | Ensure concept spans multiple chunks | Single-chunk concepts don't need synthesis |

### Source Collection Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `collect_concept_sources(concept, G, max_hops=2)` | BFS from concept node to related chunks | Configurable depth limit |
| `filter_relevant_sources(concept, sources, llm)` | LLM filters to truly relevant sources | Remove tangentially related |
| `order_sources_by_relevance(sources)` | Sort for coherent synthesis | Definition first, then details, then exceptions |
| `check_context_window(sources, max_tokens=6000)` | Ensure sources fit context | Truncate or summarize if needed |

### Construction Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `construct_conceptual_chunk(concept, sources, llm)` | LLM synthesizes unified explanation | Strict source-faithful prompting |
| `enforce_inline_citations(constructed_text)` | Ensure every claim has [source_id] citation | Post-process if needed |
| `extract_citations(text)` | Parse citation markers from text | Return list of source references |

### LLM Prompt (Conceptual Construction)
```
You are synthesizing tax guidance from official IRS source documents.
Use ONLY information from the provided sources. Do not add external knowledge.

Concept: {concept_name}

Source Documents (cite as [1], [2], etc.):
[1] {source_1_text}
[2] {source_2_text}
...

Instructions:
1. Synthesize a unified explanation of this concept
2. Cite every factual claim with [source_number]
3. Preserve ALL specific numbers, thresholds, dates, percentages
4. If filer and recipient perspectives differ, note both
5. If sources conflict or are ambiguous, state the ambiguity explicitly
6. Do not interpret, infer, or add context beyond the sources

Format:
- Start with a clear definition
- Cover key requirements and rules
- Note exceptions
- Explain how it appears on the form (which boxes)

Maximum length: 500 words
```

### Faithfulness Validation Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `validate_faithfulness(constructed, sources, llm)` | Check for unsupported claims | Different LLM call than construction |
| `extract_claims(text)` | Split constructed text into individual claims | For per-claim validation |
| `check_claim_support(claim, sources)` | Verify claim is supported by a source | Return supporting source or flag |
| `compute_faithfulness_score(validation_result)` | Aggregate score | % of claims supported |

### LLM Prompt (Faithfulness Validation)
```
Compare this constructed explanation to its source documents.

Constructed explanation:
{constructed_text}

Source documents:
[1] {source_1}
[2] {source_2}
...

For each claim in the constructed explanation:
1. Is it supported by a source? Which one?
2. Is it accurately represented (not distorted)?
3. Are there source details omitted that change meaning?

Return JSON:
{
  "overall_faithful": bool,
  "claims": [
    {
      "claim_text": "...",
      "supported": bool,
      "supporting_source": "[1]" or null,
      "accurate": bool,
      "notes": "..."
    }
  ],
  "omissions": ["important source info not included"],
  "additions": ["info added not in sources"]
}
```

### Derived-From Edge Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `create_derived_from_edges(conceptual_chunk, source_ids)` | Link concept to sources | Edge type: `derived_from` |
| `add_to_graph(G, conceptual_chunk, edges)` | Add concept node and edges to graph | Update NetworkX object |

### Conceptual Chunk Schema
```python
{
    "chunk_id": str,                  # "concept_qualified_dividends"
    "chunk_type": "conceptual",
    "concept_name": str,
    "doc_id": str,
    "text": str,                      # LLM-constructed
    "source_chunk_ids": [str],
    "citations": [
        {"position": int, "source_chunk_id": str}
    ],
    "faithfulness_score": float,
    "faithfulness_details": {
        "supported_claims": int,
        "total_claims": int,
        "omissions": [str],
        "additions": [str]
    },
    "flags": [str],                   # Any warnings
    "constructed_at": timestamp,
    "model_used": str,
    "prompt_version": str
}
```

### Outputs
- Delta table: `catalog.tax_embeddings.conceptual_chunks`
- Updated graph pickle with concept nodes
- Validation report: `dbfs:/reports/{doc_id}_conceptual_validation.json`

### Validation Checks
- [ ] All conceptual chunks have faithfulness_score > 0.95
- [ ] All claims have supporting citations
- [ ] No additions flagged (no external knowledge added)
- [ ] Manual review of each conceptual chunk (small number, ~10)

---

## Phase 6: Graph Validation & Evaluation

### File: `06_graph_eval.py`

### Purpose
Comprehensive validation before proceeding to pair generation. Gate checkpoint.

### Graph Structure Metrics

| Function | Purpose | Target |
|----------|---------|--------|
| `compute_connectivity(G)` | % of nodes in largest connected component | > 95% |
| `count_isolated_nodes(G)` | Nodes with no edges | < 5% of total |
| `compute_density(G)` | Edge count / possible edges | Document for comparison |
| `compute_avg_degree(G)` | Average edges per node | Document for comparison |
| `compute_clustering_coefficient(G)` | Local clustering | Document for comparison |

### Coverage Metrics

| Function | Purpose | Target |
|----------|---------|--------|
| `check_box_coverage(G, expected_boxes)` | All boxes have nodes | 100% |
| `check_cross_doc_coverage(G)` | All boxes aligned across doc types | > 90% |
| `check_concept_coverage(G)` | All major concepts have conceptual chunks | 100% of conservative list |
| `check_edge_type_distribution(G)` | Distribution of edge types | No type should dominate unfairly |

### Quality Metrics

| Function | Purpose | Target |
|----------|---------|--------|
| `compute_avg_edge_confidence(G)` | Mean confidence across edges | > 0.7 |
| `count_low_confidence_edges(G, threshold=0.5)` | Edges needing review | < 10% |
| `check_hierarchy_validity(G)` | Hierarchy edges form valid DAG | No cycles |
| `check_bidirectional_consistency(G)` | Bidirectional edges have both directions | 100% |

### Human Validation Sample

| Function | Purpose | Notes |
|----------|---------|-------|
| `sample_edges_for_review(G, n=50)` | Stratified sample across edge types | For manual review |
| `format_edge_for_review(edge)` | Human-readable format | Show source text, target text, relationship |
| `create_review_interface(samples)` | Generate review spreadsheet/form | Track accept/reject |
| `compute_human_agreement(reviews)` | Agreement rate with automated edges | Target > 90% |

### Visualization

| Function | Purpose | Notes |
|----------|---------|-------|
| `visualize_full_graph(G)` | Interactive graph viz | Use pyvis, output HTML |
| `visualize_subgraph(G, root, depth=2)` | Focused view around a node | For debugging |
| `visualize_cross_doc_alignment(G)` | Show alignment structure | Color by doc_type |
| `export_for_gephi(G)` | Export for advanced viz | GEXF format |

### Gate Decision

| Check | Criterion | Action if Fail |
|-------|-----------|----------------|
| Connectivity | > 95% | Review Phase 4 edge extraction |
| Box coverage | 100% | Review Phase 1 extraction |
| Cross-doc alignment | > 90% | Review Phase 3 alignment |
| Edge confidence | avg > 0.7 | Review low-confidence edges manually |
| Human validation | > 90% agreement | Iterate on edge classification |
| Faithfulness | 100% conceptual chunks > 0.95 | Revise conceptual construction |

### Outputs
- Eval report: `dbfs:/reports/{doc_id}_graph_eval.json`
- Visualizations: `dbfs:/reports/{doc_id}_graph_viz.html`
- Review samples: `dbfs:/reports/{doc_id}_edge_review.csv`
- Decision: PASS/FAIL with reasons

---

## Phase 7: Pair Generation

### File: `07_pair_generation.py`

### Purpose
Generate training pairs for embedding fine-tuning. Multiple pair types for different learning objectives.

### Pair Types

| Type | Purpose | Positive | Negative |
|------|---------|----------|----------|
| Query-Passage (atomic) | Direct retrieval | LLM-generated question → atomic chunk | Unrelated atomic chunks |
| Query-Passage (conceptual) | Concept retrieval | LLM-generated question → conceptual chunk | Unrelated concepts |
| Similarity (intra-doc) | Learn reference relationships | Connected nodes (graph neighbors) | Distant nodes |
| Similarity (cross-doc) | Learn cross-doc alignment | same_field aligned chunks | Different boxes, same doc_type |
| Hierarchical | Learn containment | Parent-child pairs | Non-ancestor pairs |
| Cross-level | Concept-to-atomic relationship | Conceptual chunk ↔ source atomics | Unrelated atomics |

### Query Generation Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `generate_queries_for_chunk(chunk, n=5, llm)` | Generate diverse questions for a chunk | Vary question types |
| `filter_low_quality_queries(queries, chunk)` | Remove trivial/unanswerable questions | LLM or heuristic filtering |
| `deduplicate_queries(all_queries)` | Remove near-duplicate questions | Embedding similarity threshold |
| `balance_query_types(queries)` | Ensure diversity | Factual, procedural, conceptual |

### LLM Prompt (Query Generation)
```
Generate 5 diverse questions that this IRS tax document chunk answers.

Chunk ({doc_type}, {level}):
{chunk_text}

Requirements:
1. Questions should be natural (how a user would ask)
2. Mix question types: factual, procedural, clarification, edge-case
3. Questions must be answerable from THIS chunk
4. Vary specificity: some direct, some requiring inference
5. Include at least one question a tax preparer might ask
6. Include at least one question a taxpayer might ask

Return JSON:
{
  "questions": [
    {"question": "...", "type": "factual|procedural|clarification|edge_case", "perspective": "filer|recipient|general"}
  ]
}
```

### Graph-Based Pair Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `sample_positive_pairs_by_edge(G, edge_type, n)` | Sample connected pairs | Positive pairs from graph structure |
| `sample_negative_pairs_by_distance(G, min_distance=3, n)` | Sample distant pairs | Hard negatives from graph |
| `sample_hard_negatives(chunk, candidates, embedding_model)` | High similarity but unrelated | Embedding-based hard negative mining |
| `weight_pair_by_edge_confidence(pair, edge)` | Weight training signal | Higher confidence edges = higher weight |

### Cross-Level Pair Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `create_concept_to_atomic_pairs(conceptual_chunk)` | Link concept to sources | Use derived_from edges |
| `create_concept_negative_pairs(conceptual_chunk, unrelated_atomics)` | Negatives for concept | Atomics not in source set |

### Pair Schema
```python
{
    "pair_id": str,
    "pair_type": str,             # query_passage_atomic, similarity_intra, etc.
    "anchor": str,                # chunk_id or query text
    "anchor_type": str,           # "query" | "chunk"
    "positive": str,              # chunk_id
    "negative": str,              # chunk_id (for triplet) or null
    "weight": float,              # Training weight
    "metadata": {
        "edge_type": str,         # If graph-based
        "edge_confidence": float,
        "query_type": str,        # If query-passage
        "distance": int           # If distance-based negative
    },
    "doc_id": str,
    "created_at": timestamp
}
```

### Balance & Sampling Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `compute_pair_distribution(pairs)` | Count by pair type | Identify imbalances |
| `balance_pair_types(pairs, target_distribution)` | Resample to target distribution | Upsample rare, downsample common |
| `create_train_val_test_split(pairs, chunks)` | Split without chunk leakage | Same chunk can't be in train and val |
| `export_for_training(pairs, format)` | Format for training framework | sentence-transformers format |

### Outputs
- Delta table: `catalog.tax_embeddings.training_pairs`
- Split tables: `training_pairs_train`, `training_pairs_val`, `training_pairs_test`
- Distribution report: `dbfs:/reports/{doc_id}_pair_distribution.json`

### Validation Checks
- [ ] No chunk appears in both train and val/test
- [ ] Pair type distribution is reasonable
- [ ] Query quality sample check (manual review of 20 queries)
- [ ] Hard negatives are actually hard (similarity check)

---

## Phase 8: Training Pipeline

### File: `08_embedding_training.py`

### Purpose
Fine-tune BERT-class embedding model on generated pairs.

### High-Level Steps

1. **Model Selection**
   - Base model options: `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`
   - Consider domain: financial/legal pre-training if available

2. **Data Loading**
   - Load pairs from Delta
   - Format for loss function (triplets vs. pairs)
   - DataLoader with appropriate batching

3. **Loss Function Selection**
   - MultipleNegativesRankingLoss (for pairs)
   - TripletLoss (for explicit negatives)
   - Potentially combine with in-batch negatives

4. **Training Configuration**
   - Learning rate: 2e-5 (typical)
   - Batch size: 32-64 (memory dependent)
   - Epochs: 3-10 (monitor validation)
   - Warmup: 10% of steps

5. **Evaluation During Training**
   - Held-out query-passage pairs
   - Track: MRR, Recall@k, NDCG

6. **MLflow Tracking**
   - Log hyperparameters
   - Log metrics per epoch
   - Log model artifacts

### Outputs
- Trained model in MLflow Model Registry
- Training metrics and curves
- Best checkpoint

---

## Phase 9: Vector Store & Retrieval

### File: `09_vector_indexing.py`

### Purpose
Index embeddings and implement retrieval pipeline.

### High-Level Steps

1. **Embed All Chunks**
   - Atomic chunks
   - Conceptual chunks
   - Store embeddings in Delta

2. **Index to Databricks Vector Search**
   - Create vector search index
   - Configure similarity metric (cosine)
   - Include metadata for filtering

3. **Retrieval Pipeline Options**
   - Option A: Search conceptual → expand to atomic
   - Option B: Search both → re-rank
   - Option C: Hybrid (vector + keyword for box numbers)

4. **Retrieval Evaluation**
   - Held-out test queries
   - Metrics: MRR, Recall@1/5/10, NDCG
   - Compare to baseline (off-the-shelf embeddings)

### Outputs
- Vector search index
- Retrieval evaluation metrics
- Comparison to baseline

---

## Phase 10: Expansion

### File: `10_expansion.py`

### Purpose
Scale from 1099-DIV to additional forms, then full corpus.

### Expansion Stages

**Stage 1: 1099-DIV** (Phases 1-9)
- Single form, full pipeline validation
- Establish baseline metrics

**Stage 2: 3 Forms**
- Add: 1099-INT (similar structure)
- Add: 1099-MISC or 1099-NEC (different structure)
- Run Phases 1-5 on each
- Identify cross-form references
- Incremental training (continue from 1099-DIV checkpoint)
- Re-index, re-evaluate

**Stage 3: Full 1099 Series**
- All 1099 variants
- Cross-form conceptual chunks (e.g., "backup withholding" spans forms)
- Full retraining vs. continued training decision

**Stage 4: Extended Corpus**
- 1098 series
- W-2, W-9
- Publications (Pub 550, etc.)
- General Instructions for Certain Information Returns

### Cross-Form Considerations
- Forms reference each other explicitly
- Shared concepts (backup withholding, TIN requirements)
- Need cross-form edges in graph
- Conceptual chunks may aggregate across forms

### Eval at Each Stage
- Retrieval metrics on expanding query set
- Coverage metrics on new forms
- Regression testing on previous forms

---

## Appendix A: LLM Configuration

### Model Selection
- **Primary**: Claude 3.5 Sonnet via Amazon Bedrock
  - Model ID: `anthropic.claude-3-5-sonnet-20240620-v1:0`
  - Region: Check available regions
- **Backup**: Azure OpenAI GPT-4o
  - For rate limit overflow or comparison

### Rate Limiting & Retry
```python
# Bedrock client configuration
retry_config = {
    "max_attempts": 3,
    "retry_mode": "adaptive"
}

# Token budgeting
MAX_INPUT_TOKENS = 8000   # Leave room for response
MAX_OUTPUT_TOKENS = 2000
BATCH_DELAY_SECONDS = 1   # Between batches
```

### Prompt Versioning
- Store prompts in Delta table: `catalog.tax_embeddings.prompt_versions`
- Log prompt_version_id with every LLM output
- Enable A/B testing of prompts

### Response Validation
```python
def validate_llm_response(response, expected_schema):
    """
    1. Check JSON parseable
    2. Validate against expected schema
    3. Check for hallucination signals
    4. Return validated data or raise
    """
```

---

## Appendix B: Error Handling

### PDF Extraction Failures
- Log failed extractions with error details
- Fallback to simpler extraction if complex fails
- Manual review queue for persistent failures

### LLM Failures
- Retry with exponential backoff
- Fallback to backup model
- Log failed requests for debugging
- Graceful degradation (skip chunk if LLM fails, flag for review)

### Graph Inconsistencies
- Validate all edges have valid source/target
- Remove dangling references
- Log inconsistencies for investigation

### Training Failures
- Checkpoint frequently
- Automatic restart from last checkpoint
- Alert on loss divergence

---

## Appendix C: Monitoring & Observability

### Delta Table Metrics
- Row counts per table per run
- Schema drift detection
- Quality score distributions

### Pipeline Metrics
- Extraction success rate
- LLM call success rate
- Average confidence scores
- Processing time per phase

### MLflow Tracking
- All training experiments
- Model artifacts
- Evaluation metrics

### Alerting
- Pipeline failure notifications
- Quality degradation alerts
- Cost monitoring (LLM usage)
