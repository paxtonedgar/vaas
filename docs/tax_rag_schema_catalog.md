# Tax Document RAG Pipeline: Schema Catalog

**Version:** 1.1
**Last Updated:** 2026-01-06
**Catalog:** 112557_prefetch_ctg_prd_exp
**Schema:** 112557_prefetch_raw

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Tables](#2-core-tables)
3. [Intermediate DataFrames](#3-intermediate-dataframes)
4. [Training Tables](#4-training-tables)
5. [Evaluation Tables](#5-evaluation-tables)
6. [Relationships](#6-relationships)

---

## 1. Overview

### 1.1 Table Categories

| Category | Tables | Purpose |
|----------|--------|---------|
| **Core** | box_registry, graph_nodes, graph_edges | Knowledge graph storage |
| **Intermediate** | spans_df, elements_df, anchors_df, sections_df | Pipeline processing |
| **Training** | training_pairs, hard_negatives, scenario_queries | Model training |
| **Evaluation** | eval_queries, eval_results, metrics_history | Performance tracking |

### 1.2 Naming Conventions

- **Delta tables:** `snake_case` (e.g., `graph_nodes`, `box_registry`)
- **DataFrame variables:** `snake_case_df` (e.g., `spans_df`, `elements_df`)
- **Columns:** `snake_case` (e.g., `node_id`, `canonical_id`)
- **IDs:** `{scope}_{type}_{identifier}` (e.g., `1099div_filer_box_1a`)

---

## 2. Core Tables

### 2.1 box_registry

**Purpose:** Canonical mapping of box keys to labels, aliases, and metadata. Enables cross-document alignment and natural language lookup.

**Location:** `catalog.schema.box_registry`

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| doc_id | string | No | Form identifier | "1099-DIV" |
| doc_version | string | No | Form revision | "Rev. January 2024" |
| registry_version | string | No | Registry version | "1.0.0" |
| box_key | string | No | **Primary key** - Normalized box identifier | "box_1b" |
| canonical_id | string | No | Cross-form canonical identifier | "qualified_dividends" |
| label | string | No | Natural language label | "Qualified Dividends" |
| aliases | array[string] | Yes | Alternative names for lookup | ["qualified dividends", "QD", "1b"] |
| group_id | string | Yes | Group membership for related boxes | "box_1_group" |
| data_type | string | Yes | Field data type | "currency", "percentage", "text" |
| source | string | No | Registry entry source | "irs_form", "manual", "extracted" |
| confidence | double | No | Trust score (0.0-1.0) | 1.0 |
| created_at | timestamp | No | Creation timestamp | 2026-01-05T10:00:00Z |

**Indexes:**
- Primary: `(doc_id, box_key)`
- Secondary: `canonical_id`

**Sample Data:**

| doc_id | box_key | canonical_id | label | aliases |
|--------|---------|--------------|-------|---------|
| 1099-DIV | box_1a | total_ordinary_dividends | Total Ordinary Dividends | ["ordinary dividends", "1a"] |
| 1099-DIV | box_1b | qualified_dividends | Qualified Dividends | ["qualified dividends", "QD", "1b"] |
| 1099-DIV | box_2e | section_897_ordinary | Section 897 Ordinary Dividends | ["section 897", "897 ordinary", "2e"] |
| 1099-DIV | box_4 | federal_withholding | Federal Income Tax Withheld | ["withholding", "backup withholding", "4"] |

---

### 2.2 graph_nodes

**Purpose:** Knowledge graph nodes representing document chunks at various granularities.

**Location:** `catalog.schema.graph_nodes`

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| node_id | string | No | **Primary key** - Unique node identifier | "1099div_filer_box_1a_001" |
| doc_id | string | No | Source document | "1099-DIV" |
| doc_type | string | No | Document type | "filer_instructions", "recipient_instructions", "form" |
| node_type | string | No | Node category | "box_section", "concept", "section", "preamble" |
| anchor_id | string | No | Source anchor ID | "box_1a", "sub_qualified_dividends" |
| box_key | string | Yes | Box key for box_section nodes | "1a", "2e" |
| label | string | Yes | Human-readable label | "Qualified Dividends" |
| canonical_id | string | Yes | Registry link for alignment | "qualified_dividends" |
| level | string | No | Hierarchy level | "form", "anchor", "paragraph", "sentence" |
| parent_node_id | string | Yes | Parent in hierarchy (FK to node_id) | "1099div_filer_box_1a" |
| depth | int | No | Hierarchy depth (0=form) | 2 |
| text | string | No | Normalized text content | "Enter the portion..." |
| text_raw | string | Yes | Original text before normalization | "Enter the portion..." |
| pages | array[int] | No | Pages containing this node | [4, 5] |
| element_count | int | No | Number of source elements | 5 |
| char_count | int | No | Character count | 619 |
| bbox | struct | Yes | Bounding box {x0, y0, x1, y1} | {72.0, 144.0, 540.0, 200.0} |
| doc_version | string | Yes | Document revision | "Rev. January 2024" |
| extraction_confidence | double | No | Quality score (0.0-1.0) | 0.95 |
| content_hash | string | No | SHA256 hash for deduplication | "a1b2c3d4e5f6..." |
| extracted_at | timestamp | No | Extraction timestamp | 2026-01-05T10:00:00Z |
| extraction_version | string | No | Pipeline version | "1.0.0" |
| concept_role | string | Yes | Semantic role for concept nodes | "definition", "qualification", "condition", "exception", "procedure" |
| concepts | array[string] | Yes | Concept tags | ["qualified_dividends", "holding_period"] |
| source_node_ids | array[string] | Yes | Source nodes (for conceptual chunks) | ["node_1", "node_2"] |

**Node Types:**

| node_type | Description | Example anchor_id |
|-----------|-------------|-------------------|
| box_section | Box instruction section | "box_1a", "box_2e" |
| concept | Semantic concept (promoted subsection) | "sub_qualified_dividends" |
| section | Major document section | "sec_specific_instructions" |
| preamble | Document preamble | "preamble" |

**Concept Roles (for node_type="concept"):**

| concept_role | Description | Trigger Pattern Example |
|--------------|-------------|------------------------|
| definition | Defines a term or concept | "X refers to...", "X means..." |
| qualification | Limits scope or applicability | "only if", "limited to" |
| condition | Specifies when rules apply | "if...", "when...", "provided that" |
| exception | Specifies exclusions | "except", "does not include" |
| procedure | Describes how to do something | "report...", "enter...", "file..." |

**Indexes:**
- Primary: `node_id`
- Secondary: `(doc_id, doc_type)`
- Secondary: `canonical_id`
- Secondary: `parent_node_id`

**Constraints:**
- `parent_node_id` must reference existing `node_id` (or be NULL for root)
- `depth` must equal `parent.depth + 1` (or 0 for roots)

---

### 2.3 graph_edges

**Purpose:** Knowledge graph edges representing relationships between nodes.

**Location:** `catalog.schema.graph_edges`

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| edge_id | string | No | **Primary key** - Unique edge identifier | "edge_incl_001" |
| doc_id | string | No | Source document | "1099-DIV" |
| source_node_id | string | No | Edge source (FK to graph_nodes) | "1099div_filer_box_1a" |
| target_node_id | string | No | Edge target (FK to graph_nodes) | "1099div_filer_box_1b" |
| edge_type | string | No | Relationship type | "includes", "parent_of", "same_field" |
| direction | string | No | Directionality | "directed", "bidirectional" |
| confidence | double | No | Trust score (0.0-1.0) | 0.92 |
| source_evidence | string | Yes | Text supporting this edge | "Box 1a includes amounts in box 1b" |
| created_by | string | No | Extraction method | "structural", "regex", "llm", "registry_alignment" |
| created_at | timestamp | No | Creation timestamp | 2026-01-05T10:00:00Z |
| metadata | map[string, string] | Yes | Extensible fields | {"pair_weight": "1.0", "reference_type": "box"} |

**Indexes:**
- Primary: `edge_id`
- Secondary: `source_node_id`
- Secondary: `target_node_id`
- Secondary: `edge_type`

**Edge Type Reference:**

| edge_type | Direction | Description | created_by |
|-----------|-----------|-------------|------------|
| parent_of | directed | Hierarchy containment | structural |
| anchor_has_content | directed | Anchor owns content | structural |
| follows | directed | Reading order | structural |
| same_group | directed | Grouped box membership (14-16) | structural |
| references_box | directed | Explicit box reference | regex |
| references_section | directed | Explicit section reference | regex |
| includes | directed | "Box 1a includes 1b" | regex, llm |
| see_also | directed | General cross-reference | regex, llm |
| same_field | bidirectional | Same box across doc types | registry_alignment |
| elaborates | directed | Filer elaborates recipient | registry_alignment |
| derived_from | directed | Conceptual chunk source | synthesis |
| external_ref | directed | Reference outside corpus | regex |

**Semantic Edge Types (Phase 2a):**

These typed edges capture semantic relationships between concepts and boxes.
Direction: edges point from the rule/concept to the thing being ruled on.

| edge_type | Direction | Polarity | Description | Pattern Example |
|-----------|-----------|----------|-------------|-----------------|
| excludes | concept → box | negative | Negation/exception | "do not include X in box Y" |
| defines | concept → box | positive | Semantic meaning | "box Y is/means X" |
| qualifies | concept → box | positive | Scope/constraint | "box Y includes only X" |
| applies_if | concept → box | positive | Conditional applicability | "report X in box Y if Z" |
| requires | box → box | positive | Computational dependency | "include amounts from box X" |

**Edge Columns (additional for semantic edges):**

| Column | Type | Description |
|--------|------|-------------|
| pattern_matched | string | Regex pattern that triggered extraction |
| polarity | string | "positive" or "negative" |
| source_evidence | string | Text snippet supporting the edge |

---

## 3. Intermediate DataFrames

These DataFrames exist during pipeline execution but are not persisted to Delta (unless debugging).

### 3.1 spans_df

**Purpose:** Raw text spans extracted from PDF.

| Column | Type | Description |
|--------|------|-------------|
| page | int | Page number (0-indexed) |
| block_id | int | Block index within page |
| line_id | int | Line index within block |
| span_id | int | Span index within line |
| text | string | Span text content |
| font | string | Font name |
| size | float | Font size in points |
| flags | int | Font flags (bold=16, italic=2) |
| x0 | float | Left bound |
| y0 | float | Top bound |
| x1 | float | Right bound |
| y1 | float | Bottom bound |

**Typical Size:** ~3,000-5,000 rows for 5-page instructions PDF

---

### 3.2 elements_df

**Purpose:** Line-level elements with role classification.

| Column | Type | Description |
|--------|------|-------------|
| doc_id | string | Document identifier |
| page | int | Page number |
| element_id | string | Unique element ID |
| text | string | Normalized text |
| role | string | Classified role (DocTitle, SectionHeader, BoxHeader, BodyText, ListBlock, TableBlock, PageArtifact) |
| role_hint | string | Detection hint (e.g., "BoxHeadingCandidate") |
| role_conf | float | Classification confidence |
| size_mode | string | Font size category ("body", "header", "subheader") |
| is_bold | bool | Bold flag |
| fonts | string | Concatenated font names |
| x_min | float | Left bound |
| y_min | float | Top bound |
| x_max | float | Right bound |
| y_max | float | Bottom bound |
| reading_order | int | Sequential position in document |

**Typical Size:** ~500-800 rows for 5-page instructions PDF

---

### 3.3 anchors_df

**Purpose:** Detected structural anchors (boxes, sections).

| Column | Type | Description |
|--------|------|-------------|
| doc_id | string | Document identifier |
| anchor_id | string | Unique anchor ID |
| key | string | Normalized key ("box_1a", "section_qualified_dividends") |
| parent_anchor_id | string | Parent anchor for grouped boxes |
| title_text | string | Full anchor title |
| page | int | Page where anchor appears |
| y_position | float | Vertical position for timeline |
| role_conf | float | Detection confidence |

**Typical Size:** ~20-30 rows for 1099-DIV

---

### 3.4 sections_df

**Purpose:** Assembled content per anchor.

| Column | Type | Description |
|--------|------|-------------|
| anchor_id | string | Owning anchor |
| section_text | string | Concatenated element text |
| element_ids | array[string] | Source element IDs |
| element_count | int | Number of elements |
| page_start | int | First page |
| page_end | int | Last page |

**Typical Size:** Same as anchors_df (~20-30 rows)

---

### 3.5 refs_df

**Purpose:** Extracted cross-references.

| Column | Type | Description |
|--------|------|-------------|
| source_anchor_id | string | Anchor containing reference |
| reference_text | string | Original reference text |
| reference_type | string | "box", "section", "publication", "irc", "form" |
| target_key | string | Resolved target key |
| confidence | float | Extraction confidence |
| extraction_method | string | "regex" or "llm" |

**Typical Size:** ~50-100 rows for 1099-DIV

---

## 4. Training Tables

### 4.1 training_pairs

**Purpose:** Generated training pairs for embedding fine-tuning.

**Location:** `catalog.schema.training_pairs`

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| pair_id | string | No | **Primary key** | "pair_0001" |
| anchor_id | string | No | Anchor node ID | "1099div_filer:box_1a" |
| anchor_text | string | No | Anchor text content | "Box 1a. Total Ordinary Dividends..." |
| positive_id | string | No | Positive node ID | "1099div_filer:box_1b" |
| positive_text | string | No | Positive text content | "Box 1b. Qualified Dividends..." |
| edge_type | string | No | Source edge type | "references_box", "excludes", "defines" |
| pair_category | string | No | Stratified category | "hierarchical", "cross_reference", "semantic", "negative" |
| confidence | double | No | Edge confidence (training weight) | 0.95 |
| evidence_text | string | Yes | Evidence text | "Box 1a includes amounts in box 1b" |
| negative_id | string | Yes | Negative node ID (for triplets) | "1099div_filer:box_9" |
| negative_text | string | Yes | Negative text content | "Box 9. Cash liquidation..." |
| negative_type | string | Yes | Negative mining method | "in_batch", "bm25_hard", "graph_distant" |
| graph_distance | int | Yes | Graph distance for hard negatives | 5 |
| doc_id | string | No | Source document | "1099-DIV" |
| created_at | timestamp | No | Creation timestamp | 2026-01-06T10:00:00Z |
| generation_version | string | No | Pipeline version | "1.1.0" |
| quality_score | double | Yes | LLM-as-judge score | 0.95 |
| quality_flags | array[string] | Yes | Quality flags | ["potential_false_negative"] |

**Pair Categories (Stratified Sampling):**

| pair_category | Source Edge Types | Purpose |
|---------------|-------------------|---------|
| hierarchical | parent_of, includes, same_group | Structural relationships |
| cross_reference | references_box, same_field | Document cross-references |
| semantic | defines, qualifies, applies_if, requires | Semantic relationships |
| negative | excludes | Negative knowledge (what NOT to associate) |

**Indexes:**
- Primary: `pair_id`
- Secondary: `pair_type`
- Secondary: `doc_id`

---

### 4.2 scenario_queries

**Purpose:** LLM-generated queries for scenario-based training.

**Location:** `catalog.schema.scenario_queries`

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| query_id | string | No | **Primary key** | "query_001" |
| query_text | string | No | Generated query | "What dividends qualify for reduced capital gains rates?" |
| query_type | string | No | Query category | "factual", "procedural", "clarification", "edge_case" |
| perspective | string | No | User perspective | "filer", "recipient", "general" |
| difficulty | string | No | Query difficulty | "easy", "medium", "hard" |
| source_node_id | string | No | Node this query was generated from | "1099div_filer_qualified_divs" |
| doc_id | string | No | Source document | "1099-DIV" |
| created_at | timestamp | No | Generation timestamp | 2026-01-05T10:00:00Z |
| model_used | string | No | LLM used for generation | "claude-3.5-sonnet" |
| prompt_version | string | No | Prompt template version | "1.0.0" |

---

## 5. Evaluation Tables

### 5.1 eval_queries

**Purpose:** Gold standard evaluation queries with expected results.

**Location:** `catalog.schema.eval_queries`

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| query_id | string | No | **Primary key** | "eval_001" |
| query_text | string | No | Evaluation query | "What are qualified dividends?" |
| query_type | string | No | Query category | "concept", "exact_anchor", "scenario" |
| gold_node_ids | array[string] | No | Expected top results | ["node_qd_def", "node_qd_rules"] |
| near_miss_node_ids | array[string] | Yes | Related but wrong results | ["node_box_1a"] |
| coverage_threshold | double | Yes | Required coverage for multi-hop | 0.8 |
| doc_id | string | No | Target document | "1099-DIV" |
| created_at | timestamp | No | Creation timestamp | 2026-01-05T10:00:00Z |
| created_by | string | No | Creator | "manual", "generated" |

---

### 5.2 eval_results

**Purpose:** Evaluation run results.

**Location:** `catalog.schema.eval_results`

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| result_id | string | No | **Primary key** | "result_001" |
| run_id | string | No | Evaluation run ID | "run_2026-01-05_001" |
| query_id | string | No | Query evaluated (FK to eval_queries) | "eval_001" |
| retrieved_node_ids | array[string] | No | Actual retrieved results | ["node_qd_def", "node_box_1b"] |
| retrieved_scores | array[double] | No | Retrieval scores | [0.92, 0.87] |
| recall_at_1 | double | No | Recall@1 | 1.0 |
| recall_at_5 | double | No | Recall@5 | 0.8 |
| recall_at_10 | double | No | Recall@10 | 1.0 |
| mrr | double | No | Mean Reciprocal Rank | 1.0 |
| ndcg_at_10 | double | No | NDCG@10 | 0.85 |
| concept_coverage | double | Yes | Multi-hop coverage | 0.8 |
| latency_ms | int | No | Query latency | 150 |
| created_at | timestamp | No | Evaluation timestamp | 2026-01-05T10:00:00Z |

---

### 5.3 metrics_history

**Purpose:** Aggregate metrics over time for tracking improvement.

**Location:** `catalog.schema.metrics_history`

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| run_id | string | No | **Primary key** | "run_2026-01-05_001" |
| run_type | string | No | Run type | "baseline", "fine_tuned", "hybrid" |
| model_version | string | No | Model version | "cohere-v3", "fine_tuned_v1" |
| doc_ids | array[string] | No | Documents evaluated | ["1099-DIV"] |
| query_count | int | No | Number of queries | 75 |
| avg_recall_at_1 | double | No | Average Recall@1 | 0.65 |
| avg_recall_at_5 | double | No | Average Recall@5 | 0.82 |
| avg_recall_at_10 | double | No | Average Recall@10 | 0.91 |
| avg_mrr | double | No | Average MRR | 0.72 |
| avg_ndcg_at_10 | double | No | Average NDCG@10 | 0.78 |
| avg_concept_coverage | double | Yes | Average concept coverage | 0.75 |
| avg_latency_ms | double | No | Average latency | 145.0 |
| p95_latency_ms | double | No | P95 latency | 280.0 |
| created_at | timestamp | No | Run timestamp | 2026-01-05T10:00:00Z |
| config | map[string, string] | Yes | Run configuration | {"rerank": "true", "hybrid": "true"} |

---

## 6. Relationships

### 6.1 Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   box_registry  │       │   graph_nodes   │       │   graph_edges   │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ PK box_key      │◄──────│ FK canonical_id │       │ PK edge_id      │
│    doc_id       │       │ PK node_id      │◄──────│ FK source_node  │
│    canonical_id │       │ FK parent_node  │◄──────│ FK target_node  │
│    label        │       │    doc_id       │       │    edge_type    │
│    aliases[]    │       │    doc_type     │       │    confidence   │
│    group_id     │       │    chunk_type   │       │    evidence     │
│    data_type    │       │    level        │       │    created_by   │
│    confidence   │       │    depth        │       │    metadata{}   │
└─────────────────┘       │    text         │       └─────────────────┘
                          │    concepts[]   │
                          │    source_ids[] │
                          └─────────────────┘
                                  │
                                  │ derived from
                                  ▼
┌─────────────────┐       ┌─────────────────┐
│ training_pairs  │       │scenario_queries │
├─────────────────┤       ├─────────────────┤
│ PK pair_id      │       │ PK query_id     │
│ FK anchor_id    │───────│ FK source_node  │
│ FK positive_id  │       │    query_text   │
│ FK negative_id  │       │    query_type   │
│    pair_type    │       │    perspective  │
│    weight       │       └─────────────────┘
│    quality      │
└─────────────────┘
         │
         │ evaluated by
         ▼
┌─────────────────┐       ┌─────────────────┐
│   eval_queries  │       │  eval_results   │
├─────────────────┤       ├─────────────────┤
│ PK query_id     │◄──────│ FK query_id     │
│    query_text   │       │ PK result_id    │
│    query_type   │       │    run_id       │
│    gold_nodes[] │       │    retrieved[]  │
│    near_miss[]  │       │    recall@k     │
└─────────────────┘       │    mrr, ndcg    │
                          └─────────────────┘
                                  │
                                  │ aggregated to
                                  ▼
                          ┌─────────────────┐
                          │ metrics_history │
                          ├─────────────────┤
                          │ PK run_id       │
                          │    run_type     │
                          │    avg_metrics  │
                          │    config{}     │
                          └─────────────────┘
```

### 6.2 Key Relationships

| Relationship | From | To | Cardinality | Description |
|--------------|------|-----|-------------|-------------|
| Registry → Nodes | box_registry.canonical_id | graph_nodes.canonical_id | 1:N | Registry entry aligns multiple nodes |
| Nodes → Nodes | graph_nodes.parent_node_id | graph_nodes.node_id | N:1 | Hierarchy (tree) |
| Nodes ↔ Edges | graph_nodes.node_id | graph_edges.source/target | 1:N | Edges connect nodes |
| Nodes → Pairs | graph_nodes.node_id | training_pairs.anchor/positive/negative | 1:N | Nodes become pair elements |
| Nodes → Queries | graph_nodes.node_id | scenario_queries.source_node_id | 1:N | Queries generated from nodes |
| Queries → Results | eval_queries.query_id | eval_results.query_id | 1:N | Results per query |
| Results → History | eval_results.run_id | metrics_history.run_id | N:1 | Results aggregated |

### 6.3 Data Integrity Rules

1. **Hierarchy validity:** `graph_nodes.parent_node_id` must reference existing node or be NULL
2. **Edge validity:** `graph_edges.source_node_id` and `target_node_id` must reference existing nodes
3. **Alignment consistency:** If `graph_nodes.canonical_id` is set, it should exist in `box_registry`
4. **Pair validity:** `training_pairs` anchor/positive/negative IDs must reference existing nodes
5. **No cycles:** `parent_of` edges must not form cycles (DAG constraint)
6. **Bidirectional pairs:** If `edge_type = "same_field"` and `direction = "bidirectional"`, both direction edges should exist

---

## Appendix: SQL Examples

### A.1 Get all nodes for a box

```sql
SELECT n.*
FROM graph_nodes n
JOIN box_registry r ON n.canonical_id = r.canonical_id
WHERE r.box_key = 'box_1b'
```

### A.2 Get cross-document aligned nodes

```sql
SELECT n1.node_id as filer_node, n2.node_id as recipient_node, n1.canonical_id
FROM graph_nodes n1
JOIN graph_nodes n2 ON n1.canonical_id = n2.canonical_id
WHERE n1.doc_type = 'filer_instructions'
  AND n2.doc_type = 'recipient_instructions'
```

### A.3 Get all edges from a node

```sql
SELECT e.*, 
       src.text as source_text, 
       tgt.text as target_text
FROM graph_edges e
JOIN graph_nodes src ON e.source_node_id = src.node_id
JOIN graph_nodes tgt ON e.target_node_id = tgt.node_id
WHERE e.source_node_id = '1099div_filer_box_1a'
```

### A.4 Generate hierarchical pairs

```sql
SELECT 
    src.node_id as anchor_id,
    src.text as anchor_text,
    tgt.node_id as positive_id,
    tgt.text as positive_text,
    'hierarchical' as pair_type,
    e.confidence as weight
FROM graph_edges e
JOIN graph_nodes src ON e.source_node_id = src.node_id
JOIN graph_nodes tgt ON e.target_node_id = tgt.node_id
WHERE e.edge_type = 'parent_of'
```

### A.5 Calculate metrics per query type

```sql
SELECT 
    q.query_type,
    COUNT(*) as query_count,
    AVG(r.recall_at_5) as avg_recall_5,
    AVG(r.mrr) as avg_mrr
FROM eval_results r
JOIN eval_queries q ON r.query_id = q.query_id
WHERE r.run_id = 'run_2026-01-05_001'
GROUP BY q.query_type
```

---

*Document generated: 2026-01-06*
*Schema version: 1.1.0*

**Changelog:**
- v1.1.0 (2026-01-06): Added concept nodes with semantic roles, typed edges (excludes, defines, qualifies, applies_if, requires), stratified pair categories
- v1.0.0 (2026-01-05): Initial schema catalog
