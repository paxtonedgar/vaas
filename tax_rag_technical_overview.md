# VaaS Tax Document Intelligence: Technical Overview

**Version:** 1.0  
**Last Updated:** 2026-01-05  
**Epic References:** UTILITIESPLATFORM-5327, UTILITIESPLATFORM-5326

---

## Why This Document Exists

This document explains what we're building, why we're building it this way, and how the pieces fit together. It's written for engineers joining the project, architects reviewing the design, and product partners who want to understand the technical approach.

---

## Part 1: The Problem

### What Breaks When You Use Standard RAG on Tax Documents

If you take a tax document like the 1099-DIV instructions, chunk it naively, embed it with an off-the-shelf model, and retrieve via vector similarity, you get answers that are **locally correct but globally wrong**.

Here's a concrete example:

**User query:** "What dividends qualify for reduced capital gains rates?"

**What standard RAG returns:**
> "Enter the portion of dividends that qualifies for reduced capital gains rates."

**Why this is useless:** The chunk is literally from the right section, but it doesn't tell you *what qualifies*. The definition is in a different chunk. The holding period rules are in another. The list of exceptions spans three pages. Standard RAG grabbed a sentence that contains the query terms but lacks the context to answer the question.

This happens because tax documents have properties that break the assumptions of standard RAG:

**1. Cross-referential structure**

Box 1a's instructions say "Include amounts in boxes 1b and 2e." If you ask about Box 1a and only retrieve Box 1a's chunk, you've missed the dependency. The embedding of "Box 1a" doesn't encode that it semantically includes 1b and 2e.

**2. Precise terminology that embeddings compress away**

"Box 2e" and "Box 2f" are completely different fields (Section 897 ordinary dividends vs. Section 897 capital gain). They're one character apart. Single-vector embeddings map similar strings to similar vectors. For tax documents, that's wrong.

**3. Multi-perspective content with no alignment mechanism**

The filer instructions for Box 1b are 400 words. The recipient instructions for Box 1b are 40 words. They describe the same field but serve different purposes. General RAG treats them as unrelated chunks. We need them aligned so "What is Box 1b?" can pull both perspectives.

**4. Implicit context that disappears in chunking**

A paragraph that says "Enter the portion that qualifies for reduced rates" makes no sense without knowing it's inside the "Qualified Dividends" section of the "1099-DIV Filer Instructions." Standard chunking loses this context.

**5. Hierarchical meaning encoded in visual layout**

Indentation, font size, and nesting encode relationships. A bullet point under "Exceptions" means something different than the same text under "Requirements." Flat chunking destroys these signals.

### Why Fine-Tuning Is Necessary, Not Optional

Some RAG approaches try to solve these problems with better prompting or retrieval tricks. That's not sufficient here. The research is clear:

- **FinMTEB benchmark:** General embedding models score 82% on average tasks but drop to 52% on legal/financial domains.
- **BEIR domain analysis:** Out-of-domain performance degrades 20-40% compared to in-domain.
- **FAA fine-tuning study:** For technically dense domains, baseline embeddings are insufficient.

Tax documents are legal, financial, and technically dense. We're firmly in the failure zone.

The question isn't *whether* to fine-tune. It's *how to fine-tune productively*—which means building the infrastructure to generate high-quality training pairs, measure improvement, and iterate.

---

## Part 2: The Solution

### Core Thesis

We build a **knowledge graph** from tax documents that captures:
- Document hierarchy (sections contain subsections contain paragraphs)
- Cross-references ("Box 1a includes 1b")
- Cross-document alignment (Box 1b in filer instructions ↔ Box 1b in recipient instructions)
- Concept definitions (the "Qualified Dividends" section defines what goes in Box 1b)

This graph serves two purposes:

1. **Training pair generation:** We traverse the graph to create pairs that teach the embedding model domain relationships. "Box 1a" and "Box 1b" should be close because there's an `includes` edge between them.

2. **Retrieval augmentation:** When we retrieve a chunk, we expand via the graph to pull in related context. If you retrieve Box 1a, the graph tells us to also surface Box 1b and 2e.

One graph, two uses, unified schema.

### How the Pieces Fit Together

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   IRS PDFs                                                                   │
│      │                                                                       │
│      ▼                                                                       │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                     EXTRACTION PIPELINE                               │  │
│   │                                                                       │  │
│   │  PDF → Spans → Elements → Anchors → Sections → References            │  │
│   │                                                                       │  │
│   │  We extract text while preserving layout signals (font, position,    │  │
│   │  nesting). We detect structural anchors (boxes, sections). We        │  │
│   │  assign content to anchors. We extract cross-references.             │  │
│   └───────────────────────────────┬──────────────────────────────────────┘  │
│                                   │                                          │
│                                   ▼                                          │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                      KNOWLEDGE GRAPH                                  │  │
│   │                                                                       │  │
│   │  ┌─────────┐      ┌─────────┐      ┌─────────┐                       │  │
│   │  │ graph_  │      │ graph_  │      │  box_   │                       │  │
│   │  │ nodes   │◄────►│ edges   │◄────►│registry │                       │  │
│   │  └─────────┘      └─────────┘      └─────────┘                       │  │
│   │                                                                       │  │
│   │  Nodes are chunks at various granularities (anchor, paragraph).      │  │
│   │  Edges are relationships (parent_of, includes, same_field).          │  │
│   │  Registry maps box keys to canonical IDs and aliases.                │  │
│   └───────────────────────────────┬──────────────────────────────────────┘  │
│                                   │                                          │
│                   ┌───────────────┴───────────────┐                          │
│                   │                               │                          │
│                   ▼                               ▼                          │
│   ┌───────────────────────────┐   ┌───────────────────────────┐             │
│   │    TRAINING PIPELINE      │   │    RETRIEVAL PIPELINE     │             │
│   │                           │   │                           │             │
│   │  Query graph edges to     │   │  Hybrid search (BM25 +    │             │
│   │  generate training pairs: │   │  dense + rerank).         │             │
│   │  - Hierarchical           │   │                           │             │
│   │  - Cross-reference        │   │  Graph expansion to pull  │             │
│   │  - Same-field             │   │  related context.         │             │
│   │  - Hard negatives         │   │                           │             │
│   │                           │   │  Contextualized chunks    │             │
│   │  Fine-tune embedding      │   │  with form/section prefix.│             │
│   │  model on these pairs.    │   │                           │             │
│   └───────────────────────────┘   └───────────────────────────┘             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: The Extraction Pipeline

### What We Extract and Why

The extraction pipeline converts a PDF into structured data that can populate the knowledge graph. Each stage builds on the previous one.

**Stage 1: Span Extraction**

We use PyMuPDF to extract every text span with its font, size, position, and styling. This gives us the raw material.

Why this matters: Font size tells us what's a header vs. body text. Position tells us reading order. Bold text often indicates important terms.

**Stage 2: Element Construction**

We group spans into line-level elements and classify each element's role:

- **DocTitle:** "Instructions for Form 1099-DIV"
- **SectionHeader:** "General Instructions", "Qualified Dividends"
- **BoxHeader:** "Box 1a. Total Ordinary Dividends"
- **BodyText:** Regular paragraphs
- **ListBlock:** Bulleted or numbered items
- **PageArtifact:** Headers, footers, page numbers (ignore these)

Why this matters: Role classification lets us treat headers as structural anchors and body text as content that belongs to those anchors.

**Stage 3: Anchor Detection**

We identify structural anchors—the organizational units that own content:

- **Box anchors:** "Box 1a", "Box 2e", "Boxes 14-16"
- **Section anchors:** "Qualified Dividends", "RICs and REITs"

Box anchors follow regex patterns. Section anchors are SectionHeaders that match known concept names from the registry.

Why this matters: Anchors become the primary nodes in our graph. They're what users ask about ("What goes in Box 1b?") and what we need to align across documents.

**Stage 4: Content Assignment**

We assign each element to its owning anchor based on reading order. Content after "Box 1a" but before "Box 1b" belongs to Box 1a.

Why this matters: This builds the hierarchy. Box 1a (anchor) contains paragraphs 1-5 (content). This becomes `parent_of` edges in the graph.

**Stage 5: Reference Extraction**

We scan section text for cross-references:

- "Box 1a includes amounts in boxes 1b and 2e" → `includes` edge
- "See Qualified Dividends, earlier" → `references_section` edge
- "See Pub. 550" → `external_ref` edge

We use regex for high-confidence patterns and LLM for ambiguous ones.

Why this matters: These references become edges that training pairs traverse. They're also how we expand retrieval results.

**Stage 6: Graph Emission**

We write everything to Delta tables: `graph_nodes`, `graph_edges`, with foreign keys to `box_registry`.

### The Box Registry: How Alignment Works

The registry is a table that maps box keys to canonical identifiers and aliases:

| box_key | canonical_id | label | aliases |
|---------|--------------|-------|---------|
| box_1b | qualified_dividends | Qualified Dividends | ["qualified dividends", "QD", "1b"] |

When we extract a section header "Qualified Dividends" from the filer instructions, we look it up in the registry and get `canonical_id = qualified_dividends`. We assign that to the node.

When we later extract "Box 1b" from the recipient instructions, same lookup, same `canonical_id`.

Now both nodes share a canonical ID. We automatically create a `same_field` edge between them. Cross-document alignment falls out of the schema.

This also enables natural language lookup. If a user asks about "qualified dividends", we can resolve that to `box_1b` via the aliases array.

---

## Part 4: The Knowledge Graph

### Node Design

Nodes represent chunks of content at various granularities:

**Anchor nodes** (level 1): Box headers and section headers. These are the primary organizational units.

**Paragraph nodes** (level 2): Individual paragraphs within an anchor section. These are what we actually retrieve.

**Sentence nodes** (level 3, optional): For particularly dense sections where paragraph-level is too coarse.

Every node carries:
- `canonical_id`: Links to the registry for alignment
- `parent_node_id`: Links to its parent in the hierarchy
- `text`: The actual content
- `concepts`: Array of concept tags for multi-concept content

The hierarchy is explicit. You can traverse from a paragraph to its anchor to its parent section.

### Edge Design

Edges represent relationships. They're typed so we can query by relationship kind:

**Structural edges** (from extraction):
- `parent_of`: Anchor contains paragraph
- `follows`: Reading order
- `part_of_group`: Box 14 is part of Boxes 14-16 group

**Reference edges** (from text analysis):
- `includes`: "Box 1a includes 1b"
- `references_box`: Explicit box reference
- `references_section`: Explicit section reference
- `defines`: Section defines what goes in a box

**Alignment edges** (from registry):
- `same_field`: Same box across filer/recipient instructions

Every edge has:
- `confidence`: How sure are we this relationship exists
- `source_evidence`: The text that supports this edge (critical for debugging and demos)
- `created_by`: How we found this edge (structural, regex, llm)

### Why One Graph Serves Both Training and Retrieval

**For training:** We query edges to generate pairs.

```
Hierarchical pairs: SELECT * FROM edges WHERE type = 'parent_of'
Cross-ref pairs:    SELECT * FROM edges WHERE type = 'includes'
Same-field pairs:   SELECT * FROM nodes n1 JOIN nodes n2 ON n1.canonical_id = n2.canonical_id
```

Each edge type teaches the model a different relationship. Parent-child pairs teach containment. Cross-reference pairs teach navigation. Same-field pairs teach alignment.

**For retrieval:** We traverse edges to expand context.

When we retrieve "Box 1a", we follow `includes` edges to also return Box 1b and 2e. We follow `parent_of` inverse to get sibling paragraphs. We follow `same_field` to offer the recipient perspective.

The graph structure doesn't change—we just query it differently for different purposes.

---

## Part 5: Training Pair Generation

### The Pair Type Taxonomy

Different edge types generate different training signals:

**Hierarchical pairs** (from `parent_of` edges)

Anchor: "Box 1b. Qualified Dividends. Report here the portion..."
Positive: "To meet the holding period requirement, you must have held the stock..."

This teaches the model that a paragraph is semantically close to its section header.

**Cross-reference pairs** (from `includes`, `references_box` edges)

Anchor: "Box 1a. Total Ordinary Dividends. Include amounts in boxes 1b and 2e..."
Positive: "Box 1b. Qualified Dividends..."

This teaches the model that referenced content is semantically related.

**Same-field pairs** (from `canonical_id` match across doc types)

Anchor: [Box 1b filer instructions - 400 words of detail]
Positive: [Box 1b recipient instructions - 40 word summary]

This teaches the model that the same field described from different perspectives should be close.

**Hard negatives** (from graph distance + BM25 similarity)

Anchor: "Box 1a. Total Ordinary Dividends..."
Negative: "Box 9. Cash liquidation distributions..."

Both are about distributions. BM25 thinks they're similar. But they're graph-distant (no path within 3 hops) and semantically unrelated. This teaches the model to discriminate.

**Scenario queries** (from LLM generation)

Anchor: "What dividends qualify for reduced capital gains rates?"
Positive: [Qualified Dividends section]

We use an LLM to generate natural-language questions that each chunk answers. This teaches the model to handle real user queries.

### Hard Negative Mining

This is critical and easy to get wrong. The naive approach: use BM25 to find similar documents, use those as negatives.

The problem: ~70% of BM25 top candidates are **false negatives**. "Box 1b" appears in Box 1a's text because Box 1a references it. BM25 returns Box 1b as similar to Box 1a. But Box 1b is actually a positive (they're related via `includes`).

Our approach: **BM25 + graph distance + positive-aware filtering**

1. Find BM25 candidates (high lexical similarity)
2. Filter to graph-distant nodes (no path within 3 hops)
3. Check embedding similarity to known positives
4. If similarity to any positive > 0.85, it's probably a false negative—discard

This gives us hard negatives that are actually hard and actually negative.

### Pair Quality Validation

We use LLM-as-judge to validate pair quality before training:

```
Is this a valid {pair_type} pair?
Anchor: {anchor_text}
Positive: {positive_text}
Are these semantically related in a useful way?
```

Pass criteria: >90% of sampled pairs rated valid. If we're below that, we have a data quality problem to fix before fine-tuning.

---

## Part 6: Retrieval Architecture

### The Hybrid Pipeline

Single retrieval methods fail on tax documents:

- **BM25 alone:** Gets exact box matches but misses semantic queries
- **Dense alone:** Gets semantic matches but misses exact terminology
- **Either alone:** No mechanism to expand context

Our pipeline combines them:

```
Query
  │
  ├──► BM25 (top 100)
  │         │
  │         ├──► Lexical matches
  │         │    "Box 1b" exact hit
  │
  ├──► Dense (top 100)
  │         │
  │         ├──► Semantic matches
  │         │    "qualified dividends" → Box 1b section
  │
  └──► Fusion (RRF)
           │
           ├──► Deduplicated top 100
           │
           ▼
      Cross-encoder rerank
           │
           ├──► Top 20 by relevance
           │
           ▼
      Graph expansion
           │
           ├──► Add related via edges
           │    (siblings, references, cross-doc)
           │
           ▼
      Final results with context
```

### Contextual Embedding

Before indexing, we prepend context to each chunk:

**Raw text:** "Enter the portion of dividends that qualifies for reduced capital gains rates."

**Contextualized:** "Form 1099-DIV, Qualified Dividends (filer instructions): Enter the portion of dividends that qualifies for reduced capital gains rates."

Now the embedding encodes not just the content but where it lives. Queries about "qualified dividends" will match not just on lexical overlap but on the context prefix.

This is cheaper than LLM-generated summaries (Anthropic's contextual retrieval approach) because we pull the context from the registry and hierarchy—structured data we already have.

### Graph Expansion

After reranking, we expand the top results via graph edges:

- Follow `parent_of` inverse to get sibling chunks
- Follow `includes` to get referenced content
- Follow `same_field` to offer cross-document perspective

This is how we answer multi-hop questions. "What are qualified dividends?" retrieves the definition chunk. Graph expansion adds the holding period rules, the exception list, and the recipient-side summary.

### Graph as Ranking Signal (Phase B)

In Phase B, we add graph proximity as a ranking factor:

```
final_score = α * dense_sim + β * bm25_sim + γ * graph_proximity
```

If multiple retrieved chunks are connected via edges, they reinforce each other. If a chunk is isolated, it gets less boost.

We defer this to Phase B because using untrusted edges for ranking can hurt precision. We need to validate edge quality first.

---

## Part 7: Evaluation Framework

### Why We Need a Domain Evaluation Set

MTEB and BEIR are useless for measuring improvement on tax documents. They test general retrieval. We need to test tax retrieval.

Our evaluation set has 75 queries across six types:

| Type | Example | What It Tests |
|------|---------|---------------|
| Exact anchor | "What goes in Box 2e?" | Direct box lookup |
| Concept | "What are qualified dividends?" | Definition + supporting detail |
| Procedural | "How do I report foreign tax paid?" | Right box + instructions |
| Scenario | "I received REIT dividends. Are they qualified?" | Multi-hop reasoning |
| Comparative | "Difference between Box 1a and Box 1b?" | Multiple chunks + relationship |
| Edge case | "60-day holding period for qualified dividends?" | Exception rules |

Each query has:
- **Gold chunks:** What should be in the top results
- **Near-miss chunks:** What's related but wrong (should not be top-1)
- **Coverage threshold:** For multi-hop, what % of gold chunks must appear

### Metrics

**Recall@k:** Fraction of gold chunks in top-k results. Our primary metric.

**MRR:** Mean reciprocal rank of first gold chunk. How quickly do we surface something useful?

**Concept coverage:** For multi-hop queries, what fraction of required chunks did we retrieve?

### Checkpoint Validation

We don't just evaluate at the end. Each pipeline stage has checkpoints:

| Stage | Check | Pass Criteria |
|-------|-------|---------------|
| Extraction | Role classification accuracy | >95% on sample |
| Anchor detection | Box coverage | 100% of expected boxes |
| Content assignment | Correct ownership | >90% on LLM eval |
| Reference extraction | Valid references | >90% precision |
| Graph integrity | DAG, connectivity | No cycles, >95% connected |
| Pair quality | Valid pairs | >90% on LLM-as-judge |
| Hard negatives | Not false negatives | <10% false negative rate |

If any checkpoint fails, we stop and fix before proceeding. No point fine-tuning on bad data.

---

## Part 8: Infrastructure

### What We Have

| Component | Technology | Purpose |
|-----------|------------|---------|
| Compute | Databricks notebooks | Pipeline orchestration |
| Storage | Delta Lake | Tables (nodes, edges, pairs) |
| Vector search | Databricks Vector Search | Embedding index |
| LLM | Amazon Bedrock (Claude 3.5 Sonnet) | Extraction, generation, eval |

### Constraints

| Constraint | Impact | How We Handle It |
|------------|--------|------------------|
| No arbitrary model downloads | Can't use LayoutLM, custom Tesseract | PyMuPDF + heuristics for layout |
| No external API during training | Synthetic data must be pre-generated | Cache all LLM outputs in Delta |
| Databricks-only compute | Must use Spark for scale | Design for DataFrame operations |

### Libraries

- **PyMuPDF (fitz):** PDF extraction with layout
- **spaCy:** Pattern matching, NER
- **NetworkX:** Graph construction and validation
- **sentence-transformers:** Embedding and training
- **pyvis:** Graph visualization for demos

---

## Part 9: Current Status and Next Steps

### What's Working

- Cells 1-4: PDF extraction, element construction, role classification, anchor detection
- 22 anchors detected for 1099-DIV including grouped boxes (14-16)
- Box registry schema defined and partially populated
- Graph schema defined and tables created

### What's In Progress

- Cell 6: Merge collision bug (duplicate element IDs)
- Concept section detection (sections without box numbers like "Qualified Dividends")

### What's Next

**Week 1-2:** Fix Cell 6, complete Cells 7-10, emit first page graph to Delta

**Week 3-4:** Reference extraction, graph integrity validation, pyvis demo

**Week 5-6:** Pair generation, hard negative mining, LLM eval on pair quality

**Week 7-8:** Baseline retrieval metrics, fine-tuning experiments

---

## Appendix: Key Decisions and Rationale

| Decision | Options Considered | What We Chose | Why |
|----------|-------------------|---------------|-----|
| Chunking granularity | Fixed-size, semantic, hierarchical | Hierarchical (anchor + paragraph) | Preserves document structure, enables training at anchor level and retrieval at paragraph level |
| Context injection | LLM-generated summaries, structured metadata | Structured metadata from registry | Cheaper, deterministic, sufficient for our domain |
| Hard negative mining | In-batch only, BM25 only, BM25 + graph | BM25 + graph + positive-aware filter | Avoids false negatives that plague BM25-only |
| Retrieval pipeline | Dense only, BM25 only, hybrid | Hybrid (BM25 + dense + rerank) | Handles both exact terms and semantic queries |
| Graph ranking | Context expansion only, ranking signal | Expansion first, ranking in Phase B | Need to validate edges before using for ranking |
| Single vs dual KG | Separate training and retrieval graphs | Single unified graph | Same schema serves both via different query patterns |

---

*This document will be updated as the system evolves. For implementation details, see the Implementation Reference. For schema definitions, see the Schema Catalog.*
