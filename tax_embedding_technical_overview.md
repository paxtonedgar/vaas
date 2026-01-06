# Technical Overview: Graph-Based Contrastive Learning for Tax Document Retrieval

## Executive Summary

We are fine-tuning a BERT-class embedding model to improve semantic retrieval over IRS tax forms and instructions. The core challenge is that tax documents are **cross-referential** - understanding any single piece requires context from multiple other pieces, often across different documents.

Naive chunking destroys these relationships. We solve this by:
1. Building a **knowledge graph** that captures document structure and cross-references
2. Using the graph to generate **contrastive learning pairs** that teach the model which chunks are semantically related
3. Constructing **conceptual chunks** that synthesize related information while maintaining source traceability

---

## Part 1: The Problem

### 1.1 Document Structure

For each IRS form (e.g., 1099-DIV), we have three document types:

```
┌─────────────────────────────────────────────────────────────────┐
│                         1099-DIV                                │
├─────────────────┬─────────────────────┬─────────────────────────┤
│   Form PDF      │  Filer Instructions │  Recipient Instructions │
│                 │                     │                         │
│ • Box labels    │ • Detailed guidance │ • Condensed guidance    │
│ • Field layout  │ • Edge cases        │ • What it means for     │
│ • Structure     │ • Definitions       │   your taxes            │
│                 │ • Regulatory refs   │                         │
└─────────────────┴─────────────────────┴─────────────────────────┘
```

**The same concept appears in all three**, but with different purposes:
- Form: "Box 1b - Qualified dividends"
- Filer: "Enter the portion of dividends that qualifies for reduced capital gains rates..."
- Recipient: "Shows the portion of the amount in box 1a that may be eligible for reduced capital gains rates..."

### 1.2 The Cross-Reference Problem

Tax documents are dense with internal references. From the 1099-DIV filer instructions:

> "Box 1a includes amounts entered in boxes 1b and 2e and it also includes the amount of the recipient's share of investment expenses that you report in box 6."

If we chunk by box, the Box 1a chunk is **semantically incomplete** without understanding boxes 1b, 2e, and 6. But a vector embedding of Box 1a alone won't capture these dependencies.

**Reference patterns we observe:**

| Pattern | Example | Challenge |
|---------|---------|-----------|
| Inclusion | "Box 1a includes amounts in boxes 1b and 2e" | Parent-child relationship |
| Forward reference | "See Section 897 gain, later" | Dependency on future content |
| Back reference | "See Qualified Dividends, earlier" | Dependency on prior content |
| Exception | "The following dividends are not qualified dividends" | Negation relationship |
| External | "See Pub. 550" | Points outside document |
| Cross-document | Same box discussed in filer and recipient instructions | Alignment across docs |

### 1.3 Why Naive Chunking Fails

**Standard approach:**
1. Split document into chunks (by section, paragraph, or token count)
2. Embed each chunk independently
3. Store in vector database
4. Retrieve by cosine similarity to query

**Failure modes:**

```
Query: "What's included in total ordinary dividends?"

Naive retrieval returns: Box 1a chunk
"Enter dividends, including dividends from money market funds..."

Missing context:
- Box 1b content (qualified dividends - subset of 1a)
- Box 2e content (Section 897 ordinary dividends - subset of 1a)  
- Box 6 content (investment expenses - included in 1a)
- Qualified Dividends section (defines terms used in boxes)
```

The embedding model doesn't know these chunks are related because:
1. It was trained on general text, not tax domain
2. The relationships aren't in the surface text of Box 1a
3. No signal during training taught it that "1a" and "1b" should be close

### 1.4 Query Types We Must Handle

| Query Type | Example | Required Retrieval |
|------------|---------|-------------------|
| Box-specific | "What goes in box 1b?" | Single box chunk |
| Concept-specific | "What are qualified dividends?" | Multiple chunks spanning definition, rules, exceptions |
| Procedural | "How do I report foreign tax paid?" | Box 7 + related instructions |
| Comparative | "What's the difference between 1a and 1b?" | Both boxes + their relationship |
| Edge case | "Are REIT dividends qualified?" | Qualified Dividends section + REIT subsection |

Concept queries are the hardest. "Qualified dividends" spans:
- Definition section
- Exceptions subsection
- Holding period rules
- Box 1b filer instructions
- Box 1b recipient instructions
- RICs and REITs special cases

A good retrieval system surfaces all of these. A naive system surfaces one or two.

---

## Part 2: Contrastive Learning for Embeddings

### 2.1 What is Contrastive Learning?

Contrastive learning trains a model to:
- **Pull together** embeddings of semantically similar items
- **Push apart** embeddings of dissimilar items

The model learns a representation space where related concepts cluster and unrelated concepts separate.

```
                    Embedding Space

        ┌─────────────────────────────────────────┐
        │                                         │
        │    [Box 1a] ●────────● [Box 1b]        │  ← Related (1a includes 1b)
        │              \      /                   │
        │               \    /                    │
        │                \  /                     │
        │            [Qualified                   │
        │             Dividends] ●               │
        │                                         │
        │                                         │
        │                                         │
        │    [Box 9] ●                           │  ← Unrelated (liquidation)
        │              \                          │
        │               ● [Box 10]               │
        │                                         │
        └─────────────────────────────────────────┘
```

### 2.2 Training Signal: Pairs and Triplets

**Pair-based training:**
- Positive pair: (anchor, positive) - should be similar
- In-batch negatives: other items in batch treated as negatives

**Triplet-based training:**
- (anchor, positive, negative)
- Loss pushes anchor closer to positive than to negative

```
Triplet Loss:

L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

Where d() is distance (e.g., 1 - cosine_similarity)
```

### 2.3 Why Pair Quality Matters

The model only learns what the pairs teach it. Bad pairs = bad embeddings.

**Good pair (teaches useful relationship):**
```
Anchor:   "Box 1a includes amounts entered in boxes 1b and 2e"
Positive: "Box 1b. Qualified Dividends - Enter the portion..."
Negative: "Box 9. Cash liquidation distributions..."
```

**Bad pair (teaches nothing or wrong thing):**
```
Anchor:   "Box 1a includes amounts entered in boxes 1b and 2e"
Positive: Random chunk from same page (just proximity, no semantic link)
Negative: Random chunk from different page
```

### 2.4 Our Pair Formation Strategy

We generate multiple pair types, each teaching a different relationship:

| Pair Type | What It Teaches | Example |
|-----------|-----------------|---------|
| Query-Passage | Direct retrieval | "What are qualified dividends?" → Qualified Dividends section |
| Graph Neighbor | Reference relationships | Box 1a → Box 1b (connected by "includes" edge) |
| Cross-Document | Same field across doc types | Filer Box 1b ↔ Recipient Box 1b |
| Hierarchical | Parent-child containment | Qualified Dividends section → Exceptions subsection |
| Concept-to-Atomic | Synthesis to sources | Conceptual chunk → its source atomic chunks |

**Hard negatives** are critical. Easy negatives (completely unrelated chunks) don't teach much. Hard negatives (superficially similar but semantically different) force the model to learn fine distinctions.

```
Hard negative example:
Anchor:   Box 1b (Qualified dividends)
Positive: Qualified Dividends definition section
Negative: Box 2a (Total capital gain distributions) ← Same form, numbers, but different concept
```

---

## Part 3: Graph-Based Approach

### 3.1 Why a Graph?

The cross-references in tax documents form a **knowledge graph**. Chunks are nodes. References are edges.

Building this graph explicitly lets us:
1. **Generate better pairs** - Connected nodes should be similar
2. **Mine hard negatives** - Distant nodes in same document
3. **Understand structure** - Hierarchy, containment, dependencies
4. **Construct conceptual chunks** - Traverse to collect related content

### 3.2 Graph Topology

```
                            Document Graph Structure
                            
Level 0: Form Structure
         ┌──────────────────────────────────────────────────────┐
         │  Form 1099-DIV                                       │
         │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐  │
         │  │ 1a  │ 1b  │ 2a  │ 2b  │ ... │ 9   │ 10  │ ... │  │
         │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘  │
         └──────────────────────────────────────────────────────┘
                    │ same_field edges
                    ▼
Level 1: Atomic Chunks
         ┌─────────────────────────────────────────────────────────────────┐
         │                                                                 │
         │  Filer Instructions          Recipient Instructions            │
         │  ┌─────────────────────┐     ┌─────────────────────┐           │
         │  │ Qualified Dividends │     │ Box 1a instructions │           │
         │  │ ├── Definition      │     │ Box 1b instructions │           │
         │  │ ├── Exceptions      │     │ Box 2a instructions │           │
         │  │ └── Foreign corp    │     │ ...                 │           │
         │  │                     │     └─────────────────────┘           │
         │  │ Box Instructions    │                                       │
         │  │ ├── Box 1a          │◄────────── same_field ────────────────┤
         │  │ ├── Box 1b          │                                       │
         │  │ └── ...             │                                       │
         │  └─────────────────────┘                                       │
         │           │                                                    │
         │           │ see_also, includes, exception_to edges            │
         │           ▼                                                    │
         │  Internal reference edges connect chunks within document       │
         └─────────────────────────────────────────────────────────────────┘
                    │ derived_from edges
                    ▼
Level 2: Conceptual Chunks
         ┌─────────────────────────────────────────────────────────────────┐
         │  ● Concept: Qualified Dividends                                │
         │  ● Concept: Section 897 Gain                                   │
         │  ● Concept: Backup Withholding                                 │
         │  ● ...                                                         │
         │                                                                 │
         │  (LLM-synthesized, linked back to source atomic chunks)        │
         └─────────────────────────────────────────────────────────────────┘
```

### 3.3 Node Schema

Every chunk becomes a node in the graph:

```python
Node = {
    # Identity
    "chunk_id": "1099div_filer_box_1a",      # Unique identifier
    "canonical_id": "box_1a",                 # Shared across doc types
    
    # Classification
    "doc_id": "1099-DIV",
    "doc_type": "filer_instructions",         # form | filer | recipient
    "chunk_type": "atomic",                   # atomic | conceptual
    "level": "box",                           # box | section | subsection | paragraph
    
    # Content
    "text": "Enter dividends, including...",
    "text_normalized": "...",                 # Cleaned version
    
    # Hierarchy
    "parent_chunk_id": "1099div_filer_specific_instructions",
    "depth": 2,
    
    # Metadata
    "concepts": ["ordinary_dividends", "money_market_funds"],
    "extraction_confidence": 0.95,
    
    # For conceptual chunks only
    "source_chunk_ids": [...],                # Traceability
    "faithfulness_score": 0.98
}
```

### 3.4 Edge Schema

Edges capture relationships between chunks:

```python
Edge = {
    "edge_id": "e_001",
    "source_chunk_id": "1099div_filer_box_1a",
    "target_chunk_id": "1099div_filer_box_1b",
    
    # Classification
    "edge_type": "includes",                  # See edge types below
    "direction": "directed",                  # directed | bidirectional
    
    # Provenance
    "confidence": 0.92,
    "source_evidence": "Box 1a includes amounts entered in boxes 1b...",
    "created_by": "llm",                      # spacy | llm | structure | embedding_match
    
    # Metadata
    "doc_id": "1099-DIV"
}
```

### 3.5 Edge Types

| Edge Type | Meaning | Direction | Example |
|-----------|---------|-----------|---------|
| `includes` | Source contains/subsumes target | Directed | Box 1a → Box 1b |
| `see_also` | Source references target for info | Directed | Box 2e → Section 897 gain |
| `exception_to` | Source is exception to rule in target | Directed | Exceptions → Qualified Dividends |
| `parent_of` | Structural hierarchy | Directed | Section → Subsection |
| `defines` | Source defines term used in target | Directed | Definition → Usage |
| `same_field` | Same box/field across doc types | Bidirectional | Filer Box 1a ↔ Recipient Box 1a |
| `elaborates` | Source provides more detail than target | Directed | Filer instructions → Recipient instructions |
| `derived_from` | Conceptual chunk to source atomics | Directed | Concept → Atomic |
| `external_ref` | Reference to external document | Directed | Chunk → External doc (not resolved) |

### 3.6 Graph Construction Pipeline

```
┌─────────────────┐
│   PDF Documents │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Extraction │ ──► Chunks with structural metadata
│  & Normalization │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Entity &     │ ──► Box refs, section refs, external refs
│   Reference     │     extracted from chunk text
│   Extraction    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cross-Document │ ──► same_field edges linking docs
│    Alignment    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reference     │ ──► includes, see_also, exception_to edges
│  Classification │     from extracted references
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Hierarchy    │ ──► parent_of edges from doc structure
│    Extraction   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Graph       │ ──► NetworkX graph object
│    Assembly     │     Nodes: chunks, Edges: all types
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Conceptual    │ ──► Concept nodes with derived_from edges
│     Chunks      │
└─────────────────┘
```

---

## Part 4: From Graph to Training Pairs

### 4.1 Graph-Based Pair Generation

The graph directly enables pair generation:

**Positive pairs from edges:**
```python
for edge in graph.edges:
    if edge.type in ["includes", "see_also", "same_field", "elaborates"]:
        pairs.append({
            "anchor": edge.source,
            "positive": edge.target,
            "pair_type": f"similarity_{edge.type}",
            "weight": edge.confidence
        })
```

**Negative pairs from graph distance:**
```python
for node in graph.nodes:
    # Find nodes that are far in graph but same doc
    distant_nodes = get_nodes_with_min_distance(graph, node, min_dist=3)
    for distant in sample(distant_nodes, k=3):
        pairs.append({
            "anchor": node,
            "negative": distant,
            "pair_type": "hard_negative_distant"
        })
```

### 4.2 Pair Type Details

**Type 1: Query-Passage (Atomic)**

LLM generates questions for each atomic chunk:

```
Input chunk: "Box 1b. Shows the portion of the amount in box 1a that may be 
              eligible for reduced capital gains rates..."

Generated queries:
- "What does box 1b show on form 1099-DIV?"
- "Which dividends qualify for reduced capital gains rates?"
- "Where do I find qualified dividends on my 1099?"
```

Pair: (query, chunk) = positive

**Type 2: Query-Passage (Conceptual)**

LLM generates concept-level questions:

```
Input: Conceptual chunk for "Qualified Dividends"

Generated queries:
- "What makes a dividend qualified?"
- "What are the holding period requirements for qualified dividends?"
- "Are REIT dividends considered qualified?"
```

Pair: (query, conceptual_chunk) = positive

**Type 3: Similarity (Graph Neighbors)**

Connected nodes should embed similarly:

```
Edge: Box 1a --[includes]--> Box 1b

Pair: (box_1a_chunk, box_1b_chunk) = positive
      Weight by edge confidence
```

**Type 4: Similarity (Cross-Document)**

Same field across doc types should be close:

```
Edge: Filer_Box_1b --[same_field]--> Recipient_Box_1b

Pair: (filer_box_1b, recipient_box_1b) = positive
```

**Type 5: Hierarchical**

Parent-child relationships:

```
Edge: Qualified_Dividends_Section --[parent_of]--> Exceptions_Subsection

Pair: (section, subsection) = positive
```

**Type 6: Cross-Level (Concept to Atomic)**

Conceptual chunks should be near their sources:

```
Edges: Concept_Qualified_Dividends --[derived_from]--> [source1, source2, source3]

Pairs: (concept, source1) = positive
       (concept, source2) = positive
       (concept, source3) = positive
       (concept, unrelated_atomic) = negative
```

### 4.3 Hard Negative Mining

Easy negatives don't teach much. Hard negatives force discrimination.

**Strategy 1: Graph-based hard negatives**
```
Nodes in same document, similar level, but no edge connection
Example: Box 1b vs Box 2a (both boxes, both amounts, but unrelated)
```

**Strategy 2: Embedding-based hard negatives**
```
Use base embedding model (before fine-tuning)
Find chunks with high similarity but no graph connection
These are "false friends" - similar surface form, different meaning
```

**Strategy 3: Same-concept different-answer**
```
Query: "What are qualified dividends?"
Positive: Qualified Dividends definition
Hard negative: Section about dividends that are NOT qualified (exceptions)
```

### 4.4 Training Data Balance

Different pair types teach different things. Need balance:

```
Target distribution (adjustable):
- Query-passage (atomic):     30%
- Query-passage (conceptual): 15%
- Similarity (graph):         25%
- Similarity (cross-doc):     10%
- Hierarchical:               10%
- Cross-level:                10%
```

Over-representing any type biases the model. Under-representing loses that learning signal.

---

## Part 5: Conceptual Chunks

### 5.1 Purpose

Atomic chunks are source-faithful but fragmented. Conceptual chunks provide synthesized views while maintaining traceability.

```
Atomic chunks for "Qualified Dividends":
┌─────────────────────────────────────────────────────────────────┐
│ Chunk 1: "Qualified Dividends - Except as provided below,      │
│          qualified dividends are dividends paid during..."      │
├─────────────────────────────────────────────────────────────────┤
│ Chunk 2: "Exceptions. The following dividends are not          │
│          qualified dividends..."                                │
├─────────────────────────────────────────────────────────────────┤
│ Chunk 3: "Qualified foreign corporation. A foreign corporation │
│          is a qualified foreign corporation if..."              │
├─────────────────────────────────────────────────────────────────┤
│ Chunk 4: "Box 1b. Enter the portion of the dividends in        │
│          box 1a that qualifies for reduced capital gains..."    │
└─────────────────────────────────────────────────────────────────┘

                              │
                              ▼ LLM Synthesis
                              
Conceptual chunk:
┌─────────────────────────────────────────────────────────────────┐
│ Concept: Qualified Dividends                                    │
│                                                                 │
│ Qualified dividends are dividends paid during the tax year     │
│ from domestic corporations and qualified foreign corporations  │
│ that meet specific holding period requirements [1].            │
│                                                                 │
│ Key requirements:                                               │
│ - Stock held 61+ days during 121-day period around ex-dividend │
│   date [2]                                                      │
│ - For preferred stock with dividends >366 days: 91+ days       │
│   during 181-day period [2]                                     │
│                                                                 │
│ Exclusions [2]:                                                 │
│ - Dividends on insufficiently-held stock                        │
│ - Dividends related to short sale obligations                   │
│ - RIC dividends not qualifying under section 854               │
│ - REIT dividends not qualifying under section 857(c)           │
│                                                                 │
│ Reported in Box 1b as subset of Box 1a ordinary dividends [4]. │
│                                                                 │
│ Sources: [1] Definition section, [2] Exceptions, [3] Foreign   │
│          corp section, [4] Box 1b instructions                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Construction Process

```
1. Identify concept (conservative: named sections only)
                    │
                    ▼
2. Collect source chunks (BFS from concept node, max 2 hops)
                    │
                    ▼
3. LLM synthesis (strict source-faithful prompting)
                    │
                    ▼
4. Faithfulness validation (separate LLM pass)
                    │
                    ▼
5. Citation extraction and verification
                    │
                    ▼
6. Create derived_from edges to sources
```

### 5.3 Source Faithfulness

Critical constraint: Conceptual chunks must not add information beyond sources.

**Enforced via:**
1. Explicit prompt instructions ("ONLY use information from sources")
2. Required inline citations ([1], [2], etc.)
3. Validation pass checking each claim against sources
4. Faithfulness score threshold (>0.95 required)

```python
# Faithfulness validation output
{
    "overall_faithful": True,
    "claims": [
        {
            "claim": "Stock held 61+ days during 121-day period",
            "supported": True,
            "source": "[2]",
            "accurate": True
        },
        ...
    ],
    "omissions": [],      # Important source info not included
    "additions": []       # Info added not in sources (FAIL if present)
}
```

### 5.4 Retrieval Strategy with Conceptual Chunks

**Option A: Retrieve conceptual, return atomic**
```
Query → Search conceptual chunks → Expand to source atomics → Return atomics

Pro: Better concept matching
Con: Extra step
```

**Option B: Retrieve both, re-rank**
```
Query → Search both indexes → Merge results → Re-rank → Return

Pro: Covers both query types
Con: More complex
```

**Option C: Conceptual for routing**
```
Query → Match to concepts → Filter atomics by concept → Search filtered set

Pro: Focused retrieval
Con: Requires concept detection
```

---

## Part 6: Putting It Together

### 6.1 End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   IRS PDFs  │───►│  Extraction │───►│   Chunks    │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                               │                          │
│                     ┌─────────────────────────┼─────────────────────────┐│
│                     │                         ▼                         ││
│                     │               ┌─────────────────┐                 ││
│                     │               │  Reference &    │                 ││
│                     │               │  Entity Extract │                 ││
│                     │               └────────┬────────┘                 ││
│                     │                        │                          ││
│                     │    ┌───────────────────┼───────────────────┐      ││
│                     │    │                   ▼                   │      ││
│                     │    │         ┌─────────────────┐           │      ││
│                     │    │         │ Graph Assembly  │           │      ││
│                     │    │         └────────┬────────┘           │      ││
│                     │    │                  │                    │      ││
│                     │    │    ┌─────────────┴─────────────┐      │      ││
│                     │    │    │                           │      │      ││
│                     │    │    ▼                           ▼      │      ││
│                     │    │ ┌──────────┐           ┌───────────┐  │      ││
│                     │    │ │ Concept  │           │   Pair    │  │      ││
│                     │    │ │ Chunks   │           │Generation │  │      ││
│                     │    │ └────┬─────┘           └─────┬─────┘  │      ││
│                     │    │      │                       │        │      ││
│                     │    │      └───────────┬───────────┘        │      ││
│                     │    │                  │                    │      ││
│                     │    │                  ▼                    │      ││
│                     │    │        ┌─────────────────┐            │      ││
│                     │    │        │   Training      │            │      ││
│                     │    │        │   Pairs         │            │      ││
│                     │    │        └────────┬────────┘            │      ││
│                     │    │                 │                     │      ││
│                     │    └─────────────────┼─────────────────────┘      ││
│                     │                      │                            ││
│                     └──────────────────────┼────────────────────────────┘│
│                                            │                             │
│                                            ▼                             │
│                               ┌─────────────────────┐                    │
│                               │   BERT Fine-Tuning  │                    │
│                               │  (Contrastive Loss) │                    │
│                               └──────────┬──────────┘                    │
│                                          │                               │
│                                          ▼                               │
│                               ┌─────────────────────┐                    │
│                               │   Trained Model     │                    │
│                               └─────────────────────┘                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │    Query    │───►│   Embed     │───►│   Vector    │                  │
│  │             │    │  (trained   │    │   Search    │                  │
│  │             │    │   model)    │    │             │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                               │                          │
│                     ┌─────────────────────────┴─────────────────────────┐│
│                     │                                                   ││
│                     ▼                                                   ││
│           ┌─────────────────┐                                           ││
│           │  Candidate      │                                           ││
│           │  Chunks         │                                           ││
│           │  (atomic +      │                                           ││
│           │   conceptual)   │                                           ││
│           └────────┬────────┘                                           ││
│                    │                                                    ││
│                    ▼                                                    ││
│           ┌─────────────────┐                                           ││
│           │  Expand         │  (if conceptual hit, include sources)     ││
│           │  + Re-rank      │                                           ││
│           └────────┬────────┘                                           ││
│                    │                                                    ││
│                    ▼                                                    ││
│           ┌─────────────────┐                                           ││
│           │  Retrieved      │───►  To user / RAG system                 ││
│           │  Chunks         │                                           ││
│           └─────────────────┘                                           ││
│                                                                         ││
│                     └─────────────────────────────────────────────────────┘
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 What Success Looks Like

**Before fine-tuning:**
```
Query: "What are qualified dividends?"

Top 3 results (base model):
1. Box 1b recipient instructions (partial match)
2. Some unrelated dividend mention
3. Random chunk with word "qualified"
```

**After fine-tuning:**
```
Query: "What are qualified dividends?"

Top 3 results (fine-tuned model):
1. Conceptual chunk: Qualified Dividends (complete synthesis)
2. Qualified Dividends definition section (atomic)
3. Box 1b filer instructions (atomic)

All relevant. Covers definition, rules, and form location.
```

### 6.3 Key Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| MRR (Mean Reciprocal Rank) | How high is the first relevant result? | > 0.7 |
| Recall@5 | Are relevant chunks in top 5? | > 0.85 |
| Recall@10 | Are relevant chunks in top 10? | > 0.95 |
| NDCG | Ranking quality (weighted by position) | > 0.75 |
| Concept coverage | For concept queries, do we retrieve all relevant chunks? | > 0.8 |

---

## Appendix: Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| PDF Extraction | PyMuPDF, pdfplumber | Text and structure extraction |
| Entity Extraction | spaCy + custom patterns | High-precision reference extraction |
| LLM | Claude 3.5 Sonnet (Bedrock) | Reference classification, synthesis, validation |
| Graph | NetworkX | Construction and analysis |
| Storage | Databricks Delta Lake | Chunks, edges, pairs |
| Vector Store | Databricks Vector Search | Embedding index |
| Training | sentence-transformers | BERT fine-tuning |
| Tracking | MLflow | Experiment management |
