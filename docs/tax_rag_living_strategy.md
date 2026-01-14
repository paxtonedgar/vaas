# VaaS Tax Document Intelligence: Living Strategy

**Version:** 1.0  
**Last Updated:** 2026-01-05  
**Status:** Living document — updated as we learn and validate

---

## What This Document Is

This is our strategic compass. It documents our architectural approach, the risks we've identified, and how our design addresses those risks. It's "living" because our understanding evolves—when we validate or invalidate assumptions, we update this document.

This is not a gap list. It's an integrated strategy where risks and mitigations are woven into the architecture itself.

---

## The Strategic Thesis

**We believe** that tax document retrieval requires domain-specific embeddings trained on structurally-aware pairs, combined with graph-augmented retrieval that leverages document relationships.

**We will know we're right when** fine-tuned embeddings outperform off-the-shelf models by >15% on Recall@5 against a tax-specific evaluation set, and multi-hop queries achieve >80% concept coverage.

**We will know we're wrong if** contextual retrieval with off-the-shelf embeddings achieves comparable results, in which case we should simplify the architecture and skip fine-tuning.

This strategy is expensive to build. We're betting that the domain is hard enough to justify the investment. The evaluation framework tells us if we're wrong early.

---

## Part 1: Extraction Architecture

### The Approach

We extract PDF content while preserving layout signals (font, position, nesting) that encode semantic relationships. We don't use OCR—IRS PDFs are digitally authored. We classify elements by role (header, body, list) and detect structural anchors (boxes, sections) that organize the content.

### The Risk: Layout Heuristics Are Fragile

**What could go wrong:** Our role classification depends on font size thresholds and regex patterns. A different form might use different fonts or header styles. We could misclassify headers as body text or vice versa.

**How we're addressing it:**

1. **Confidence scores on every classification.** We don't just say "this is a SectionHeader"—we say "this is a SectionHeader with 0.87 confidence." Low-confidence classifications get flagged for review.

2. **Validation checkpoint.** Before proceeding past extraction, we sample 50 elements and verify role classification manually. If accuracy is below 95%, we stop and adjust.

3. **Form-specific tuning allowed.** The extraction pipeline is parameterized. If 1099-INT needs different thresholds, we can adjust without rewriting the pipeline.

4. **Registry as ground truth.** The box registry tells us what anchors *should* exist. If anchor detection misses expected boxes, we catch it in validation.

### The Risk: Concept Sections Aren't Detected

**What could go wrong:** We detect box headers ("Box 1a") via regex, but concept sections like "Qualified Dividends" don't match box patterns. These sections define what goes in boxes—they're semantically critical but structurally different.

**How we're addressing it:**

1. **Registry-driven detection.** The box registry contains aliases for concept sections. "Qualified Dividends" maps to `canonical_id = qualified_dividends`. During anchor detection, we check SectionHeaders against registry aliases.

2. **Phase A scope limitation.** For Phase A, we're adding concept sections manually to the registry for 1099-DIV. This is acceptable for one form.

3. **Phase B generalization.** For Phase B (multi-form), we'll extend detection to recognize SectionHeaders that match registry aliases automatically.

---

## Part 2: Knowledge Graph Architecture

### The Approach

We build a knowledge graph with nodes (chunks at various granularities) and edges (relationships). The graph serves both training pair generation and retrieval augmentation from a unified schema.

### The Risk: Wrong Granularity for Different Uses

**What could go wrong:** Training wants larger chunks (enough signal per pair). Retrieval wants smaller chunks (precise hits). We could build a graph that's great for training but terrible for retrieval, or vice versa.

**How we're addressing it:**

**Hierarchical nodes.** The graph has multiple levels:
- Level 1: Anchors (box/section headers) — used for training pairs
- Level 2: Paragraphs — used for retrieval
- Level 3: Sentences (optional) — for particularly dense content

Training queries edges at level 1. Retrieval indexes at level 2. The hierarchy connects them—every paragraph knows its parent anchor.

**Why this works:** When we generate a hierarchical training pair (anchor → paragraph), we're teaching the model that the paragraph is semantically close to its anchor. When we retrieve a paragraph and expand to its siblings, we're using the same hierarchy. Same graph, different query patterns.

### The Risk: Training on Untrusted Edges

**What could go wrong:** We extract cross-references via regex and LLM. Some will be wrong. If we train on wrong edges, we teach the model bad relationships. If we use wrong edges for ranking, we degrade retrieval.

**How we're addressing it:**

**Confidence scores and provenance.** Every edge has:
- `confidence`: How sure we are (0.0-1.0)
- `created_by`: How we found it (structural, regex, llm)
- `source_evidence`: The text that supports it

**Tiered trust:**
- Structural edges (`parent_of`, `follows`) are 1.0 confidence—derived from document structure
- Regex edges (`includes` from "Box 1a includes 1b") are 0.95 confidence—pattern is unambiguous
- LLM edges are variable confidence based on model's stated certainty

**Gated usage:**
- Phase A: Use only structural edges for ranking (they're trustworthy)
- Phase B: After validating reference edges (>90% precision on manual review), add them to ranking

**Training pair filtering:** We weight training pairs by edge confidence. High-confidence edges contribute more to the loss. Low-confidence edges are downweighted or excluded.

### The Risk: Registry Coverage Gaps

**What could go wrong:** The registry maps box keys to canonical IDs and aliases. If a box isn't in the registry, we can't align it across documents. If an alias is missing, natural language lookup fails.

**How we're addressing it:**

1. **Registry validation.** Before pipeline runs, we verify registry has entries for all expected boxes. For 1099-DIV, that's boxes 1a-16.

2. **Alias expansion.** We seed aliases from form text and expand via observed usage. If users frequently ask "QDI" for qualified dividends, we add that alias.

3. **Confidence tracking.** Registry entries have a `source` field (irs_form, manual, extracted). We know which entries are authoritative vs. inferred.

4. **Graceful degradation.** If a node can't resolve to a canonical ID, it still functions—just without cross-document alignment. We log these cases for registry updates.

---

## Part 3: Contextual Retrieval Strategy

### The Approach

Before embedding, we prepend context to each chunk: form name, section title, document type. This addresses the "implicit context" problem where chunks are locally correct but globally ambiguous.

### The Risk: Context Bloat or Wrong Context

**What could go wrong:** If context is too long, it dominates the embedding. If context is wrong (we attribute content to the wrong section), we make things worse.

**How we're addressing it:**

**Structured context from registry.** We don't use LLM-generated summaries (Anthropic's original approach). We pull context from the registry and hierarchy:

```
"Form 1099-DIV, Qualified Dividends (filer instructions): {chunk text}"
```

This is:
- Deterministic (same chunk always gets same context)
- Cheap (no LLM call per chunk)
- Verifiable (context comes from structured data we control)

**Hierarchy traversal for context.** The context path comes from traversing `parent_node_id`:
```
paragraph → anchor → section → form
```

If this hierarchy is correct (validated in extraction), the context is correct.

**A/B testing.** We'll measure retrieval with and without context prefixes. If context hurts (possible if embeddings already capture structure), we drop it.

### The Risk: Single-Vector Embeddings Compress Away Precision

**What could go wrong:** "Box 2e" and "Box 2f" are one character apart. Single-vector embeddings map them to similar vectors. For tax documents, that's wrong.

**How we're addressing it:**

**BM25 for lexical precision.** The hybrid pipeline uses BM25 alongside dense retrieval. BM25 does exact matching—"Box 2e" matches "Box 2e", not "Box 2f".

**Reranking for discrimination.** The cross-encoder reranker does pairwise comparison. Even if "Box 2e" and "Box 2f" have similar embeddings, the reranker can distinguish them.

**ColBERT as benchmark.** ColBERT uses token-level late interaction—it preserves per-token precision better than single-vector models. We'll benchmark ColBERT vs. our fine-tuned single-vector model. If ColBERT wins significantly on exact-match queries, we reconsider.

---

## Part 4: Training Pair Strategy

### The Approach

We generate training pairs by traversing graph edges. Different edge types yield different training signals: hierarchical (containment), cross-reference (navigation), same-field (alignment), hard negatives (discrimination).

### The Risk: Hard Negatives Are Actually Positives

**What could go wrong:** BM25 retrieval for hard negative mining returns ~70% false negatives. "Box 1b" appears in "Box 1a" text (because 1a references 1b). BM25 thinks they're similar. But they're actually related—using 1b as a negative for 1a teaches the wrong thing.

**How we're addressing it:**

**Graph-aware mining.** We filter BM25 candidates to graph-distant nodes (no path within 3 hops). If there's an edge between them, they're not negative.

**Positive-aware threshold.** After filtering by graph distance, we compute embedding similarity to known positives. If similarity to any positive > 0.85, it's probably a false negative—discard.

**LLM validation.** For sampled hard negatives, we ask: "Are these genuinely unrelated?" Pass criterion: <10% false negative rate.

### The Risk: Pair Quality Too Low to Train

**What could go wrong:** We generate pairs mechanically from edges. Some edges are wrong. Some pairs are trivial (not useful for learning). We could have a dataset that looks large but doesn't teach the model anything.

**How we're addressing it:**

**LLM-as-judge validation.** Before training, we sample 50 pairs per type and evaluate:
- Is this pair valid?
- For positives: Are they semantically related in a useful way?
- For negatives: Are they genuinely unrelated?

Pass criterion: >90% valid. If we're below that, we have a data problem to fix.

**Diversity monitoring.** We track pair type distribution. If hierarchical pairs dominate (easy to generate), the model only learns containment. We need balance across pair types.

**Weighting by confidence.** Pairs from high-confidence edges get weight 1.0. Lower-confidence edges get downweighted. This prevents noisy edges from dominating the loss.

---

## Part 5: Retrieval Architecture

### The Approach

Hybrid retrieval: BM25 for lexical, dense for semantic, RRF fusion, cross-encoder reranking, graph expansion.

### The Risk: Graph Expansion Amplifies Wrong Retrievals

**What could go wrong:** We retrieve a wrong chunk. Graph expansion pulls in its neighbors. Now we've returned more wrong content, not less.

**How we're addressing it:**

**Expand after reranking.** We don't expand the raw retrieval results. We rerank first (cross-encoder), then expand only the top-20 reranked results. This filters noise before expansion.

**Expansion rules are edge-type specific:**
- `parent_of`: Add siblings (safe—same section)
- `includes`: Add target (intended—the reference exists for a reason)
- `same_field`: Add cross-doc equivalent (safe—same semantic field)
- `follows`: Don't expand (reading order isn't semantic)

**Boost, don't replace.** Expanded nodes get a score boost, not automatic top ranking. They still have to compete.

### The Risk: Graph Ranking Degrades Precision

**What could go wrong:** We add graph proximity as a ranking signal. Bad edges cause bad proximity scores. Precision drops.

**How we're addressing it:**

**Phased rollout:**
- Phase A: Graph for expansion only, not ranking
- Phase B: After edge precision validated at >90%, add graph to ranking

**Tunable weights.** The final score is `α * dense + β * bm25 + γ * graph`. We can set γ = 0 if graph hurts, or tune it empirically.

**Per-edge-type contribution.** Not all edges are equal for ranking. Structural edges (hierarchy) are safer than extracted edges (references). We can weight by edge type.

---

## Part 6: LLM Usage & Provenance

### The Principle

LLMs are supervised operators inside a provenance framework—not sources of truth.

The LLM can *propose* structure, edges, judgments, and routes. But every proposal must emit evidence pointers back to atomic corpus artifacts. A separate pass (deterministic or validation) checks that every claim is supported by cited sources. Downstream stages weight decisions by confidence + evidence quality.

This isn't about distrust. It's about debuggability. When an answer is wrong, we need to trace *why*—which chunk, which edge, which extraction decision. If the LLM invented something without grounding, we can't debug it.

### The Provenance Pointer Standard

Every artifact in the system should be able to roundtrip to the PDF. The test: **"Can we click 'show me the evidence' and highlight the exact rectangle(s) that support this?"**

The pointer includes:
- `doc_id`, `doc_type`, `page`
- `element_id` or `span_ids` (lowest-level truth)
- `bbox` (so we can render-highlight)
- `text_raw` + `content_hash`
- `extraction_version`

Our schema already captures most of this. The discipline is ensuring nothing downstream loses the pointer.

### Tiered LLM Usage

We already encode this implicitly. Making it explicit:

**Tier 1: Deterministic (no LLM)**
- Regex patterns for box references, cross-references
- Layout rules for role classification
- Registry lookup for alias resolution
- Emits with confidence ~1.0

**Tier 2: LLM proposes candidates**
- Only invoked when Tier 1 confidence < threshold
- Used for: ambiguous references, implicit relationships, section boundary disputes
- Must emit evidence pointers with proposal

**Tier 3: Validation pass**
- Checks LLM proposal against cited evidence
- If evidence doesn't support claim → discard or downgrade confidence
- This is how our edge confidence tiers work: structural (1.0), regex (0.95), LLM (variable based on validation)

### The LLM Output Contract

Any LLM call that changes state (creates edge, assigns anchor, validates pair, routes query, synthesizes concept) must produce:

```
{
  "decision": "...",
  "confidence": 0.0-1.0,
  "evidence": [
    {"doc_id": "...", "page": N, "element_id": "...", "quote": "..."}
  ],
  "unsupported_claims": [],  // Must be empty to pass
  "abstain_reason": null     // If not confident, explain why
}
```

If `unsupported_claims` is non-empty, the output fails validation. This is already how our conceptual chunk faithfulness validator works—we're just applying it universally.

### Where This Applies

**Extraction (Cell 9: Reference Extraction)**
- Regex handles "Box 1a includes 1b" patterns
- LLM handles "see the discussion above" (ambiguous)
- LLM must cite which element contains "the discussion above"

**Pair Validation (LLM-as-Judge)**
- Judge prompt requires: decision + minimal supporting quotes + mapping to source element IDs
- "Additions" (implied but unsupported claims) cause failure

**Query Routing**
- LLM produces: intent_type, target_canonical_ids, confidence
- Must cite alias match or query phrase that triggered the route
- If it can't cite evidence → must abstain → fall back to hybrid retrieval

**Graph Expansion**
- Every expanded node inherits provenance: `expanded_via_edge_id`, `edge_type`, `edge_confidence`, `source_evidence`
- Debug output can show: "This chunk was added because Box 1a includes 1b" with the evidence text

### The Unit of Quotable Evidence

What's the smallest unit we consider citable? This matters because LLMs can summarize across multiple sources, but citations must point to something we can highlight.

For our IRS PDF work: the atomic unit is **elements** (post-split lines with role classification) or **section chunks** (role-contiguous content under an anchor). These are where layout semantics are preserved and bboxes are meaningful.

The rule: LLM may synthesize and summarize, but every citation must resolve to an element or chunk that can be highlighted in the PDF. If a claim can't be traced to a highlightable unit, it's unsupported.

### What This Doesn't Mean

This isn't "avoid LLMs." We use them heavily:
- Scenario query generation
- Pair quality validation  
- Ambiguous reference resolution
- Conceptual chunk synthesis

But in every case, the LLM is a *proposer* that must ground its proposals in evidence we can verify. The provenance chain stays intact.

---

## Part 7: Evaluation Strategy

### The Approach

We build a domain-specific evaluation set (75 queries) and measure throughout the pipeline, not just at the end.

### The Risk: Evaluation Set Too Easy or Too Hard

**What could go wrong:** If queries are too easy (exact box lookup), we overestimate performance. If too hard (require external knowledge), we underestimate.

**How we're addressing it:**

**Stratified query types:**
- Exact anchor (15): Direct lookup, establishes baseline
- Concept (20): Definition + supporting detail
- Procedural (10): Task-oriented
- Scenario (15): Multi-hop reasoning
- Comparative (5): Multiple chunks
- Edge case (10): Exception rules

**Near-miss negatives.** Each query has "wrong but related" chunks. A system that returns Box 1a for a Box 1b question scores zero, even though they're related.

**Coverage metrics for multi-hop.** For scenario queries, we don't just check top-1. We check what fraction of required chunks appear in top-10.

### The Risk: Baseline Is Too Weak

**What could go wrong:** We compare fine-tuned model to a bad baseline, declare victory, ship—and it's actually not that good.

**How we're addressing it:**

**Strong baseline.** Our baseline is:
- Cohere Embed v3 (top-tier general model)
- With contextual prefixes
- With BM25 hybrid
- With cross-encoder reranking

This is a legitimately strong baseline. If we can't beat this, fine-tuning might not be worth it.

**Absolute thresholds.** We don't just measure "better than baseline." We have absolute targets:
- Recall@5 > 0.85
- Concept coverage > 0.80
- End-to-end accuracy > 95%

If we beat baseline but miss absolute targets, we haven't solved the problem.

---

## Part 8: Phased Delivery

### Phase A: Foundation (Weeks 1-4)

**Goal:** Working KG for 1099-DIV with validated extraction and baseline retrieval metrics.

**Scope:**
- Cells 1-10 complete
- Graph populated for 1099-DIV filer instructions
- Reference edges extracted and validated
- Evaluation set created
- Baseline metrics measured

**Exit criteria:**
- Graph integrity passes (DAG, >95% connected)
- 100% box coverage
- Reference precision >90%
- Baseline Recall@5 measured (any value—we're establishing the number to beat)

### Phase B: Training and Improvement (Weeks 5-8)

**Goal:** Fine-tuned embeddings with demonstrated improvement.

**Scope:**
- Pair generation pipeline complete
- All pair types generated and validated
- Fine-tuning experiments complete
- Graph ranking signal added (after edge validation)

**Exit criteria:**
- Pair quality >90% valid
- Hard negative FN rate <10%
- Fine-tuned Recall@5 > baseline + 10%

### Phase C: Multi-Form Expansion (Weeks 9-12)

**Goal:** Generalized pipeline for 1099 series.

**Scope:**
- 1099-INT, 1099-MISC extraction
- Cross-form concept nodes
- Expanded evaluation set

**Exit criteria:**
- Extraction accuracy maintained on new forms
- No regression on 1099-DIV metrics

### What Could Change This Plan

**We accelerate if:** Baseline metrics are already strong. In this case, focus on retrieval infrastructure, reduce fine-tuning scope.

**We slow down if:** Extraction accuracy is poor. Can't build a good graph on bad extraction. Fix foundation before proceeding.

**We pivot if:** Fine-tuning doesn't help. If Phase B shows minimal improvement, consider ColBERT, longer context models, or structured query approaches instead of custom embeddings.

---

## Part 9: Open Questions

These are things we don't know yet. We'll update this section as we learn.

---

### Dialectical Challenges (Do We Need All This?)

These questions challenge whether we're overbuilding. They should be answered with numbers, not vibes.

**Q: Do we need a KG at all to get 80% of the benefit?**

Anthropic's contextual retrieval shows that prepending structured context + hybrid BM25/dense + reranking gets substantial gains *without* graph expansion or fine-tuning. Our contextualized prefixes from registry/hierarchy might already capture most of the value.

Experiment needed: Measure baseline with contextual chunks + hybrid + rerank. If Recall@5 > 0.80 without graph expansion, the KG value proposition changes from "necessary" to "incremental."

**Q: If ColBERT/SPLADE already solve the precision failure mode, why bet on fine-tuning single-vector?**

We worry that single-vector embeddings blur "Box 2e" vs "Box 2f". But ColBERT (token-level late interaction) and SPLADE (learned sparse retrieval) already address this without custom training. 

Experiment needed: Benchmark ColBERTv2 and/or SPLADE on "near-miss" queries (2e vs 2f style). If they dominate single-vector on precision-sensitive queries, fine-tuning single-vector may not be worth it.

**Q: Do we want graph traversal for determinism, or graph summarization for discovery?**

GraphRAG is strongest for "global narrative" corpora with query-focused summarization. Our domain is highly structured—often answerable by strict navigation (anchor → section → included refs). That's graph *traversal*, not graph *summarization*. These have different costs.

Decision needed: Are we building a navigation graph or a summarization graph? If navigation only, we can skip community detection, LLM summaries, and most of GraphRAG's complexity.

**Q: Can a structured query router avoid retrieval entirely for many queries?**

Many queries are deterministic:
- "Explain Box 2a" → registry lookup → return section
- "What goes in Box 1b" → alias match → return section
- "Where do I report foreign tax paid" → alias match to Box 7

These don't need retrieval at all. They're routing problems, not search problems.

Experiment needed: What % of real queries can be resolved to a canonical box/section via aliases alone? If >50%, build a router first and only fall back to retrieval for ambiguous queries.

**Q: Does our layout extraction pipeline beat a cheap baseline?**

We have a sophisticated extraction pipeline (spans → elements → anchors with font/position heuristics). 

Experiment needed: Compare against a "cheap baseline" (pdf text extraction + regex anchors + simple page order). If cheap baseline achieves >90% of our accuracy on 1099-DIV, keep the complex pipeline as fallback, not default.

**Q: Are there cheaper pair quality proxies than LLM-as-judge?**

We propose LLM-as-judge for pair validation. But cheaper options exist:
- Self-consistency checks (does the model agree with itself?)
- Heuristic edge validity (regex references are high precision by construction)
- Automatic contradiction checks ("includes" edges should show mention patterns in text)

Experiment needed: Can heuristic checks filter 80% of bad pairs before LLM judging? If yes, use LLM only for ambiguous cases.

---

### Dialectical Checklist (Answer With Numbers)

Before declaring any component "done," answer these:

| Question | Target | How to Measure |
|----------|--------|----------------|
| Routing hit-rate | >50% | % of eval queries resolvable via registry/alias lookup |
| Baseline ceiling | Recall@5, coverage | Contextual + hybrid + rerank, no graph, no fine-tuning |
| Precision stress test | ColBERT vs single-vector | Near-miss queries (2e vs 2f style) |
| Graph marginal value | Coverage delta | Multi-hop coverage with vs without expansion after rerank |
| Fine-tuning ROI | +10-15% over baseline | Only justified if it beats strongest baseline materially |

---

### Training Questions

**Q: What loss function for fine-tuning?**
Options: MultipleNegativesRankingLoss (good for in-batch negatives), TripletLoss (good for mined hard negatives).
Decision: Deferred until we see pair distribution.

**Q: Symmetric or asymmetric encoding?**
Options: Same encoder for query and document (simpler), different representation spaces (better for short query → long document).
Decision: Start with Cohere's native `input_type` differentiation. If fine-tuning our own model, revisit.

**Q: What teacher model for negative filtering?**
The positive-aware threshold checks similarity to positives. We need embeddings for this. Use same model being trained, or a separate teacher?
Decision: Use Cohere Embed v3 as teacher. Avoid bootstrap problem of training model judging its own negatives.

---

### Retrieval Questions

**Q: ColBERT vs. single-vector?**
ColBERT preserves token-level precision but has storage overhead. SPLADE offers learned sparse retrieval with inverted-index exactness.
Decision: Benchmark ColBERTv2 and SPLADE in Phase B. If either wins on exact-match queries by >5%, reconsider single-vector fine-tuning.

**Q: Community summaries for global queries?**
GraphRAG uses community detection and summary generation for questions that span the corpus.
Decision: Not needed for Phase A (single form). Revisit in Phase C if cross-form queries are common. Note: We likely need traversal, not summarization—different cost profile.

---

### Evaluation Questions

**Q: What's the right coverage threshold for multi-hop?**
80%? 90%? Lower threshold is easier to hit but might not represent real usefulness.
Decision: Start with 80%. Adjust based on user feedback.

**Q: What defines "comparable" for baseline vs. fine-tuned, per query class?**

| Query Class | Baseline Expectation | Fine-tuning Justification |
|-------------|---------------------|---------------------------|
| Exact anchor | Should be ~perfect with routing | If not, extraction/registry is broken, not embeddings |
| Concept | Contextual retrieval should hit definition + 1 supporting rule | Fine-tuning justified if coverage jumps significantly |
| Multi-hop | Graph expansion should help | Quantify coverage uplift from expansion before building more edges |

---

*This document is updated as we validate or invalidate assumptions. Last major revision: 2026-01-05.*
