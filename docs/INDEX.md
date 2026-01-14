# VaaS Documentation Index

## Overview

VaaS (Value-as-a-Service) is a graph-based RAG system for IRS tax instruction processing. This index links all design documentation.

---

## Architecture Documents

### Semantic Foundation (Versioned)

| Document | Version | Purpose | Status |
|----------|---------|---------|--------|
| [SEMANTIC_CORE_V3.md](./SEMANTIC_CORE_V3.md) | **v3.0** | **18 Primitives** - Jurisdiction-agnostic legal reasoning infrastructure | **Current** |
| [TAX_METAMODEL_DESIGN.md](./TAX_METAMODEL_DESIGN.md) | v2.0 | Integration Architecture - graph schema, atoms, bindings | Design Complete |

### Version Evolution

```
v1.0: IRS-specific ontology (1099-DIV hardcoded)
v2.0: US tax execution ontology (IRS concepts as instances)
v3.0: Jurisdiction-agnostic legal reasoning (Authority, EconomicEvent, DecisionTrace)
```

### Form-Specific Ontologies

| Document | Purpose | Status |
|----------|---------|--------|
| [Form_1099DIV_Ontology.md](./Form_1099DIV_Ontology.md) | 1099-DIV semantic model - entities, rules, windows, relationships | Reference |

---

## Reading Order

### For US Tax Implementation (IRS focus)

1. **[SEMANTIC_CORE_V3.md](./SEMANTIC_CORE_V3.md)** - 18 primitives with Authority, EconomicEvent, DecisionTrace
2. **[TAX_METAMODEL_DESIGN.md](./TAX_METAMODEL_DESIGN.md)** - Integration architecture
3. **[Form_1099DIV_Ontology.md](./Form_1099DIV_Ontology.md)** - 1099-DIV reference

### For Jurisdiction-Agnostic Systems (Multi-authority)

1. **[SEMANTIC_CORE_V3.md](./SEMANTIC_CORE_V3.md)** - Base primitives (current)
2. Reference `docs/tax_rag_technical_overview.md` and `docs/tax_rag_schema_catalog.md` for historical IRS-specific mapping details

---

## Key Concepts Quick Reference

### The 18 Primitives (v3.0)

| # | Primitive | One-Liner | New in v3? |
|---|-----------|-----------|------------|
| 1 | **Authority** | Jurisdiction/regime that defines rules | NEW |
| 2 | **EconomicEvent** | What happened economically (pre-Item) | NEW |
| 3 | **Item** | Reportable unit derived from event | |
| 4 | **Classification** | Tax treatment category | |
| 5 | **Eligibility** | Boolean condition evaluation | NEW (from Qualification) |
| 6 | **TaxTreatmentChange** | Treatment effect when eligible | NEW (from Qualification) |
| 7 | **ReportingEffect** | Reporting consequence | NEW (from Qualification) |
| 8 | **TemporalConstraint** | Time-bounded conditions | |
| 9 | **KnowledgeBoundary** | Epistemic uncertainty with lifecycle | Extended |
| 10 | **ReportingSurface** | Authority-defined reporting target | Modified |
| 11 | **Actor** | Party in chain | |
| 12 | **RoleActivation** | Dynamic role assumption | |
| 13 | **Routing** | Item → Surface assignment | |
| 14 | **Exclusion** | Mutual impossibility | |
| 15 | **ConditionalPrecedence** | Non-linear conflict resolution | Modified |
| 16 | **LifecycleEvent** | Terminations, dissolutions | NEW |
| 17 | **DecisionTrace** | Explainability trail | NEW |
| 18 | **Derivation** | Slot arithmetic | |

### Layer Architecture (v3.0)

```
Authority Layer:   Authority defines all rules below
                        ↓
Economic Layer:    EconomicEvent → produces → Item(s)
                        ↓
Evaluation Layer:  Eligibility → TaxTreatmentChange → ReportingEffect
                        ↓
Resolution Layer:  ConditionalPrecedence (non-linear, cyclic-safe)
                        ↓
Output Layer:      ReportingSurface (form, API, ledger)
                        ↓
Audit Layer:       DecisionTrace (explainability)
```

### Design Principles (v3.0)

1. **Authority-agnostic, not form-first**
   - Every rule is scoped to an Authority (IRS, OECD CRS, etc.)

2. **EconomicEvent precedes Item**
   - One event can produce multiple Items under different authorities

3. **Qualification is decomposed**
   - Eligibility (boolean) + TaxTreatmentChange (effect) + ReportingEffect (reporting)

4. **Precedence is non-linear**
   - ConditionalPrecedence handles cycles and context-dependent ordering

5. **Uncertainty has lifecycle**
   - KnowledgeBoundary tracks expiration, transition, dispute resolution

6. **Decisions are traceable**
   - DecisionTrace provides audit trail and explainability

---

## Implementation Status

| Component | Location | Status |
|-----------|----------|--------|
| Pipeline Orchestration | `run_pipeline_v2.py` | **Active** – modular CLI replacing notebooks |
| PDF & Layout Extraction | `src/vaas/extraction/` | Cells 1-6 complete, sections/anchors in progress |
| Semantic Regime Detection | `src/vaas/semantic/` | Under active development |
| Claim Resolution & Compilation | `src/vaas/semantic/resolution.py`, `src/vaas/semantic/compiler.py` (directive IR) | Under active development |
| Graph Construction | `src/vaas/graph/` | Node/edge builders stable |
| Core Primitives & Atoms | `src/vaas/core/primitives.py`, `src/vaas/core/atoms.py` | **Complete** (18 primitives + registry) |
| Validation & Coverage | `validate_graph.py`, `output/graph_quality_report.md`, `output/ontology_coverage_report.md` | Update after each pipeline run |

---

## Related Files

- [CLAUDE.md](../CLAUDE.md) - Project instructions for Claude Code
- [output/graph_quality_report.md](../output/graph_quality_report.md) - Latest quality check results
- [output/ontology_coverage_report.md](../output/ontology_coverage_report.md) - Ontology coverage gaps
