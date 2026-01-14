# Technical Debt: Semantic Core

**Status:** Active tracking
**Last Updated:** 2026-01-12

This document tracks known architectural debt from the initial semantic core implementation.
Items are prioritized by impact on multi-form scalability.

---

## Status Legend

| Status | Meaning |
|--------|---------|
| ✅ DONE | Structure present AND behavior proven with tests |
| ⚠️ STRUCT | Structure exists, behavior unproven or untested |
| ❌ MISSING | Not implemented |

**Rule:** Never mark ✅ DONE unless both structure AND behavior are verified.

---

## Truth Table: What Actually Runs

| Component | Runs in Prod Path | Has Tests | Status |
|-----------|-------------------|-----------|--------|
| `Pipeline` | ✅ | ✅ | BEHAVIOR_PROVEN |
| `FactSet` | ✅ | ✅ | BEHAVIOR_PROVEN |
| `ATOM_REGISTRY` | ✅ | ✅ | BEHAVIOR_PROVEN |
| `InstructionCompiler` | ✅ | ✅ | BEHAVIOR_PROVEN |
| `CorpusProfile` | ✅ | ✅ | BEHAVIOR_PROVEN |
| `CitationScheme` | ✅ | ✅ | BEHAVIOR_PROVEN |
| `DecisionTrace` | ✅ | ✅ | BEHAVIOR_PROVEN |
| `Authority` | metadata only | N/A | Simplified |
| `EconomicEvent` | ❌ | ❌ | Unused |
| `Item` | ❌ | ❌ | Unused |
| `BINDING_REGISTRY` | ❌ | ❌ | Unused |

**Summary:** Core pipeline (FactSet → IR → Bindings → Trace) is tested and working.
Golden E2E test proves behavior. Citation precedence tested.

---

## Critical (Block Multi-Form Scaling)

### 1. Primitives Layer Mixing

**Problem:** `primitives.py` contains 18 "primitives" that mix ontology types with execution artifacts.

**Offenders:**
- `DecisionTrace` - execution artifact, not ontology
- `Derivation` - execution operation
- `ConditionalPrecedence` - resolution framework
- `RoleActivation` - execution context
- `ReportingSurface` - output layer

**Impact:** Freezes core too early. New forms force primitive additions.

**Solution Implemented:** Split into:
```
core_types.py     # Ontology: Authority, EconomicEvent, Item, Actor,
                  # Classification, TemporalConstraint, KnowledgeBoundary, LifecycleEvent
execution_types.py # Framework: Routing, Exclusion, Derivation, Precedence,
                   # DecisionTrace, ReportingEffect
primitives.py      # Re-export facade for backward compatibility
```

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ⚠️ Files exist | ❌ No test verifies separation |

---

### 2. Atom Inflation via Effectful AtomKinds

**Problem:** Current `AtomKind` enum includes effectful categories that are rules, not atoms:

```python
class AtomKind(Enum):
    TEMPORAL = "temporal"           # ✅ Atom (evaluates to value)
    THRESHOLD = "threshold"         # ✅ Atom (evaluates to value)
    DEFAULTING = "defaulting"       # ⚠️ Borderline (has side effect: sets default)
    SCOPE = "scope"                 # ✅ Atom (evaluates to bool)
    DERIVATION = "derivation"       # ❌ RULE (computes new value from others)
    DISCLOSURE = "disclosure"       # ❌ RULE (requires action: emit statement)
    ROUTING = "routing"             # ❌ RULE (causes side effect: redirect)
    DISQUALIFICATION = "disqualification"  # ✅ Atom (evaluates to bool)
```

**The Inflation Pattern:** Once ROUTING/DISCLOSURE/DERIVATION are "atoms", every new form adds:
- 3-5 routing atoms (where values go)
- 2-4 disclosure atoms (what statements required)
- 2-3 derivation atoms (computed fields)

This is exactly what atoms were supposed to prevent.

**Fix Options:**
1. **Remove effectful kinds** - ROUTING/DISCLOSURE/DERIVATION become rule templates, not atoms
2. **Rename the concept** - If keeping effectful kinds, rename from "Atom" to "SemanticModule" (honest about what it is)

**Recommended:** Option 1. Atoms should be:
- Pure predicates (evaluate to True/False)
- Pure values (evaluate to number/string/date)
- No side effects, no routing, no disclosure triggers

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ⚠️ AtomKind enum exists | ❌ Effectful kinds actively cause inflation |

---

### 3. ~~Authority Adapters~~ → Replaced with CorpusProfile

**Problem:** Built a multi-jurisdiction routing abstraction before needing it. For IRS-only scope, this was dead weight.

**Solution:** Replaced `AuthorityAdapter` system with `CorpusProfile`:
- Static config for document extraction (issuer, form_id, revision)
- Citation scheme for normalization (IRC, TREAS_REG, etc.)
- Surface model (what "box" means)
- No routing, no adapters, no multi-jurisdiction abstraction

**Authority is now:**
- A metadata dimension (`authority_id` on FactSet)
- Provenance tracking (`AuthorityProvenance` on KG nodes)
- NOT a routing architecture

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ✅ CorpusProfile exists | ✅ Simpler, fits actual scope |

---

## High (Will Cause Drift)

### 4. No Single Enforced Pipeline

**Problem:** Rules can be defined in multiple places:
- As atoms (ROUTING/DISCLOSURE kinds)
- As predicates
- As operator instructions
- As binding conditions
- As authority adapter policies

**Required Single Pipeline:**
```
FactSet → Predicates → IR Instructions → Resolution → Bindings → Trace
```

**Current State:** Components exist but aren't wired together. A rule author could:
- Put routing logic in an atom
- Put routing logic in a predicate
- Put routing logic in an operator instruction
- Put routing logic in a binding condition

This is semantic diffusion - the rule lives in too many places.

**Fix:** Make the pipeline the ONLY way to execute rules. Block other paths.

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ⚠️ Components exist separately | ❌ No integration, no enforcement |

---

### 5. Registry Determinism Incomplete

**Problem:** Global registries with initialization order create subtle bugs.

**Implemented:**
- `AtomRegistry.freeze()` - prevents post-init registration
- `AtomRegistry.compute_checksum()` - detects version mismatch

**Missing:**
- `BindingRegistry` doesn't have freeze/checksum
- `AuthorityAdapterRegistry` doesn't have freeze/checksum
- No `RegistryManifest` that checksums all registries together
- No way to assert "this test runs against registry version X"

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ⚠️ AtomRegistry has freeze/checksum | ❌ Other registries don't, no unified manifest |

---

### 6. Predicate Complexity Constraints Mis-Aimed

**Problem:** Current constraints (MAX_FACT_REFERENCES=2, MAX_NESTING_DEPTH=1) don't prevent bad predicates.

**What's Constrained:**
- Fact reference count
- Nesting depth

**What Should Be Constrained:**
- AST node count (total complexity)
- Allowed operators (whitelist, not blacklist)
- Totality (must handle all cases, no partial evaluation)
- No side effects (pure evaluation only)

**Example Bad Predicate That Passes Current Constraints:**
```python
"context.holding_days >= atom:holding:61_day.days"  # Passes (1 fact, depth 0)
# But what if atom:holding:61_day doesn't exist? Predicate fails at runtime.
```

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ⚠️ Limits exist | ❌ Limits don't prevent actual problems |

---

## Medium (Technical Debt)

### 7. 1099-INT Doesn't Prove Portability

**Problem:** 1099-INT is same IRS universe as 1099-DIV. Doesn't test authority-agnostic claim.

**What "Portability Proven" Requires:**
- Cross-authority test: Same economic event → different treatment under IRS vs OECD
- Adapter difference test: Authority adapter changes output
- Conflict test: Two authorities with conflicting rules, system handles it

**Current State:** 1099-INT has 7 atoms and 12 bindings. All IRS. Smoke test only.

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ⚠️ form_1099_int.py exists | ❌ Same authority, no cross-authority test |

---

### 8. DecisionTrace Not Mandatory

**Problem:** `DecisionTraceSchema` exists but operators don't require trace emission.

**Impact:** In production, traces will be optional. When debugging, traces won't exist.

**Fix:** Operators should fail if trace metadata not provided. Make it mandatory, not optional.

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ⚠️ Schema exists | ❌ Emission optional, operators don't fail without it |

---

### 9. BindingRegistry Keying Incomplete

**Problem:** `BindingRegistry` is keyed by `semantic_id` alone, not `(semantic_id, authority_id)`.

**Impact:** Can't have different bindings for same semantic concept under different authorities.

**Example:** `qualified_dividends` might map to:
- Box 1b on IRS 1099-DIV
- XML element `QualDiv` in OECD CRS schema

Current registry can't represent this.

| STRUCTURE | BEHAVIOR |
|-----------|----------|
| ⚠️ Registry exists | ❌ Wrong keying, can't support multi-authority |

---

## Addressed (Verified)

*None yet. Items move here only when BOTH structure AND behavior are proven.*

---

## Tracking Metrics

| Metric | Target | STRUCTURE | BEHAVIOR | Status |
|--------|--------|-----------|----------|--------|
| Primitives count | ≤10 ontology | 8 ontology, 10 execution | N/A | ⚠️ STRUCT |
| Atom categories | 5 pure kinds | 8 defined (3 effectful) | Violations ignored | ❌ MISSING |
| Authority adapters | 1 per authority | 2 adapters | Never called | ⚠️ STRUCT |
| Second form coverage | Cross-authority | 1099-INT (same authority) | Smoke test only | ⚠️ STRUCT |
| Operator IR coverage | 100% rules | 9 operators defined | 0 rules compile through | ⚠️ STRUCT |
| Predicate grammar | Constrained | Parser exists | No totality check | ⚠️ STRUCT |
| Typed context model | FactSet | Dataclasses exist | Used in nothing | ⚠️ STRUCT |
| Decision trace schema | Mandatory | Schema exists | Emission optional | ⚠️ STRUCT |
| Conditional bindings | Multi-authority | Method exists | Single authority only | ⚠️ STRUCT |
| Registry determinism | Unified manifest | Partial (AtomRegistry only) | No manifest | ⚠️ STRUCT |
| Single pipeline | Enforced | Components exist | Not wired, not enforced | ❌ MISSING |

---

## Next Actions (Priority Order)

**Phase 1: Stop the Bleeding**
1. Remove effectful AtomKinds (ROUTING/DISCLOSURE/DERIVATION) from AtomKind enum
2. Create rule templates for routing/disclosure/derivation (separate from atoms)
3. Add freeze/checksum to ALL registries

**Phase 2: Wire the Pipeline**
4. Define single pipeline: FactSet → Predicates → IR → Resolution → Bindings → Trace
5. Make this pipeline the ONLY execution path
6. Make trace emission mandatory (operators fail without it)

**Phase 3: Prove Behavior**
7. Create golden E2E fixture: input FactSet → expected IR → expected bindings → expected trace → expected output
8. Create cross-authority test: same facts, different output under IRS vs OECD
9. Create conflict test: two rules conflict, precedence resolves correctly
10. Create revival test: uncertainty resolves, treatment changes

**Phase 4: Expand (Only After Phase 3)**
11. Add OECD CRS form (different authority, different surface)
12. Add real precedence conflict (not toy example)
13. Add uncertainty lifecycle test

---

## Golden E2E Fixture (Required for Phase 3)

```python
# Input
facts = FactSet(
    actor=ActorFacts(tin="123-45-6789", is_us_person=True),
    instrument=InstrumentFacts(cusip="ABC123", is_qualified_dividend_eligible=True),
    holding=HoldingFacts(acquisition_date=date(2025, 1, 1), disposition_date=None),
    amount=AmountFacts(gross_amount=Decimal("1000.00")),
    temporal=TemporalFacts(tax_year=2025, payment_date=date(2025, 6, 15)),
)

# Expected IR (what predicates produce)
expected_ir = [
    QualifyInstruction(
        qualification_id="qual:qualified_dividend",
        eligibility_predicates=["holding_days >= 61", "is_qualified_dividend_eligible == True"],
    ),
    RouteInstruction(
        source_id="amt:gross_dividend",
        destination="box:1099div:1b",  # Qualified dividends
    ),
]

# Expected bindings (what IR resolves to)
expected_bindings = [
    BindingResult(
        semantic_id="qualified_dividends",
        form_slot="1099-DIV:Box1b",
        value=Decimal("1000.00"),
    ),
]

# Expected trace (explainability)
expected_trace = DecisionTraceSchema(
    entries=[
        TraceEntry(action="QUALIFY", result="qualified", rule_id="rule:qdiv:1"),
        TraceEntry(action="ROUTE", result="box:1099div:1b", rule_id="rule:qdiv:route"),
    ],
)

# Expected output (what user sees)
expected_output = {
    "form": "1099-DIV",
    "tax_year": 2025,
    "boxes": {
        "1a": Decimal("1000.00"),  # Total ordinary dividends
        "1b": Decimal("1000.00"),  # Qualified dividends
    },
}
```

This fixture, when passing, proves the pipeline works end-to-end.
