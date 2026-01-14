# Repository Guidelines

## Project Structure & Module Organization
Implementation now lives entirely in Python modules under `src/vaas/`: extraction logic in `extraction/`, semantic reasoning in `semantic/`, graph assembly in `graph/`, and primitives/bindings in `core/`. Use `python -m vaas.run_pipeline_v2` as the orchestrator, keep source PDFs in `data/`, and route generated parquet/csv assets to `output/` or `output_v2/` (both ignored). Mirror the runtime layout when creating tests under `tests/`.

## Build, Test, and Development Commands
- Bootstrap a virtualenv and run `make install-dev` to install runtime deps, pytest, black, mypy, flake8, and the spaCy model.
- `make pipeline` executes `python -m vaas.run_pipeline_v2 --validate` against `data/i1099div.pdf` and drops outputs in `output/`.
- `make test`, `make lint`, `make format`, and `make typecheck` wrap pytest, flake8 (100-char budget, ignore `W503`), black, and mypy; keep them green pre-push.
- `make clean` removes build artifacts, caches, and bytecode.

## Coding Style & Naming Conventions
Run Black for formatting (4 spaces, hanging indents) and flake8 for linting. Use `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_SNAKE` for constants. Type hints are required anywhere mypy executes; gate optional imports with `typing.TYPE_CHECKING`. Keep parsing patterns and shared literals in `src/vaas/utils`, and prefer dataclasses or TypedDicts when representing structured graph elements.

## Engineering Principles (Functional Core)
Prefer a functional style inspired by Scala FP: pure functions, immutable data, explicit inputs/outputs, and small composable transforms. Keep side effects at module edges (I/O, persistence). Favor total functions with clear defaults over implicit `None` handling, and avoid hidden global state. Use dataclasses/TypedDicts for typed, immutable-ish records and pass context explicitly instead of relying on globals.

## Evidence-First Outputs (No AI Slop)
Every semantic claim, edge, or derived field must be attributable to evidence in the source instructions. Do not invent facts, gloss over uncertainty, or hand-wave with generic language. Prefer concrete identifiers and provenance fields (`doc_id`, `anchor_id`, `element_id`, `sentence_idx`, char offsets, `ref_occurrence_id`) over narrative summaries. If evidence is missing, mark the artifact as rejected or leave it unmaterialized rather than guessing.

## Semantic-Structural Linking Rules
The structural graph is authoritative for scope. Semantic artifacts must join back to structural nodes via stable keys:
- Claims and typed edges must point to paragraph or anchor nodes, and include the source element ID and sentence offsets.
- Use the sentence index and scope overlay to attach `scope_struct_node_id` and `scope_struct_type`.
- Keep canonical IDs and labels deterministic to enable downstream joins and N-pair mining.

## Ontology & Entity Linking Standards
We are building a semantic layer to support N-pair mining and hybrid retrieval (OpenSearch + graph). Follow modern ontology practices:
- Separate mentions from entities; link mentions to canonical entity IDs with confidence and evidence.
- Disambiguate by context (section, box, form, authority type) and record the disambiguation rule.
- Keep ontology additions aligned with `docs/SEMANTIC_CORE_V3.md` and surface new atoms/regimes in `docs/Form_1099DIV_Ontology.md`.

## Testing Guidelines
Pytest discovers `test_*.py` under `tests/`; name suites after the behavior they cover (e.g., `test_anchor_timeline.py`). New parsing or semantic work must include regression tests for malformed PDFs and scenarios from the latest validation reports, seeding any stochastic logic for determinism. Execute `pytest tests/ -v` and `python validate_graph.py` before submitting changes that affect graph structure or semantics.

## Commit & Pull Request Guidelines
Write imperative, <=50-character commit subjects (`refactor anchors`, `add regime edges`) and mention tickets like `VAAS-42` when relevant. Bodies should capture risk areas or data migrations. Pull requests need: behavior summary, evidence from key commands (`python -m vaas.run_pipeline_v2`, `pytest`, validation scripts), and references to any updated docs. Call out manual follow-ups or data refresh tasks explicitly.

## Semantic Core & Documentation
`docs/SEMANTIC_CORE_V3.md` is the authoritative ontology; earlier versions are archival only. Keep `docs/INDEX.md` and `CLAUDE.md` updated whenever primitives, bindings, or pipeline stages change. Surface new semantic atoms or regime detection rules in `docs/Form_1099DIV_Ontology.md` so other agents can reuse them.

## Data & Configuration Hygiene
Never commit proprietary PDFs, creds, or generated parquet/csv outputs. Store secrets in environment variables or ignored `.env` files and document acquisition steps in `docs/` rather than inline comments. Before sharing artifacts, ensure `output/` and `output_v2/` contain only sanitized data.
