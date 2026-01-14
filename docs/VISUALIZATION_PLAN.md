# VaaS Visualization Plan

## Overview

Interactive visualization for document structure and knowledge graph exploration.

## Components

### 1. Document Viewer (Left Panel)
- PDF rendering with page navigation
- Highlight overlays for:
  - **Box anchors** (colored regions)
  - **Section boundaries** (divider lines)
  - **Element classifications** (body, header, list item)
- Click-to-select: clicking a region selects the corresponding graph node

### 2. Graph Viewer (Right Panel)
- Force-directed or hierarchical layout
- Node types distinguished by color/shape:
  - `doc_root` (diamond)
  - `box_section` (box shape)
  - `section` (rounded rectangle)
  - `paragraph` (ellipse)
- Edge types distinguished by style:
  - `parent_of` (solid arrow)
  - `references_box` (dashed arrow)
  - `excludes` (red dotted)
  - `includes` (green dotted)
- Click-to-highlight: selecting a node highlights the source region in document

### 3. Detail Panel (Bottom)
- Node properties (ID, type, text snippet)
- Edge list (incoming/outgoing)
- Provenance info (page, bbox, element_id)
- Validation status (which checks passed/failed)

## Implementation Options

### Option A: Streamlit App (Recommended for MVP)
```
pip install streamlit streamlit-pdf-viewer pyvis
```

**Pros:**
- Fastest to build (1-2 days)
- Interactive widgets out of the box
- Easy deployment

**Cons:**
- Limited PDF annotation capability
- Layout constraints

**File structure:**
```
src/vaas/viz/
├── __init__.py
├── app.py           # Streamlit entry point
├── document.py      # PDF rendering with highlights
├── graph.py         # pyvis graph generation
└── components.py    # Reusable UI components
```

### Option B: Dash + Cytoscape
```
pip install dash dash-cytoscape dash-pdf-viewer
```

**Pros:**
- More control over layout
- Better graph interactivity (cytoscape.js)

**Cons:**
- More boilerplate
- Steeper learning curve

### Option C: React Frontend (Full Product)
```
npx create-react-app vaas-viz --template typescript
npm install react-pdf cytoscape react-cytoscape
```

**Pros:**
- Full control
- Production-ready
- Best UX

**Cons:**
- Requires separate build pipeline
- 1-2 weeks development

## MVP Scope (Streamlit)

### Phase 1: Static Viewer
1. Load pipeline outputs (nodes_df, edges_df, anchors_df)
2. Display graph using pyvis
3. Show node details on click
4. Filter by node type / edge type

### Phase 2: Document Integration
1. Render PDF pages as images
2. Overlay anchor bounding boxes
3. Sync selection between graph and document

### Phase 3: Validation Dashboard
1. Run validators on loaded data
2. Display pass/fail status for each check
3. Show metrics and findings
4. Highlight problematic nodes/edges in graph

## Data Requirements

Visualization needs these pipeline outputs:
```python
{
    "nodes_df": pd.DataFrame,     # node_id, node_type, text, page, bbox
    "edges_df": pd.DataFrame,     # edge_id, source, target, edge_type
    "anchors_df": pd.DataFrame,   # anchor_id, page, geom_y0, geom_x0
    "elements_df": pd.DataFrame,  # element_id, text, role, anchor_id
    "pdf_path": str,              # Path to source PDF
}
```

## Launch Command

```bash
# After implementation
streamlit run src/vaas/viz/app.py -- --pdf data/i1099div.pdf --output output/
```

## Next Steps

1. Install Streamlit and pyvis
2. Create basic graph viewer
3. Add node type filtering
4. Integrate PDF page display
5. Add validation dashboard

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Phase 1 (Graph only) | 4-8 hours |
| Phase 2 (PDF sync) | 8-16 hours |
| Phase 3 (Validation) | 4-8 hours |

Total MVP: 2-4 days of focused development
