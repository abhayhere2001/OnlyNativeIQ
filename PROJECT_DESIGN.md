# OnlyNativeIQ Project Design

- `knowledge_base/`: authoritative PDF/DOCX/TXT source documents
- `data/processed/`: normalized/intermediate data
- `data/metadata/`: corpus governance metadata
- `data/evaluation/`: evaluation datasets
- `src/`: core application logic
- `app/`: Streamlit UI
- `vector_store/`: persisted Chroma and BM25 indexes
- `scripts/`: command-line workflows
- `tests/`: automated tests

Access-control intent:
- Public: customer-safe product facts, brand story, FAQ, public policies
- Internal: employee handbook, code of conduct, production SOPs
- Confidential: detailed recipes/formulations/ingredient ratios

Access control should be enforced programmatically.
