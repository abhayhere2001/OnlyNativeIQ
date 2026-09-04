# OnlyNativeIQ

Enterprise Knowledge & Customer Support Assistant for OnlyNative.

Supported knowledge-source formats:
- PDF
- DOCX
- TXT

Knowledge documents belong under `knowledge_base/`.

JSON/JSONL files under `data/metadata/` and `data/evaluation/`
are engineering-support artifacts and are not knowledge ingestion sources.

High-level flow:

Knowledge Base -> Load -> Parse -> Clean -> Metadata -> Chunk -> Embed
-> Chroma + BM25 -> Hybrid Retrieval -> Rerank -> Access Filter
-> Grounding -> LLM -> Citations -> Streamlit UI
