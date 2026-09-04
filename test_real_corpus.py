"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Manual Smoke Test
File         : test_real_corpus.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Performs a simple end-to-end loading test against the real OnlyNativeIQ
    knowledge_base directory.
================================================================================
"""

from src.ingestion.document_loader import DocumentLoader


def main() -> None:
    loader = DocumentLoader("knowledge_base")

    documents = loader.load_all()

    print()
    print("=" * 80)
    print("OnlyNativeIQ - Knowledge Base Loader Test")
    print("=" * 80)

    print(f"Total documents loaded: {len(documents)}")
    print()

    for document in documents:
        print("-" * 80)
        print(f"File       : {document.file_name}")
        print(f"Domain     : {document.domain}")
        print(f"Extension  : {document.file_extension}")
        print(f"Empty      : {document.is_empty}")
        print(f"Characters : {len(document.content)}")
        print(f"Path       : {document.metadata.get('relative_path')}")
        print()
        print("CONTENT PREVIEW")
        print(document.content[:300])
        print()


if __name__ == "__main__":
    main()