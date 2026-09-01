from hashlib import sha256

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore


def _element_text(element):
    """Return a stable string representation for any supported PDF element."""
    return getattr(element, "text", str(element))


def _source_metadata(element, *, doc_id, id_key, source_file, modality, chunk_index):
    """Build metadata used both for retrieval and for evaluation labels."""
    element_metadata = getattr(element, "metadata", None)
    detected_source = getattr(element_metadata, "filename", None)
    resolved_source = str(source_file or detected_source or "unknown")
    page_number = getattr(element_metadata, "page_number", None)

    # This is stable for a given document. It makes labels reusable across
    # evaluation runs, unlike a random UUID.
    source_material = ":".join(
        [
            resolved_source,
            str(modality),
            str(page_number or ""),
            str(chunk_index),
            _element_text(element),
        ]
    )
    source_id = sha256(source_material.encode("utf-8")).hexdigest()[:16]

    metadata = {
        id_key: doc_id,
        "source_id": source_id,
        "source_file": resolved_source,
        "modality": str(modality),
        "chunk_index": chunk_index,
    }
    if page_number is not None:
        metadata["page_number"] = page_number
    return metadata


def store_documents(
    retriever,
    elements,
    summaries,
    id_key,
    *,
    source_file=None,
    modality="text",
):
    """Store summary vectors and originals with traceable source metadata.

    ``doc_id`` links a summary vector to its original element. The remaining
    metadata supports source display and reproducible retrieval evaluation.
    """
    if not elements or not summaries:
        return []

    pairs = [
        (chunk_index, element, summary)
        for chunk_index, (element, summary) in enumerate(zip(elements, summaries))
        if summary and summary.strip()
    ]
    if not pairs:
        return []

    source_metadata = []
    for chunk_index, element, _ in pairs:
        doc_id = sha256(
            f"{source_file}:{modality}:{chunk_index}:{_element_text(element)}".encode("utf-8")
        ).hexdigest()
        source_metadata.append(
            _source_metadata(
                element,
                doc_id=doc_id,
                id_key=id_key,
                source_file=source_file,
                modality=modality,
                chunk_index=chunk_index,
            )
        )

    summary_docs = [
        Document(page_content=summary, metadata=metadata)
        for (_, _, summary), metadata in zip(pairs, source_metadata)
    ]
    retriever.vectorstore.add_documents(summary_docs)

    # Store full elements with identical metadata so source mode and the
    # evaluator can trace every result back to its PDF location.
    retriever.docstore.mset(
        [
            (
                metadata[id_key],
                Document(page_content=_element_text(element), metadata=metadata),
            )
            for (_, element, _), metadata in zip(pairs, source_metadata)
        ]
    )
    return summary_docs


def setup_retriever(vectorstore):
    store = InMemoryStore()
    return MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
