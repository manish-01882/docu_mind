from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
import uuid

def store_documents(retriever, elements, summaries, id_key):
    if not elements or not summaries:
        return

    pairs = [
        (element, summary)
        for element, summary in zip(elements, summaries)
        if summary and summary.strip()
    ]

    if not pairs:
        return

    ids = [str(uuid.uuid4()) for _ in pairs]

    # Create summary documents with metadata
    summary_docs = [
        Document(page_content=summary, metadata={id_key: ids[i]})
        for i, (_, summary) in enumerate(pairs)
    ]

    # # ✅ Debug print to inspect documents being stored
    # for i, doc in enumerate(summary_docs):
    #     print(f"[{i}] Page Content: {repr(doc.page_content)}, Metadata: {doc.metadata}")

    # ✅ Add summaries to vector store
    retriever.vectorstore.add_documents(summary_docs)

    # Store full elements WITH metadata (so it can show up in retrieval)
    retriever.docstore.mset([
        (
            ids[i],
            Document(
                page_content=getattr(element, "text", str(element)),
                metadata={id_key: ids[i]},
            ),
        )
        for i, (element, _) in enumerate(pairs)
    ])

def setup_retriever(vectorstore):
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    return retriever
