import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def get_vectorstore(collection_name=None):
    """Return a Chroma store backed by a collection unique to this call.

    Chroma's in-memory client is process-global: two stores created with the
    same collection name share one collection, even from different Streamlit
    sessions. A fixed name would therefore mix every uploaded document together
    and expose one visitor's PDF to the next. Callers that genuinely want a
    stable collection, such as the evaluator, pass the name explicitly.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    resolved_name = collection_name or f"multi_modal_rag_{uuid.uuid4().hex}"
    return Chroma(collection_name=resolved_name, embedding_function=embeddings)


def release_vectorstore(vectorstore):
    """Drop a collection once its document is no longer needed.

    Each upload creates a new collection, so without this the process would
    accumulate every document indexed since start-up.
    """
    if vectorstore is None:
        return
    try:
        vectorstore.delete_collection()
    except Exception as error:  # pragma: no cover - cleanup must never block indexing
        print(f"[WARN] Could not release vector store collection: {error}")

# def test_vectorstore():
#     vectorstore = get_vectorstore()

#     documents = [
#         Document(page_content="Transformers are neural networks for sequence modeling."),
#         Document(page_content="LangChain enables LLM applications."),
#         Document(page_content="Sentence Transformers are great for embedding queries."),
#     ]
    
#     vectorstore.add_documents(documents)

#     query = "How do sentence transformers work?"
#     results = vectorstore.similarity_search(query, k=2)

#     print("🔍 Similar documents to query:")
#     for i, doc in enumerate(results):
#         print(f"{i+1}. {doc.page_content}")

# if __name__ == "__main__":
#     test_vectorstore()
