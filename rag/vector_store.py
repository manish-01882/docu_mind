from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def get_vectorstore(collection_name="multi_modal_rag"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(collection_name=collection_name, embedding_function=embeddings)

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
