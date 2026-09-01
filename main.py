import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from extraction.extract_pdf import extract_pdf_elements
from summarization.summarize_text_table import summarize_texts, summarize_tables
from summarization.summarize_image import summarize_images
from rag.vector_store import get_vectorstore
from rag.retrieval import setup_retriever, store_documents
from rag.rag_chain import get_rag_chain, get_rag_chain_with_sources

print("Libraries Imported.....")

# Load and extract
file_path = "./data/attention.pdf"
chunks = extract_pdf_elements(file_path)

texts, tables, images = [], [], []

for chunk in chunks:
    chunk_type = type(chunk).__name__

    if chunk_type == "CompositeElement":
        texts.append(chunk)

        orig_elements = getattr(chunk.metadata, "orig_elements", [])

        for el in orig_elements:
            el_type = type(el).__name__

            if el_type == "Image":
                image_b64 = getattr(el.metadata, "image_base64", None)
                if image_b64:
                    images.append(image_b64)

            elif el_type == "Table":
                tables.append(el)

print("Chunks made....")

# Summarize
text_summaries = summarize_texts([t.text for t in texts])
table_summaries = summarize_tables([t.metadata.text_as_html for t in tables])
image_summaries = summarize_images(images)


print("Summaries have been generated....")
print(len(text_summaries))
print(len(table_summaries))
print(len(image_summaries))

# Vector DB
vectorstore = get_vectorstore()
retriever = setup_retriever(vectorstore)

print("Database have been setup....")

store_documents(retriever, texts, text_summaries, "doc_id", source_file=file_path, modality="text")
print("text summaries uploaded....")
store_documents(retriever, tables, table_summaries, "doc_id", source_file=file_path, modality="table")
print("table summaries uploaded....")
store_documents(retriever, images, image_summaries, "doc_id", source_file=file_path, modality="image")
print("image summaries uploaded....")

print("Entries have been pushed in vector store.....")

# Run RAG
chain = get_rag_chain(retriever)
response = chain.invoke({"question": "Whaat are the points of ai email assistant?"})
print(response)
