# import streamlit as st
# import sys
# import os
# import logging
# from tempfile import NamedTemporaryFile

# # Ensure the root directory is in the path so backend modules can be imported
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # 🧠 RAG and summary modules
# from extraction.extract_pdf import extract_pdf_elements
# from summarization.summarize_text_table import summarize_texts, summarize_tables
# from summarization.summarize_image import summarize_images
# from rag.vector_store import get_vectorstore
# from rag.retrieval import setup_retriever, store_documents
# from rag.rag_chain import get_rag_chain, get_rag_chain_with_sources
# from groq_api import TEXT_MODEL, query_groq

# # --- Streamlit Page Config ---
# st.set_page_config(page_title="MultiModal RAG", layout="centered")
# st.title("📄 MultiModal PDF QA App")

# # --- File Upload ---
# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# # --- Query Input ---
# query = st.text_input("Ask a question:")

# # --- RAG Mode Selection (shown only if a file is uploaded) ---
# rag_mode = None
# if uploaded_file:
#     rag_mode = st.radio("Select RAG Mode", ["Only response", "Response and source"])

# # --- Submit Button ---
# submit = st.button("Submit")

# # --- PDF Processor Function (previously in backend/api.py) ---
# def process_pdf(file_path_or_bytes):
#     if isinstance(file_path_or_bytes, bytes):
#         with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#             tmp.write(file_path_or_bytes)
#             file_path = tmp.name
#     else:
#         file_path = file_path_or_bytes

#     st.info("📚 Chunking PDF into smaller sections...")
#     chunks = extract_pdf_elements(file_path)

#     st.info("🧩 Splitting chunks into text, tables, and images...")
#     texts, tables, images = [], [], []
#     for chunk in chunks:
#         if type(chunk).__name__ == "CompositeElement":
#             texts.append(chunk)
#             orig_elements = getattr(chunk.metadata, "orig_elements", [])
#             for el in orig_elements:
#                 el_type = type(el).__name__
#                 if el_type == "Image":
#                     image_b64 = getattr(el.metadata, "image_base64", None)
#                     if image_b64:
#                         images.append(image_b64)
#                 elif el_type == "Table":
#                     tables.append(el)

#     st.info("📝 Preparing semantic summaries for each chunk...")
#     text_summaries = summarize_texts([t.text for t in texts])
#     table_summaries = summarize_tables([t.metadata.text_as_html for t in tables])
#     image_summaries = summarize_images(images)

#     print(len(text_summaries))
#     print(len(table_summaries))
#     print(len(image_summaries))

#     st.info("🗂️ Setting up the vector store with embeddings...")
#     vectorstore = get_vectorstore()
#     retriever = setup_retriever(vectorstore)

#     st.info("💾 Storing processed data into the database...")
#     store_documents(retriever, texts, text_summaries, "doc_id")
#     store_documents(retriever, tables, table_summaries, "doc_id")
#     store_documents(retriever, images, image_summaries, "doc_id")

#     return retriever

# # --- Main Logic ---
# if submit and query:
#     if uploaded_file:
#         st.info("🔧 Processing PDF and preparing vector store...")
#         retriever = process_pdf(uploaded_file.read())

#         st.info("🧠 Generating answer using selected RAG mode...")
#         if rag_mode == "Only response":
#             chain = get_rag_chain(retriever)
#             result = chain.invoke({"question": query})
#             st.success("✅ Response:")
#             st.markdown(result)

#         else:  # "Response and source"
#             chain = get_rag_chain_with_sources(retriever)
#             output = chain.invoke({"question": query})

#             response = output.get("response", "")
#             context = output.get("context", {})

#             # Display the response
#             st.success("✅ Response:")
#             st.markdown(response)

#             # Display textual context
#             text_context = context.get("texts", [])
#             if text_context:
#                 st.info("📚 Context (Text):")
#                 for i, doc in enumerate(text_context):
#                     try:
#                         content = doc.page_content
#                     except:
#                         content = str(doc)
#                     st.markdown(f"**Document {i+1}:**\n{content}")

#             # Display image context
#             image_context = context.get("images", [])
#             if image_context:
#                 st.info("🖼️ Context (Images):")
#                 for i, img_b64 in enumerate(image_context):
#                     if "," in img_b64:
#                         img_b64 = img_b64.split(",")[1]  # Strip data URI prefix
#                     st.image(f"data:image/jpeg;base64,{img_b64}", caption=f"Image {i+1}")

#     else:
#         # No file uploaded: fallback to Groq model
#         st.info("💬 No document provided. Using direct LLM query...")
#         payload = {
#             "model": TEXT_MODEL,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [{"type": "text", "text": query}]
#                 }
#             ]
#         }
#         result = query_groq(payload)
#         st.success("✅ Response:")
#         st.markdown(result)



import streamlit as st
import sys
import os
import re
from tempfile import NamedTemporaryFile

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extraction.extract_pdf import extract_pdf_elements
from summarization.summarize_text_table import summarize_texts, summarize_tables
from summarization.summarize_image import summarize_images
from rag.vector_store import get_vectorstore
from rag.retrieval import setup_retriever, store_documents
from rag.rag_chain import get_rag_chain, get_rag_chain_with_sources
from groq_api import TEXT_MODEL, query_groq

# --- Streamlit Page Config ---
st.set_page_config(page_title="MultiModal RAG", layout="centered")
st.title("📄 MultiModal PDF QA App")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False


def split_model_response(raw_response):
    """Return the final answer and any model reasoning as separate strings."""
    response = "" if raw_response is None else str(raw_response)
    reasoning_parts = []

    # Reasoning models commonly wrap their hidden work in one of these tags.
    # Remove only complete blocks so an unrelated, malformed response is not lost.
    pattern = re.compile(
        r"<(?:think|thinking|analysis)>\s*(.*?)\s*</(?:think|thinking|analysis)>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def collect_reasoning(match):
        reasoning = match.group(1).strip()
        if reasoning:
            reasoning_parts.append(reasoning)
        return ""

    answer = pattern.sub(collect_reasoning, response).strip()
    return answer, "\n\n".join(reasoning_parts)


def show_model_response(answer, reasoning=""):
    """Render the user-facing answer without mixing in model reasoning."""
    st.success("✅ Response:")
    st.markdown(answer or "The model did not return a final answer.")

    if reasoning:
        with st.expander("Model reasoning", expanded=False):
            st.markdown(reasoning)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question:")
rag_mode = st.radio("Select RAG Mode", ["Only response", "Response and source"]) if uploaded_file else None
submit = st.button("Submit")

# --- Preprocessing Function ---
def process_pdf(file_bytes, source_file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        file_path = tmp.name

    st.info("📚 Chunking PDF into smaller sections...")
    chunks = extract_pdf_elements(file_path)

    st.info("🧩 Splitting chunks into text, tables, and images...")
    texts, tables, images = [], [], []
    for chunk in chunks:
        if type(chunk).__name__ == "CompositeElement":
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

    st.info("📝 Preparing semantic summaries...")
    text_summaries = summarize_texts([t.text for t in texts])
    table_summaries = summarize_tables([t.metadata.text_as_html for t in tables])
    image_summaries = summarize_images(images)

    st.info("🗂️ Setting up the vector store...")
    vectorstore = get_vectorstore()
    retriever = setup_retriever(vectorstore)

    st.info("💾 Storing data in database...")
    store_documents(retriever, texts, text_summaries, "doc_id", source_file=source_file, modality="text")
    store_documents(retriever, tables, table_summaries, "doc_id", source_file=source_file, modality="table")
    store_documents(retriever, images, image_summaries, "doc_id", source_file=source_file, modality="image")

    return retriever

# --- Submit Logic ---
if submit and query:
    # If PDF was just uploaded for the first time
    if uploaded_file and not st.session_state.pdf_uploaded:
        st.session_state.retriever = process_pdf(uploaded_file.read(), uploaded_file.name)
        st.session_state.pdf_uploaded = True
        st.success("✅ PDF Processed and stored!")

    retriever = st.session_state.retriever

    if uploaded_file:
        # Try retrieving documents
        if rag_mode == "Only response":
            chain = get_rag_chain(retriever)
            output = chain.invoke({"question": query})
            raw_response = output
            sources = None
        else:
            chain = get_rag_chain_with_sources(retriever)
            output = chain.invoke({"question": query})
            raw_response = output.get("response", "")
            sources = output.get("context", {})

        response, reasoning = split_model_response(raw_response)

        # Check if relevant documents were retrieved
        if not response.strip():
            st.warning("⚠️ No relevant context found in PDF. Falling back to Groq.")
            payload = {
                "model": TEXT_MODEL,
                "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            }
            response, reasoning = split_model_response(query_groq(payload))

        # Keep only the final answer in the conversation context and history.
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", response))

        show_model_response(response, reasoning)

        # If source context exists, show it
        if sources:
            text_ctx = sources.get("texts", [])
            img_ctx = sources.get("images", [])

            if text_ctx:
                st.info("📚 Context (Text):")
                for i, doc in enumerate(text_ctx):
                    try:
                        content = doc.page_content
                    except:
                        content = str(doc)
                    st.markdown(f"**Text {i+1}:**\n{content}")

            if img_ctx:
                st.info("🖼️ Context (Images):")
                for i, img_b64 in enumerate(img_ctx):
                    if "," in img_b64:
                        img_b64 = img_b64.split(",")[1]
                    st.image(f"data:image/jpeg;base64,{img_b64}", caption=f"Image {i+1}")

    else:
        # No PDF uploaded: use Groq directly
        st.info("💬 No PDF found — using Groq to answer the question...")

        # Reconstruct conversation history (last 10 messages)
        st.info("📜 Building conversation context from history...")
        messages = []
        for role, msg in st.session_state.chat_history[-10:]:  # last 10 messages
            messages.append({"role": role, "content": [{"type": "text", "text": msg}]})

        # Append current user message at the end
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})

        # Prepare payload and call ChatGroq
        st.info("🤖 Querying Groq model for response...")
        payload = {"model": TEXT_MODEL, "messages": messages}
        response, reasoning = split_model_response(query_groq(payload))

        # Keep only the final answer in the conversation context and history.
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", response))

        show_model_response(response, reasoning)



# --- Show Chat History ---
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🕘 Chat History")

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            # Show only the first line of the response, full content inside an expander
            first_line = msg.strip().split('\n')[0]
            with st.expander(f"🤖 **Assistant:** {first_line}"):
                st.markdown(msg)
