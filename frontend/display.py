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
# from rag.vector_store import get_vectorstore, release_vectorstore
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
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extraction.extract_pdf import extract_pdf_elements
from summarization.summarize_text_table import summarize_texts, summarize_tables
from summarization.summarize_image import summarize_images
from rag.vector_store import get_vectorstore
from rag.retrieval import setup_retriever, store_documents
from rag.rag_chain import generate_answer, retrieve_context
from groq_api import is_configured
from inference import answer_chat, use_local_backend

# --- Streamlit Page Config ---
st.set_page_config(page_title="DocuMind", layout="centered")
st.title("📄 DocuMind — MultiModal PDF QA")

if not use_local_backend() and not is_configured():
    st.warning(
        "No GROQ_API_KEY is configured, so answers come from a limited offline "
        "fallback instead of the model. Set it in `.streamlit/secrets.toml` locally, "
        "or under App settings → Secrets once deployed."
    )

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Track which upload produced the current retriever so that selecting a
# different PDF rebuilds the index instead of answering from the previous one.
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None


@st.cache_resource(show_spinner=False)
def get_local_model_resource():
    """Keep local Gemma loaded across reruns, but only when it is the backend.

    On a hosted CPU tier the weights cannot be loaded at all, so this is a no-op
    unless a local backend was explicitly selected.
    """
    if not use_local_backend():
        return None

    from local_model import load_local_model

    return load_local_model()


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
retrieval_k = st.slider(
    "Sources to retrieve (k)", min_value=1, max_value=10, value=4,
    help="How many indexed elements are pulled in as context for each answer.",
) if uploaded_file else 4
submit = st.button("Submit")

# --- Preprocessing Function ---
@contextmanager
def temporary_pdf(file_bytes):
    """Write an upload to disk for the parser, then always remove it.

    ``unstructured`` needs a real path rather than a buffer, but the file must
    not outlive the parse: uploads would otherwise accumulate on the server for
    the lifetime of the container.
    """
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError as error:
            print(f"[WARN] Could not remove temporary upload: {error}")


def process_pdf(file_bytes, source_file):
    st.info("📚 Chunking PDF into smaller sections...")
    with temporary_pdf(file_bytes) as file_path:
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

    if use_local_backend():
        st.info("🧠 Loading local Gemma model...")
        get_local_model_resource()

    st.info("📝 Preparing semantic summaries... (one model call per element)")
    text_summaries = summarize_texts([t.text for t in texts])
    table_summaries = summarize_tables([t.metadata.text_as_html for t in tables])
    image_summaries = summarize_images(images)

    # A summary that failed is stored as an empty string and dropped from the
    # index, so report it rather than letting the document be silently partial.
    attempted = len(text_summaries) + len(table_summaries) + len(image_summaries)
    indexed = sum(
        1
        for summary in text_summaries + table_summaries + image_summaries
        if summary and summary.strip()
    )
    if indexed < attempted:
        st.warning(
            f"⚠️ {attempted - indexed} of {attempted} elements could not be summarised "
            "and were left out of the index, so answers may miss parts of this "
            "document. The app log gives the reason for each one — common causes are "
            "Groq rate limits (retry in a minute) and a configured model your API key "
            "cannot access (the log shows HTTP 404)."
        )

    st.info("🗂️ Setting up the vector store...")
    # Each upload gets its own collection, so drop the previous document's
    # collection rather than letting it accumulate for the life of the process.
    previous = st.session_state.get("retriever")
    if previous is not None:
        release_vectorstore(previous.vectorstore)

    vectorstore = get_vectorstore()
    retriever = setup_retriever(vectorstore)

    st.info("💾 Storing data in database...")
    store_documents(retriever, texts, text_summaries, "doc_id", source_file=source_file, modality="text")
    store_documents(retriever, tables, table_summaries, "doc_id", source_file=source_file, modality="table")
    store_documents(retriever, images, image_summaries, "doc_id", source_file=source_file, modality="image")

    return retriever

# --- Submit Logic ---
if submit and query:
    # Index on first use, and again whenever a different PDF is selected.
    if uploaded_file:
        file_key = (uploaded_file.name, uploaded_file.size)
        if st.session_state.indexed_file != file_key:
            st.session_state.retriever = process_pdf(uploaded_file.read(), uploaded_file.name)
            st.session_state.indexed_file = file_key
            st.success("✅ PDF Processed and stored!")

    retriever = st.session_state.retriever

    if uploaded_file:
        # Retrieve first, so an empty index is detected before a model call is
        # spent answering from no context at all.
        context = retrieve_context(retriever, query, k=retrieval_k)
        retrieved_texts = context.get("texts", [])
        retrieved_images = context.get("images", [])
        sources = context if rag_mode == "Response and source" else None

        if not retrieved_texts and not retrieved_images:
            st.warning(
                "⚠️ Nothing was retrieved from this PDF, so the answer below is not "
                "grounded in it. This usually means indexing was incomplete — "
                "re-upload the file and check for rate-limit warnings."
            )
            sources = None
            get_local_model_resource()
            response, reasoning = split_model_response(answer_chat([("user", query)]))
        else:
            get_local_model_resource()
            raw_response = generate_answer({"question": query, "context": context})
            response, reasoning = split_model_response(raw_response)

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
        # No PDF uploaded: answer directly from the chat history.
        st.info("💬 No PDF found — answering from the conversation directly...")

        # Reconstruct conversation history (last 10 messages)
        st.info("📜 Building conversation context from history...")
        messages = st.session_state.chat_history[-10:] + [("user", query)]

        st.info("🤖 Querying the model for a response...")
        get_local_model_resource()
        response, reasoning = split_model_response(answer_chat(messages))

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
