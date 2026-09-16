# rag_chain.py
import sys
import os
from base64 import b64decode
from io import BytesIO
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from inference import answer_question
from PIL import Image, UnidentifiedImageError

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def is_image_base64(value):
    """Return whether a string decodes to a real image, not merely Base64 bytes."""
    if not isinstance(value, str) or not value.strip():
        return False
    payload = value.split(",", 1)[-1] if "," in value else value
    try:
        image_data = b64decode(payload, validate=True)
        with Image.open(BytesIO(image_data)) as image:
            image.verify()
    except (ValueError, UnidentifiedImageError):
        return False
    return True


# --------------------------
# ✅ Preprocessor
# --------------------------
def parse_docs(docs, docstore=None):
    b64, text = [], []
    for doc in docs:
        if isinstance(doc, str):
            if is_image_base64(doc):
                b64.append(doc)
            else:
                text.append(doc)
        else:
            doc_id = doc.metadata.get("doc_id")
            if doc_id and docstore:
                full_doc = docstore.mget([doc_id])[0]
                if full_doc is not None:
                    content = full_doc.page_content
                    if is_image_base64(content):
                        b64.append(content)
                    else:
                        text.append(full_doc)
                else:
                    text.append(doc)
            else:
                text.append(doc)
    return {"images": b64, "texts": text}



def retrieve_context(retriever, question, k=4):
    """Return retrieved context split into ``texts`` and ``images``.

    Exposed separately so a caller can see whether anything was retrieved
    before spending a model call on an answer.
    """
    return parse_docs(
        retriever.vectorstore.similarity_search(question, k=k),
        retriever.docstore,
    )


def generate_answer(kwargs):
    """Answer from the retrieved text and image context via the active backend."""
    context = kwargs["context"]
    return answer_question(
        kwargs["question"],
        texts=context["texts"],
        images=context["images"],
    )

# --------------------------
# ✅ RAG chain (Simple)
# --------------------------
def get_rag_chain(retriever, k=4):
    """Return the standard RAG chain using an explicit retrieval depth."""
    return (
        {
            "context": RunnableLambda(lambda x: retrieve_context(retriever, x["question"], k)),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | RunnableLambda(generate_answer)
        | StrOutputParser()
    )


# --------------------------
# ✅ RAG chain With Image Saving
# --------------------------
def get_rag_chain_with_sources(retriever, k=4):
    """Return the source-display RAG chain using an explicit retrieval depth."""
    def log_context_size(response_with_context):
        """Trace retrieval volume only.

        Document text and figures belong to the person who uploaded them, so
        they are never written to the server log or to disk.
        """
        context = response_with_context.get("context", {})
        print(
            f"[INFO] Retrieved {len(context.get('texts', []))} text chunk(s) "
            f"and {len(context.get('images', []))} image(s)."
        )
        return response_with_context

    return (
        {
            "context": RunnableLambda(lambda x: retrieve_context(retriever, x["question"], k)),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | RunnablePassthrough().assign(
            response=(RunnableLambda(generate_answer) | StrOutputParser())
        )
        | RunnableLambda(log_context_size)
    )
