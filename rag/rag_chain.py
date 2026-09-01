# rag_chain.py
import sys
import os
import uuid
from base64 import b64decode
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from local_model import answer_question

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --------------------------
# ✅ Image Save Utility
# --------------------------
def save_image_if_relevant(image_b64, folder="saved_images", prefix="matched_image"):
    try:
        os.makedirs(folder, exist_ok=True)  # Ensure directory exists
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]  # Remove data URL prefix if present
        image_data = b64decode(image_b64)
        filename = os.path.join(folder, f"{prefix}_{uuid.uuid4().hex[:8]}.jpg")
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"[INFO] ✅ Image saved: {filename}")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to save image: {e}")

# --------------------------
# ✅ Preprocessor
# --------------------------
def parse_docs(docs, docstore=None):
    b64, text = [], []
    for doc in docs:
        if isinstance(doc, str):
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        else:
            doc_id = doc.metadata.get("doc_id")
            if doc_id and docstore:
                full_doc = docstore.mget([doc_id])[0]
                if full_doc is not None:
                    content = full_doc.page_content
                    try:
                        b64decode(content)
                        b64.append(content)
                    except Exception:
                        text.append(full_doc)
                else:
                    text.append(doc)
            else:
                text.append(doc)
    return {"images": b64, "texts": text}



def generate_answer(kwargs):
    """Run local Gemma against the retrieved text and image context."""
    context = kwargs["context"]
    return answer_question(
        kwargs["question"],
        texts=context["texts"],
        images=context["images"],
    )

# --------------------------
# ✅ Local Gemma chain (Simple)
# --------------------------
def get_rag_chain(retriever, k=4):
    """Return the standard RAG chain using an explicit retrieval depth."""
    return (
        {
            "context": RunnableLambda(
                lambda x: parse_docs(
                    retriever.vectorstore.similarity_search(x["question"], k=k),
                    retriever.docstore,
                )
            ),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | RunnableLambda(generate_answer)
        | StrOutputParser()
    )


# --------------------------
# ✅ Local Gemma chain With Image Saving
# --------------------------
def get_rag_chain_with_sources(retriever, k=4):
    """Return the source-display RAG chain using an explicit retrieval depth."""
    def process_and_save(response_with_context):
        context = response_with_context.get("context", {})
        images = context.get("images", [])
        texts = context.get("texts", [])
        response = response_with_context.get("response", "")

        # ✅ Print Response
        print("\n🧠 [RESPONSE]:\n", response)

        # ✅ Print Text Context
        print("\n📚 [TEXT CONTEXT]:")
        for i, doc in enumerate(texts):
            print(f"\n--- Text Document {i+1} ---")
            try:
                print(doc.page_content)
            except:
                print(doc)

        # ✅ Save all images, regardless of whether 'image' is in response
        if images:
            print(f"\n🖼️ [IMAGE CONTEXT]: {len(images)} image(s) found.")
            for idx, img in enumerate(images):
                save_image_if_relevant(img, prefix=f"matched_image_{idx}")
        else:
            print("[INFO] ❌ No images found in context.")

        return response_with_context


    return (
        {
            "context": RunnableLambda(
                lambda x: parse_docs(
                    retriever.vectorstore.similarity_search(x["question"], k=k),
                    retriever.docstore,
                )
            ),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | RunnablePassthrough().assign(
            response=(RunnableLambda(generate_answer) | StrOutputParser())
        )
        | RunnableLambda(process_and_save)
    )
