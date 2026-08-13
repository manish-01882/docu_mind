# rag_chain.py
import sys
import os
import uuid
from base64 import b64decode
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from perplexity_api import query_perplexity

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



# --------------------------
# ✅ Prompt Builder
# --------------------------
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = "".join([t.page_content for t in docs_by_type["texts"]])

    prompt_content = [
        {
            "type": "text",
            "text": f"Answer the question based only on the following context.\nContext:\n{context_text}\nQuestion: {user_question}"
        }
    ]

    for image in docs_by_type["images"]:
        prompt_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
        })

    return {
        "model": "sonar",  # Perplexity model
        "messages": [
            {
                "role": "user",
                "content": prompt_content
            }
        ]
    }

# --------------------------
# ✅ Perplexity Chain (Simple)
# --------------------------
def get_rag_chain(retriever):
    return (
        {
            "context": RunnableLambda(lambda x: retriever.vectorstore.similarity_search(x["question"])) | RunnableLambda(parse_docs),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(build_prompt)
        | RunnableLambda(query_perplexity)
        | StrOutputParser()
    )


# --------------------------
# ✅ Perplexity Chain With Image Saving
# --------------------------
def get_rag_chain_with_sources(retriever):
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
            "context": RunnableLambda(lambda x: parse_docs(retriever.vectorstore.similarity_search(x["question"]), retriever.docstore)),
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough().assign(
            response=(RunnableLambda(build_prompt) | RunnableLambda(query_perplexity) | StrOutputParser())
        )
        | RunnableLambda(process_and_save)
    )
