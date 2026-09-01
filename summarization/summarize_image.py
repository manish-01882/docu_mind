

import base64
from PIL import Image
from io import BytesIO
from groq_api import VISION_MODEL, query_groq

# Decode base64 to PIL Image
def decode_base64_to_image(image_b64):
    image_data = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_data)).convert("RGB")

# Encode PIL Image to base64 string
def encode_image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Summarize a single image
def get_image_summary(image_base64):
    try:
        image = decode_base64_to_image(image_base64)
        encoded_image = encode_image_to_base64(image)

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Summarize this image briefly."
                        }
                    ]
                }
            ]
        }

        return query_groq(payload)

    except Exception as e:
        return f"⚠️ Exception occurred: {str(e)}"


def is_indexable_image_summary(summary):
    """Exclude fallback/error messages from the vector index.

    ``query_groq`` deliberately returns a readable offline response when a key
    is unavailable. That response is useful to a caller, but it is not a
    description of an image and must not compete with real document summaries
    during retrieval.
    """
    if not isinstance(summary, str) or not summary.strip():
        return False
    return not summary.lstrip().startswith(("Offline fallback response", "⚠️ Exception occurred:"))


# Batch summarization (for a list of base64 strings)
def summarize_images(images_base64):
    summaries = [get_image_summary(img_b64) for img_b64 in images_base64]
    return [summary if is_indexable_image_summary(summary) else "" for summary in summaries]
