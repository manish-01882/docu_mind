"""Image summarisation for retrieval indexing."""

from inference import InferenceError, summarize_image


def get_image_summary(image_base64):
    try:
        return summarize_image(image_base64)
    except InferenceError as error:
        # A failed description must not become a searchable document.
        print(f"[WARN] Skipping image summary: {error}")
        return ""


def is_indexable_image_summary(summary):
    return isinstance(summary, str) and bool(summary.strip())


def summarize_images(images_base64):
    summaries = [get_image_summary(image_base64) for image_base64 in images_base64]
    return [summary if is_indexable_image_summary(summary) else "" for summary in summaries]
