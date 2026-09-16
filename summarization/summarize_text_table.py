"""Text and table summarisation for retrieval indexing."""

from inference import InferenceError, summarize_text


def safe_batch(inputs):
    """Summarise valid inputs while preserving their original positions.

    A failed summary becomes an empty string rather than an error message, so
    ``store_documents`` drops it instead of embedding the failure text as a
    searchable document.
    """
    if not inputs:
        return []

    results = []
    for value in inputs:
        if not isinstance(value, str) or not value.strip():
            results.append("")
            continue
        try:
            results.append(summarize_text(value))
        except InferenceError as error:
            print(f"[WARN] Skipping summary: {error}")
            results.append("")
    return results


def summarize_texts(texts):
    return safe_batch(texts)


def summarize_tables(tables_html):
    return safe_batch(tables_html)
