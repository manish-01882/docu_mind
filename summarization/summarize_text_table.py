"""Local text and table summarisation for retrieval indexing."""

from local_model import summarize_text


def safe_batch(inputs):
    """Summarise valid inputs while preserving their original positions."""
    if not inputs:
        return []

    results = []
    for value in inputs:
        if not isinstance(value, str) or not value.strip():
            results.append("")
            continue
        results.append(summarize_text(value))
    return results


def summarize_texts(texts):
    return safe_batch(texts)


def summarize_tables(tables_html):
    return safe_batch(tables_html)
