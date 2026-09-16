"""PDF partitioning into text, table, and image elements."""

import os

from unstructured.partition.pdf import partition_pdf


# ``hi_res`` is what makes table structure and embedded images available, so it
# stays the default. Hosted CPU tiers with little RAM can fall back to "fast"
# via the environment, at the cost of losing image and table extraction.
DEFAULT_STRATEGY = "hi_res"


def extract_pdf_elements(file_path: str, strategy: str = None):
    """Partition a PDF into title-based chunks with tables and images inlined."""
    resolved_strategy = strategy or os.getenv("UNSTRUCTURED_STRATEGY", DEFAULT_STRATEGY)

    return partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy=resolved_strategy,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
