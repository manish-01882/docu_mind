from unstructured.partition.pdf import partition_pdf

file_path = "./data/attention_short.pdf"
def extract_pdf_elements(file_path: str, output_path: str = "./content"):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return chunks

chunk=extract_pdf_elements(file_path)
