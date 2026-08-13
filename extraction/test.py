from extract_pdf import chunk

texts, tables, images = [], [], []

for chunk in chunk:
    chunk_type = type(chunk).__name__

    if chunk_type == "CompositeElement":
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


print(f"\n✅ Extracted: {len(texts)} texts, {len(tables)} tables, {len(images)} images")

# Print 1 sample from each (if available)
if texts:
    print("\n📝 Sample Text Chunk:")
    print(texts[0])

if tables:
    print("\n📊 Sample Table Chunk:")
    print(tables[0])

if images:
    print("\n🖼️ Sample Image (base64 preview):")
    print(images[0][:100] + "...")  # Print first 100 characters of base64 string


import base64
from IPython.display import Image, display

if images:
    with open("sample_image.jpg", "wb") as f:
        f.write(base64.b64decode(images[0]))
    print("\n🖼️ Image saved as 'sample_image.jpg' — open it to view.")
else:
    print("\nNo image found.")