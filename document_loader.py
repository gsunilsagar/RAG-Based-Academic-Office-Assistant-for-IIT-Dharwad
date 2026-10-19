
from pathlib import Path
import fitz  # PyMuPDF

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf_documents():
    """Extract text from every PDF in the data folder."""
    documents = []

    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data folder.")
        return documents

    for pdf_path in pdf_files:
        try:
            with fitz.open(str(pdf_path)) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()

                    if text:
                        documents.append({
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_number
                        })

            print("Loaded:", pdf_path.name)

        except Exception as error:
            print("Could not read", pdf_path.name, "-", error)

    return documents


def split_into_chunks(documents):
    """Split each page's text into overlapping character chunks."""
    chunks = []

    chunk_size = int(CHUNK_SIZE)
    overlap = int(CHUNK_OVERLAP)

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
        )

    for document in documents:
        text = document["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"]
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks


if __name__ == "__main__":
    pages = load_pdf_documents()
    chunks = split_into_chunks(pages)

    print("\nPages extracted:", len(pages))
    print("Text chunks created:", len(chunks))

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0]["text"][:300])
        print("Source:", chunks[0]["source"])
        print("Page:", chunks[0]["page"])
