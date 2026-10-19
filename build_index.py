
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, INDEX_DIR, EMBEDDING_MODEL
from document_loader import load_pdf_documents, split_into_chunks


def build_index():
    print("Loading PDF documents...")
    pages = load_pdf_documents()
    chunks = split_into_chunks(pages)

    if not chunks:
        print("No text chunks found. Add PDFs to the data folder first.")
        return

    print("Loading embedding model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings for", len(texts), "chunks...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Normalize vectors so inner-product search acts like cosine similarity.
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "faiss.index"
    metadata_path = INDEX_DIR / "chunks.json"

    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print("\nIndex built successfully!")
    print("Chunks indexed:", index.ntotal)
    print("FAISS index:", index_path)
    print("Chunk metadata:", metadata_path)


if __name__ == "__main__":
    build_index()
