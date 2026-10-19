
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, EMBEDDING_MODEL, TOP_K


class LocalSearch:
    def __init__(self):
        index_path = INDEX_DIR / "faiss.index"
        metadata_path = INDEX_DIR / "chunks.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS files not found. Run build_index.py first."
            )

        print("Loading saved FAISS index...")
        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("Local search is ready.")

    def search(self, question, top_k=None):
        if not question or not question.strip():
            return []

        k = min(top_k or TOP_K, self.index.ntotal)

        if k <= 0:
            return []

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = self.chunks[int(idx)]

            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score)
            })

        return results


if __name__ == "__main__":
    searcher = LocalSearch()

    question = input("\nAsk an academic question: ")
    results = searcher.search(question)

    print("\n--- Search Results ---")

    for number, result in enumerate(results, start=1):
        print("\nResult", number)
        print("Similarity:", round(result["score"], 3))
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Text:", result["text"][:500])
