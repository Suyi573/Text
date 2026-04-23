from sentence_transformers import SentenceTransformer
from typing import List


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        if not query or not query.strip():
            return []

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()


if __name__ == "__main__":
    # Simple smoke test for the embedding module
    embedder = TextEmbedder()

    docs = [
        "Mediterranean cuisine emphasizes olive oil, vegetables, and legumes.",
        "Greek cuisine often includes feta cheese, olives, and grilled fish."
    ]
    query = "What foods are common in Mediterranean cuisine?"

    doc_vecs = embedder.embed_documents(docs)
    query_vec = embedder.embed_query(query)

    print(f"Number of document embeddings: {len(doc_vecs)}")
    print(f"Dimension of first document embedding: {len(doc_vecs[0])}")
    print(f"Query embedding dimension: {len(query_vec)}")