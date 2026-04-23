import json
import re
from pathlib import Path
from typing import List, Dict

import numpy as np

from src.retrieval.embedder import TextEmbedder


def lexical_overlap_score(query: str, text: str) -> float:
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))

    if not query_words:
        return 0.0

    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words)


class Retriever:
    def __init__(
        self,
        index_path: str,
        model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    ):
        self.index_path = Path(index_path)
        self.embedder = TextEmbedder(model_name=model_name)
        self.index_data = self._load_index()

        self.chunk_ids = [item.get("chunk_id") for item in self.index_data]
        self.texts = [item.get("text", "") for item in self.index_data]
        self.metadata = [item.get("metadata", {}) for item in self.index_data]
        self.embeddings = np.array(
            [item["embedding"] for item in self.index_data],
            dtype=np.float32
        )

    def _load_index(self) -> List[Dict]:
        with open(self.index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query or not query.strip():
            return []

        query_embedding = np.array(
            self.embedder.embed_query(query),
            dtype=np.float32
        )

        # embeddings were normalized in embedder.py
        # so dot product here is equivalent to cosine similarity
        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(scores[idx])
            })

        return results

    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        lexical_weight: float = 0.2
    ) -> List[Dict]:
        candidates = self.retrieve(query, top_k=top_k)

        reranked = []
        for item in candidates:
            lex_score = lexical_overlap_score(query, item["text"])
            final_score = item["score"] + lexical_weight * lex_score

            new_item = dict(item)
            new_item["lexical_score"] = lex_score
            new_item["final_score"] = final_score
            reranked.append(new_item)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k]


def main():
    project_root = Path(__file__).resolve().parents[2]
    index_path = project_root / "data" / "retrieval" / "index_multiqa.json"

    retriever = Retriever(str(index_path))

    query = "What are common ingredients in Mediterranean cuisine?"
    results = retriever.retrieve_with_rerank(query, top_k=5, lexical_weight=0.35)

    print(f"Query: {query}")
    print(f"Top {len(results)} reranked results:")

    for i, item in enumerate(results, 1):
        print(f"\nResult {i}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Dense score: {item['score']:.4f}")
        print(f"Lexical score: {item['lexical_score']:.4f}")
        print(f"Final score: {item['final_score']:.4f}")
        print(f"Text: {item['text'][:300]}")
        print(f"Metadata: {item['metadata']}")


if __name__ == "__main__":
    main()